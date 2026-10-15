#!/bin/sh
# 3090 pursuit server lifecycle + per-start log archive.
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ -f "$SCRIPT_DIR/pursuit_server.py" ]; then
    # 3090 部署目录：脚本与 pursuit_server.py 均位于 ~/track。
    TRACK_ROOT=$SCRIPT_DIR
else
    # 本地仓库布局：二者位于 track/3090。
    TRACK_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
fi
PYTHON_BIN=${PURSUIT_PYTHON:-/home/sjtu/3090/.venv/bin/python}
PORT=${PURSUIT_PORT:-8021}
DEVICE=${PURSUIT_DEVICE:-cuda}
LOG_ROOT=${PURSUIT_SERVER_LOG_DIR:-$TRACK_ROOT/logs/server}
PID_FILE="$LOG_ROOT/current.pid"
URL="http://127.0.0.1:$PORT"

mkdir -p "$LOG_ROOT"

health_ok() {
    curl -fsS --max-time 2 "$URL/health" >/dev/null 2>&1
}

matching_pids() {
    pgrep -u "$(id -u)" -f "pursuit_server.py.*--port $PORT" 2>/dev/null || true
}

start_server() {
    if health_ok; then
        echo "[server] already healthy: $URL/"
        return 0
    fi

    stamp=$(date +%Y%m%d_%H%M%S)
    run_dir="$LOG_ROOT/$stamp"
    mkdir -p "$run_dir"
    printf '%s\n' "$run_dir" > "$LOG_ROOT/LATEST.txt"
    ln -sfn "$stamp" "$LOG_ROOT/latest" 2>/dev/null || true
    {
        echo "started_at=$(date --iso-8601=seconds)"
        echo "hostname=$(hostname)"
        echo "port=$PORT"
        echo "python=$PYTHON_BIN"
        printf 'device=%s\n' "$DEVICE"
        printf 'depth=disabled\n'
        printf 'command=%s %s --port %s --device %s --no-depth' \
            "$PYTHON_BIN" "$SCRIPT_DIR/pursuit_server.py" "$PORT" "$DEVICE"
        printf ' %s' "$@"
        printf '\n'
    } > "$run_dir/run.env"

    nohup "$PYTHON_BIN" "$SCRIPT_DIR/pursuit_server.py" \
        --port "$PORT" --device "$DEVICE" --no-depth "$@" \
        > "$run_dir/server.log" 2>&1 </dev/null &
    pid=$!
    printf '%s\n' "$pid" > "$PID_FILE"
    printf '%s\n' "$pid" > "$run_dir/pid"
    echo "[server] starting pid=$pid log=$run_dir/server.log"

    attempts=0
    while [ "$attempts" -lt 30 ]; do
        if health_ok; then
            echo "[server] healthy: $URL/"
            return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "[server] exited during startup; inspect $run_dir/server.log" >&2
            return 1
        fi
        attempts=$((attempts + 1))
        sleep 1
    done
    echo "[server] health timeout; process is still running, inspect $run_dir/server.log" >&2
    return 1
}

stop_server() {
    pids=$(matching_pids)
    if [ -z "$pids" ]; then
        rm -f "$PID_FILE"
        echo "[server] not running"
        return 0
    fi
    echo "[server] stopping pid(s): $pids"
    for pid in $pids; do
        kill "$pid" 2>/dev/null || true
    done
    attempts=0
    while [ "$attempts" -lt 15 ]; do
        [ -z "$(matching_pids)" ] && break
        attempts=$((attempts + 1))
        sleep 1
    done
    remaining=$(matching_pids)
    if [ -n "$remaining" ]; then
        # uvicorn 会等待 MJPEG 长连接关闭，浏览器不退出时可能永久卡在
        # "Waiting for connections to close"。这里只强制结束已经通过
        # pursuit_server.py + 指定端口匹配验证的本用户进程。
        echo "[server] graceful shutdown timed out; force-stopping: $remaining"
        for pid in $remaining; do
            cmd=$(tr '\000' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)
            case "$cmd" in
                *pursuit_server.py*--port*"$PORT"*) kill -KILL "$pid" 2>/dev/null || true ;;
                *) echo "[server] skip unvalidated pid=$pid cmd=$cmd" >&2 ;;
            esac
        done
        sleep 1
    fi
    if [ -n "$(matching_pids)" ]; then
        echo "[server] process still exists after validated force-stop" >&2
        return 1
    fi
    rm -f "$PID_FILE"
    echo "[server] stopped"
}

show_status() {
    if health_ok; then
        echo "[server] healthy: $URL/"
        curl -fsS "$URL/health"
        echo
        echo "[server] pid(s): $(matching_pids)"
    else
        echo "[server] unavailable: $URL/"
        pids=$(matching_pids)
        [ -z "$pids" ] || echo "[server] process exists but health failed: $pids"
        return 1
    fi
    [ ! -f "$LOG_ROOT/LATEST.txt" ] || echo "[server] latest log: $(cat "$LOG_ROOT/LATEST.txt")/server.log"
}

follow_logs() {
    if [ ! -f "$LOG_ROOT/LATEST.txt" ]; then
        echo "[server] no archived logs under $LOG_ROOT" >&2
        return 1
    fi
    run_dir=$(cat "$LOG_ROOT/LATEST.txt")
    echo "[server] following $run_dir/server.log"
    tail -n 100 -F "$run_dir/server.log"
}

action=${1:-status}
[ "$#" -eq 0 ] || shift
case "$action" in
    start) start_server "$@" ;;
    stop) stop_server ;;
    restart) stop_server; start_server "$@" ;;
    status) show_status ;;
    logs) follow_logs ;;
    *)
        echo "usage: $0 {start|stop|restart|status|logs} [server arguments]" >&2
        exit 2
        ;;
esac
