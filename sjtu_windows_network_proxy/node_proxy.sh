#!/usr/bin/env bash
# 必须用 source 加载：
#   source ~/node_proxy_v2.sh
#
# Windows 端先运行 windows_start_proxy_v2.ps1，并保持窗口开启。

PROXY_HOST="127.0.0.1"
PROXY_USER="${WINDOWS_PROXY_USER:-raleproxy}"
PROXY_PORT_FILE="${HOME}/.windows_proxy_port"

_load_proxy_port() {
    if [[ -n "${WINDOWS_PROXY_PORT:-}" ]]; then
        PROXY_PORT="${WINDOWS_PROXY_PORT}"
    elif [[ -r "${PROXY_PORT_FILE}" ]]; then
        PROXY_PORT="$(tr -d '[:space:]' < "${PROXY_PORT_FILE}")"
    else
        PROXY_PORT="8899"
    fi

    if [[ ! "${PROXY_PORT}" =~ ^[0-9]+$ ]] ||
       (( PROXY_PORT < 1024 || PROXY_PORT > 65535 )); then
        echo "无效的代理端口：${PROXY_PORT}" >&2
        return 1
    fi
}

_proxy_port_open() {
    _load_proxy_port || return 1
    (exec 3<>"/dev/tcp/${PROXY_HOST}/${PROXY_PORT}") >/dev/null 2>&1
}

_proxy_require_enabled() {
    if [[ -z "${http_proxy:-}" || -z "${https_proxy:-}" ]]; then
        echo "代理尚未启用。请先运行：proxy_on"
        return 1
    fi
}

proxy_on() {
    _load_proxy_port || return 1

    if ! _proxy_port_open; then
        echo "节点上的 ${PROXY_HOST}:${PROXY_PORT} 尚未监听。"
        echo "请确认 Windows 的 windows_start_proxy_v2.ps1 正在运行。"
        echo "当前端口来自：${PROXY_PORT_FILE}"
        return 1
    fi

    local proxy_password
    read -r -s -p "请输入与 Windows 脚本相同的临时代理密码：" proxy_password
    echo

    if [[ ! "${proxy_password}" =~ ^[A-Za-z0-9._-]{8,64}$ ]]; then
        echo "密码格式不符合要求。"
        unset proxy_password
        return 1
    fi

    local proxy_url="http://${PROXY_USER}:${proxy_password}@${PROXY_HOST}:${PROXY_PORT}"

    export http_proxy="${proxy_url}"
    export https_proxy="${proxy_url}"
    export HTTP_PROXY="${proxy_url}"
    export HTTPS_PROXY="${proxy_url}"
    export no_proxy="localhost,127.0.0.1,::1,192.168.55.0/24"
    export NO_PROXY="${no_proxy}"

    unset proxy_password
    unset proxy_url

    echo "当前终端已启用 Windows 网络代理：${PROXY_HOST}:${PROXY_PORT}"
    echo "请先运行 proxy_test，再运行 apt_update_via_windows。"
}

proxy_off() {
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
    echo "当前终端的代理环境变量已清除。"
}

proxy_status() {
    _load_proxy_port || return 1

    echo "端口配置文件：${PROXY_PORT_FILE}"
    echo "远程代理地址：${PROXY_HOST}:${PROXY_PORT}"

    if [[ -n "${http_proxy:-}" ]]; then
        echo "代理环境变量：已启用"
    else
        echo "代理环境变量：未启用"
    fi

    if _proxy_port_open; then
        echo "SSH 反向端口：正在监听"
    else
        echo "SSH 反向端口：未监听"
    fi
}

proxy_test() {
    _proxy_require_enabled || return 1

    echo "正在通过 Windows 测试 HTTPS..."
    wget -4 --spider --timeout=15 --tries=1 https://www.baidu.com
    local result=$?

    if [[ ${result} -eq 0 ]]; then
        echo "测试成功：节点已能通过 Windows 访问外部网络。"
    else
        echo "测试失败。请检查 Windows 脚本窗口、代理密码和 SSH 隧道。"
    fi

    return ${result}
}

apt_update_via_windows() {
    _proxy_require_enabled || return 1

    sudo apt-get \
        -o Acquire::ForceIPv4=true \
        -o "Acquire::http::Proxy=${http_proxy}" \
        -o "Acquire::https::Proxy=${https_proxy}" \
        update
}

apt_install_via_windows() {
    _proxy_require_enabled || return 1

    if [[ $# -eq 0 ]]; then
        echo "用法：apt_install_via_windows 软件包1 [软件包2 ...]"
        return 2
    fi

    sudo apt-get \
        -o Acquire::ForceIPv4=true \
        -o "Acquire::http::Proxy=${http_proxy}" \
        -o "Acquire::https::Proxy=${https_proxy}" \
        install -y "$@"
}

pip_install_via_windows() {
    _proxy_require_enabled || return 1

    if [[ $# -eq 0 ]]; then
        echo "用法：pip_install_via_windows Python包1 [Python包2 ...]"
        return 2
    fi

    python3 -m pip install "$@"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "不要直接执行 bash node_proxy_v2.sh。"
    echo "请使用：source ~/node_proxy_v2.sh"
    exit 1
fi

case "${1:-on}" in
    on)
        proxy_on
        ;;
    off)
        proxy_off
        ;;
    status)
        proxy_status
        ;;
    *)
        echo "可用方式："
        echo "  source ~/node_proxy_v2.sh"
        echo "  source ~/node_proxy_v2.sh off"
        echo "  source ~/node_proxy_v2.sh status"
        return 2
        ;;
esac
