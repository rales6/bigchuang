"""Lightweight live map dashboard for the Raspberry Pi mapping process."""

from __future__ import annotations

from datetime import datetime
import copy
import json
import math
import socket
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import numpy as np

from raspberry_pi.mapping.map_visualization import encode_trajectory_png


DASHBOARD_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CAR·LAB 实车建图监控</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0b1118;
      --panel: #121b25;
      --panel-2: #172331;
      --line: #27384a;
      --text: #ecf4f7;
      --muted: #91a4b5;
      --cyan: #41d6c3;
      --orange: #ff9c5a;
      --green: #65d991;
      --red: #ff6b68;
      --blue: #65a5ff;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background:
        radial-gradient(circle at 80% 0%, rgba(39, 100, 112, .22), transparent 35%),
        var(--bg);
      color: var(--text);
      font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
    }
    main {
      width: min(1480px, calc(100% - 28px));
      margin: 0 auto;
      padding: 22px 0 34px;
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 20px;
      margin-bottom: 16px;
    }
    h1, h2, p { margin: 0; }
    h1 { font-size: clamp(23px, 3vw, 34px); letter-spacing: -.02em; }
    h2 { font-size: 15px; color: var(--muted); font-weight: 650; }
    .subtitle { color: var(--muted); margin-top: 5px; }
    .connection {
      display: flex;
      align-items: center;
      gap: 8px;
      white-space: nowrap;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: rgba(18, 27, 37, .8);
      padding: 7px 12px;
    }
    .dot {
      width: 9px;
      height: 9px;
      border-radius: 50%;
      background: var(--red);
      box-shadow: 0 0 12px currentColor;
    }
    .connection.online .dot { background: var(--green); }
    .layout {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      gap: 16px;
      align-items: start;
    }
    .panel {
      border: 1px solid var(--line);
      border-radius: 15px;
      background: rgba(18, 27, 37, .94);
      box-shadow: 0 16px 50px rgba(0, 0, 0, .24);
      overflow: hidden;
    }
    .panel-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      padding: 13px 16px;
      border-bottom: 1px solid var(--line);
    }
    .source {
      color: var(--cyan);
      font-size: 12px;
      letter-spacing: .04em;
      text-transform: uppercase;
    }
    .map-stage {
      position: relative;
      display: grid;
      place-items: center;
      min-height: min(75vh, 900px);
      background:
        linear-gradient(rgba(255,255,255,.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,.025) 1px, transparent 1px),
        #0d141d;
      background-size: 24px 24px;
      padding: 18px;
    }
    #mapImage {
      display: block;
      width: min(100%, 920px);
      height: auto;
      max-height: 78vh;
      object-fit: contain;
      image-rendering: pixelated;
      border: 1px solid #314252;
      box-shadow: 0 12px 38px rgba(0, 0, 0, .38);
    }
    .empty {
      position: absolute;
      max-width: 340px;
      text-align: center;
      color: var(--muted);
      padding: 20px;
    }
    .legend {
      display: flex;
      flex-wrap: wrap;
      gap: 14px;
      padding: 11px 16px;
      color: var(--muted);
      border-top: 1px solid var(--line);
    }
    .legend span::before {
      content: "";
      display: inline-block;
      width: 9px;
      height: 9px;
      margin-right: 6px;
      border-radius: 2px;
      background: var(--swatch);
    }
    aside { display: grid; gap: 12px; }
    .status-card { padding: 15px; }
    .badge {
      display: inline-flex;
      border-radius: 999px;
      padding: 4px 9px;
      margin-top: 9px;
      color: var(--green);
      background: rgba(101, 217, 145, .11);
      border: 1px solid rgba(101, 217, 145, .3);
      font-weight: 700;
    }
    .badge.warn {
      color: var(--orange);
      border-color: rgba(255, 156, 90, .35);
      background: rgba(255, 156, 90, .10);
    }
    .metrics {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1px;
      background: var(--line);
    }
    .metric {
      min-width: 0;
      padding: 13px 14px;
      background: var(--panel);
    }
    .metric label {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 2px;
    }
    .metric strong {
      display: block;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-size: 18px;
      font-variant-numeric: tabular-nums;
    }
    .detail-list { padding: 7px 15px; }
    .detail {
      display: grid;
      grid-template-columns: 104px minmax(0, 1fr);
      gap: 10px;
      padding: 8px 0;
      border-bottom: 1px solid rgba(39, 56, 74, .7);
    }
    .detail:last-child { border-bottom: 0; }
    .detail span:first-child { color: var(--muted); }
    .detail span:last-child {
      text-align: right;
      overflow-wrap: anywhere;
      font-variant-numeric: tabular-nums;
    }
    footer {
      color: var(--muted);
      margin-top: 14px;
      text-align: center;
      font-size: 12px;
    }
    @media (max-width: 980px) {
      .layout { grid-template-columns: 1fr; }
      aside { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .map-stage { min-height: 58vh; }
    }
    @media (max-width: 650px) {
      main { width: min(100% - 18px, 1480px); padding-top: 14px; }
      header { display: block; }
      .connection { width: fit-content; margin-top: 11px; }
      aside { grid-template-columns: 1fr; }
      .map-stage { min-height: 48vh; padding: 8px; }
      .panel-head { align-items: flex-start; }
    }
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>实车建图 · Live SLAM</h1>
      <p class="subtitle">显示树莓派 LidarSlam 的实际估计结果，不使用仿真真值</p>
    </div>
    <div id="connection" class="connection">
      <span class="dot"></span><span id="connectionText">正在连接</span>
    </div>
  </header>

  <div class="layout">
    <section class="panel">
      <div class="panel-head">
        <h2>占据栅格与估计轨迹</h2>
        <span class="source">Raspberry Pi · N10 · ICP</span>
      </div>
      <div class="map-stage">
        <img id="mapImage" alt="LidarSlam 实时占据栅格地图" hidden>
        <div id="emptyState" class="empty">
          正在等待第一帧有效雷达数据。地图将在算法完成首次融合后出现。
        </div>
      </div>
      <div class="legend">
        <span style="--swatch:#171717">占用/墙体</span>
        <span style="--swatch:#fefefe">自由空间</span>
        <span style="--swatch:#cdcdcd">未知区域</span>
        <span style="--swatch:#dc1e1e">估计轨迹</span>
        <span style="--swatch:#14b428">起点</span>
        <span style="--swatch:#1e5ae6">当前位置</span>
      </div>
    </section>

    <aside>
      <section class="panel status-card">
        <h2>自主状态</h2>
        <div id="motionState" class="badge warn">等待数据</div>
        <div class="detail-list" style="padding:10px 0 0">
          <div class="detail"><span>状态说明</span><span id="motionReason">—</span></div>
          <div class="detail"><span>地图门控</span><span id="mapStatus">—</span></div>
          <div class="detail"><span>更新时间</span><span id="updatedAt">—</span></div>
        </div>
      </section>

      <section class="panel metrics">
        <div class="metric"><label>扫描帧数</label><strong id="scanCount">0</strong></div>
        <div class="metric"><label>接受率</label><strong id="acceptance">—</strong></div>
        <div class="metric"><label>地图融合</label><strong id="integrated">0</strong></div>
        <div class="metric"><label>匹配 RMSE</label><strong id="rmse">—</strong></div>
        <div class="metric"><label>已观测格</label><strong id="observed">0</strong></div>
        <div class="metric"><label>占用格</label><strong id="occupied">0</strong></div>
      </section>

      <section class="panel detail-list">
        <div class="detail"><span>估计位姿</span><span id="pose">—</span></div>
        <div class="detail"><span>控制命令</span><span id="command">—</span></div>
        <div class="detail"><span>测量速度</span><span id="velocity">—</span></div>
        <div class="detail"><span>匹配点数</span><span id="matches">—</span></div>
        <div class="detail"><span>内点比例</span><span id="inliers">—</span></div>
        <div class="detail"><span>运行时间</span><span id="elapsed">—</span></div>
        <div class="detail"><span>地图尺寸</span><span id="mapSize">—</span></div>
      </section>
    </aside>
  </div>
  <footer>实时画面仅用于监控；底盘急停与短 TTL 安全保护不依赖本网页。</footer>
</main>
<script>
  const ids = {};
  for (const id of [
    "connection", "connectionText", "mapImage", "emptyState", "motionState",
    "motionReason", "mapStatus", "updatedAt", "scanCount", "acceptance",
    "integrated", "rmse", "observed", "occupied", "pose", "command",
    "velocity", "matches", "inliers", "elapsed", "mapSize"
  ]) ids[id] = document.getElementById(id);

  let lastMapVersion = -1;
  const valid = (value) =>
    value !== null && value !== undefined && value !== ""
    && Number.isFinite(Number(value));
  const number = (value, digits = 2, fallback = "—") =>
    valid(value) ? Number(value).toFixed(digits) : fallback;
  const integer = (value) =>
    valid(value) ? Math.round(Number(value)).toLocaleString() : "—";

  function text(id, value) { ids[id].textContent = value; }

  function updateView(data) {
    ids.connection.classList.add("online");
    text("connectionText", "实时数据在线");
    text("scanCount", integer(data.scan_count || 0));
    text("acceptance", valid(data.accepted_ratio)
      ? `${(Number(data.accepted_ratio) * 100).toFixed(1)}%` : "—");
    text("integrated", integer(data.mapping?.integrated_count || 0));
    text("rmse", valid(data.rmse_m)
      ? `${number(data.rmse_m, 3)} m` : "—");
    text("observed", integer(data.map?.observed_cells || 0));
    text("occupied", integer(data.map?.occupied_cells || 0));

    const pose = data.pose || {};
    text("pose", valid(pose.x_m)
      ? `x ${number(pose.x_m)} · y ${number(pose.y_m)} · ${number(pose.yaw_deg, 1)}°`
      : "—");
    const command = data.command || {};
    text("command", valid(command.linear_mm_s)
      ? `${integer(command.linear_mm_s)} mm/s · ${integer(command.angular_mrad_s)} mrad/s`
      : "停车");
    text("velocity", `${number(data.linear_speed_m_s)} m/s · ${number(data.angular_speed_rad_s)} rad/s`);
    text("matches", `${integer(data.correspondences)} / ${integer(data.scan_points)} 点`);
    text("inliers", valid(data.inlier_ratio)
      ? `${(Number(data.inlier_ratio) * 100).toFixed(0)}%` : "—");
    text("elapsed", `${number(data.elapsed_s, 1)} s`);
    text("mapSize", data.map
      ? `${integer(data.map.width_cells)} × ${integer(data.map.height_cells)} · ${number(data.map.resolution_m, 3)} m/格`
      : "—");
    text("motionReason", command.reason || data.rejection_reason || "—");
    text("mapStatus", data.map_status || "—");
    text("updatedAt", data.updated_at || "—");

    const state = command.state || (data.running ? "定位中" : "已停止");
    text("motionState", state);
    ids.motionState.classList.toggle("warn",
      !data.accepted || state.includes("blocked") || state.includes("stopped"));

    if (data.map_version > lastMapVersion) {
      lastMapVersion = data.map_version;
      ids.mapImage.onload = () => {
        ids.mapImage.hidden = false;
        ids.emptyState.hidden = true;
      };
      ids.mapImage.src = `/api/map.png?v=${data.map_version}`;
    }
  }

  async function refresh() {
    try {
      const response = await fetch("/api/status", { cache: "no-store" });
      if (!response.ok) throw new Error("status unavailable");
      updateView(await response.json());
    } catch (_error) {
      ids.connection.classList.remove("online");
      text("connectionText", "数据连接中断");
    }
  }
  refresh();
  setInterval(refresh, 500);
</script>
</body>
</html>
"""


def _json_safe(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


class _LiveMapStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._status = {
            "running": False,
            "scan_count": 0,
            "accepted_count": 0,
            "rejected_count": 0,
            "accepted_ratio": 0.0,
        }
        self._png = None
        self._map_version = 0
        self._status_version = 0

    def publish(self, status, png=None):
        with self._lock:
            self._status.update(_json_safe(status))
            self._status["updated_at"] = datetime.now().astimezone().isoformat(
                timespec="seconds"
            )
            self._status_version += 1
            if png is not None:
                self._png = bytes(png)
                self._map_version += 1

    def publish_png(self, png):
        with self._lock:
            self._png = bytes(png)
            self._map_version += 1

    def status(self):
        with self._lock:
            result = dict(self._status)
            result["map_version"] = self._map_version
            result["status_version"] = self._status_version
            result["map_available"] = self._png is not None
            return result

    def png(self):
        with self._lock:
            return self._png


class _LiveMapHttpServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class RealtimeMapServer:
    """Serve in-memory LidarSlam map snapshots to a local dashboard."""

    def __init__(self, bind="0.0.0.0", port=8766, refresh_hz=2.0):
        self.bind = str(bind)
        self.requested_port = int(port)
        self.refresh_hz = float(refresh_hz)
        if not 1 <= self.requested_port <= 65535 and self.requested_port != 0:
            raise ValueError("port must be 0 or within 1..65535")
        if not 0.2 <= self.refresh_hz <= 10.0:
            raise ValueError("refresh_hz must be within 0.2..10.0")
        self._minimum_render_interval_s = 1.0 / self.refresh_hz
        self._last_render_s = float("-inf")
        self._store = _LiveMapStore()
        self._httpd = None
        self._thread = None
        self._render_thread = None
        self._render_condition = threading.Condition()
        self._render_job = None
        self._render_stopping = False

    @property
    def port(self):
        if self._httpd is None:
            return self.requested_port
        return int(self._httpd.server_address[1])

    @property
    def local_url(self):
        return f"http://127.0.0.1:{self.port}"

    @property
    def network_url(self):
        host = self.bind
        if host in ("0.0.0.0", "::"):
            host = _preferred_local_ip()
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        return f"http://{host}:{self.port}"

    def start(self):
        if self._httpd is not None:
            return self.network_url
        store = self._store

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                path = urlparse(self.path).path
                if path == "/":
                    self._send(
                        HTTPStatus.OK,
                        DASHBOARD_HTML.encode("utf-8"),
                        "text/html; charset=utf-8",
                    )
                    return
                if path == "/api/status":
                    payload = json.dumps(
                        store.status(),
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                    self._send(
                        HTTPStatus.OK,
                        payload,
                        "application/json; charset=utf-8",
                    )
                    return
                if path == "/api/map.png":
                    png = store.png()
                    if png is None:
                        self._send(
                            HTTPStatus.SERVICE_UNAVAILABLE,
                            "地图尚未生成".encode("utf-8"),
                            "text/plain; charset=utf-8",
                        )
                    else:
                        self._send(HTTPStatus.OK, png, "image/png")
                    return
                if path == "/api/health":
                    self._send(
                        HTTPStatus.OK,
                        b'{"ok":true}',
                        "application/json",
                    )
                    return
                if path == "/favicon.ico":
                    self.send_response(HTTPStatus.NO_CONTENT)
                    self.end_headers()
                    return
                self._send(
                    HTTPStatus.NOT_FOUND,
                    b"not found",
                    "text/plain; charset=utf-8",
                )

            def _send(self, status, body, content_type):
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, _format, *_args):
                return

        self._httpd = _LiveMapHttpServer(
            (self.bind, self.requested_port),
            Handler,
        )
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="realtime-map-http",
            daemon=True,
        )
        self._thread.start()
        self._render_stopping = False
        self._render_thread = threading.Thread(
            target=self._render_loop,
            name="realtime-map-render",
            daemon=True,
        )
        self._render_thread.start()
        return self.network_url

    def publish(self, slam, status, force=False):
        """Publish metrics now and queue PNG work outside the control loop."""
        now = time.monotonic()
        should_render = (
            force
            or self._store.png() is None
            or now - self._last_render_s >= self._minimum_render_interval_s
        )
        grid = slam.grid
        occupied_threshold = math.log(0.65 / 0.35)
        map_status = {
            "map": {
                "width_cells": int(grid.width),
                "height_cells": int(grid.height),
                "resolution_m": float(grid.resolution_m),
                "origin_x_m": float(grid.origin_x_m),
                "origin_y_m": float(grid.origin_y_m),
                "observed_cells": int(np.count_nonzero(grid.observed)),
                "occupied_cells": int(np.count_nonzero(
                    grid.observed
                    & (grid.log_odds >= occupied_threshold)
                )),
                "trajectory_samples": len(slam.trajectory),
            }
        }
        self._store.publish({**status, **map_status})
        if not should_render:
            return False

        # Copy only the mutable arrays used by rendering. The copy is quick
        # and keeps the worker isolated while SLAM continues expanding and
        # updating the live grid in the control thread.
        grid_snapshot = copy.copy(grid)
        grid_snapshot.log_odds = grid.log_odds.copy()
        grid_snapshot.observed = grid.observed.copy()
        grid_snapshot._significant_obstacles = (
            grid._significant_obstacles.copy()
        )
        trajectory_snapshot = tuple(slam.trajectory)
        with self._render_condition:
            # Keep only the newest pending frame. A slow Raspberry Pi should
            # drop stale dashboard frames, never queue seconds of old work.
            self._render_job = (
                grid_snapshot,
                trajectory_snapshot,
            )
            self._render_condition.notify()
        self._last_render_s = now
        return True

    def _render_loop(self):
        while True:
            with self._render_condition:
                while (
                    self._render_job is None
                    and not self._render_stopping
                ):
                    self._render_condition.wait()
                if (
                    self._render_job is None
                    and self._render_stopping
                ):
                    return
                grid, trajectory = self._render_job
                self._render_job = None
            try:
                png = encode_trajectory_png(grid, trajectory)
                self._store.publish_png(png)
            except Exception:
                # Dashboard rendering must never stop localization or motion.
                continue

    def stop(self):
        if self._httpd is None:
            return
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        with self._render_condition:
            self._render_stopping = True
            self._render_condition.notify_all()
        if self._render_thread is not None:
            self._render_thread.join(timeout=2.0)
        self._httpd = None
        self._thread = None
        self._render_thread = None


def _preferred_local_ip():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        address = sock.getsockname()[0]
        if address:
            return address
    except OSError:
        pass
    finally:
        sock.close()
    try:
        return socket.gethostbyname(socket.gethostname())
    except OSError:
        return "127.0.0.1"
