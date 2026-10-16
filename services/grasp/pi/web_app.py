"""Robot Arm Web Control — 树莓派端"""
import base64, json, os, queue, re, subprocess, threading, time
from pathlib import Path
from urllib import request as urlrequest
from urllib.error import HTTPError
import cv2, numpy as np
from flask import Flask, Response, render_template_string, request, jsonify

try:
    from .structured_log import EventLogger
except ImportError:
    from structured_log import EventLogger

EVENT_LOG = EventLogger("pi_web", "web_app.jsonl")

NUS_RX = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"
BLE_ADDR = "AA:BB:CC:DD:EE:01"
_ble_q = queue.Queue()
_ble_ok = False
_ARM_STATE_TRACKER = None

class _BleWriteRequest:
    def __init__(self, command):
        self.command = command
        self.done = threading.Event()
        self.ok = False
        self.error = ""
        self.cancelled = False

# ── BLE ───────────────────────────────────────────────────
def _ble(addr):
    global _ble_ok
    import asyncio
    from bleak import BleakScanner, BleakClient
    async def r():
        global _ble_ok
        while True:
            c = None
            was_connected = False
            try:
                d = await BleakScanner.discover(timeout=4)
                for x in d:
                    if x.address.upper() == addr.upper():
                        c = BleakClient(x.address, timeout=8); await c.connect(); break
                if c is None: c = BleakClient(addr, timeout=8); await c.connect()
            except Exception as exc:
                EVENT_LOG.throttled("ble_connect_error", 5.0, "ble_connect_error", error=repr(exc))
            if c and c.is_connected:
                _ble_ok = True
                was_connected = True
                EVENT_LOG.event("ble_connected", address=addr)
                while c.is_connected:
                    try:
                        item = _ble_q.get(timeout=0.02)
                        if item is None: await c.disconnect(); return
                        request_item = item if isinstance(item, _BleWriteRequest) else None
                        if request_item is not None and request_item.cancelled:
                            request_item.error = "cancelled before BLE write"
                            request_item.done.set()
                            continue
                        payload = request_item.command if request_item is not None else item
                        if isinstance(payload, str): payload = (payload + "\n").encode()
                        try:
                            await c.write_gatt_char(NUS_RX, payload, response=True)
                            if request_item is not None:
                                request_item.ok = True
                            EVENT_LOG.event(
                                "ble_written",
                                command=(request_item.command if request_item is not None else repr(item)),
                            )
                        except Exception as exc:
                            if request_item is not None:
                                request_item.error = repr(exc)
                            raise
                        finally:
                            if request_item is not None:
                                request_item.done.set()
                    except queue.Empty: continue
                    except Exception as exc:
                        EVENT_LOG.event("ble_write_error", error=repr(exc))
                        break
                try: await c.disconnect()
                except: pass
            _ble_ok = False; await asyncio.sleep(2)
            if was_connected:
                EVENT_LOG.event("ble_disconnected", address=addr)
    asyncio.run(r())

def _ble_send(d):
    if _ble_ok:
        item = _BleWriteRequest(d)
        _ble_q.put(item)
        EVENT_LOG.event("ble_queued", command=d)
        if not item.done.wait(timeout=10.0):
            item.cancelled = True
            EVENT_LOG.event("ble_write_timeout", command=d)
            return "TIMEOUT"
        if item.ok:
            if _ARM_STATE_TRACKER is not None:
                _ARM_STATE_TRACKER.apply(d)
            return "OK"
        EVENT_LOG.event("ble_write_failed", command=d, error=item.error)
        return "ERROR"
    EVENT_LOG.event("ble_rejected", command=d, reason="not_connected")
    return "SIM"

# ── Camera ────────────────────────────────────────────────
_latest_jpeg = {"left": b"", "right": b""}  # 共享给 grasp_runner
_jpeg_conditions = {"left": threading.Condition(), "right": threading.Condition()}
_camera_meta = {
    "left": {"frames": 0, "last_frame_at": 0.0, "jpeg_bytes": 0},
    "right": {"frames": 0, "last_frame_at": 0.0, "jpeg_bytes": 0},
}


def _latest_left_jpeg():
    """Return the exact MJPEG frame owned by this running web_app process."""
    return _latest_jpeg.get("left", b"")


def _latest_right_jpeg():
    return _latest_jpeg.get("right", b"")


def _http_json(url: str, payload: dict | None = None, timeout: float = 90.0) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if data is not None else "GET",
    )
    try:
        with urlrequest.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(body).get("error", body)
        except Exception:
            detail = body
        raise RuntimeError(str(detail)) from exc

class Cam:
    def __init__(s, l=2, r=0, w=640, h=480, fps=15):
        # left/right are defined from behind the vehicle looking forward.
        s.ids = {"left":l,"right":r}; s.w=w; s.h=h; s.f=fps
        s._started = set()
        s._lock = threading.Lock()
    def start(s, name):
        with s._lock:
            if name in s._started: return
            s._started.add(name)
        threading.Thread(target=s._capture_loop, args=(name,), daemon=True).start()
    def _capture_loop(s, name):
        dev = f"/dev/video{s.ids.get(name,0)}"
        while True:
            p = None
            try:
                p = subprocess.Popen([
                    "ffmpeg", "-f", "v4l2", "-input_format", "mjpeg",
                    "-video_size", f"{s.w}x{s.h}", "-framerate", str(s.f),
                    "-i", dev, "-c:v", "copy", "-f", "mjpeg", "pipe:1",
                ], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                EVENT_LOG.event(
                    "camera_started", name=name, device=dev,
                    format="mjpeg", width=s.w, height=s.h, fps=s.f,
                )
                buf = b""
                while True:
                    ch = p.stdout.read(4096)
                    if not ch: break
                    buf += ch
                    while True:
                        soi = buf.find(b'\xff\xd8')
                        eoi = buf.find(b'\xff\xd9', soi+2 if soi>=0 else 0)
                        if soi < 0 or eoi <= soi: break
                        jpeg = buf[soi:eoi+2]; buf = buf[eoi+2:]
                        cond = _jpeg_conditions[name]
                        with cond:
                            _latest_jpeg[name] = jpeg
                            meta = _camera_meta[name]
                            meta["frames"] += 1
                            meta["last_frame_at"] = time.time()
                            meta["jpeg_bytes"] = len(jpeg)
                            cond.notify_all()
                    if len(buf)>500000: buf=buf[-100000:]
            except Exception as exc:
                EVENT_LOG.event("camera_error", name=name, device=dev, error=repr(exc))
                time.sleep(0.3)
            finally:
                if p is not None: p.terminate()
                EVENT_LOG.event("camera_restarting", name=name, device=dev)
    def mjpeg(s, name):
        s.start(name)
        last = None
        while True:
            cond = _jpeg_conditions[name]
            with cond:
                cond.wait_for(lambda: _latest_jpeg[name] is not last, timeout=1.0)
                jpeg = _latest_jpeg[name]
            if not jpeg or jpeg is last: continue
            last = jpeg
            yield (b"--frame\r\nContent-Type: image/jpeg\r\nCache-Control: no-store\r\nContent-Length: "+str(len(jpeg)).encode()+b"\r\n\r\n"+jpeg+b"\r\n")

# ── 爪子调试检测 (运动+黑色: 画面中移动的黑色物体=爪子) ──
_dbg_blobs = {}  # id -> {cx,cy,disp,age,last_seen,box}

def _dbg_detect(frame):
    """暗色斑块检测 + 多帧运动轨迹锁定。返回 (jpeg_bytes, locked, n_candidates)。"""
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dark = cv2.inRange(hsv[:, :, 2], 0, 90)
    kernel = np.ones((5, 5), np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    now = time.time()
    locked = None
    candidates = []
    used = set()
    for c in contours:
        area = cv2.contourArea(c)
        if area < 80 or area > w * h * 0.25:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        cx, cy = x + bw / 2.0, y + bh / 2.0
        candidates.append((x, y, bw, bh))
        # 与历史轨迹匹配
        best_key, best_d = None, 80.0
        for k, t in _dbg_blobs.items():
            if k in used:
                continue
            d = ((cx - t["cx"]) ** 2 + (cy - t["cy"]) ** 2) ** 0.5
            if d < best_d:
                best_d, best_key = d, k
        if best_key is not None:
            used.add(best_key)
            t = _dbg_blobs[best_key]
            t["disp"] += best_d
            t["cx"], t["cy"] = cx, cy
            t["age"] += 1
            t["last_seen"] = now
            t["box"] = (x, y, bw, bh)
        else:
            _dbg_blobs[id(c)] = {"cx": cx, "cy": cy, "disp": 0.0,
                                  "age": 1, "last_seen": now, "box": (x, y, bw, bh)}
    stale = [k for k, t in _dbg_blobs.items() if now - t["last_seen"] > 2.0]
    for k in stale:
        del _dbg_blobs[k]
    # 累计位移最大者 = 锁定爪子
    best_disp = 12.0
    for t in _dbg_blobs.values():
        if t["age"] >= 2 and t["disp"] > best_disp:
            best_disp = t["disp"]
            locked = t["box"]
    # 绘制
    out = frame.copy()
    for (x, y, bw, bh) in candidates:
        cv2.rectangle(out, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
    if locked is not None:
        x, y, bw, bh = locked
        cv2.rectangle(out, (x, y), (x + bw, y + bh), (0, 0, 255), 3)
        cv2.putText(out, "CLAW LOCKED", (x, max(22, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    ok, jpg = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return (jpg.tobytes(), locked, len(candidates)) if ok else (None, locked, len(candidates))

def _latest_left_frame():
    jpeg = _latest_jpeg.get("left", b"")
    if not jpeg or len(jpeg) < 1000:
        return None
    arr = np.frombuffer(jpeg, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

# ── HTML ──────────────────────────────────────────────────
H = r"""<!doctype html><html lang=zh><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>Arm</title>
<style>
:root{color-scheme:dark;--bg:#0d131c;--f:#e8ecf1;--m:#9fb0c4;--a:#4fd1c5;--r:#be3455;--g:#10b981;--b:#3b82f6;--l:#2b3b50}
*{box-sizing:border-box;margin:0}body{background:var(--bg);color:var(--f);font-family:Arial,sans-serif}
.t{height:44px;border-bottom:1px solid var(--l);display:flex;align-items:center;gap:10px;padding:0 12px;background:#111827;font-size:16px;font-weight:700}
.bd{border:1px solid #40536b;border-radius:99px;padding:3px 8px;font-size:11px;background:#0c1421}
.bd.g{color:var(--g);border-color:var(--g)}.bd.r{color:var(--r);border-color:var(--r)}
.w{display:grid;grid-template-columns:1fr 360px;height:calc(100vh - 44px)}
.m{display:grid;grid-template-rows:1fr 1fr;gap:4px;padding:4px;overflow:hidden}
.p{border:1px solid var(--l);background:#050a13;overflow:hidden;display:flex;align-items:center;justify-content:center;position:relative}
.p img{max-width:100%;max-height:100%;object-fit:contain}
.p.sel{border-color:#22d3ee;box-shadow:inset 0 0 0 1px #22d3ee}
.p canvas{position:absolute;pointer-events:none;z-index:2}
.p .lb{position:absolute;top:4px;left:8px;font-size:11px;font-weight:700;color:var(--a);background:rgba(0,0,0,.7);padding:2px 5px;border-radius:3px}
.s{border-left:1px solid var(--l);overflow-y:auto;padding:10px;background:#0f1622;display:flex;flex-direction:column;gap:8px}
.c{border:1px solid #33465d;border-radius:8px;padding:9px;background:#111c2a}
.c h3{font-size:12px;font-weight:700;margin-bottom:5px;color:var(--a)}
textarea{width:100%;padding:7px;border:1px solid #465970;border-radius:6px;background:#172234;color:var(--f);resize:vertical;min-height:38px;font:12px Arial}
.br{display:flex;gap:5px;flex-wrap:wrap;margin-top:5px}
button{padding:6px 10px;border-radius:6px;border:1px solid #465970;background:#172234;color:var(--f);font-weight:600;cursor:pointer;font-size:11px}
button:hover{filter:brightness(1.2)}
.go{background:#064e3b;border-color:var(--g)}.st{background:#581c2b;border-color:var(--r)}.ac{background:#1e3a5f;border-color:var(--b)}
.mt{color:var(--m);font-size:10px;margin-top:3px}
@media(max-width:800px){.w{grid-template-columns:1fr}.m{height:50vh}.s{border-left:0;border-top:1px solid var(--l)}}
</style></head><body>
<div class=t>Robot Arm<span class=bd id=b1>--</span><span class=bd id=b2>--</span></div>
<div class=w><div class=m>
<div class=p id=lp><span class=lb>LEFT</span><img id=lc src="/video/left"><canvas id=lo></canvas></div>
<div class=p id=rp><span class=lb>RIGHT</span><img id=rc src="/video/right"><canvas id=ro></canvas></div>
</div><div class=s>
<div class=c><h3>Arm</h3><div class=br>
<button onclick=a('D5')>前伸5</button><button onclick=a('U5')>后退5</button>
<button onclick=a('B3')>上升3</button><button onclick=a('F3')>下降3</button>
<button onclick=a('L3')>L+3</button><button onclick=a('R3')>R-3</button>
<button onclick=a('C90')>ClawOpen</button><button class=st onclick=a('C2')>ClawClose</button>
<button class=ac onclick=a('HOME')>归位</button><button class=st onclick=h()>STOP</button>
</div><div class=mt id=as>--</div></div>
<div class=c><h3>Qwen</h3><textarea id=tx placeholder="物体描述"></textarea>
  <div class=br><button class=go onclick=q()>识别</button><button class=ac onclick=g()>抓取</button><button class=ac onclick=r()>释放</button><button class=st onclick=s()>停止</button></div><div class=mt id=cs>--</div></div>
<div class=c><h3>Result</h3><div id=rs style="font-size:11px;color:#cbd5e1;max-height:150px;overflow-y:auto">--</div></div>
<div class=c><h3>Dataset</h3><a href="/arm-mapping" target="_blank"><button class=ac>双摄抓取标定与映射</button></a> <a href="/gripper-dataset" target="_blank"><button class=go>夹爪采集与YOLO训练</button></a><div class=mt>数量不限；Qwen推荐+人工确认</div></div>
</div></div>
<script>
async function P(u,d){return (await fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(d||{})})).json()}
async function G(u){return (await fetch(u,{cache:'no-store'})).json()}
function B(i,t,c){let e=document.getElementById(i);e.textContent=t;e.className='bd '+c}
function box(c,b,color,label){
  if(!b)return;
  let x=Number(b.x1),y=Number(b.y1),w=Number(b.width),h=Number(b.height);
  if(![x,y,w,h].every(Number.isFinite)||w<=0||h<=0)return;
  let sx=c.width/640,sy=c.height/480,ctx=c.getContext('2d');
  x*=sx;y*=sy;w*=sx;h*=sy;
  ctx.strokeStyle=color;ctx.lineWidth=3;ctx.strokeRect(x,y,w,h);
  ctx.font='bold 14px Arial';let tw=ctx.measureText(label).width+10;
  let ly=Math.max(0,y-21);ctx.fillStyle='rgba(2,8,18,.82)';ctx.fillRect(x,ly,tw,21);
  ctx.fillStyle=color;ctx.fillText(label,x+5,ly+15);
}
function jawPoints(c,points){
  if(!Array.isArray(points))return;
  let sx=c.width/640,sy=c.height/480,ctx=c.getContext('2d');
  let visible=points.filter(p=>Number(p.v||p.confidence)>0);
  if(visible.length===2){ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(visible[0].x*sx,visible[0].y*sy);ctx.lineTo(visible[1].x*sx,visible[1].y*sy);ctx.stroke()}
  points.forEach((p,i)=>{if(Number(p.v||p.confidence)<=0)return;ctx.fillStyle=i===0?'#ef4444':'#3b82f6';ctx.beginPath();ctx.arc(Number(p.x)*sx,Number(p.y)*sy,6,0,Math.PI*2);ctx.fill()});
}
function overlay(cam,g){
  let im=document.getElementById(cam==='left'?'lc':'rc');
  let cv=document.getElementById(cam==='left'?'lo':'ro');
  let p=document.getElementById(cam==='left'?'lp':'rp');
  let selected=g.active&&g.camera===cam;
  p.classList.toggle('sel',selected);
  let w=Math.round(im.clientWidth),h=Math.round(im.clientHeight);
  cv.style.left=im.offsetLeft+'px';cv.style.top=im.offsetTop+'px';
  if(cv.width!==w)cv.width=w;if(cv.height!==h)cv.height=h;
  let ctx=cv.getContext('2d');ctx.clearRect(0,0,cv.width,cv.height);
  if(!selected||!w||!h)return;
  let t=g.tracking||{};
  if(t.target_ok)box(cv,t.target_bbox,'#22d3ee','目标 TARGET');
  if(t.claw_ok)box(cv,t.claw_bbox,'#facc15','机械爪 CLAW');
  if(t.claw_ok)jawPoints(cv,t.claw_keypoints);
}
function overlays(g){overlay('left',g);overlay('right',g)}
async function a(c){document.getElementById('as').textContent=c;try{await P('/api/arm/raw',{ascii:c});B('b2','OK','g')}catch(e){B('b2','ERR','r')}}
async function h(){try{await P('/api/arm/stop');B('b2','STOP','r')}catch(e){}}
async function q(){let t=document.getElementById('tx').value.trim();if(!t)return;document.getElementById('cs').textContent='...';B('b1','busy','g');let t0=Date.now();try{let r=await P('/api/command',{instruction:t});document.getElementById('cs').textContent=((Date.now()-t0)/1000).toFixed(1)+'s';if(r.ok){B('b1','ok','g');let h='';for(let x of[...(r.landmarks||[]),...(r.objects||[])].slice(0,10))h+=`<b>${x.name||'?'}</b> ${x.type||''} ${x.camera_id||''} c=${(x.confidence||0).toFixed(2)} ${x.appearance||''}<br>`;document.getElementById('rs').innerHTML=h||'none'}else{B('b1','err','r');document.getElementById('rs').innerHTML=r.error||'err'}}catch(e){B('b1','err','r')}}
async function g(){let t=document.getElementById('tx').value.trim();if(!t)return;document.getElementById('cs').textContent='grasp...';try{let r=await P('/api/grasp/start',{target:t});document.getElementById('cs').textContent=r.ok?'started':'fail: '+(r.error||'')}catch(e){}}
  async function s(){try{await P('/api/grasp/stop');document.getElementById('cs').textContent='stopped'}catch(e){}}
  async function r(){try{let x=await P('/api/grasp/release');document.getElementById('cs').textContent=x.ok?'release requested':'release rejected'}catch(e){}}
setInterval(async()=>{try{let j=await G('/api/status');B('b1',j.ble?'BLE':'SIM','g');let g=j.grasp||{};overlays(g);if(g.active){B('b2',(g.phase||'servo')+' · '+(g.camera||'left'),'g');document.getElementById('cs').textContent=(g.step||g.phase)+' · camera='+(g.camera||'left');let c=g.arm_cmd||{};document.getElementById('as').textContent=(c.action||'-')+' F:'+(c.forward_mm||0)+' U:'+(c.up_mm||0)+' L:'+(c.left_deg||0)+' | '+(c.reason||'').slice(0,40);let t=g.tracking||{};let rs=document.getElementById('rs');rs.innerHTML='camera:'+(g.camera||'left')+' claw:'+(t.claw_ok?'OK':'LOST')+' target:'+(t.target_ok?'OK':'LOST')+'<br>'+(g.log||[]).slice(-5).join('<br>')}else{B('b2',j.ble?'BLE':'SIM','g')}}catch(e){}},250)
window.addEventListener('resize',()=>overlays({active:false}));
</script></body></html>"""

GRIPPER_DATASET_HTML = r"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>夹爪双夹点标注</title>
<style>body{margin:0;background:#0b1320;color:#e8eef7;font-family:Arial,"Microsoft YaHei",sans-serif}.top{display:flex;gap:16px;align-items:center;padding:12px 16px;background:#111d2c}.top b,h3,.top a{color:#4fd1c5}.grid{display:grid;grid-template-columns:minmax(560px,1fr) 400px;gap:12px;padding:12px}.card{background:#111c2a;border:1px solid #33465d;border-radius:10px;padding:11px}.stage{position:relative;background:#050a13;min-height:480px;display:flex;align-items:center;justify-content:center;overflow:hidden}.stage img{max-width:100%;max-height:72vh}.stage canvas{position:absolute;cursor:crosshair}.row{display:flex;gap:7px;flex-wrap:wrap;margin:8px 0}button,select,input{background:#172234;color:#e8ecf1;border:1px solid #52677f;border-radius:6px;padding:8px 11px}.go{background:#065f46}.blue{background:#1e3a5f}.bad{background:#6b1830}.active{outline:3px solid #facc15}.note{font-size:12px;color:#a9b8ca;line-height:1.6}.status{white-space:pre-wrap;font-size:12px;max-height:230px;overflow:auto;background:#0a111d;padding:8px}.warn{color:#facc15}.cand{display:grid;grid-template-columns:repeat(2,1fr);gap:6px}@media(max-width:980px){.grid{grid-template-columns:1fr}.stage{min-height:340px}}</style></head><body>
<div class="top"><b>夹爪双夹点 Pose 标注</b><span id="progress">加载中…</span><a href="/">返回抓取页</a></div><div class="grid"><div class="card"><div class="row"><select id="cam" onchange="live()"><option value="left">左摄像头</option><option value="right">右摄像头</option></select><button class="go" onclick="capture()">拍摄新图</button><button onclick="loadLegacy()">载入原数据待补点</button><button onclick="previousSample()">上一张</button><button onclick="nextSample()">下一张</button></div><div class="stage" id="stage"><img id="img" src="/video/left"><canvas id="cv"></canvas></div>
<div class="row"><button id="boxMode" class="active" onclick="setMode('box')">拖动画夹爪框</button><button id="leftMode" onclick="setMode('left')">点左夹指夹点</button><button id="rightMode" onclick="setMode('right')">点右夹指夹点</button><button onclick="hidePoint(0)">左夹点不可见</button><button onclick="hidePoint(1)">右夹点不可见</button></div><div class="note">黄色框=夹爪局部框；红点=图像左侧夹点；蓝点=图像右侧夹点。照片只有一个夹指时，只标可见点并把另一点设为“不可见”。运行时系统只做 1° 左右试探，直到两个点都出现。</div></div>
<div class="card"><h3>当前样本</h3><div id="sample" class="note">可直接载入原数据集，不必重新拍摄</div><div id="candidates" class="cand"></div><div class="row"><button onclick="qwen()">Qwen 推荐夹爪框</button><button class="go" onclick="confirmPose(false)">保存夹点标注</button><button onclick="confirmPose(true)">负样本</button><button class="bad" onclick="discard()">删除图片</button></div><div id="msg" class="status warn"></div><hr style="border-color:#33465d"><h3>Pose 训练</h3><div class="note">训练只使用已补至少一个夹点的正样本和负样本；建议每个摄像头 50 张，多数照片包含两个夹点。</div><div class="row"><a href="/api/gripper-dataset/export"><button>下载 YOLO Pose ZIP</button></a><label>轮次 <input id="epochs" type="number" value="40" min="5" max="100" style="width:70px"></label><button class="go" onclick="trainModel()">低显存训练</button><button class="blue" onclick="promote()">部署训练模型</button></div><div id="train" class="status">训练未开始</div></div></div>
<script>
var S=null,B=null,K=[{x:0,y:0,v:0},{x:0,y:0,v:0}],MODE='box',drawing=false,start=null,LEGACY=[],CUR=-1;
async function P(u,d){var r=await fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(d||{})});var j=await r.json();if(!r.ok||j.ok===false)throw Error(j.error||r.statusText);return j}
async function refresh(){var j=await (await fetch('/api/gripper-dataset/status',{cache:'no-store'})).json();progress.textContent='左 '+j.left+' · 右 '+j.right+' · 总计 '+(j.left+j.right)+' · 双点 '+j.two_points+' · 单点 '+j.one_point+' · 旧框待补 '+j.legacy_bbox+' · 待确认 '+j.pending}
function live(){if(!S)img.src='/video/'+cam.value}function setMode(m){MODE=m;['box','left','right'].forEach(function(n){document.getElementById(n+'Mode').classList.toggle('active',n===m)})}
function canvasFit(){var r=img.getBoundingClientRect(),p=stage.getBoundingClientRect();cv.style.left=(r.left-p.left)+'px';cv.style.top=(r.top-p.top)+'px';cv.width=Math.round(r.width);cv.height=Math.round(r.height);draw()}
function normBox(b){if(!b||!S)return null;return{x1:b.x1*cv.width/S.width,y1:b.y1*cv.height/S.height,x2:b.x2*cv.width/S.width,y2:b.y2*cv.height/S.height}}function denormBox(b){return{x1:b.x1*S.width/cv.width,y1:b.y1*S.height/cv.height,x2:b.x2*S.width/cv.width,y2:b.y2*S.height/cv.height}}function denormPoint(p){return{x:p.x*S.width/cv.width,y:p.y*S.height/cv.height,v:2}}function rect(x,b,c,w){b=normBox(b);if(!b)return;x.strokeStyle=c;x.lineWidth=w||2;x.strokeRect(b.x1,b.y1,b.x2-b.x1,b.y2-b.y1)}function pointCanvas(p){return{x:p.x*cv.width/S.width,y:p.y*cv.height/S.height}}
function draw(){var x=cv.getContext('2d');x.clearRect(0,0,cv.width,cv.height);if(!S)return;(S.candidates||[]).forEach(function(c){rect(x,c,'#22c55e',1)});rect(x,S.qwen_bbox,'#22d3ee',2);rect(x,B,'#facc15',3);var a=K[0]&&K[0].v>0?pointCanvas(K[0]):null,b=K[1]&&K[1].v>0?pointCanvas(K[1]):null;if(a&&b){x.strokeStyle='#fff';x.lineWidth=2;x.beginPath();x.moveTo(a.x,a.y);x.lineTo(b.x,b.y);x.stroke()}[[a,'#ef4444','L'],[b,'#3b82f6','R']].forEach(function(z){if(!z[0])return;x.fillStyle=z[1];x.beginPath();x.arc(z[0].x,z[0].y,7,0,Math.PI*2);x.fill();x.fillStyle='#fff';x.fillText(z[2],z[0].x+9,z[0].y-7)})}
function setSample(s){S=s;B=s.bbox||s.qwen_bbox||(s.candidates||[])[0]||null;K=(s.keypoints||s.qwen_keypoints||[{x:0,y:0,v:0},{x:0,y:0,v:0}]).map(function(p){return Object.assign({},p)});while(K.length<2)K.push({x:0,y:0,v:0});img.src='/api/gripper-dataset/image/'+s.sample_id+'?t='+Date.now();sample.textContent=s.sample_id+' · '+s.camera+' · '+s.width+'×'+s.height+' · 可见夹点 '+K.filter(function(p){return p.v>0}).length;candidates.innerHTML='';(s.candidates||[]).forEach(function(c,i){var b=document.createElement('button');b.textContent='候选 '+(i+1)+' '+c.width+'×'+c.height;b.onclick=function(){B=c;draw()};candidates.appendChild(b)});img.onload=canvasFit;draw()}
async function capture(){try{var j=await P('/api/gripper-dataset/capture',{camera:cam.value});setSample(j.sample);msg.textContent='先画框，再分别点两个夹点';refresh()}catch(e){msg.textContent=e.message}}async function loadLegacy(){try{var j=await (await fetch('/api/gripper-dataset/samples?pose_pending=1',{cache:'no-store'})).json();LEGACY=j.samples||[];CUR=0;if(!LEGACY.length){msg.textContent='没有尚待补点的旧框样本';return}setSample(LEGACY[0]);msg.textContent='原数据 1/'+LEGACY.length}catch(e){msg.textContent=e.message}}function previousSample(){if(!LEGACY.length)return;CUR=(CUR-1+LEGACY.length)%LEGACY.length;setSample(LEGACY[CUR]);msg.textContent='原数据 '+(CUR+1)+'/'+LEGACY.length}function nextSample(){if(!LEGACY.length)return;CUR=(CUR+1)%LEGACY.length;setSample(LEGACY[CUR]);msg.textContent='原数据 '+(CUR+1)+'/'+LEGACY.length}function hidePoint(i){K[i]={x:0,y:0,v:0};draw()}
async function qwen(){if(!S)return;try{var j=await P('/api/gripper-dataset/qwen',{sample_id:S.sample_id});S=j.sample;if(S.qwen_bbox)B=S.qwen_bbox;draw();msg.textContent='Qwen 只推荐框；精确夹点请人工点击'}catch(e){msg.textContent=e.message}}async function confirmPose(negative){if(!S)return;if(!negative&&!B)return msg.textContent='请先画夹爪框';if(!negative&&!K.some(function(p){return p.v>0}))return msg.textContent='至少标一个可见夹点';try{await P('/api/gripper-dataset/confirm',{sample_id:S.sample_id,bbox:B,keypoints:K,negative:negative});msg.textContent=negative?'负样本已保存':'夹点已保存';S=null;B=null;K=[{x:0,y:0,v:0},{x:0,y:0,v:0}];img.src='/video/'+cam.value;await refresh();if(LEGACY.length)await loadLegacy()}catch(e){msg.textContent=e.message}}async function discard(){if(!S)return;await P('/api/gripper-dataset/delete',{sample_id:S.sample_id});S=null;img.src='/video/'+cam.value;refresh()}async function trainModel(){try{var j=await P('/api/gripper-dataset/train',{epochs:Number(epochs.value)});msg.textContent=j.remote.message}catch(e){msg.textContent=e.message}}async function promote(){try{var j=await P('/api/gripper-dataset/train/promote',{});msg.textContent='已部署：'+j.remote.model}catch(e){msg.textContent=e.message}}async function trainStatus(){try{var j=await (await fetch('/api/gripper-dataset/train/status')).json();train.textContent='状态: '+(j.status||'idle')+'\n结果: '+(j.result||'')+'\n错误: '+(j.error||'')+'\n'+(j.log||[]).slice(-12).join('\n')}catch(e){}}
cv.onmousedown=function(e){if(!S)return;var r=cv.getBoundingClientRect(),p={x:e.clientX-r.left,y:e.clientY-r.top};if(MODE==='box'){start=p;drawing=true}else{K[MODE==='left'?0:1]=denormPoint(p);draw()}};cv.onmousemove=function(e){if(!drawing||MODE!=='box')return;var r=cv.getBoundingClientRect(),p={x:e.clientX-r.left,y:e.clientY-r.top};B=denormBox({x1:Math.min(start.x,p.x),y1:Math.min(start.y,p.y),x2:Math.max(start.x,p.x),y2:Math.max(start.y,p.y)});draw()};cv.onmouseup=function(){drawing=false};window.onresize=canvasFit;refresh();setInterval(refresh,3000);setInterval(trainStatus,3000);
</script></body></html>"""

def create_app(qwen_url="", lc=0, rc=2, w=640, h=480,
               grasp_http_url="http://192.0.2.20:8020"):
    app = Flask(__name__)
    cam = Cam(l=lc, r=rc, w=w, h=h)
    try:
        from .gripper_dataset import GripperDataset
    except ImportError:
        from gripper_dataset import GripperDataset
    dataset = GripperDataset(
        os.environ.get("GRIPPER_DATASET_DIR", str(Path(__file__).resolve().parent / "gripper_dataset")),
        target_per_camera=None,
    )
    try:
        from .arm_mapping_dataset import ArmMappingDataset, ArmStateTracker
    except ImportError:
        from arm_mapping_dataset import ArmMappingDataset, ArmStateTracker
    data_root = Path(os.environ.get(
        "ARM_MAPPING_DATASET_DIR",
        str(Path(__file__).resolve().parent / "arm_mapping_dataset"),
    ))
    mapping_dataset = ArmMappingDataset(data_root)
    arm_state = ArmStateTracker(
        os.environ.get("ARM_STATE_PATH", str(data_root / "arm_state.json"))
    )
    global _ARM_STATE_TRACKER
    _ARM_STATE_TRACKER = arm_state
    # Capture starts independently of browser viewers. Grasp automation and
    # MJPEG HTTP clients consume the same latest compressed frame.
    cam.start("left")
    cam.start("right")
    from werkzeug.serving import run_simple

    @app.route("/video/<name>")
    def video(name):
        return Response(cam.mjpeg(name), mimetype="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control":"no-store","X-Accel-Buffering":"no"})

    _cmd_log = []  # 记录所有命令用于标定

    @app.route("/api/arm/raw", methods=["POST"])
    def arm_raw():
        d = request.get_json(silent=True) or {}
        c = str(d.get("ascii","")).strip()
        if not c: return jsonify(ok=False,error="empty")
        if not c.startswith("#"):
            if c.upper() in ("HOME", "STOP", "PING"):
                c = f"#{c.upper()}!"
            else:
                c = f"#ARM{c}!"
        _cmd_log.append(f"{time.strftime('%H:%M:%S')} {c}")
        r = _ble_send(c)
        return jsonify(ok=True,result=r)

    @app.route("/api/arm/log")
    def arm_log():
        return jsonify(log=_cmd_log[-30:])

    @app.route("/api/arm/stop", methods=["POST"])
    def arm_stop():
        _ble_send("#STOP!")
        return jsonify(ok=True)

    @app.route("/api/status")
    def status():
        gs = {}
        try:
            try:
                from .grasp_runner import status as gs_dict
            except ImportError:
                from grasp_runner import status as gs_dict
            gs = dict(gs_dict)
        except: pass
        now = time.time()
        cameras = {
            name: {
                **meta,
                "age_s": (round(now - meta["last_frame_at"], 3)
                          if meta["last_frame_at"] else None),
            }
            for name, meta in _camera_meta.items()
        }
        return jsonify(ble=_ble_ok, grasp=gs, cameras=cameras)

    def _grasp_active():
        try:
            try:
                from .grasp_runner import status as runner_status
            except ImportError:
                from grasp_runner import status as runner_status
            return bool(runner_status.get("active"))
        except Exception:
            return False

    @app.route("/api/arm/state")
    def arm_state_status():
        return jsonify(ok=True, state=arm_state.snapshot())

    @app.route("/api/arm/state/baseline", methods=["POST"])
    def arm_state_baseline():
        state = arm_state.mark_baseline("manual_web_baseline")
        EVENT_LOG.event("arm_state_baseline", state=state)
        return jsonify(ok=True, state=state)

    @app.route("/api/arm/state/set", methods=["POST"])
    def arm_state_set():
        try:
            state = arm_state.set_state(request.get_json(silent=True) or {})
            EVENT_LOG.event("arm_state_set", state=state)
            return jsonify(ok=True, state=state)
        except Exception as exc:
            return jsonify(ok=False, error=str(exc)), 400

    @app.route("/api/arm/fine-move", methods=["POST"])
    def arm_fine_move():
        data = request.get_json(silent=True) or {}
        axis = str(data.get("axis", ""))
        try:
            direction = 1 if float(data.get("direction", 1)) >= 0 else -1
            step = int(data.get("step", 1))
        except (TypeError, ValueError):
            return jsonify(ok=False, error="invalid direction or step"), 400
        limits = {"forward": 5, "up": 3, "yaw": 3}
        letters = {
            ("forward", 1): "D", ("forward", -1): "U",
            ("up", 1): "B", ("up", -1): "F",
            ("yaw", 1): "L", ("yaw", -1): "R",
        }
        if axis not in limits or step < 1 or step > limits[axis]:
            return jsonify(ok=False, error=f"{axis} step must be 1..{limits.get(axis, 0)}"), 400
        command = f"#ARM{letters[(axis, direction)]}{step}!"
        result = _ble_send(command)
        _cmd_log.append(f"{time.strftime('%H:%M:%S')} {command}")
        EVENT_LOG.event("arm_fine_move", axis=axis, direction=direction, step=step,
                        command=command, result=result)
        if result != "OK":
            return jsonify(ok=False, error=f"BLE command was not written: {result}",
                           result=result, state=arm_state.snapshot()), 409
        return jsonify(ok=True, result=result, command=command, state=arm_state.snapshot())

    @app.route("/api/arm-mapping/status")
    def arm_mapping_status():
        return jsonify(ok=True, status=mapping_dataset.status(), state=arm_state.snapshot())

    @app.route("/api/arm-mapping/samples")
    def arm_mapping_samples():
        return jsonify(ok=True, samples=mapping_dataset.list_samples(
            request.args.get("object_name", "")
        ))

    @app.route("/api/arm-mapping/capture", methods=["POST"])
    def arm_mapping_capture():
        if _grasp_active():
            return jsonify(ok=False, error="请先停止自动抓取任务"), 409
        data = request.get_json(silent=True) or {}
        object_name = str(data.get("object_name", "")).strip()
        if not object_name:
            return jsonify(ok=False, error="请输入物体名称"), 400
        left_jpeg = bytes(_latest_jpeg.get("left", b""))
        right_jpeg = bytes(_latest_jpeg.get("right", b""))
        if not left_jpeg or not right_jpeg:
            return jsonify(ok=False, error="左右摄像头画面尚未就绪"), 503
        try:
            sample = mapping_dataset.capture_pair(
                left_jpeg, right_jpeg, object_name=object_name,
                arm_state=arm_state.snapshot(), motion_steps=data.get("steps") or {},
            )
            EVENT_LOG.event("arm_mapping_capture", sample_id=sample["sample_id"],
                            object_name=object_name, arm_state=sample["arm_state"])
            return jsonify(ok=True, sample=sample, status=mapping_dataset.status())
        except Exception as exc:
            return jsonify(ok=False, error=str(exc)), 400

    @app.route("/api/arm-mapping/image/<sample_id>/<camera>")
    def arm_mapping_image(sample_id, camera):
        if camera not in ("left", "right"):
            return jsonify(ok=False, error="invalid camera"), 400
        path = mapping_dataset._image_path(Path(sample_id).name, camera)
        if not path.exists():
            return jsonify(ok=False, error="sample image not found"), 404
        return Response(path.read_bytes(), mimetype="image/jpeg",
                        headers={"Cache-Control": "no-store"})

    @app.route("/api/arm-mapping/analyze", methods=["POST"])
    def arm_mapping_analyze():
        if _grasp_active():
            return jsonify(ok=False, error="请先停止自动抓取任务"), 409
        data = request.get_json(silent=True) or {}
        try:
            sample_id = str(data.get("sample_id", ""))
            mapping_dataset.get(sample_id)
            payload = {
                "left_image_b64": base64.b64encode(
                    mapping_dataset._image_path(sample_id, "left").read_bytes()
                ).decode("ascii"),
                "right_image_b64": base64.b64encode(
                    mapping_dataset._image_path(sample_id, "right").read_bytes()
                ).decode("ascii"),
                "target_description": str(data.get("target_description", "")),
                "use_qwen": bool(data.get("use_qwen", True)),
            }
            result = _http_json(
                grasp_http_url.rstrip("/") + "/api/calibration/analyze",
                payload, timeout=120,
            )
            if result.get("ok") is False:
                raise RuntimeError(result.get("error", "analysis failed"))
            sample = mapping_dataset.set_analysis(sample_id, result)
            EVENT_LOG.event("arm_mapping_analyzed", sample_id=sample_id,
                            use_qwen=payload["use_qwen"])
            return jsonify(ok=True, sample=sample)
        except Exception as exc:
            return jsonify(ok=False, error=str(exc)), 400

    def _jaw_box_from_view(view):
        box = view.get("jaw_bbox")
        if box:
            return box
        points = [p for p in (view.get("jaw_points") or []) if int(p.get("v", 0)) > 0]
        if not points:
            return None
        width, height = float(view.get("width", 640)), float(view.get("height", 480))
        margin = 35.0
        return {
            "x1": max(0.0, min(p["x"] for p in points) - margin),
            "y1": max(0.0, min(p["y"] for p in points) - margin),
            "x2": min(width, max(p["x"] for p in points) + margin),
            "y2": min(height, max(p["y"] for p in points) + margin),
        }

    @app.route("/api/arm-mapping/confirm", methods=["POST"])
    def arm_mapping_confirm():
        data = request.get_json(silent=True) or {}
        try:
            sample_id = str(data.get("sample_id", ""))
            existing_sample = mapping_dataset.get(sample_id)
            views = data.get("views") or {}
            grasp_ready = bool(data.get("grasp_ready", True))
            jaw_verified = bool(data.get("jaw_verified", False))
            sample = mapping_dataset.confirm(
                sample_id, views=views, arm_state=arm_state.snapshot(),
                grasp_ready=grasp_ready, jaw_verified=jaw_verified,
                notes=str(data.get("notes", "")),
            )
            gripper_ids = dict(existing_sample.get("gripper_sample_ids") or {})
            if (jaw_verified and bool(data.get("add_to_gripper_dataset", True))
                    and not gripper_ids):
                for camera in ("left", "right"):
                    view = sample["views"][camera]
                    box = _jaw_box_from_view(view)
                    points = view.get("jaw_points") or []
                    if box is None or not any(int(p.get("v", 0)) > 0 for p in points):
                        raise ValueError(f"{camera} camera needs a jaw box and at least one visible jaw point")
                    captured = dataset.capture(
                        camera, mapping_dataset._image_path(sample_id, camera).read_bytes()
                    )
                    saved = dataset.save(captured["sample_id"], box, False, points)
                    gripper_ids[camera] = saved["sample_id"]
                sample = mapping_dataset.confirm(
                    sample_id, views=views, arm_state=arm_state.snapshot(),
                    grasp_ready=grasp_ready, jaw_verified=jaw_verified,
                    notes=str(data.get("notes", "")), gripper_sample_ids=gripper_ids,
                )
            EVENT_LOG.event("arm_mapping_confirm", sample_id=sample_id,
                            object_name=sample.get("object_name"), grasp_ready=grasp_ready,
                            jaw_verified=jaw_verified, gripper_sample_ids=gripper_ids)
            return jsonify(ok=True, sample=sample, gripper_sample_ids=gripper_ids,
                           status=mapping_dataset.status(), gripper_status=dataset.status())
        except Exception as exc:
            return jsonify(ok=False, error=str(exc)), 400

    @app.route("/api/arm-mapping/train", methods=["POST"])
    def arm_mapping_train():
        data = request.get_json(silent=True) or {}
        try:
            model = mapping_dataset.train(
                str(data.get("object_name", "")).strip(),
                ridge=float(data.get("ridge", 0.01)),
                quantization=data.get("quantization") or {},
            )
            EVENT_LOG.event("arm_mapping_trained", object_name=model["object_name"],
                            samples=model["samples"], test_mae=model["test_mae"])
            return jsonify(ok=True, model=model)
        except Exception as exc:
            return jsonify(ok=False, error=str(exc)), 400

    @app.route("/api/arm-mapping/predict", methods=["POST"])
    def arm_mapping_predict():
        try:
            prediction = mapping_dataset.predict(
                (request.get_json(silent=True) or {}).get("views") or {}
            )
            return jsonify(ok=True, prediction=prediction)
        except Exception as exc:
            return jsonify(ok=False, error=str(exc)), 400

    @app.route("/api/arm-mapping/export")
    def arm_mapping_export():
        try:
            archive, stats = mapping_dataset.build_zip()
            return Response(archive, mimetype="application/zip", headers={
                "Content-Disposition": "attachment; filename=arm_mapping_dataset.zip",
                "X-Dataset-Samples": str(stats["samples"]),
            })
        except Exception as exc:
            return jsonify(ok=False, error=str(exc)), 400
    @app.route("/api/gripper-dataset/status")
    def gripper_dataset_status():
        return jsonify(ok=True, **dataset.status())

    @app.route("/api/gripper-dataset/samples")
    def gripper_dataset_samples():
        pending = request.args.get("pose_pending", "0") in ("1", "true", "yes")
        return jsonify(ok=True, samples=dataset.list_samples(pose_pending_only=pending))
    @app.route("/api/gripper-dataset/capture", methods=["POST"])
    def gripper_dataset_capture():
        if _grasp_active():
            return jsonify(ok=False, error="请先停止抓取任务"), 409
        camera = str((request.get_json(silent=True) or {}).get("camera", "left"))
        jpeg = _latest_jpeg.get(camera, b"")
        if not jpeg:
            return jsonify(ok=False, error="camera frame unavailable"), 503
        try:
            meta = dataset.capture(camera, bytes(jpeg))
            EVENT_LOG.event("gripper_dataset_capture", camera=camera,
                            sample_id=meta["sample_id"], candidate_count=len(meta["candidates"]))
            return jsonify(ok=True, sample=meta)
        except Exception as exc:
            return jsonify(ok=False, error=str(exc)), 400

    @app.route("/api/gripper-dataset/image/<sample_id>")
    def gripper_dataset_image(sample_id):
        path = dataset._image_path(Path(sample_id).name)
        if not path.exists():
            return jsonify(ok=False, error="sample not found"), 404
        return Response(path.read_bytes(), mimetype="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.route("/api/gripper-dataset/qwen", methods=["POST"])
    def gripper_dataset_qwen():
        sample_id = str((request.get_json(silent=True) or {}).get("sample_id", ""))
        try:
            meta = dataset.get(sample_id)
            jpeg = dataset._image_path(sample_id).read_bytes()
            result = _http_json(
                grasp_http_url.rstrip("/") + "/api/gripper/qwen",
                {"camera": meta["camera"],
                 "image_b64": base64.b64encode(jpeg).decode("ascii")},
                timeout=90,
            )
            meta = dataset.set_qwen(sample_id, result.get("bbox"), result.get("raw", ""))
            return jsonify(ok=True, sample=meta, confidence=result.get("confidence", 0.0))
        except Exception as exc:
            return jsonify(ok=False, error=str(exc)), 400

    @app.route("/api/gripper-dataset/confirm", methods=["POST"])
    def gripper_dataset_confirm():
        data = request.get_json(silent=True) or {}
        try:
            meta = dataset.save(str(data.get("sample_id", "")), data.get("bbox"),
                                bool(data.get("negative")), data.get("keypoints"))
            EVENT_LOG.event("gripper_dataset_confirm", sample_id=meta["sample_id"],
                            camera=meta["camera"], negative=meta["negative"], bbox=meta.get("bbox"))
            return jsonify(ok=True, sample=meta, status=dataset.status())
        except Exception as exc:
            return jsonify(ok=False, error=str(exc)), 400

    @app.route("/api/gripper-dataset/delete", methods=["POST"])
    def gripper_dataset_delete():
        dataset.delete(str((request.get_json(silent=True) or {}).get("sample_id", "")))
        return jsonify(ok=True, status=dataset.status())

    @app.route("/api/gripper-dataset/export")
    def gripper_dataset_export():
        try:
            data, stats = dataset.build_zip()
            return Response(data, mimetype="application/zip", headers={
                "Content-Disposition": "attachment; filename=gripper_yolo_dataset.zip",
                "X-Dataset-Samples": str(stats["samples"]),
            })
        except Exception as exc:
            return jsonify(ok=False, error=str(exc)), 400

    @app.route("/api/gripper-dataset/train", methods=["POST"])
    def gripper_dataset_train():
        if _grasp_active():
            return jsonify(ok=False, error="请先停止抓取任务再训练"), 409
        try:
            archive, stats = dataset.build_zip()
            params = request.get_json(silent=True) or {}
            result = _http_json(
                grasp_http_url.rstrip("/") + "/api/gripper/train",
                {"dataset_zip_b64": base64.b64encode(archive).decode("ascii"),
                 "epochs": int(params.get("epochs", 40)), "imgsz": 416},
                timeout=60,
            )
            return jsonify(ok=True, dataset=stats, remote=result)
        except Exception as exc:
            return jsonify(ok=False, error=str(exc)), 400

    @app.route("/api/gripper-dataset/train/promote", methods=["POST"])
    def gripper_dataset_train_promote():
        try:
            remote = _http_json(
                grasp_http_url.rstrip("/") + "/api/gripper/train/promote", {},
                timeout=30,
            )
            return jsonify(ok=True, remote=remote)
        except Exception as exc:
            return jsonify(ok=False, error=str(exc)), 400
    @app.route("/api/gripper-dataset/train/status")
    def gripper_dataset_train_status():
        try:
            return jsonify(_http_json(
                grasp_http_url.rstrip("/") + "/api/gripper/train/status", timeout=5,
            ))
        except Exception as exc:
            return jsonify(active=False, status="unavailable", error=str(exc))

    @app.route("/api/command", methods=["POST"])
    def cmd():
        return jsonify(ok=False,error="not implemented")

    @app.route("/api/grasp/start", methods=["POST"])
    def gs():
        d = request.get_json(silent=True) or {}
        t = str(d.get("target","")).strip()
        if not t: return jsonify(ok=False,error="target required")
        EVENT_LOG.event("api_grasp_start", target=t, remote=request.remote_addr)
        try:
            from .grasp_runner import start, status
        except ImportError:
            from grasp_runner import start, status
        ok = start(
            t,
            ble_send=_ble_send,
            captures={
                "left": _latest_left_jpeg,
                "right": _latest_right_jpeg,
            },
            initial_camera="left",
        )
        return jsonify(ok=ok, message="started" if ok else "already running")

    @app.route("/api/grasp/stop", methods=["POST"])
    def gp():
        EVENT_LOG.event("api_grasp_stop", remote=request.remote_addr)
        try:
            from .grasp_runner import stop
        except ImportError:
            from grasp_runner import stop
        stop()
        _ble_send("#STOP!")
        return jsonify(ok=True)

    @app.route("/api/grasp/release", methods=["POST"])
    def gr():
        EVENT_LOG.event("api_grasp_release", remote=request.remote_addr)
        try:
            from .grasp_runner import release
        except ImportError:
            from grasp_runner import release
        return jsonify(ok=bool(release()))

    @app.route("/api/grasp/status")
    def gst():
        try:
            from .grasp_runner import status as gs
        except ImportError:
            from grasp_runner import status as gs
        return jsonify(dict(gs))

    @app.route("/")
    def index(): return render_template_string(H)

    @app.route("/arm-mapping")
    def arm_mapping_page():
        try:
            from .arm_mapping_page import ARM_MAPPING_HTML
        except ImportError:
            from arm_mapping_page import ARM_MAPPING_HTML
        return render_template_string(ARM_MAPPING_HTML)
    @app.route("/gripper-dataset")
    def gripper_dataset_page():
        return render_template_string(GRIPPER_DATASET_HTML)

    # ── 爪子锁定调试页 ────────────────────────────────────
    @app.route("/debug/claw")
    def debug_claw():
        frame = _latest_left_frame()
        if frame is None:
            return jsonify(ok=False, error="no frame"), 503
        jpg, locked, n = _dbg_detect(frame)
        resp = Response(jpg, mimetype="image/jpeg")
        resp.headers["X-Claw-Locked"] = "1" if locked else "0"
        resp.headers["X-Candidates"] = str(n)
        return resp

    @app.route("/debug")
    def debug_page():
        return render_template_string(r"""<!doctype html><html lang=zh><head><meta charset=utf-8>
<title>Claw Debug</title><style>
body{background:#0d131c;color:#e8ecf1;font-family:Arial;margin:0;padding:12px}
h2{color:#4fd1c5}
img{max-width:96vw;max-height:78vh;border:2px solid #2b3b50;border-radius:8px}
.st{font-size:14px;margin:8px 0}
.locked{color:#10b981;font-weight:700}.lost{color:#be3455;font-weight:700}
</style></head><body>
<h2>爪子运动锁定调试 (黑色+移动=爪子)</h2>
<div class=st id=st>...</div>
<img id=im src="/debug/claw">
<script>
setInterval(async()=>{
  try{
    let r=await fetch('/debug/claw',{cache:'no-store'});
    let locked=r.headers.get('X-Claw-Locked')==='1';
    let n=r.headers.get('X-Candidates')||'0';
    document.getElementById('st').innerHTML=
      (locked?'<span class=locked>✅ 爪子已锁定</span>':'<span class=lost>❌ 未锁定 (移动机械臂几下)</span>')
      +` | 黑色斑块候选: ${n}`;
    let b=await r.blob();
    let u=URL.createObjectURL(b);
    let im=document.getElementById('im');
    im.onload=()=>URL.revokeObjectURL(u);
    im.src=u;
  }catch(e){}
},500);
</script></body></html>""")

    # 机械臂示教夹取 (独立系统): 教学采集页 + 模型夹取执行
    try:
        from .arm_teach import register as arm_teach_register
    except ImportError:
        from arm_teach import register as arm_teach_register
    arm_teach_register(
        app,
        jpegs=_latest_jpeg,
        ble_send=_ble_send,
        teach_url=os.environ.get("ARM_TEACH_URL", "http://192.0.2.20:8022"),
    )

    return app

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--qwen-url", default="")
    p.add_argument("--grasp-http-url", default="http://192.0.2.20:8020")
    p.add_argument("--left-camera", type=int, default=0,
                   help="left camera when viewed from vehicle rear toward front")
    p.add_argument("--right-camera", type=int, default=2,
                   help="right camera when viewed from vehicle rear toward front")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--ble-address", default=BLE_ADDR)
    args = p.parse_args()
    EVENT_LOG.event(
        "web_start", host=args.host, port=args.port,
        left_camera=args.left_camera, right_camera=args.right_camera,
        ble_address=args.ble_address, log=str(EVENT_LOG.path),
    )
    threading.Thread(target=_ble, args=(args.ble_address,), daemon=True).start()
    app = create_app(qwen_url=args.qwen_url, lc=args.left_camera, rc=args.right_camera,
                     grasp_http_url=args.grasp_http_url)
    from werkzeug.serving import run_simple
    run_simple(args.host, args.port, app, threaded=True, use_reloader=False)

if __name__ == "__main__": main()
