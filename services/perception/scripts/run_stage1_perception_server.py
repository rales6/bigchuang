from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path
import re
import threading
import time
from typing import Any

from fastapi import FastAPI, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
import numpy as np
from PIL import Image
import requests
import uvicorn


app = FastAPI(title="Stage 1 Stereo Semantic Perception")
SERVICE: "Stage1PerceptionService | None" = None
PI_URL = ""
PI_SESSION = requests.Session()
PI_SESSION.trust_env = False


HTML = """<!doctype html><html lang=zh-CN><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>3090 阶段1感知服务</title><style>
:root{color-scheme:dark}body{margin:0;background:#0b1118;color:#eef4fb;font-family:Arial,"Microsoft YaHei",sans-serif}header{padding:12px 16px;background:#111827;border-bottom:1px solid #31445c;font-size:20px;font-weight:800}.grid{display:grid;grid-template-columns:minmax(0,1fr)420px;min-height:calc(100vh - 52px)}main{display:grid;grid-template-rows:minmax(420px,1fr)300px;gap:10px;padding:10px}.panel{border:1px solid #31445c;background:#050a12;display:flex;justify-content:center;align-items:center;overflow:hidden}.panel img{max-width:100%;max-height:100%;object-fit:contain}.map img{image-rendering:pixelated}#mapwrap{position:relative;width:100%;height:100%;cursor:crosshair}#markers{position:absolute;inset:0;pointer-events:none}.marker{position:absolute;transform:translate(-50%,-50%);width:16px;height:16px;border:2px solid #ffb020;background:rgba(255,176,32,.22);border-radius:50%;pointer-events:auto;cursor:pointer}.marker:hover{background:rgba(255,176,32,.65)}.marker .count{position:absolute;top:-8px;right:-8px;min-width:15px;height:15px;padding:0 3px;border-radius:8px;background:#d63d2f;color:#fff;font-size:10px;line-height:15px;text-align:center;font-weight:700}#popup{position:absolute;z-index:6;min-width:240px;max-width:360px;background:#0f1723;border:1px solid rgba(255,176,32,.45);border-radius:10px;padding:10px 12px;box-shadow:0 8px 28px rgba(0,0,0,.55);display:none;pointer-events:auto;font-size:13px}#popup h4{margin:0 0 6px;font-size:14px;color:#ffd27a}#popup .item{border-top:1px solid #23344d;padding:6px 0}#popup .close{float:right;cursor:pointer;color:#8fa3bd;font-size:13px}aside{padding:12px;border-left:1px solid #31445c;overflow:auto}.card{border:1px solid #33475f;background:#111b29;padding:8px 10px;border-radius:8px;margin:8px 0}.muted{color:#9db0c7;font-size:12px}pre{white-space:pre-wrap;word-break:break-word;font-size:12px}.badge{display:inline-block;border:1px solid #48617d;border-radius:999px;padding:6px 10px;margin:0 5px;background:#101a28}@media(max-width:1000px){.grid{grid-template-columns:1fr}aside{border-left:0;border-top:1px solid #31445c}main{height:72vh}}</style></head><body><header>3090 阶段1感知服务 <span id=health class=badge>连接中</span><span id=pi class=badge>Pi -</span><span id=lm class=badge>landmarks -</span></header><div class=grid><main><div class="panel map"><div id=mapwrap><img id=map src="/proxy/map.png"><div id=markers></div><div id=popup></div></div></div><div class=panel><img src="/proxy/camera.mjpg"></div></main><aside><h3>已定位地标</h3><div id=landmarks></div><h3>服务状态</h3><pre id=status></pre><h3>树莓派状态</h3><pre id=pistatus></pre></aside></div><script>
const esc=s=>String(s??'').replace(/[&<>\"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const card=x=>`<div class=card><b>${esc(x.name||x.class_name||x.type)}</b><div class=muted>${esc(x.type||'')} · seen ${x.seen_count||1} · conf ${Number(x.map_confidence||x.confidence||0).toFixed(2)}</div><div class=xy>${x.map_xy?'('+Number(x.map_xy[0]).toFixed(2)+', '+Number(x.map_xy[1]).toFixed(2)+')':''}</div></div>`;
let mapView=null,clusters=[];
function markerPos(px,py){const img=document.getElementById('map'),wrap=document.getElementById('mapwrap');const nw=img.naturalWidth||(mapView?mapView.width:0),nh=img.naturalHeight||(mapView?mapView.height:0);const w=wrap.clientWidth,h=wrap.clientHeight;if(!w||!h||!nw||!nh)return null;const scale=Math.min(w/nw,h/nh);return {x:(w-nw*scale)/2+px*scale,y:(h-nh*scale)/2+py*scale};}
function renderMarkers(){const box=document.getElementById('markers');box.innerHTML='';for(const c of clusters||[]){const pos=markerPos(c.pixel[0],c.pixel[1]);if(!pos)continue;const d=document.createElement('div');d.className='marker';d.style.left=pos.x+'px';d.style.top=pos.y+'px';d.title=(c.names||[]).join(', ');if(c.count>1){const b=document.createElement('span');b.className='count';b.textContent=c.count;d.appendChild(b);}d.addEventListener('click',ev=>{ev.stopPropagation();showPopup(c,pos);});box.appendChild(d);}}
function showPopup(cluster,pos){const pop=document.getElementById('popup'),wrap=document.getElementById('mapwrap');let html='<span class=close onclick="hidePopup()">✕ 关闭</span>';html+='<h4>此处 '+(cluster.count||1)+' 个标注</h4>';for(const it of cluster.items||[]){html+='<div class=item><b>'+esc(it.name||it.class_name||it.type||'?')+'</b> <span class=muted>'+esc(it.canonical_class||it.type||'')+' · seen '+esc(it.seen_count??1)+' · '+esc(it.localization_source||it.source||'')+' · conf '+Number(it.map_confidence||it.confidence||0).toFixed(2)+'</span>';if(it.map_xy)html+='<div class=muted>位置 ('+Number(it.map_xy[0]).toFixed(2)+', '+Number(it.map_xy[1]).toFixed(2)+') m · σ '+Number(it.position_sigma_m||0).toFixed(2)+'m</div>';const at=it.attributes||{};const desc=[at.color,at.material].filter(Boolean).join(' ');if(desc||(at.mobility&&at.mobility!=='unknown'))html+='<div class=muted>'+esc(desc)+(at.mobility&&at.mobility!=='unknown'?' · 移动性:'+esc(at.mobility):'')+'</div>';if(it.visual_description)html+='<div class=muted>'+esc(String(it.visual_description).slice(0,120))+'</div>';html+='</div>';}pop.innerHTML=html;let x=pos.x+16,y=pos.y-12;const pw=pop.offsetWidth,ph=pop.offsetHeight;if(x+pw>wrap.clientWidth)x=pos.x-pw-16;if(y+ph>wrap.clientHeight)y=wrap.clientHeight-ph-4;if(y<4)y=4;pop.style.left=x+'px';pop.style.top=y+'px';pop.style.display='block';}
function hidePopup(){document.getElementById('popup').style.display='none';}
function onMapClick(ev){const wrap=document.getElementById('mapwrap'),img=document.getElementById('map');const rect=wrap.getBoundingClientRect();const cx=ev.clientX-rect.left,cy=ev.clientY-rect.top;const nw=img.naturalWidth||(mapView?mapView.width:0),nh=img.naturalHeight||(mapView?mapView.height:0);if(!nw||!nh)return;const scale=Math.min(wrap.clientWidth/nw,wrap.clientHeight/nh);const ox=(wrap.clientWidth-nw*scale)/2,oy=(wrap.clientHeight-nh*scale)/2;const ix=(cx-ox)/scale,iy=(cy-oy)/scale;let best=null,bd=1e9;for(const c of clusters||[]){const d=Math.hypot(c.pixel[0]-ix,c.pixel[1]-iy);if(d<bd){bd=d;best=c;}}if(best&&bd<=60)showPopup(best,{x:ox+best.pixel[0]*scale,y:oy+best.pixel[1]*scale});else hidePopup();}
async function tick(){try{const h=await fetch('/health',{cache:'no-store'}).then(r=>r.json());health.textContent=h.ok?'感知就绪':'异常';status.textContent=JSON.stringify(h,null,2)}catch(e){health.textContent='断开';status.textContent=String(e)}try{const [s,m,p]=await Promise.all([fetch('/proxy/status',{cache:'no-store'}).then(r=>r.json()),fetch('/proxy/semantic',{cache:'no-store'}).then(r=>r.json()),fetch('/proxy/map.json',{cache:'no-store'}).then(r=>r.json())]);pi.textContent=s.running?'Pi运行中':'Pi未运行';pistatus.textContent=JSON.stringify(s,null,2);lm.textContent='landmarks '+(m.landmarks||[]).length;landmarks.innerHTML=(m.landmarks||[]).slice().reverse().slice(0,40).map(card).join('');if(p&&p.map_view){mapView=p.map_view;clusters=p.map_clusters||[];renderMarkers();}map.src='/proxy/map.png?t='+Date.now()}catch(e){pi.textContent='Pi断开';pistatus.textContent=String(e)}}setInterval(tick,800);tick();document.getElementById('mapwrap').addEventListener('click',onMapClick);</script></body></html>"""


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.I).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        value = json.loads(cleaned)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if not match:
            return {}
        try:
            value = json.loads(match.group(0))
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            return {}


def clamp(value: Any, lower: float, upper: float, default: float = 0.0) -> float:
    try:
        return max(lower, min(upper, float(value)))
    except (TypeError, ValueError):
        return default


def strict_bool(value: Any) -> bool:
    if value is True or value == 1:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "是"}
    return False


def normalize_bbox(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return []
    try:
        bbox = [float(v) for v in value]
    except (TypeError, ValueError):
        return []
    if max(abs(v) for v in bbox) <= 1.5:
        bbox = [v * 1000.0 for v in bbox]
    x1, y1, x2, y2 = bbox
    if not (0.0 <= x1 < x2 <= 1000.0 and 0.0 <= y1 < y2 <= 1000.0):
        return []
    return [round(v, 2) for v in bbox]


def _confidence_value(value: Any) -> float:
    """Qwen 常把示例里的 confidence 0.0 当占位符照抄，0.0 视为"未提供"回落 0.5。"""
    conf = clamp(value, 0.0, 1.0, 0.5)
    return 0.5 if conf <= 0.0 else conf


def normalize_detection(value: Any, default_camera: str = "") -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    camera_id = str(value.get("camera_id", default_camera)).strip().lower()
    if camera_id not in {"left", "right"}:
        return None
    bbox = normalize_bbox(value.get("bbox_1000", value.get("bbox", value.get("camera_bbox"))))
    if not bbox:
        return None
    class_name = str(value.get("class_name", value.get("name", value.get("type", "unknown")))).strip()
    if not class_name:
        class_name = "unknown"
    attrs = value.get("attributes", {})
    if not isinstance(attrs, dict):
        attrs = {}
    for key in ("color", "material", "description", "mobility"):
        if key in value and key not in attrs:
            attrs[key] = value.get(key)
    # 外观指纹：颜色_材质_相对大小，树莓派用它区分同一位置同一类别的多个实例。
    appearance_key = str(value.get("appearance_key", "") or "").strip().lower()
    if not appearance_key and ("color" in attrs or "material" in attrs):
        appearance_key = "_".join([
            str(attrs.get("color", "")).strip().lower(),
            str(attrs.get("material", "")).strip().lower(),
        ]).strip("_")
    return {
        "detection_id": str(value.get("detection_id", "")),
        "camera_id": camera_id,
        "class_name": class_name[:80],
        "confidence": _confidence_value(value.get("confidence", 0.5)),
        "bbox_1000": bbox,
        "source": str(value.get("source", "qwen"))[:32],
        "appearance_key": appearance_key[:80],
        "count": int(value.get("count", 0) or 0) if value.get("count") not in (None, "", 0) else 0,
        "attributes": {
            "color": str(attrs.get("color", ""))[:60],
            "material": str(attrs.get("material", ""))[:80],
            "description": str(attrs.get("description", ""))[:240],
            "mobility": str(attrs.get("mobility", "unknown"))[:40],
        },
    }


def iou(a: list[float], b: list[float]) -> float:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    inter = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(1e-9, area_a + area_b - inter)


class YoloDetector:
    def __init__(self, model_path: str, device: str, confidence: float, image_size: int) -> None:
        self.model_path = str(model_path).strip()
        self.device = device
        self.confidence = float(confidence)
        self.image_size = int(image_size)
        self.model: Any | None = None
        self.error = ""
        self._lock = threading.Lock()

    def enabled(self) -> bool:
        return bool(self.model_path)

    def _load(self) -> None:
        if self.model is not None or not self.enabled():
            return
        path = Path(self.model_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"YOLO模型不存在: {path}")
        from ultralytics import YOLO

        self.model = YOLO(str(path))
        print(f"[stage1-yolo] model ready: {path}")

    def detect(self, image: Image.Image, camera_id: str) -> list[dict[str, Any]]:
        if not self.enabled():
            return []
        with self._lock:
            try:
                self._load()
                assert self.model is not None
                results = self.model.predict(
                    source=np.asarray(image.convert("RGB")),
                    conf=self.confidence,
                    imgsz=self.image_size,
                    device=self.device,
                    verbose=False,
                )
                output: list[dict[str, Any]] = []
                width, height = image.size
                for result in results:
                    names = result.names
                    boxes = getattr(result, "boxes", None)
                    if boxes is None:
                        continue
                    for index, xyxy in enumerate(boxes.xyxy.detach().cpu().numpy()):
                        cls_id = int(boxes.cls[index].item())
                        confidence = float(boxes.conf[index].item())
                        x1, y1, x2, y2 = [float(v) for v in xyxy]
                        output.append(
                            {
                                "camera_id": camera_id,
                                "class_name": str(names.get(cls_id, cls_id)),
                                "confidence": confidence,
                                "bbox_1000": [
                                    x1 / width * 1000.0,
                                    y1 / height * 1000.0,
                                    x2 / width * 1000.0,
                                    y2 / height * 1000.0,
                                ],
                                "source": "yolo",
                                "attributes": {},
                            }
                        )
                self.error = ""
                return [d for item in output if (d := normalize_detection(item)) is not None]
            except Exception as exc:
                self.error = str(exc)
                print(f"[stage1-yolo] disabled for this request: {exc}")
                return []


class QwenDetector:
    def __init__(
        self,
        model_path: str,
        device: str,
        local_files_only: bool,
        max_new_tokens: int,
        mock: bool,
    ) -> None:
        self.model_path = str(model_path)
        self.device = str(device)
        self.local_files_only = bool(local_files_only)
        self.max_new_tokens = int(max_new_tokens)
        self.mock = bool(mock)
        self.model: Any | None = None
        self.processor: Any | None = None
        self.torch: Any | None = None
        self.error = ""
        self._lock = threading.Lock()

    def _load(self) -> None:
        if self.model is not None:
            return
        path = Path(self.model_path).expanduser()
        if self.local_files_only and not path.exists():
            raise FileNotFoundError(f"Qwen模型不存在: {path}")
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        use_cuda = self.device.lower().startswith("cuda")
        if use_cuda and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        dtype = torch.bfloat16 if use_cuda else torch.float32
        kwargs = {"local_files_only": self.local_files_only, "low_cpu_mem_usage": True}
        try:
            model = AutoModelForImageTextToText.from_pretrained(str(path), dtype=dtype, **kwargs)
        except TypeError:
            model = AutoModelForImageTextToText.from_pretrained(str(path), torch_dtype=dtype, **kwargs)
        model.to("cuda" if use_cuda else "cpu")
        model.eval()
        self.model = model
        self.processor = AutoProcessor.from_pretrained(str(path), local_files_only=self.local_files_only)
        self.torch = torch
        print(f"[stage1-qwen] model ready: {path} on {self.device}")

    @staticmethod
    def prompt(context: dict[str, Any]) -> str:
        return f"""
你是双目移动机器人中的视觉语义检测模块。输入 LEFT CAMERA 和 RIGHT CAMERA 两张独立图像。
目标：帮助机器人把"这里有什么"标注到地图上。机器人会把你给的 bbox 换算成方位角再定位，
所以 bbox 的几何准确性直接决定标注是否落在真实物体上。

硬性限制：
1. 禁止输出地图坐标、距离、角度、深度、雷达特征编号、SLAM位姿或运动建议。
2. bbox_1000 是单张图内 0-1000 相对坐标 [x1,y1,x2,y2]。
3. bbox 必须紧贴物体本身：
   - 框的下边放在物体与地面/墙面的接触线（落地基线）上，不要悬空、不要被画面截断。
   - 不要框整个场景/画面；不确定物体范围时降低 confidence，而不是给一个大框。
   - 物体被画面边缘截断（贴边/只剩一部分）时，该 bbox 不可靠：降低 confidence 或不返回。
4. 同一个物体在左右图都可见时，分别返回 left 和 right 两条，class_name 保持一致；
   只在一侧可见时在 description 里注明。
5. 只返回对地图有用的物体：柜子、桌子、椅子、垃圾桶、纸箱、门、门框、柱子、插座、
   扫帚、簸箕、瓶子、箱子、固定设备、墙体/墙角结构。不返回人、手、屏幕、窗外物体、纯墙面纹理。
6. 低位摄像头可能只看到椅腿/桌腿/柜体底部，可保守标注 chair-like/table-like/cabinet-like 并降低 confidence。
7. 不要臆造：看不清就返回空 detections。confidence 是 0.0~1.0 的实数，代表把握程度：
   很确定 0.8~0.95，比较确定 0.6~0.8，勉强看清 0.4~0.6；把握低于 0.4 就不返回该物体。
   0.0 只表示占位符，实际输出不要写 0.0。
8. 每张图选 3-6 个最有把握、最有用、位置最明确的候选；同一物体只输出一次。
9. appearance_key：用"颜色_材质_相对大小"（英文小写，如 black_plastic_medium / white_metal_large）
   描述该实例，帮助机器人跨视角区分同一类别下的不同物体；看不清就用 ""。
10. count：这一类别在当前图像中可见的实例总数（如两把挨着的椅子填 2）；不确定时省略。
    它是软提示，不要因为画面只显示一部分就硬报数量。

上下文仅用于绑定 observation_id，不用于几何推理：
{json.dumps(context, ensure_ascii=False)}

只返回JSON：
{{
  "scene_summary": "简短说明",
  "detections": [
    {{
      "camera_id": "left|right",
      "class_name": "chair",
      "confidence": 0.85,
      "bbox_1000": [x1,y1,x2,y2],
      "appearance_key": "black_plastic_medium",
      "count": 1,
      "attributes": {{
        "color": "",
        "material": "",
        "description": "",
        "mobility": "fixed|occasionally_moved|frequently_moved|unknown"
      }}
    }}
  ]
}}
""".strip()

    def detect(self, left: Image.Image, right: Image.Image, context: dict[str, Any]) -> tuple[list[dict[str, Any]], str, str]:
        if self.mock:
            raw = context.get("mock_detections", [])
            detections = [d for item in raw if (d := normalize_detection(item)) is not None]
            return detections, "mock", ""
        with self._lock:
            try:
                self._load()
                assert self.model is not None and self.processor is not None and self.torch is not None
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": left.convert("RGB")},
                            {"type": "text", "text": "LEFT CAMERA image."},
                            {"type": "image", "image": right.convert("RGB")},
                            {"type": "text", "text": "RIGHT CAMERA image."},
                            {"type": "text", "text": self.prompt(context)},
                        ],
                    }
                ]
                inputs = self.processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(self.model.device)
                with self.torch.inference_mode():
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                    )
                trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, output_ids)]
                raw_text = self.processor.batch_decode(
                    trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
                parsed = parse_json_object(raw_text)
                detections = [
                    d
                    for item in parsed.get("detections", [])
                    if (d := normalize_detection(item)) is not None
                ]
                self.error = ""
                return detections, str(parsed.get("scene_summary", ""))[:300], raw_text[:2000]
            except Exception as exc:
                self.error = str(exc)
                raise

    @staticmethod
    def verification_prompt(
        mode: str,
        target_query: str,
        destination_query: str,
        context: dict[str, Any],
    ) -> str:
        return f"""
你是移动机械臂的视觉安全复核器。只判断两张当前相机图像里的可见事实，不规划动作，
不输出舵机值、运动命令、地图坐标、距离或思维过程。看不清时必须返回 false。

复核阶段：{mode}
目标物体：{target_query}
目的地：{destination_query}
上下文：{json.dumps(context, ensure_ascii=False)}

字段含义：
- target_visible：目标物体当前清楚可见；
- target_centered：目标位于夹爪/图像中央的抓放区域；
- object_in_gripper：有直接视觉证据表明目标已被夹爪夹住并离开原位置；
- target_in_destination：有直接视觉证据表明目标已落入指定容器或放置区域；
- confidence：对上述判断的总体置信度。遮挡、画面过期、夹爪不可见或目标相似时降低置信度。

严格只返回一个 JSON：
{{"mode":"{mode}","target_visible":false,"target_centered":false,
"object_in_gripper":false,"target_in_destination":false,"confidence":0.0,
"evidence":"不超过80字的可见证据"}}
""".strip()

    @staticmethod
    def normalize_verification(payload: Any, requested_mode: str) -> dict[str, Any]:
        value = payload if isinstance(payload, dict) else {}
        mode = str(value.get("mode") or requested_mode).strip()
        if mode != requested_mode:
            mode = requested_mode
        return {
            "mode": mode,
            "target_visible": strict_bool(value.get("target_visible", False)),
            "target_centered": strict_bool(value.get("target_centered", False)),
            "object_in_gripper": strict_bool(value.get("object_in_gripper", False)),
            "target_in_destination": strict_bool(value.get("target_in_destination", False)),
            "confidence": clamp(value.get("confidence"), 0.0, 1.0, 0.0),
            "evidence": str(value.get("evidence") or "")[:240],
        }

    def verify(
        self,
        left: Image.Image,
        right: Image.Image,
        mode: str,
        target_query: str,
        destination_query: str,
        context: dict[str, Any],
    ) -> tuple[dict[str, Any], str]:
        if mode not in {"align_for_grasp", "grasp", "verify_grasp", "release", "verify_completion"}:
            raise ValueError(f"unsupported verification mode: {mode}")
        if self.mock:
            result = self.normalize_verification(context.get("mock_verification", {}), mode)
            return result, "mock"
        with self._lock:
            self._load()
            assert self.model is not None and self.processor is not None and self.torch is not None
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": left.convert("RGB")},
                        {"type": "text", "text": "LEFT CAMERA current image."},
                        {"type": "image", "image": right.convert("RGB")},
                        {"type": "text", "text": "RIGHT CAMERA current image."},
                        {
                            "type": "text",
                            "text": self.verification_prompt(
                                mode, target_query, destination_query, context
                            ),
                        },
                    ],
                }
            ]
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)
            with self.torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=min(160, self.max_new_tokens),
                    do_sample=False,
                )
            trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, output_ids)]
            raw_text = self.processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            result = self.normalize_verification(parse_json_object(raw_text), mode)
            self.error = ""
            return result, raw_text[:2000]


class Stage1PerceptionService:
    def __init__(self, args: argparse.Namespace) -> None:
        self.yolo = YoloDetector(
            args.yolo_model,
            args.yolo_device,
            args.yolo_confidence,
            args.yolo_image_size,
        )
        self.qwen = None if args.disable_qwen else QwenDetector(
            args.model_path,
            args.device,
            not args.allow_download,
            args.max_new_tokens,
            args.mock,
        )
        self.started_s = time.monotonic()
        self.requests = 0
        self.last_elapsed_s = 0.0
        self.last_error = ""

    @staticmethod
    def merge(yolo: list[dict[str, Any]], qwen: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged = [dict(item) for item in yolo]
        for candidate in qwen:
            best_index = -1
            best_iou = 0.0
            for index, existing in enumerate(merged):
                if existing["camera_id"] != candidate["camera_id"]:
                    continue
                overlap = iou(existing["bbox_1000"], candidate["bbox_1000"])
                if overlap > best_iou:
                    best_iou = overlap
                    best_index = index
            if best_index >= 0 and best_iou >= 0.45:
                existing = merged[best_index]
                # YOLO保留更确定的几何框；Qwen补充细粒度名称和属性。
                if candidate["confidence"] >= 0.45 and candidate["class_name"] != "unknown":
                    existing["class_name"] = candidate["class_name"]
                existing["attributes"] = candidate.get("attributes", {})
                existing["source"] = "yolo+qwen"
                existing["confidence"] = max(existing["confidence"], candidate["confidence"])
                # Qwen 补充外观指纹；YOLO 框保持几何，外观用于跨视角区分同类实例。
                if candidate.get("appearance_key"):
                    existing["appearance_key"] = candidate["appearance_key"]
                if candidate.get("count"):
                    existing["count"] = candidate["count"]
            else:
                merged.append(dict(candidate))
        for index, item in enumerate(merged, start=1):
            item["detection_id"] = f"det_{index:03d}"
        return merged

    def perceive(
        self,
        observation_id: str,
        left: Image.Image,
        right: Image.Image,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        started = time.perf_counter()
        self.requests += 1
        yolo_detections = self.yolo.detect(left, "left") + self.yolo.detect(right, "right")
        qwen_detections: list[dict[str, Any]] = []
        summary = ""
        raw_text = ""
        qwen_error = ""
        if self.qwen is not None:
            try:
                qwen_detections, summary, raw_text = self.qwen.detect(left, right, context)
            except Exception as exc:
                qwen_error = str(exc)
                print(f"[stage1-qwen] request failed: {exc}")
        detections = self.merge(yolo_detections, qwen_detections)
        elapsed = time.perf_counter() - started
        self.last_elapsed_s = elapsed
        self.last_error = qwen_error
        print(
            f"[stage1-perception] observation={observation_id} elapsed={elapsed:.2f}s "
            f"yolo={len(yolo_detections)} qwen={len(qwen_detections)} merged={len(detections)}",
            flush=True,
        )
        return {
            "ok": True,
            "schema": "semantic_stage1_perception_v1",
            "observation_id": observation_id,
            "elapsed_s": round(elapsed, 3),
            "scene_summary": summary,
            "detections": detections,
            "diagnostics": {
                "yolo_count": len(yolo_detections),
                "qwen_count": len(qwen_detections),
                "qwen_error": qwen_error,
                "raw_qwen_text": raw_text,
                "geometry_fields_returned": False,
            },
        }

    def verify_manipulation(
        self,
        left: Image.Image,
        right: Image.Image,
        mode: str,
        target_query: str,
        destination_query: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        if self.qwen is None:
            return {
                "ok": False,
                "schema": "manipulation_visual_verification.v1",
                "mode": mode,
                "error": "Qwen verification is disabled",
            }
        started = time.perf_counter()
        self.requests += 1
        result, raw_text = self.qwen.verify(
            left,
            right,
            mode,
            target_query,
            destination_query,
            context,
        )
        elapsed = time.perf_counter() - started
        self.last_elapsed_s = elapsed
        return {
            "ok": True,
            "schema": "manipulation_visual_verification.v1",
            **result,
            "elapsed_s": round(elapsed, 3),
            "diagnostics": {"raw_qwen_text": raw_text},
        }

    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "service": "stage1_stereo_semantic_perception",
            "contract": "returns detections only; no map_xy/lidar_feature_id",
            "requests": self.requests,
            "last_elapsed_s": self.last_elapsed_s,
            "last_error": self.last_error,
            "qwen_enabled": self.qwen is not None,
            "qwen_error": "" if self.qwen is None else self.qwen.error,
            "manipulation_verification_supported": self.qwen is not None,
            "yolo_enabled": self.yolo.enabled(),
            "yolo_error": self.yolo.error,
            "pi_url": PI_URL or None,
        }


@app.get("/")
def index() -> HTMLResponse:
    return HTMLResponse(HTML)


@app.get("/health")
def health() -> dict[str, Any]:
    if SERVICE is None:
        return {"ok": False, "error": "not_initialized"}
    return SERVICE.health()


@app.post("/perceive")
async def perceive(
    left_image: UploadFile,
    right_image: UploadFile,
    observation_id: str = Form(...),
    context_json: str = Form("{}"),
) -> JSONResponse:
    if SERVICE is None:
        return JSONResponse({"ok": False, "error": "not_initialized"}, status_code=503)
    try:
        context = json.loads(context_json)
        if not isinstance(context, dict):
            context = {}
    except json.JSONDecodeError:
        context = {}
    try:
        left = Image.open(io.BytesIO(await left_image.read())).convert("RGB")
        right = Image.open(io.BytesIO(await right_image.read())).convert("RGB")
        if left.size != right.size:
            raise ValueError(f"左右图尺寸不一致: {left.size} vs {right.size}")
        result = SERVICE.perceive(str(observation_id), left, right, context)
        return JSONResponse(result)
    except Exception as exc:
        return JSONResponse(
            {"ok": False, "observation_id": str(observation_id), "error": repr(exc)},
            status_code=500,
        )


@app.post("/verify")
async def verify_manipulation(
    left_image: UploadFile,
    right_image: UploadFile,
    mode: str = Form(...),
    target_query: str = Form(""),
    destination_query: str = Form(""),
    context_json: str = Form("{}"),
) -> JSONResponse:
    if SERVICE is None:
        return JSONResponse({"ok": False, "error": "not_initialized"}, status_code=503)
    try:
        context = json.loads(context_json)
        if not isinstance(context, dict):
            context = {}
    except json.JSONDecodeError:
        context = {}
    try:
        left = Image.open(io.BytesIO(await left_image.read())).convert("RGB")
        right = Image.open(io.BytesIO(await right_image.read())).convert("RGB")
        if left.size != right.size:
            raise ValueError(f"左右图尺寸不一致: {left.size} vs {right.size}")
        result = SERVICE.verify_manipulation(
            left,
            right,
            str(mode),
            str(target_query),
            str(destination_query),
            context,
        )
        return JSONResponse(result, status_code=200 if result.get("ok") else 503)
    except ValueError as exc:
        return JSONResponse({"ok": False, "mode": str(mode), "error": str(exc)}, status_code=400)
    except Exception as exc:
        return JSONResponse(
            {"ok": False, "mode": str(mode), "error": repr(exc)},
            status_code=500,
        )


def require_pi() -> str:
    if not PI_URL:
        raise RuntimeError("未配置 --pi-url")
    return PI_URL.rstrip("/")


@app.get("/proxy/status")
def proxy_status() -> JSONResponse:
    try:
        response = PI_SESSION.get(require_pi() + "/api/status", timeout=(0.5, 1.5))
        response.raise_for_status()
        return JSONResponse(response.json())
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=502)


@app.get("/proxy/semantic")
def proxy_semantic() -> JSONResponse:
    try:
        response = PI_SESSION.get(require_pi() + "/api/semantic", timeout=(0.5, 1.5))
        response.raise_for_status()
        return JSONResponse(response.json())
    except Exception as exc:
        return JSONResponse({"landmarks": [], "error": str(exc)}, status_code=502)


@app.get("/proxy/map.json")
def proxy_map_json() -> JSONResponse:
    try:
        response = PI_SESSION.get(require_pi() + "/api/map.json", timeout=(0.5, 2.0))
        response.raise_for_status()
        return JSONResponse(response.json())
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=502)


@app.get("/proxy/map.png")
def proxy_map() -> Response:
    try:
        response = PI_SESSION.get(require_pi() + "/api/map.png", timeout=(0.5, 2.0))
        response.raise_for_status()
        return Response(response.content, media_type="image/png")
    except Exception as exc:
        return Response(str(exc).encode(), status_code=502, media_type="text/plain")


@app.get("/proxy/camera.mjpg")
def proxy_camera() -> StreamingResponse:
    def frames():
        while True:
            try:
                response = PI_SESSION.get(require_pi() + "/api/camera.jpg", timeout=(0.4, 1.2))
                response.raise_for_status()
                body = response.content
            except Exception:
                body = b""
            if body:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n"
                yield f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
                yield body + b"\r\n"
            time.sleep(0.10)

    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")


def main() -> None:
    global SERVICE, PI_URL
    parser = argparse.ArgumentParser(description="阶段1：YOLO/Qwen双目语义检测服务")
    parser.add_argument("--model-path", default="~/3090/models/qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=900)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--disable-qwen", action="store_true")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--yolo-model", default="~/3090/models/yolo/yolo11n-seg.pt")
    parser.add_argument("--yolo-device", default="0")
    parser.add_argument("--yolo-confidence", type=float, default=0.25)
    parser.add_argument("--yolo-image-size", type=int, default=960)
    parser.add_argument("--pi-url", default="")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()
    args.model_path = str(Path(args.model_path).expanduser())
    args.yolo_model = str(Path(args.yolo_model).expanduser()) if args.yolo_model else ""
    PI_URL = args.pi_url.strip().rstrip("/")
    SERVICE = Stage1PerceptionService(args)
    print("[stage1-perception] geometry contract: detections only; no map coordinates")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
