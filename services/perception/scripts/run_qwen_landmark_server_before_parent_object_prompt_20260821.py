from __future__ import annotations

import argparse
import io
import json
import re
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from PIL import Image
import requests
import uvicorn


app = FastAPI(title="Standalone Qwen Landmark Server")
LANDMARKER: "QwenLandmarker | None" = None
PI_MAP_URL = ""
PI_SESSION = requests.Session()
PI_SESSION.trust_env = False
PI_CACHE: dict[str, tuple[bytes, str, float]] = {}


HUB_HTML = """<!doctype html><html lang="zh-CN"><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>Robot Semantic Map Hub</title>
<style>
:root{color-scheme:dark;--bg:#0d131c;--panel:#121b28;--line:#2b3b50;--text:#f4f7fb;--muted:#9fb0c4;--green:#10b981;--red:#be3455;--cyan:#4fd1c5}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:Arial,"Microsoft YaHei",sans-serif}.top{height:58px;border-bottom:1px solid var(--line);display:flex;align-items:center;gap:14px;padding:0 18px;background:#111827}.title{font-size:20px;font-weight:800}.badge{border:1px solid #40536b;border-radius:999px;padding:7px 12px;background:#0c1421;white-space:nowrap}.wrap{display:grid;grid-template-columns:minmax(0,1fr)460px;height:calc(100vh - 59px)}main{display:grid;grid-template-rows:minmax(0,1fr)280px;gap:10px;padding:10px;overflow:hidden}.panel{border:1px solid var(--line);background:#050a13;overflow:hidden;display:flex;align-items:center;justify-content:center}.panel img{max-width:100%;max-height:100%;object-fit:contain}.panel.map img{image-rendering:pixelated;background:#d0d0d0}.mapbox{position:relative;display:inline-flex;max-width:100%;max-height:100%}.mapbox img{display:block}.mapbox canvas{position:absolute;inset:0;width:100%;height:100%;cursor:pointer}.cluster-card{border-color:#8a5b05;background:#1d1720}.cluster-items{display:grid;gap:8px}.pill{display:inline-block;border:1px solid #4a627d;border-radius:999px;padding:2px 8px;margin:2px 4px 2px 0;color:#cfe1ff;background:#101a28}.desc{margin-top:5px;color:#d7e3f2}aside{border-left:1px solid var(--line);overflow:auto;padding:14px;background:#0f1622}.buttons{display:grid;grid-template-columns:1fr 1fr;gap:8px}.buttons button{padding:11px;border-radius:8px;border:1px solid #465970;background:#172234;color:#fff;font-weight:800}.buttons .go{background:#064e3b;border-color:var(--green)}.buttons .stop{background:#581c2b;border-color:var(--red)}.card{border:1px solid #33465d;border-radius:8px;padding:10px;margin-bottom:10px;background:#111c2a}.muted{color:var(--muted);font-size:12px}.name{font-size:17px;font-weight:800}.xy{font-family:Consolas,monospace;color:#d7e3f2}pre{white-space:pre-wrap;word-break:break-word;font-size:12px;color:#cbd5e1}@media(max-width:1050px){.wrap{grid-template-columns:1fr;height:auto;min-height:calc(100vh - 59px)}main{height:70vh}.top{overflow-x:auto}aside{border-left:0;border-top:1px solid var(--line)}}@media(max-width:650px){main{grid-template-rows:minmax(0,1fr)210px}.buttons{grid-template-columns:1fr}.title{font-size:17px}}
</style></head><body>
<div class=top><span class=title>Robot Semantic Map Hub</span><span class=badge id=conn>connecting</span><span class=badge id=scan>scan -</span><span class=badge id=acc>accepted -</span><span class=badge id=rmse>rmse -</span><span class=badge id=labels>labels -</span><span class=badge id=semantic>semantic -</span><span class=badge id=explore>explore -</span></div>
<div class=wrap><main><div class="panel map"><div class=mapbox><img id=map src="/proxy/map.png"><canvas id=marks></canvas></div></div><div class=panel><img id=camera src="/proxy/camera.mjpg"></div></main>
<aside><h3>巡航控制</h3><div class="card buttons"><button class=go onclick="post('/proxy/patrol/start')">开始巡航</button><button class=stop onclick="post('/proxy/patrol/stop')">停止</button><button onclick="post('/proxy/drive/forward')">前进一步</button><button onclick="post('/proxy/drive/left')">左转一步</button><button onclick="post('/proxy/drive/right')">右转一步</button><button onclick="post('/proxy/drive/stop')">单步停止</button><button class=stop onclick="post('/proxy/map/reset')">重新画地图</button></div><div class=muted id=ctrl></div><h3>Map Point Detail</h3><div class="card cluster-card" id=clusterDetail>Click an orange map point to inspect nearby labels.</div>
<h3>可导航/地图标签</h3><div id=targets></div><h3>全部语义记录</h3><div id=list></div><h3>状态 JSON</h3><pre id=json></pre></aside></div>
<script>
async function getJson(u){const r=await fetch(u,{cache:'no-store'});if(!r.ok)throw new Error(await r.text());return await r.json()}
async function post(u){const r=await fetch(u,{method:'POST'});ctrl.textContent=JSON.stringify(await r.json());tick()}
let clusters=[];let selectedCluster=null;let latestStatus=null;let latestSemantic=null;
function tagList(x){return [x.type,x.color,x.material,x.mobility,x.behind_glass?'behind glass':'',x.reachable===false?'unreachable':'',x.navigation_allowed===false?'not navigable':''].filter(Boolean).map(v=>`<span class=pill>${v}</span>`).join('')}
function card(x){const tags=tagList(x);return `<div class=card><div class=name>${x.name||x.id||'unknown'}</div><div>${tags}</div><div class=muted>${x.role||'object'} / map ${x.map_annotation?'yes':'no'} / stability ${Number(x.stability||0).toFixed(2)} / score ${Number(x.anchor_score||0).toFixed(2)} / seen ${x.seen_count||1}</div><div class=xy>(${Number(x.map_xy?.[0]||0).toFixed(2)}, ${Number(x.map_xy?.[1]||0).toFixed(2)})</div><div class=desc>${x.appearance||x.visual_description||x.reason||''}</div><div class=muted>${x.vision_lidar_alignment||''}</div></div>`}
function itemDetail(x){const tags=tagList(x);return `<div><div class=name>${x.name||x.id||'unknown'}</div><div>${tags}</div><div class=xy>(${Number(x.map_xy?.[0]||0).toFixed(2)}, ${Number(x.map_xy?.[1]||0).toFixed(2)})</div><div class=desc>${x.appearance||x.visual_description||x.reason||''}</div></div>`}
function showCluster(c){selectedCluster=c;if(!c){clusterDetail.textContent='Click an orange map point to inspect nearby labels.';return}clusterDetail.innerHTML=`<div class=name>${c.label||'Map labels'} - ${c.count||0} items</div><div class=xy>(${Number(c.map_xy?.[0]||0).toFixed(2)}, ${Number(c.map_xy?.[1]||0).toFixed(2)})</div><div class=cluster-items>${(c.items||[]).map(itemDetail).join('')}</div>`}
function mapPointToPixel(xy){const v=latestSemantic?.map_view;if(!v||!xy)return null;const col=(Number(xy[0])-Number(v.origin_x_m))/Number(v.resolution_m);const row=(Number(xy[1])-Number(v.origin_y_m))/Number(v.resolution_m);return [col,Number(v.height)-1-row]}
function drawClusters(){const canvas=document.getElementById('marks');const img=document.getElementById('map');if(!canvas||!img||!img.naturalWidth)return;const rect=img.getBoundingClientRect();const ratio=window.devicePixelRatio||1;canvas.width=Math.max(1,Math.round(rect.width*ratio));canvas.height=Math.max(1,Math.round(rect.height*ratio));canvas.style.width=rect.width+'px';canvas.style.height=rect.height+'px';const ctx=canvas.getContext('2d');ctx.setTransform(ratio,0,0,ratio,0,0);ctx.clearRect(0,0,rect.width,rect.height);const sx=rect.width/img.naturalWidth,sy=rect.height/img.naturalHeight;for(const c of clusters){const p=c.pixel||[0,0];const x=p[0]*sx,y=p[1]*sy;const r=12+Math.min(10,(c.count||1)*2);ctx.beginPath();ctx.arc(x,y,r,0,Math.PI*2);ctx.fillStyle=c===selectedCluster?'rgba(255,180,0,.35)':'rgba(255,160,0,.22)';ctx.fill();ctx.lineWidth=2;ctx.strokeStyle=c===selectedCluster?'#ffd166':'#f59e0b';ctx.stroke();ctx.fillStyle='#fff7cc';ctx.font='bold 12px Arial';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(String(c.count||1),x,y)}const target=mapPointToPixel(latestStatus?.exploration?.target_xy_m||latestStatus?.command?.target_xy_m);if(target){const x=target[0]*sx,y=target[1]*sy;ctx.save();ctx.strokeStyle='#22d3ee';ctx.fillStyle='rgba(34,211,238,.20)';ctx.lineWidth=3;ctx.beginPath();ctx.arc(x,y,15,0,Math.PI*2);ctx.fill();ctx.stroke();ctx.beginPath();ctx.moveTo(x-22,y);ctx.lineTo(x+22,y);ctx.moveTo(x,y-22);ctx.lineTo(x,y+22);ctx.stroke();ctx.fillStyle='#cffafe';ctx.font='bold 12px Arial';ctx.textAlign='left';ctx.fillText('explore target',x+18,y-16);ctx.restore()}}
marks.addEventListener('click',ev=>{const img=document.getElementById('map');const rect=img.getBoundingClientRect();const sx=rect.width/img.naturalWidth,sy=rect.height/img.naturalHeight;let best=null,bd=1e9;for(const c of clusters){const p=c.pixel||[0,0];const d=Math.hypot(ev.offsetX-p[0]*sx,ev.offsetY-p[1]*sy);if(d<bd){best=c;bd=d}}if(best&&bd<32){showCluster(best)}else{showCluster(null)}drawClusters()});
window.addEventListener('resize',drawClusters);map.addEventListener('load',drawClusters);
async function tick(){try{const [s,sem]=await Promise.all([getJson('/proxy/status'),getJson('/proxy/semantic')]);latestStatus=s;latestSemantic=sem;conn.textContent=(s._stale?'Pi cached':'Pi mapping connected');scan.textContent='scan '+(s.scan_count??0);acc.textContent='accepted '+(((s.accepted_ratio??0)*100).toFixed(1))+'%';rmse.textContent='rmse '+(s.rmse_m??'-');labels.textContent=(s.control?.patrol_active?'patrol ON ':'patrol OFF ')+'labels '+(s.semantic_labels??s.semantic_anchors??0);semantic.textContent=s.semantic_observe?.active?'Qwen observe '+Number(s.semantic_observe.remaining_s||0).toFixed(1)+'s':(s.qwen?.busy?'Qwen running':'Qwen idle');explore.textContent=s.exploration?.label||s.command?.reason||'explore -';json.textContent=JSON.stringify(s,null,2);clusters=sem.map_clusters||[];if(selectedCluster){selectedCluster=clusters.find(c=>c.id===selectedCluster.id)||null;showCluster(selectedCluster)}drawClusters();targets.innerHTML=(sem.navigation_targets||[]).map(card).join('');list.innerHTML=(sem.landmarks||[]).slice().reverse().map(card).join('');map.src='/proxy/map.png?t='+Date.now();ctrl.textContent=s.exploration?.label||s.control?.last_error||''}catch(e){conn.textContent='Pi disconnected';json.textContent=String(e)}}setInterval(tick,600);tick();
</script></body></html>"""


class QwenLandmarker:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        local_files_only: bool = True,
        mock: bool = False,
        max_new_tokens: int = 2048,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.local_files_only = local_files_only
        self.mock = mock
        self.max_new_tokens = max_new_tokens
        self.model: Any | None = None
        self.processor: Any | None = None
        self.torch: Any | None = None

    def detect(self, camera: Image.Image, lidar: Image.Image, context: dict[str, Any]) -> dict[str, Any]:
        if self.mock:
            return self._mock(context)
        self._load()
        assert self.model is not None and self.processor is not None and self.torch is not None

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": camera.convert("RGB")},
                    {"type": "text", "text": "Robot camera image. It may be a mono image or a stereo side-by-side image labeled LEFT and RIGHT."},
                    {"type": "image", "image": lidar.convert("RGB")},
                    {"type": "text", "text": "Local lidar top-down image. Markers show geometric candidates."},
                    {"type": "text", "text": build_prompt(context)},
                ],
            }
        ]
        started = time.perf_counter()
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
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
        raw = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        parsed = parse_json_object(raw)
        landmarks = normalize_items(parsed.get("landmarks", []))
        objects = normalize_items(parsed.get("objects", []))
        fallback_used = False
        if not landmarks and not objects and bool(context.get("allow_lidar_fallback", False)):
            landmarks = fallback_landmarks_from_context(context)
            fallback_used = bool(landmarks)
        print(
            "[qwen-landmark] "
            f"elapsed={time.perf_counter() - started:.1f}s "
            f"landmarks={len(landmarks)} objects={len(objects)} "
            f"fallback={fallback_used} "
            f"summary={str(parsed.get('scene_summary', ''))[:120]} "
            f"raw={raw[:240].replace(chr(10), ' ')}",
            flush=True,
        )
        return {
            "ok": True,
            "elapsed_s": round(time.perf_counter() - started, 3),
            "scene_summary": (
                str(parsed.get("scene_summary", ""))[:300]
                or ("Qwen returned no candidates; conservative lidar feature fallback was used." if fallback_used else "")
            ),
            "observation_advice": normalize_observation_advice(parsed.get("observation_advice", {})),
            "landmarks": landmarks,
            "objects": objects,
            "raw_text": raw[:2000],
        }

    def _load(self) -> None:
        if self.model is not None and self.processor is not None:
            return
        path = Path(self.model_path)
        if self.local_files_only and not path.exists():
            raise FileNotFoundError(f"model path not found: {path}")
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        use_cuda = self.device.lower().startswith("cuda")
        if use_cuda and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        dtype = torch.bfloat16 if use_cuda else torch.float32
        kwargs = {"local_files_only": self.local_files_only, "low_cpu_mem_usage": True}
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(self.model_path, dtype=dtype, **kwargs)
        except TypeError:
            self.model = AutoModelForImageTextToText.from_pretrained(self.model_path, torch_dtype=dtype, **kwargs)
        self.model.to("cuda" if use_cuda else "cpu")
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=self.local_files_only)
        print(f"[qwen-landmark] model ready: {self.model_path} on {self.device}")

    @staticmethod
    def _mock(context: dict[str, Any]) -> dict[str, Any]:
        landmarks = []
        for feature in list(context.get("lidar_features", []))[:3]:
            ftype = str(feature.get("type", "structure"))
            landmarks.append(
                {
                    "name": ftype,
                    "type": "wall_corner" if ftype == "corner" else "fixed_structure",
                    "stable": True,
                    "mobility": "fixed",
                    "long_term_place": True,
                    "anchor_score": 0.65,
                    "confidence": 0.55,
                    "lidar_feature_id": feature.get("id", ""),
                    "camera_bbox": [],
                    "reason": "mock from lidar feature",
                }
            )
        return {"ok": True, "elapsed_s": 0.0, "scene_summary": "mock", "landmarks": landmarks, "objects": []}


def build_component_inspection_prompt(context: dict[str, Any]) -> str:
    """Conservative map-component label prompt used by stopped inspections."""
    return f"""
You are labeling exactly one permanent black connected component in a saved
2D robot map.  The robot is stopped and points the centre of both cameras at
that component.  The camera image may be a LEFT/RIGHT side-by-side composite.

Current deterministic map context:
{json.dumps(context, ensure_ascii=False)}

Identify only the central mapped component, not incidental objects at the
image edge and not an example suggested by the context.  Valid semantic types
are fixed_cabinet, cabinet, fixed_table, table, desk, chair, stool,
fixed_trash_bin, trash_bin, box, fixed_structure, door_frame, wall_corner,
column, cleaning_tool, broom, dustpan, and unknown.

Strict evidence rules:
- Empty landmarks and objects are correct when the component cannot be seen
  clearly.  Never add a candidate merely to avoid an empty answer.
- Do not infer a whole table/cabinet from floor texture, glare, one vague line,
  a tiny image-edge fragment, or the context text.
- confidence >= 0.80 only when the same component category is clearly visible
  in both camera halves.  Otherwise confidence must be below 0.70.
- camera_bbox must tightly bound visible pixels of the component in the full
  composite 0-1000 coordinate system.  Never use [0,0,1000,1000], a generic
  corner box, or a box copied from this instruction.
- Set stereo_consistency="both" only after comparing the two halves.
- Return at most one landmark: the central component.  Incidental scene items
  may be mentioned only in scene_summary.
- observation_advice should request a closer view when evidence is too small,
  occluded, overexposed, or at a bad angle.

Return JSON only, using the existing landmark schema:
{{
  "scene_summary": "what is actually visible",
  "observation_advice": {{
    "quality": "good|too_far|bad_angle|occluded|unknown",
    "action": "none|backward|turn_left|turn_right",
    "retry_after_motion": false,
    "reason": "evidence assessment"
  }},
  "landmarks": [
    {{
      "name": "short concrete name",
      "type": "one allowed type",
      "stable": true,
      "mobility": "fixed|occasionally_moved|unknown",
      "long_term_place": true,
      "anchor_score": 0.0,
      "confidence": 0.0,
      "lidar_feature_id": "",
      "camera_bbox": [0,0,0,0],
      "lidar_visible": true,
      "ground_contact": true,
      "stereo_consistency": "both|left_only|right_only|unknown",
      "camera_lidar_match": "unknown",
      "map_memory_allowed": true,
      "behind_glass": false,
      "reachable": true,
      "navigation_allowed": true,
      "appearance": "visible evidence",
      "color": "",
      "material": "",
      "reason": "why this category is supported"
    }}
  ],
  "objects": []
}}
""".strip()


def build_prompt(context: dict[str, Any]) -> str:
    if str(context.get("inspection_mode") or "") == "component_label":
        return build_component_inspection_prompt(context)
    # 如果 Pi 端提供了标定信息，加入 prompt 帮助 Qwen 做更好的空间推理
    calib_info = context.get("camera_calibration", {})
    calib_lines = ""
    if calib_info:
        calib_lines = f"""
额外相机标定信息（已由树莓派端提供，用于帮助空间推理）：
- 左相机焦距 fx={calib_info.get('left_fx', '?')}px, fy={calib_info.get('left_fy', '?')}px, HFOV≈{calib_info.get('left_hfov_deg', '?')}°
- 右相机焦距 fx={calib_info.get('right_fx', '?')}px, fy={calib_info.get('right_fy', '?')}px, HFOV≈{calib_info.get('right_hfov_deg', '?')}°
- 双目基线 ≈{calib_info.get('stereo_baseline_m', '?')}m
- 左相机 yaw 偏移（相对车头）≈{calib_info.get('camera_yaw_left_deg', '?')}°，右相机 ≈{calib_info.get('camera_yaw_right_deg', '?')}°
- 相机离地高度 ≈{calib_info.get('camera_height_m', '?')}m
- 这意味着：画面中物体的水平像素位置对应实际方位角；近距离物体在左右相机中的视差约为 {calib_info.get('stereo_baseline_m', 0.14)}m / 距离 × {calib_info.get('left_fx', 781)}px。
- 请利用这些参数帮助判断物体的大致距离和空间位置，提高 camera_bbox 坐标精度。

"""
    return f"""
Hard rules for this low-mounted robot camera:
- The camera is close to the floor. Many objects are only visible as lower parts: stool legs, chair legs, table legs, cabinet bottom, trash-bin base, broom/dustpan bottom, cables near the floor, or outlet near the wall.
- If only legs/base are visible, infer the likely object category when reasonable, but say it in appearance/reason, e.g. "only chair legs are visible", "table-like legs near the floor", "cabinet bottom edge".
- Prefer concrete objects over generic wall labels. Broom, dustpan, cleaning tool, power outlet, cabinet, table, chair, stool, trash bin, box, bottle, fruit, garbage, paper ball, cable should be returned as themselves, not renamed as wall/corner/structure.
- Fill color, material, and appearance whenever visible. For example: purple plastic dustpan, metal broom handle, white plastic outlet, gray metal cabinet.
- If glass door, glass wall, window, or transparent partition is visible, label only the glass boundary itself with type glass_door/glass_wall/window/transparent_partition. Set that boundary map_memory_allowed=true, reachable=false, navigation_allowed=false. Do not return objects outside or behind glass in landmarks or objects; mention them only in scene_summary if useful.
- Do not return only wall/corner labels when task-useful objects are visible.

你是移动机器人语义参考点选择模块。

机器人正在用激光雷达建二维地图，但底盘转弯和速度不准，雷达拼接会漂移。
你要结合摄像头图像、局部雷达图、雷达候选几何特征，尽可能标注地图里当前能看到的物体、结构和任务目标。
long_term_place 只是物体是否长期稳定的属性说明，不再决定是否能被标注到地图上。
如果 camera_mode 是 stereo_side_by_side，摄像头图像左半边是小车左相机，右半边是小车右相机。
请分别比较 LEFT/RIGHT 中目标的位置和可见性；同一物体在两边都可见时视为更可靠，只在一边可见时也可以输出，但 reason 要说明在哪一边更明显。
注意摄像头视野通常比雷达有效扇区更宽；如果物体出现在画面边缘但局部雷达图没有对应几何候选，要输出为低置信度或放弃。可靠标注应优先填写真实存在的 lidar_feature_id。

{calib_lines}
当前上下文：
{json.dumps(context, ensure_ascii=False)}

重要输出策略：
- 这次任务不是只找“最稳定长期记忆点”，而是尽量给地图增加可用语义标注。
- 如果画面里清楚可见柜子、桌子、椅子、插座、扫帚、簸箕、垃圾桶、纸箱、门框、墙角等，请输出 3-6 个最有用、位置最明确的候选；不要一次标太多，后续巡航会继续补充。
- 普通任务物体也可以简单记录，例如水果、瓶子、杯子、纸团、垃圾、塑料袋、纸箱、椅子、桌子。它们可以 confidence/anchor_score 较低、long_term_place=false，但只要落地位置大致明确，就应该作为地图语义标签返回，后续观察可更新。
- 有雷达候选可以绑定时优先放 landmarks 并填写真实 lidar_feature_id。
- 没有明确雷达候选但物体可见、接触地面/墙面、位置明确时，也可以放 landmarks：camera_lidar_match="weak" 或 "unknown"，map_memory_allowed=true，anchor_score 0.35-0.55，reason 说明是视觉估计位置。
- 柜子/桌子/插座这类靠墙或固定的物体不要因为雷达扇区没有单独凸起就丢弃；它们对后续“去柜子旁/去桌子旁”很有用。
- 可移动但任务有用的物体，如扫帚、簸箕、清洁工具、垃圾桶、纸箱，也可以标到地图；long_term_place 可以是 false，但 map_memory_allowed 应按是否可定位来决定。

landmarks 放适合直接标到雷达地图上的结构或物体。它们不一定必须长期固定，但必须尽量满足：可见、位置明确、接触地面/墙体/固定结构、和雷达候选或雷达扇区方向大致一致。

优先选择：
- 墙角、门框、开口、柱子
- 固定柜子、固定桌子、固定架子、固定设备
- 垃圾桶、纸箱、椅子、桌腿、瓶子、待夹取目标等对导航或任务有用的可见物体

不要标注：
- 人、手、窗外物体、墙上纯图案、无法确定落地位置的悬空物体
- 没有唯一性的白墙平面
- 纯纹理但雷达没有对应几何特征的区域

objects 放任务有用但不适合直接画到雷达地图的可见物体；如果它能和雷达/地面位置对应，优先放 landmarks。
对于每个候选都判断 mobility：
- fixed：墙角、门框、柱子、内置柜、固定设备等，通常不会移动
- occasionally_moved：垃圾桶、椅子、大纸箱、可移动柜等，可能被人移动
- frequently_moved：瓶子、垃圾、手、包、小盒子等，常被移动
- unknown：看不清或无法判断

long_term_place 只表示这个物体是否大概率长期存在；即使 long_term_place=false，只要位置能和雷达/地面对应，也可以标注到地图。
anchor_score 表示是否适合标注到雷达地图上：有明确 lidar_feature_id 或视觉-雷达匹配强通常 0.7-1.0；只有角度重叠但无明确雷达候选通常不超过 0.5；画面边缘、悬空、无地面接触通常不超过 0.2。
额外判断视觉和雷达是否一致：
- lidar_visible：这个物体是否位于雷达平面附近，理论上能被 2D 雷达扫到。
- ground_contact：物体是否接触地面/墙面下沿/固定结构，而不是悬空、墙上图案、窗外物体。
- stereo_consistency：both|left_only|right_only|unknown。
- camera_lidar_match：strong|weak|none|unknown，表示视觉物体和雷达候选是否匹配。
- map_memory_allowed：这里表示是否允许标注到地图；如果没有雷达候选且画面位置边缘，要保守给 false。

如果雷达候选里有对应点，请填写 lidar_feature_id。不要编造 feature id。
如果环境单一，也应输出清楚可见且对导航/任务有用的候选；不要因为缺少完美雷达匹配而返回空数组，但要用 confidence、anchor_score、camera_lidar_match 区分可靠程度。
每次适中选择对导航/任务最有用的 3-6 个物体，例如墙角、柜子、桌子、椅子、门框、插座、清洁工具、扫帚、簸箕、线缆、垃圾桶、水果、瓶子、纸团、垃圾等；长期巡航会逐步补齐地图。
但 line/corner 这类雷达墙体几何特征只能绑定到 wall_corner、door_frame、opening、fixed_structure 等结构标签。
不要把扫帚、簸箕、清洁工具、瓶子、盒子、垃圾等可移动物体改名成墙角或墙线；它们应该保留自己的物体名称。
如果扫帚/簸箕/清洁工具靠墙或靠柜子放置、接触地面、位置明确，可以作为 cleaning_tool/broom/dustpan 标注到地图；
此时可以引用附近的 line/corner 作为“靠近的结构参考”，camera_lidar_match 设为 weak，map_memory_allowed=true，但 long_term_place=false。
同一个真实物体只输出一次；如果同一区域有多个候选名称，选择最可靠、最具体的一个，不要重复标注。
同时判断当前拍照位置是否适合做地图标注：
- 如果物体离相机太近、只看到局部、画面边缘严重截断、角度不好导致难以识别或难以定位，给 observation_advice.action 建议 backward 或轻微 turn_left/turn_right。
- 如果当前画面已经能识别并定位主要物体，action 必须是 none。
- 只有确实需要换视角重拍时才建议动作，避免巡航频繁打断。

只返回 JSON，不要 Markdown：
{{
  "scene_summary": "简短场景说明",
  "observation_advice": {{
    "quality": "good|too_close|too_far|bad_angle|blurred|too_few_features|occluded|unknown",
    "action": "none|backward|turn_left|turn_right",
    "retry_after_motion": false,
    "reason": "是否需要换一个拍照位置"
  }},
  "landmarks": [
    {{
      "name": "短名称",
      "type": "wall_corner|door_frame|opening|column|fixed_cabinet|fixed_table|fixed_trash_bin|power_outlet|cleaning_tool|broom|dustpan|cable|chair|stool|desk|table|fruit|food|garbage|trash|paper_ball|bottle|cup|bag|box|fixed_structure",
      "stable": true,
      "mobility": "fixed|occasionally_moved|frequently_moved|unknown",
      "long_term_place": true,
      "anchor_score": 0.0,
      "confidence": 0.0,
      "lidar_feature_id": "候选id或空",
      "camera_bbox": [x1,y1,x2,y2],
      "lidar_visible": true,
      "ground_contact": true,
      "stereo_consistency": "both|left_only|right_only|unknown",
      "camera_lidar_match": "strong|weak|none|unknown",
      "map_memory_allowed": true,
      "behind_glass": false,
      "reachable": true,
      "navigation_allowed": true,
      "appearance": "一句话外观描述，例如紫色扫帚和簸箕靠在墙角",
      "color": "主要颜色，不确定可留空",
      "material": "可见材质，例如金属/塑料/木质/布面，不确定可留空",
      "reason": "为什么稳定"
    }}
  ],
  "objects": [
    {{
      "name": "短名称",
      "type": "trash_bin|doorway|desk|table|cabinet|power_outlet|cleaning_tool|broom|dustpan|cable|chair|stool|fruit|food|garbage|trash|paper_ball|bottle|cup|bag|box|pickup_target|other",
      "stable": true,
      "mobility": "fixed|occasionally_moved|frequently_moved|unknown",
      "long_term_place": false,
      "anchor_score": 0.0,
      "confidence": 0.0,
      "lidar_feature_id": "可选",
      "camera_bbox": [x1,y1,x2,y2],
      "lidar_visible": true,
      "ground_contact": true,
      "stereo_consistency": "both|left_only|right_only|unknown",
      "camera_lidar_match": "strong|weak|none|unknown",
      "map_memory_allowed": true,
      "behind_glass": false,
      "reachable": true,
      "navigation_allowed": true,
      "appearance": "一句话外观描述",
      "color": "主要颜色，不确定可留空",
      "material": "可见材质，不确定可留空",
      "reason": "为什么对任务有用"
    }}
  ]
}}

camera_bbox 使用 0-1000 相对坐标；不确定就填 []。
必须尽量避免空返回：
- 如果画面中只有墙、柜体、桌椅边缘、墙角、插座、清洁工具、垃圾、箱子等普通环境元素，也要返回 2-4 个最有用标签。
- 如果无法精确识别具体名称，请返回保守名称，例如 wall surface、wall corner、cabinet/table/chair-like structure、cleaning tool-like object，并用 confidence 0.30-0.55。
- 如果 lidar_features 中有 line/corner/cluster 候选，请至少选择 1 个可解释的结构标签放入 landmarks，并填写对应 lidar_feature_id。
- 只有当相机完全无有效画面、严重模糊或没有任何可定位物体时，才允许 landmarks 和 objects 同时为空。
""".strip()


def fallback_landmarks_from_context(context: dict[str, Any]) -> list[dict[str, Any]]:
    """Qwen 完全空返回时，用雷达几何生成保守地图标签，避免语义链路静默失效。"""
    features = context.get("lidar_features", [])
    if not isinstance(features, list):
        return []
    landmarks: list[dict[str, Any]] = []
    for feature in features[:3]:
        if not isinstance(feature, dict):
            continue
        fid = str(feature.get("id", "") or "")
        ftype = str(feature.get("type", "") or "").lower()
        desc = str(feature.get("description", "") or "")
        if not fid:
            continue
        if ftype == "corner":
            name, ltype, appearance = "墙角", "wall_corner", "雷达检测到的墙角结构"
        elif ftype == "line":
            name, ltype, appearance = "墙面结构", "fixed_structure", "雷达检测到的墙面或柜体边缘线段"
        elif ftype == "cluster":
            name, ltype, appearance = "前方结构", "fixed_structure", "雷达检测到的小型结构候选"
        else:
            name, ltype, appearance = "环境结构", "fixed_structure", "雷达检测到的环境几何候选"
        landmarks.append(
            {
                "name": name,
                "type": ltype,
                "stable": True,
                "mobility": "fixed",
                "long_term_place": True,
                "anchor_score": 0.42 if ftype == "cluster" else 0.55,
                "confidence": 0.42 if ftype == "cluster" else 0.50,
                "lidar_feature_id": fid,
                "camera_bbox": [],
                "lidar_visible": True,
                "ground_contact": True,
                "stereo_consistency": "unknown",
                "camera_lidar_match": "weak",
                "map_memory_allowed": True,
                "behind_glass": False,
                "reachable": True,
                "navigation_allowed": True,
                "appearance": appearance,
                "color": "",
                "material": "",
                "reason": f"Qwen returned no visual candidates; fallback from lidar feature {fid} ({desc}).",
            }
        )
    return landmarks


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.I).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            return {}
        try:
            obj = json.loads(match.group(0))
            return obj if isinstance(obj, dict) else {}
        except json.JSONDecodeError:
            return {}


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "yes", "1", "是"}:
            return True
        if text in {"false", "no", "0", "否"}:
            return False
    return default


def normalize_observation_advice(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        value = {}
    action = str(value.get("action", "none") or "none").strip().lower()
    if action not in {"none", "backward", "turn_left", "turn_right"}:
        action = "none"
    quality = str(value.get("quality", "good") or "good").strip().lower()
    if quality not in {"good", "too_close", "too_far", "bad_angle", "blurred", "too_few_features", "occluded", "unknown"}:
        quality = "unknown"
    return {
        "quality": quality,
        "action": action,
        "retry_after_motion": _bool(value.get("retry_after_motion", action != "none"), action != "none"),
        "reason": str(value.get("reason", ""))[:240],
    }


def normalize_items(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    result = []
    for item in items:
        if not isinstance(item, dict):
            continue
        bbox = item.get("camera_bbox", [])
        if isinstance(bbox, list) and len(bbox) == 4:
            try:
                bbox = [float(v) for v in bbox]
            except (TypeError, ValueError):
                bbox = []
            if bbox:
                max_abs = max(abs(v) for v in bbox)
                if max_abs <= 1.5:
                    # Qwen 偶尔会按 0-1 返回 bbox；统一转成网页和 Pi 端使用的 0-1000。
                    bbox = [v * 1000.0 for v in bbox]
                x1, y1, x2, y2 = bbox
                if x2 <= x1 or y2 <= y1 or min(bbox) < 0.0 or max(bbox) > 1000.0:
                    bbox = []
                else:
                    bbox = [round(v, 1) for v in bbox]
        else:
            bbox = []
        try:
            conf = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        behind_glass = _bool(item.get("behind_glass", False), False)
        reachable = _bool(
            item.get("reachable", not behind_glass),
            not behind_glass,
        )
        navigation_allowed = _bool(
            item.get("navigation_allowed", reachable and not behind_glass),
            reachable and not behind_glass,
        )
        result.append(
            {
                "name": str(item.get("name", ""))[:80],
                "type": str(item.get("type", "fixed_structure"))[:60],
                "stable": _bool(item.get("stable", True), True),
                "mobility": str(item.get("mobility", "unknown"))[:32],
                "long_term_place": _bool(item.get("long_term_place", False), False),
                "anchor_score": max(0.0, min(1.0, _float(item.get("anchor_score", 0.0)))),
                "confidence": max(0.0, min(1.0, conf)),
                "lidar_feature_id": str(item.get("lidar_feature_id", ""))[:40],
                "camera_bbox": bbox,
                "lidar_visible": _bool(item.get("lidar_visible", False), False),
                "ground_contact": _bool(item.get("ground_contact", True), True),
                "stereo_consistency": str(item.get("stereo_consistency", "unknown"))[:32],
                "camera_lidar_match": str(item.get("camera_lidar_match", "unknown"))[:32],
                "map_memory_allowed": _bool(item.get("map_memory_allowed", True), True),
                "behind_glass": behind_glass,
                "reachable": reachable,
                "navigation_allowed": navigation_allowed,
                "appearance": str(item.get("appearance", item.get("description", "")))[:220],
                "color": str(item.get("color", ""))[:60],
                "material": str(item.get("material", ""))[:80],
                "reason": str(item.get("reason", ""))[:300],
            }
        )
    return result


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "service": "standalone_qwen_landmark", "pi_url": PI_MAP_URL or None}


def _require_pi_url() -> str:
    if not PI_MAP_URL:
        raise RuntimeError("pi-url is not configured; restart with --pi-url http://<PiIP>:8766")
    return PI_MAP_URL.rstrip("/")


def _pi_get(path: str, timeout: float | tuple[float, float] = (0.6, 2.0)) -> requests.Response:
    # 3090 上常有 http_proxy 环境变量；访问树莓派内网服务时必须直连，
    # 否则网页会出现“一会儿能打开一会儿 502”的假故障。
    response = PI_SESSION.get(_require_pi_url() + path, timeout=timeout)
    response.raise_for_status()
    return response


def _pi_post(path: str, timeout: float | tuple[float, float] = (0.6, 2.0)) -> requests.Response:
    response = PI_SESSION.post(_require_pi_url() + path, timeout=timeout)
    response.raise_for_status()
    return response


def _cached_pi_get(path: str, content_type: str, timeout: float | tuple[float, float] = (0.6, 2.0)) -> tuple[bytes, str, bool]:
    try:
        response = _pi_get(path, timeout=timeout)
        body = bytes(response.content)
        media_type = response.headers.get("content-type", content_type).split(";")[0]
        PI_CACHE[path] = (body, media_type or content_type, time.time())
        return body, media_type or content_type, False
    except Exception:
        cached = PI_CACHE.get(path)
        if cached is None:
            raise
        return cached[0], cached[1], True


@app.get("/")
def hub_index() -> HTMLResponse:
    return HTMLResponse(HUB_HTML)


@app.get("/proxy/status")
def proxy_status() -> JSONResponse:
    try:
        body, _media_type, stale = _cached_pi_get("/api/status", "application/json", timeout=(0.5, 1.2))
        payload = json.loads(body.decode("utf-8"))
        if stale and isinstance(payload, dict):
            payload["_stale"] = True
        return JSONResponse(payload)
    except Exception as exc:
        return JSONResponse({"ok": False, "error": repr(exc), "pi_url": PI_MAP_URL or None}, status_code=502)


@app.get("/proxy/semantic")
def proxy_semantic() -> JSONResponse:
    try:
        body, _media_type, stale = _cached_pi_get("/api/semantic", "application/json", timeout=(0.5, 1.5))
        payload = json.loads(body.decode("utf-8"))
        if stale and isinstance(payload, dict):
            payload["_stale"] = True
        return JSONResponse(payload)
    except Exception as exc:
        return JSONResponse({"landmarks": [], "navigation_targets": [], "error": repr(exc), "pi_url": PI_MAP_URL or None}, status_code=502)


@app.get("/proxy/map.png")
def proxy_map_png() -> Response:
    try:
        body, media_type, stale = _cached_pi_get("/api/map.png", "image/png", timeout=(0.5, 2.0))
        return Response(content=body, media_type=media_type, headers={"X-Proxy-Stale": "1" if stale else "0"})
    except Exception as exc:
        return Response(content=str(exc).encode("utf-8"), media_type="text/plain", status_code=502)


@app.get("/proxy/camera.jpg")
def proxy_camera_jpg() -> Response:
    try:
        body, media_type, stale = _cached_pi_get("/api/camera.jpg", "image/jpeg", timeout=(0.5, 2.0))
        return Response(content=body, media_type=media_type, headers={"X-Proxy-Stale": "1" if stale else "0"})
    except Exception as exc:
        return Response(content=str(exc).encode("utf-8"), media_type="text/plain", status_code=502)


@app.get("/proxy/camera.mjpg")
def proxy_camera_mjpg() -> StreamingResponse:
    def frames():
        while True:
            try:
                body, _media_type, _stale = _cached_pi_get(
                    "/api/camera.jpg",
                    "image/jpeg",
                    timeout=(0.25, 0.8),
                )
            except Exception:
                body = b""
            if body:
                yield b"--frame\r\n"
                yield b"Content-Type: image/jpeg\r\n"
                yield f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
                yield body
                yield b"\r\n"
            time.sleep(1.0 / 15.0)

    return StreamingResponse(
        frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.post("/proxy/{group}/{action}")
def proxy_control(group: str, action: str) -> JSONResponse:
    if group not in {"patrol", "drive", "map"}:
        return JSONResponse({"ok": False, "error": "bad proxy group"}, status_code=404)
    try:
        return JSONResponse(_pi_post(f"/api/{group}/{action}").json())
    except Exception as exc:
        return JSONResponse({"ok": False, "error": repr(exc), "pi_url": PI_MAP_URL or None}, status_code=502)


@app.post("/depth")
async def estimate_depth(
    camera_image: UploadFile,
    bboxes_json: str = Form("[]"),
    baseline_mm: float = Form(140.0),
    fx: float = Form(781.0),
) -> JSONResponse:
    """对 side-by-side 双目图像中的 bbox 列表估算深度。

    接受单张左右拼接的 JPEG，在 3090 GPU 上做局部立体匹配。
    返回每个 bbox 的估计深度和置信度。
    """
    try:
        bboxes = json.loads(bboxes_json)
        if not isinstance(bboxes, list):
            bboxes = []
    except json.JSONDecodeError:
        bboxes = []

    if not bboxes:
        return JSONResponse({"depths": [], "note": "no bboxes provided"})

    try:
        import cv2
        import numpy as np

        img_bytes = await camera_image.read()
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        full_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if full_img is None:
            return JSONResponse({"ok": False, "error": "cannot decode image"}, status_code=400)

        h, w = full_img.shape[:2]
        half_w = w // 2
        left_img = full_img[:, :half_w]
        right_img = full_img[:, half_w:]

        gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

        stereo_matcher = cv2.StereoSGBM_create(
            minDisparity=-32,
            numDisparities=112,
            blockSize=7,
            P1=8 * 3 * 7 ** 2,
            P2=32 * 3 * 7 ** 2,
            disp12MaxDiff=2,
            uniquenessRatio=8,
            speckleWindowSize=80,
            speckleRange=2,
        )
        disparity_full = stereo_matcher.compute(gray_left, gray_right).astype(np.float32) / 16.0

        baseline_m = float(baseline_mm) / 1000.0
        fx_val = float(fx)
        depths = []

        for item in bboxes:
            if not isinstance(item, dict):
                continue
            bbox = item.get("camera_bbox", item.get("bbox", []))
            if not isinstance(bbox, list) or len(bbox) != 4:
                depths.append({"depth_m": None, "error": "invalid bbox"})
                continue

            # bbox 0-1000 归一化坐标（左半 0-500，右半 500-1000）
            max_val = max(abs(v) for v in bbox)
            if max_val <= 2.0:
                scale_x, scale_y = float(half_w), float(h)
            else:
                scale_x, scale_y = float(half_w) / 1000.0, float(h) / 1000.0
                # 右半 bbox 映射回右图像素坐标
                if bbox[0] >= 500:
                    bbox = [bbox[0] - 500, bbox[1], bbox[2] - 500, bbox[3]]

            cx_px = int((float(bbox[0]) + float(bbox[2])) / 2.0 * scale_x)
            cy_px = int((float(bbox[1]) + float(bbox[3])) / 2.0 * scale_y)
            cx_px = max(0, min(half_w - 1, cx_px))
            cy_px = max(0, min(h - 1, cy_px))

            # 局部取样
            r = 20
            y1, y2 = max(0, cy_px - r), min(h, cy_px + r)
            x1, x2 = max(0, cx_px - r), min(half_w, cx_px + r)
            local_disp = disparity_full[y1:y2, x1:x2]
            valid = local_disp[local_disp > 0]
            if len(valid) < 15:
                depths.append({"depth_m": None, "error": "insufficient valid disparity"})
                continue

            median_disp = float(np.median(valid))
            disp_std = float(np.std(valid))

            if median_disp <= 0.5:
                depths.append({"depth_m": None, "error": "disparity too small (object too far)"})
                continue

            depth_m = (fx_val * baseline_m) / median_disp
            depth_sigma = (fx_val * baseline_m / (median_disp ** 2)) * disp_std

            depths.append({
                "depth_m": round(depth_m, 3),
                "depth_sigma_m": round(depth_sigma, 3),
                "disparity_px": round(median_disp, 2),
                "disparity_std_px": round(disp_std, 2),
                "pixel_xy": [cx_px, cy_px],
                "confidence": round(max(0.1, min(0.95, 0.7 - depth_sigma / max(depth_m, 0.5))), 2),
            })

        return JSONResponse({"ok": True, "depths": depths})

    except Exception as exc:
        return JSONResponse({"ok": False, "error": repr(exc)}, status_code=500)
@app.post("/landmarks")
async def landmarks(
    camera_image: UploadFile,
    lidar_image: UploadFile,
    context_json: str = Form("{}"),
) -> JSONResponse:
    if LANDMARKER is None:
        return JSONResponse({"ok": False, "error": "server not initialized"}, status_code=503)
    try:
        context = json.loads(context_json)
        if not isinstance(context, dict):
            context = {}
    except json.JSONDecodeError:
        context = {}
    try:
        camera = Image.open(io.BytesIO(await camera_image.read()))
        lidar = Image.open(io.BytesIO(await lidar_image.read()))
        return JSONResponse(LANDMARKER.detect(camera, lidar, context))
    except Exception as exc:
        return JSONResponse({"ok": False, "error": repr(exc)}, status_code=500)


def main() -> None:
    global LANDMARKER, PI_MAP_URL
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="~/3090/models/qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument(
        "--pi-url",
        "--pi-map-url",
        dest="pi_url",
        default="",
        help="optional Raspberry Pi live map URL, e.g. http://192.168.55.44:8766; enables the 3090 web hub",
    )
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()
    PI_MAP_URL = args.pi_url.strip().rstrip("/")
    LANDMARKER = QwenLandmarker(
        model_path=str(Path(args.model_path).expanduser()),
        device=args.device,
        local_files_only=not args.allow_download,
        mock=args.mock,
    )
    if PI_MAP_URL:
        print(f"[qwen-landmark] web hub proxying Pi map: {PI_MAP_URL}")
        print(f"[qwen-landmark] open http://{args.host}:{args.port}/ from other computers via the 3090 IP")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
