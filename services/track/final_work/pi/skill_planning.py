"""Use the already-consented Pi DeepSeek configuration; never expose its key."""
import json
import os
from pathlib import Path
from urllib import request


def plan_contract(payload, adapter):
    consent = adapter.runtime_root.parent / "deepseek_semantic_upload.allowed"
    if os.environ.get("FINAL_WORK_ALLOW_DEEPSEEK_SEMANTIC_UPLOAD") != "1" and (
            not consent.is_file() or consent.read_text(encoding="utf-8").strip() != "1"):
        raise PermissionError("未授权 DeepSeek 语义任务外发")
    # Fixed path; the HTTP caller cannot nominate files or API endpoints.
    path = adapter.car1_root / "secrets/deepseek_vision.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    key = str(config.get("api_key") or "").strip()
    if not config.get("enabled", True) or not key:
        raise RuntimeError("现有 DeepSeek 配置未启用或密钥未填写")
    url = str(config.get("base_url") or "https://api.deepseek.com").rstrip("/")
    if url not in {"https://api.deepseek.com", "https://api.deepseek.com/v1"}:
        raise ValueError("DeepSeek 配置地址不在允许的 HTTPS 服务范围内")
    prompt = str(payload.get("prompt") or "")
    if not prompt or len(prompt) > 30000:
        raise ValueError("技能规划提示为空或过长")
    body = {"model": str(config.get("model") or "deepseek-flash"),
            "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": 800,
            "thinking": {"type": "disabled"}, "response_format": {"type": "json_object"}}
    req = request.Request(url + "/chat/completions",
                          data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                          headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    with request.urlopen(req, timeout=min(90, float(config.get("timeout_s") or 90))) as response:
        value = json.load(response)
    raw = value["choices"][0]["message"]["content"]
    if not isinstance(raw, str):
        raise RuntimeError("DeepSeek 未返回文本契约")
    return {"ok": True, "raw": raw, "source": "deepseek_robot_skills", "hardware_started": False}
