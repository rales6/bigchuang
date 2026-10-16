"""Adapter around the existing local Qwen semantic task planner."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import threading
from typing import Any
import json
from urllib import request as urlrequest


API_PROVIDERS = {
    "deepseek_api": "https://api.deepseek.com",
    "deepseek_flash_api": "https://api.deepseek.com",
    "qwen_api": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "gpt_api": "https://api.openai.com/v1",
}


class CompatibleChatBackend:
    """Minimal OpenAI-compatible text backend with fixed provider endpoints."""

    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.model = model

    def generate(self, prompt: str) -> str:
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 360,
        }, ensure_ascii=False).encode("utf-8")
        request = urlrequest.Request(
            self.url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json; charset=utf-8",
            },
        )
        with urlrequest.urlopen(request, timeout=90) as response:
            value = json.loads(response.read().decode("utf-8"))
        choices = value.get("choices") if isinstance(value, dict) else None
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("模型 API 未返回 choices")
        content = choices[0].get("message", {}).get("content")
        if isinstance(content, list):
            content = "".join(
                str(item.get("text") or "") for item in content
                if isinstance(item, dict)
            )
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("模型 API 返回内容为空")
        return content


class AgentPlanner:
    def __init__(self) -> None:
        self.backend_name = os.environ.get("FINAL_WORK_PLANNER_BACKEND", "shared_http")
        self.shared_url = os.environ.get(
            "FINAL_WORK_QWEN_HUB_URL", "http://127.0.0.1:8010"
        ).rstrip("/")
        self.model_path = os.environ.get(
            "FINAL_WORK_QWEN_MODEL",
            "/home/gpu/3090/models/qwen/Qwen3-VL-2B-Instruct",
        )
        self.device = os.environ.get("FINAL_WORK_QWEN_DEVICE", "cuda")
        self.source_root = Path(os.environ.get("FINAL_WORK_3090_SRC", "/home/gpu/3090/src"))
        self._planner = None
        self._lock = threading.RLock()

    def _ensure_planner(self):
        if self._planner is not None:
            return self._planner
        with self._lock:
            if self._planner is not None:
                return self._planner
            if str(self.source_root) not in sys.path:
                sys.path.insert(0, str(self.source_root))
            from qwen_grounded_tracker.semantic_task_planner import (  # type: ignore
                QwenTransformersBackend,
                SemanticTaskPlanner,
            )

            backend = None
            if self.backend_name == "qwen":
                backend = QwenTransformersBackend(
                    self.model_path,
                    device=self.device,
                    local_files_only=True,
                    max_new_tokens=320,
                    max_time_seconds=60.0,
                )
            self._planner = SemanticTaskPlanner(backend)
            return self._planner

    @staticmethod
    def _safe_error_plan(instruction: str, provider: str, exc: Exception) -> dict[str, Any]:
        return {
            "schema": "robot_task_plan.v1",
            "task_id": "planner_unavailable",
            "instruction": instruction,
            "intent": "unsupported",
            "summary": "规划模型暂时不可用，任务保持在安全停止状态。",
            "assumptions": [],
            "global_constraints": ["规划失败时不得驱动底盘或机械臂"],
            "steps": [{
                "id": "step_01", "action": "report_result",
                "description": "报告规划器错误并等待人工处理",
                "depends_on": [], "target": {}, "sensors": [],
                "success_condition": "用户收到错误信息",
                "on_failure": "safe_stop", "max_retries": 0,
                "condition": "always",
            }],
            "source": "safe_fallback",
            "warnings": [f"{provider} 调用失败：{type(exc).__name__}: {exc}"],
            "model_interpretation": {},
        }

    def plan(
        self,
        instruction: str,
        semantic_map: dict[str, Any] | None,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        config = config or {}
        provider = str(config.get("provider") or "local_qwen")
        try:
            from .qwen_hub import SharedAgentPlanner

            compiler = SharedAgentPlanner(None)  # type: ignore[arg-type]
            direct = compiler._direct_motion_contract(instruction)
            if direct is not None:
                return compiler._compile(instruction, direct, "deterministic_motion_parser")
            if provider == "local_qwen" and self.backend_name == "shared_http":
                body = json.dumps(
                    {"instruction": instruction, "semantic_map": semantic_map or {}},
                    ensure_ascii=False,
                ).encode("utf-8")
                request = urlrequest.Request(
                    f"{self.shared_url}/agent/plan",
                    data=body,
                    method="POST",
                    headers={"Content-Type": "application/json; charset=utf-8"},
                )
                with urlrequest.urlopen(request, timeout=90) as response:
                    value = json.loads(response.read().decode("utf-8"))
                if not isinstance(value, dict) or not value.get("ok") or not isinstance(value.get("plan"), dict):
                    raise RuntimeError(str(value.get("error") if isinstance(value, dict) else value))
                return value["plan"]
            if provider in API_PROVIDERS:
                api_key = str(config.get("api_key") or "").strip()
                model = str(config.get("model") or "").strip()
                if provider == "deepseek_flash_api" and not model:
                    model = "deepseek-flash"
                if not api_key:
                    raise RuntimeError("尚未配置该服务的 API 密钥")
                if not model:
                    raise RuntimeError("尚未填写模型 ID")
                backend = CompatibleChatBackend(API_PROVIDERS[provider], api_key, model)
                compiler = SharedAgentPlanner(backend)  # type: ignore[arg-type]
                raw = backend.generate(compiler._prompt(instruction, semantic_map or {}))
                contract = compiler._extract(raw)
                return compiler._compile(
                    instruction, contract, f"{provider}_compiled"
                )
            result = self._ensure_planner().plan(instruction, semantic_map)
            return result.to_dict()
        except Exception as exc:
            return self._safe_error_plan(instruction, provider, exc)

    @staticmethod
    def client_prompt(instruction: str, semantic_map: dict[str, Any] | None) -> str:
        """Build the constrained prompt returned to a user-device API client."""
        from .qwen_hub import SharedAgentPlanner

        compiler = SharedAgentPlanner(None)  # type: ignore[arg-type]
        return compiler._prompt(instruction, semantic_map or {})

    @staticmethod
    def compile_client_output(instruction: str, raw: str, provider: str) -> dict[str, Any]:
        """Validate and compile untrusted client-side model output."""
        from .qwen_hub import SharedAgentPlanner

        compiler = SharedAgentPlanner(None)  # type: ignore[arg-type]
        contract = compiler._extract(raw)
        return compiler._compile(instruction, contract, f"{provider}_client_compiled")
