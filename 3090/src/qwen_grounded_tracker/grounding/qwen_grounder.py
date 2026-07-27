from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np
from PIL import Image

from qwen_grounded_tracker.domain import GroundingResult
from qwen_grounded_tracker.grounding.parser import parse_grounding_output


class Qwen3VLTargetGrounder:
    """Use Qwen3-VL once to ground a user-described target in a key frame.

    The model is loaded lazily and reused for subsequent re-grounding requests.
    Coordinates are requested on a relative 1000x1000 grid.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        local_files_only: bool = True,
        max_new_tokens: int = 64,
        max_time_seconds: float = 90.0,
        min_visual_tokens: int = 64,
        max_visual_tokens: int = 256,
        minimum_box_area_ratio: float = 0.001,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.local_files_only = local_files_only
        self.max_new_tokens = max_new_tokens
        self.max_time_seconds = max_time_seconds
        self.min_visual_tokens = min_visual_tokens
        self.max_visual_tokens = max_visual_tokens
        self.minimum_box_area_ratio = minimum_box_area_ratio

        self.model: Any | None = None
        self.processor: Any | None = None
        self.torch: Any | None = None

    def _ensure_loaded(self) -> None:
        if self.model is not None and self.processor is not None:
            return

        model_path = Path(self.model_path)
        if self.local_files_only and not model_path.exists():
            raise FileNotFoundError(
                f"Local Qwen model directory not found: {model_path.resolve()}"
            )

        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        requested_device = self.device.lower()
        use_cuda = requested_device.startswith("cuda")
        if use_cuda and not torch.cuda.is_available():
            raise RuntimeError("Qwen device is CUDA, but torch.cuda.is_available() is False")

        dtype = torch.bfloat16 if use_cuda else torch.float32
        load_kwargs: dict[str, Any] = {
            "local_files_only": self.local_files_only,
            "low_cpu_mem_usage": True,
        }

        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_path,
                dtype=dtype,
                **load_kwargs,
            )
        except TypeError:
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_path,
                torch_dtype=dtype,
                **load_kwargs,
            )

        self.model.to("cuda" if use_cuda else "cpu")
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            local_files_only=self.local_files_only,
        )

        image_processor = getattr(self.processor, "image_processor", None)
        if image_processor is not None and hasattr(image_processor, "size"):
            image_processor.size = {
                "longest_edge": int(self.max_visual_tokens * 32 * 32),
                "shortest_edge": int(self.min_visual_tokens * 32 * 32),
            }

        print(
            f"[Qwen grounder] model ready on {self.device}; "
            f"visual_tokens={self.min_visual_tokens}-{self.max_visual_tokens}"
        )

    @staticmethod
    def _to_pil(frame: np.ndarray) -> Image.Image:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def _build_messages(
        self,
        frame: np.ndarray,
        instruction: str,
        reference_image: np.ndarray | None,
    ) -> list[dict[str, Any]]:
        prompt = f"""
You are the visual grounding module of a mobile robot.
The user's full instruction is:
{instruction}

Locate exactly one visible target referred to by the instruction.
The target may be a small object, a large object, or a scene region such as a window,
door, tabletop, floor area, wall area, container opening, trash bin opening, or bright panel.
If the target is a large region, return the tightest visible rectangle covering that region.
Ignore requested motion such as follow, approach, orbit, or stop; only locate the visual target.
If multiple similar objects exist, use attributes and spatial relations such as left/right,
color, size, and nearby objects to select the intended instance.

If the instruction asks for a number of objects, for example "two bottles" or "两个水瓶",
locate up to that many concrete instances and return them from left to right.

Return only one compact JSON object with one of these schemas:
{{"found":true,"bbox_2d":[x1,y1,x2,y2],"coordinate_system":"relative_1000","target_name":"short name","confidence":0.0}}
{{"found":true,"coordinate_system":"relative_1000","targets":[{{"target_name":"short name 1","bbox_2d":[x1,y1,x2,y2],"confidence":0.0}},{{"target_name":"short name 2","bbox_2d":[x1,y1,x2,y2],"confidence":0.0}}]}}
Coordinates must use a 0-1000 grid relative to the CURRENT scene, origin at top-left.
Before answering, verify bbox_2d has four numbers and x2>x1, y2>y1.
Do not refuse just because the target is not graspable. Approximate a visible box if needed.
If and only if the intended target is not visible at all, return:
{{"found":false,"bbox_2d":[],"coordinate_system":"relative_1000","target_name":"unknown","confidence":0.0}}
Do not output Markdown or explanations.
""".strip()

        content: list[dict[str, Any]] = []
        if reference_image is not None and reference_image.size > 0:
            content.extend(
                [
                    {"type": "text", "text": "Reference crop of the previously selected target:"},
                    {"type": "image", "image": self._to_pil(reference_image)},
                    {"type": "text", "text": "CURRENT scene to search:"},
                ]
            )
        content.extend(
            [
                {"type": "image", "image": self._to_pil(frame)},
                {"type": "text", "text": prompt},
            ]
        )
        return [{"role": "user", "content": content}]

    def ground(
        self,
        frame: np.ndarray,
        instruction: str,
        reference_image: np.ndarray | None = None,
    ) -> GroundingResult:
        self._ensure_loaded()
        assert self.model is not None
        assert self.processor is not None
        assert self.torch is not None

        height, width = frame.shape[:2]
        messages = self._build_messages(frame, instruction, reference_image)
        started = perf_counter()

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        with self.torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                max_time=self.max_time_seconds,
                do_sample=False,
            )

        generated_trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        raw_text = self.processor.batch_decode(
            generated_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        elapsed = perf_counter() - started
        print(f"[Qwen grounder] finished in {elapsed:.1f}s: {raw_text}")

        return parse_grounding_output(
            raw_text=raw_text,
            frame_width=width,
            frame_height=height,
            minimum_box_area_ratio=self.minimum_box_area_ratio,
        )
