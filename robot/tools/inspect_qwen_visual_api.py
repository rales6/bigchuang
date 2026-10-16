"""Read-only inspection of the deployed Qwen3-VL visual encoder API."""

from __future__ import annotations

import os

from tools.deploy_object_search import connect, run_checked


def main() -> None:
    client = connect(
        "198.51.100.20",
        "sjtu",
        os.environ["RTX_SSH_PASSWORD"],
        port=2020,
    )
    try:
        command = (
            "/home/gpu/3090/.venv/bin/python -c \""
            "from transformers.models.qwen3_vl import modeling_qwen3_vl as m; "
            "import inspect; "
            "print([n for n in dir(m) if 'Vision' in n or 'Visual' in n]); "
            "print('VISION_FORWARD', inspect.signature(m.Qwen3VLVisionModel.forward)); "
            "print('VISION_SOURCE', inspect.getsource(m.Qwen3VLVisionModel.forward)); "
            "print('MODEL_ATTRS', [n for n in dir(m.Qwen3VLForConditionalGeneration) if 'image' in n.lower() or 'visual' in n.lower()]); "
            "print('IMAGE_FEATURES', inspect.getsource(m.Qwen3VLForConditionalGeneration.get_image_features))"
            "\""
        )
        print(run_checked(client, command, timeout=90))
    finally:
        client.close()


if __name__ == "__main__":
    main()
