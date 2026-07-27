from __future__ import annotations

import cv2
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
print("opencv:", cv2.__version__)
print(
    "CSRT available:",
    hasattr(cv2, "TrackerCSRT_create")
    or hasattr(getattr(cv2, "legacy", object()), "TrackerCSRT_create"),
)
