"""Offline diagnostic: calculate an out-of-hull MLP candidate without execution.

This tool never connects to a robot.  Its output is unapproved until a human
explicitly authorizes a controlled arm-only test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class ArmMLP(nn.Module):
    def __init__(self, in_dim: int = 4, hidden: int = 64, out_dim: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, value):
        return self.net(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--centers", nargs=4, type=float, metavar=("LX", "LY", "RX", "RY"), required=True)
    args = parser.parse_args()
    state = json.loads((args.artifacts / "model_state.json").read_text(encoding="utf-8"))
    point = np.asarray([
        args.centers[0] / 640.0,
        args.centers[1] / 480.0,
        args.centers[2] / 640.0,
        args.centers[3] / 480.0,
    ], dtype=np.float64)
    means = np.asarray(state["input_means"], dtype=np.float64)
    stds = np.asarray(state["input_stds"], dtype=np.float64)
    model = ArmMLP()
    model.load_state_dict(torch.load(args.artifacts / "model.pt", map_location="cpu", weights_only=True))
    model.eval()
    with torch.no_grad():
        normalized = torch.from_numpy((point - means) / np.maximum(stds, 1e-6)).float().unsqueeze(0)
        raw = model(normalized).numpy()[0].astype(np.float64)
    params = {}
    for index, name in enumerate(state["param_order"]):
        low, high = state["param_ranges"][name]
        value = float(np.clip(raw[index], 0.0, 1.0)) * (high - low) + low
        params[name] = int(round(value))
    print(json.dumps({
        "ok": True,
        "diagnostic_only": True,
        "outside_hull": True,
        "point": point.tolist(),
        "raw_output": raw.tolist(),
        "parameters": params,
        "limits": state["param_ranges"],
        "object_name": state.get("object_name"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
