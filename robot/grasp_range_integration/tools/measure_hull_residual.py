from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch


def project_simplex(value: torch.Tensor) -> torch.Tensor:
    ordered, _ = torch.sort(value, descending=True)
    cssv = torch.cumsum(ordered, dim=0) - 1.0
    indices = torch.arange(1, value.numel() + 1, dtype=torch.float64)
    valid = ordered - cssv / indices > 0
    rho = int(torch.nonzero(valid).max().item()) + 1 if valid.any() else 1
    return torch.clamp(value - cssv[rho - 1] / float(rho), min=0.0)


def residual(point: np.ndarray, hull: np.ndarray) -> float:
    h = torch.from_numpy(hull)
    x = torch.from_numpy(point)
    gram = h @ h.T
    step = 1.0 / max(torch.linalg.eigvalsh(gram)[-1].item(), 1e-6)
    weights = torch.full((h.shape[0],), 1.0 / h.shape[0], dtype=torch.float64)
    accelerated = weights.clone()
    scalar = 1.0
    for _ in range(50000):
        new_weights = project_simplex(accelerated - step * (gram @ accelerated - h @ x))
        new_scalar = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * scalar * scalar))
        accelerated = new_weights + ((scalar - 1.0) / new_scalar) * (new_weights - weights)
        weights, scalar = new_weights, new_scalar
        value = float(((h.T @ weights - x) ** 2).sum().item())
        if value <= 1e-10:
            return value
    return float(((h.T @ weights - x) ** 2).sum().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--centers", nargs=4, type=float, required=True)
    args = parser.parse_args()
    state = json.loads(args.state.read_text(encoding="utf-8"))
    point = np.asarray([
        args.centers[0] / 640.0, args.centers[1] / 480.0,
        args.centers[2] / 640.0, args.centers[3] / 480.0,
    ], dtype=np.float64)
    value = residual(point, np.asarray(state["hull"], dtype=np.float64))
    print(json.dumps({
        "centers": args.centers,
        "residual": value,
        "distance_px": math.sqrt(max(value, 0.0)) * 640.0,
        "within_12px_margin": value <= (12.0 / 640.0) ** 2,
    }, indent=2))


if __name__ == "__main__":
    main()
