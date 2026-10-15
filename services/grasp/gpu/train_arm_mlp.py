"""训练机械臂示教夹取 MLP (3090)。

用法:
  python train_arm_mlp.py --dataset <样本目录> [--epochs 3000] [--hidden 64]

样本目录结构 (arm_teach 采集产生):
  samples/<id>.json  {left_center:[x,y], right_center:[x,y],
                      s0, s1, s2, object_name}
  samples/<id>_left.jpg / <id>_right.jpg  (原图仅存档, 训练只用中心坐标)

产物:
  arm_teach_artifacts/model.pt          MLP 权重
  arm_teach_artifacts/model_state.json  归一化参数 + 4D 凸包点 + 评估指标

模型: 4 维归一化中心 (Lx/640, Ly/480, Rx/640, Ry/480) -> MLP -> 三舵机脉宽
(s0/s1/s2, 即树莓派发送的 #ARMG 绝对脉宽指令值; 3 号舵机镜像联动不训练)。
凸包: 训练样本的 4 维归一化中心构成; 推理时凸包外直接拒绝 (无法实现)。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from arm_teach_model import (
    ArmMLP,
    ARTIFACT_DIR,
    CAMERA_H,
    CAMERA_W,
    PARAM_ORDER,
    PARAM_RANGES,
    point_in_convex_hull,
)

SEED = 42


def load_samples(dataset_dir: Path, object_name: str) -> list[dict]:
    records = []
    for path in sorted((dataset_dir / "samples").glob("*.json")):
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if object_name and meta.get("object_name") != object_name:
            continue
        if not meta.get("left_center") or not meta.get("right_center"):
            continue
        if not all(name in meta for name in PARAM_ORDER):
            continue
        records.append(meta)
    # 完全相同中心的样本 = 物块没动只调了机械臂 -> 标签互相矛盾;
    # 保留最新的一条 (用户后调的姿态更可能是最终可夹取姿态)。
    deduped: dict[tuple, dict] = {}
    for meta in records:
        key = (
            round(float(meta["left_center"][0]), 1),
            round(float(meta["left_center"][1]), 1),
            round(float(meta["right_center"][0]), 1),
            round(float(meta["right_center"][1]), 1),
        )
        existing = deduped.get(key)
        if existing is None or meta.get("created_at", 0) >= existing.get("created_at", 0):
            deduped[key] = meta
    if len(deduped) < len(records):
        print(f"[train] 去重: {len(records)} -> {len(deduped)} (相同中心保留最新)")
    return sorted(deduped.values(), key=lambda m: m.get("created_at", 0))


def build_features(records: list[dict]) -> np.ndarray:
    x = np.asarray([[
        float(r["left_center"][0]) / CAMERA_W,
        float(r["left_center"][1]) / CAMERA_H,
        float(r["right_center"][0]) / CAMERA_W,
        float(r["right_center"][1]) / CAMERA_H,
    ] for r in records], dtype=np.float64)
    return x


def build_labels(records: list[dict]) -> np.ndarray:
    # 每轴按 PARAM_RANGES 归一化到 [0,1]
    y = np.zeros((len(records), len(PARAM_ORDER)), dtype=np.float64)
    for i, record in enumerate(records):
        for j, name in enumerate(PARAM_ORDER):
            low, high = PARAM_RANGES[name]
            value = float(record[name])
            y[i, j] = (min(high, max(low, value)) - low) / (high - low)
    return y


def train_model(
    x: np.ndarray,
    y: np.ndarray,
    *,
    hidden: int,
    epochs: int,
    seed: int = SEED,
) -> tuple[ArmMLP, dict]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x))
    test_count = max(1, int(round(len(x) * 0.2)))
    test_idx, train_idx = order[:test_count], order[test_count:]
    x_test, y_test = x[test_idx], y[test_idx]
    x_train, y_train = x[train_idx], y[train_idx]

    means = x_train.mean(axis=0)
    stds = x_train.std(axis=0)
    xn_train = (x_train - means) / np.maximum(stds, 1e-6)
    xn_test = (x_test - means) / np.maximum(stds, 1e-6)

    torch.manual_seed(seed)
    model = ArmMLP(in_dim=4, hidden=hidden, out_dim=len(PARAM_ORDER))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.MSELoss()

    xt = torch.from_numpy(xn_train).float()
    yt = torch.from_numpy(y_train).float()
    xv = torch.from_numpy(xn_test).float()
    yv = torch.from_numpy(y_test).float()

    best_val = float("inf")
    best_state = None
    patience = 300
    stale = 0
    noise_std = float(max(0.002, stds.min()) * 0.1)  # 输入噪声增强 (坐标抖动)

    for epoch in range(1, epochs + 1):
        model.train()
        noise = torch.randn_like(xt) * noise_std
        loss = loss_fn(model(xt + noise), yt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if epoch % 200 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                val = loss_fn(model(xv), yv).item()
            if val < best_val - 1e-7:
                best_val = val
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if stale >= patience:
                print(f"[train] early stop at epoch {epoch}, best val mse={best_val:.6f}")
                break
    model.load_state_dict(best_state or model.state_dict())
    model.eval()
    with torch.no_grad():
        pred = model(xv).numpy()
    mae = {}
    for j, name in enumerate(PARAM_ORDER):
        low, high = PARAM_RANGES[name]
        span = high - low
        mae[name] = round(
            float(np.mean(np.abs(pred[:, j] - y_test[:, j])) * span), 2,
        )
    return model, {"mae": mae, "val_mse": float(best_val)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--object-name", default="")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--hidden", type=int, default=64)
    args = parser.parse_args()

    records = load_samples(Path(args.dataset), args.object_name)
    if len(records) < 6:
        raise SystemExit(
            f"至少需要 6 个有效样本, 当前 {len(records)} 个 (建议 20-40 个)"
        )
    x = build_features(records)
    y = build_labels(records)

    # 凸包用全部样本的 4 维归一化中心
    hull = x.tolist()
    for i in range(len(records)):
        if not point_in_convex_hull(x[i], x, tol=1e-6):
            raise SystemExit("内部错误: 样本自身不在凸包内")

    model, metrics = train_model(x, y, hidden=args.hidden, epochs=args.epochs)

    means = x.mean(axis=0)
    stds = x.std(axis=0)
    output_min = np.zeros(len(PARAM_ORDER))
    output_max = np.ones(len(PARAM_ORDER))

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ARTIFACT_DIR / "model.pt")
    state = {
        "version": 2,
        "object_name": args.object_name or (records[0].get("object_name") or ""),
        "num_samples": len(records),
        "param_order": list(PARAM_ORDER),
        "param_ranges": {name: list(rng) for name, rng in PARAM_RANGES.items()},
        "input_means": means.tolist(),
        "input_stds": stds.tolist(),
        "output_min": output_min.tolist(),
        "output_max": output_max.tolist(),
        "hull": hull,
        "hull_tol": 1e-6,
        "camera_size": [CAMERA_W, CAMERA_H],
        "test_mae": metrics["mae"],
        "val_mse": metrics["val_mse"],
        "sample_ids": [r.get("sample_id", "") for r in records],
    }
    (ARTIFACT_DIR / "model_state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"[train] 样本 {len(records)} 个, 凸包点 {len(hull)} 个")
    print(f"[train] 测试 MAE: {metrics['mae']} (脉宽 us/us/us)")
    print(f"[train] 产物已写入 {ARTIFACT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
