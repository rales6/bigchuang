"""Render recorded lidar evidence without opening any hardware."""
import argparse
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from raspberry_pi.lidar.base import LaserScan
from raspberry_pi.open_space_panorama import _filter_body_returns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scans")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with gzip.open(args.scans, "rt", encoding="utf-8") as stream:
        rows = [r for line in stream if (r := json.loads(line))["type"] == "scan"]
    kept, masked = [], []
    for row in rows:
        scan = LaserScan(np.asarray(row["angles_rad"], dtype=float),
                         np.asarray(row["distances_m"], dtype=float), row["timestamp_s"])
        filtered = _filter_body_returns(scan, 0.19, 0.14, 0, 0.125)
        points = np.column_stack((0.14 + scan.distances_m * np.cos(scan.angles_rad),
                                  scan.distances_m * np.sin(scan.angles_rad)))
        valid = np.isfinite(points).all(axis=1) & (scan.distances_m > 0) & (scan.distances_m <= 8)
        excluded = valid & ~np.isfinite(filtered.distances_m)
        kept.append(points[valid & ~excluded])
        masked.append(points[excluded])
    kept, masked = np.concatenate(kept), np.concatenate(masked)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    for ax in axes:
        ax.scatter(kept[:, 1], kept[:, 0], s=3, alpha=0.10, color="#246bb0", label="保留的扫描点")
        ax.scatter(masked[:, 1], masked[:, 0], s=4, alpha=0.20, color="#e37819", label="候选车体回波（被过滤）")
        ax.scatter([0], [0], c="black", marker="+", s=70, linewidth=2, zorder=5, label="底盘坐标原点")
        ax.scatter([0], [.14], c="#9b39b8", marker="^", s=55, zorder=5, label="雷达（前置 0.14 m）")
        ax.set_aspect("equal")
        ax.grid(alpha=.25)
        ax.set_xlabel("横向 y / m（左侧为正）")
        ax.set_ylabel("前后 x / m（前方为正）")
        ax.invert_xaxis()
    axes[0].set_xlim(3.7, -3.7)
    axes[0].set_ylim(-3.7, 3.7)
    axes[0].set_title(f"周围雷达点云：{len(rows)} 帧静止扫描叠加")
    axes[0].text(.03, .97, "↑ 车头方向\n左侧 ← 车体 → 右侧", transform=axes[0].transAxes, va="top", fontsize=10)
    axes[1].set_xlim(.36, -.36)
    axes[1].set_ylim(-.32, .40)
    axes[1].set_title("近场放大：侧后方 8～12 cm 回波")
    lateral, forward = np.meshgrid(np.linspace(-.36, .36, 350), np.linspace(-.32, .4, 350))
    region = (lateral**2 + forward**2 < .19**2) & (np.abs(lateral) < .125) & (forward <= .14)
    axes[1].contourf(lateral, forward, region.astype(int), levels=[.5, 1.5], colors=["#e37819"], alpha=.10)
    axes[1].contour(lateral, forward, region.astype(int), levels=[.5], colors=["#b96820"], linewidths=.8)
    axes[1].text(.02, .02, "浅橙轮廓 = 当前配置的过滤范围\n不是测量得到的真实车体轮廓", transform=axes[1].transAxes, fontsize=9)
    axes[1].legend(loc="upper right", fontsize=8)
    fig.suptitle("2026-09-16 20:20 采集的静止雷达数据（非实时、未叠加旧地图）", fontsize=13)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    print(str(output.resolve()))


if __name__ == "__main__":
    main()
