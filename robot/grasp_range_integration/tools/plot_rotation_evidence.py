"""Render recorded local sensor-frame point clouds, never a guessed room map."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_file')
    parser.add_argument('output')
    args = parser.parse_args()
    data = json.loads(Path(args.raw_file).read_text(encoding='utf-8'))
    before = next(f for f in data['frames'] if f['stage'] == 'during')
    after = [f for f in data['frames'] if f['stage'] == 'after'][-1]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    all_points = []
    for ax, frame, label in zip(axes, [before, after], ['Before turn', 'After STOP']):
        angles = np.asarray(frame['angles_rad'])
        ranges = np.asarray(frame['distances_m'], dtype=float)
        valid = np.isfinite(ranges) & (ranges > 0) & (ranges <= 4)
        body = np.zeros(len(ranges), dtype=bool)
        body[frame['legacy_mask_indices'] + frame['installation_patch_indices']] = True
        x, y = ranges*np.cos(angles), ranges*np.sin(angles)
        ax.scatter(x[valid & ~body], y[valid & ~body], s=7, label='Exterior raw returns')
        ax.scatter(x[valid & body], y[valid & body], s=5, color='gray', label='Documented body returns')
        ax.plot(0, 0, '+', color='black', ms=9)
        ax.add_patch(plt.Circle((0, 0), .30, fill=False, color='red', lw=1, label='0.30 m sensor guard'))
        ax.set(title=label, xlabel='Forward x (m)', ylabel='Left y (m)')
        ax.set_aspect('equal', adjustable='box'); ax.grid(alpha=.2)
        all_points.append(np.column_stack((x[valid], y[valid])))
    points = np.concatenate(all_points)
    for ax in axes:
        ax.set_xlim(points[:, 0].min()-.2, points[:, 0].max()+.2)
        ax.set_ylim(points[:, 1].min()-.2, points[:, 1].max()+.2)
    axes[0].legend(loc='lower left', fontsize=8)
    fig.suptitle('Recorded lidar change: sensor coordinates, not global localization\n'
                 'Post-STOP ICP estimate ~36 deg clockwise; raw-range validation did not pass', fontsize=11)
    destination = Path(args.output); destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=160)
    print(destination.resolve())


if __name__ == '__main__':
    main()
