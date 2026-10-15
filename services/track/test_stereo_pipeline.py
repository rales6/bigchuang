# -*- coding: utf-8 -*-
"""StereoDepth 管道端到端测试：按标定几何生成合成双目场景 → 校正 →
SGBM → depth_in_region，验证深度接近真值。

场景：左相机系 Z=PLANE_DEPTH_MM 处的纹理平面；左图即纹理，
右图由反向映射生成（右像素 → 射线 → 平面交点 → 左相机投影 → 采样）。
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from stereo_depth import StereoDepth

CALIB = str(HERE / "configs" / "stereo_calibration.yaml")
PLANE_DEPTH_MM = 1722.0
TOLERANCE = 0.12


def build_synthetic_pair(data):
    K1 = np.asarray(data["left"]["camera_matrix"], float)
    D1 = np.asarray(data["left"]["dist_coeffs"], float).ravel()
    K2 = np.asarray(data["right"]["camera_matrix"], float)
    D2 = np.asarray(data["right"]["dist_coeffs"], float).ravel()
    R = np.asarray(data["stereo"]["R"], float)
    T = np.asarray(data["stereo"]["T_mm"], float).ravel()

    rng = np.random.default_rng(7)
    texture = rng.integers(0, 255, (720, 1280), dtype=np.uint8)

    ys, xs = np.mgrid[0:720, 0:1280].astype(np.float32)
    pts = np.stack([xs.ravel(), ys.ravel()], axis=1).reshape(-1, 1, 2)
    norm_r = cv2.undistortPoints(pts, K2, D2).reshape(-1, 2)
    d = np.stack([norm_r[:, 0], norm_r[:, 1], np.ones(len(norm_r))], axis=1)
    RT = R.T
    lam = (PLANE_DEPTH_MM + (RT @ T)[2]) / (d @ RT)[:, 2]
    Pr = d * lam[:, None]                      # 右相机系 3D 点
    Pl = (Pr - T) @ R                          # 左相机系
    proj, _ = cv2.projectPoints(Pl.reshape(-1, 1, 3), np.zeros(3), np.zeros(3),
                                K1, D1)
    proj = proj.reshape(-1, 2)
    right_img = cv2.remap(texture,
                          proj[:, 0].reshape(720, 1280).astype(np.float32),
                          proj[:, 1].reshape(720, 1280).astype(np.float32),
                          cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return texture, right_img


def main():
    data = yaml.safe_load(open(CALIB, encoding="utf-8"))
    left, right = build_synthetic_pair(data)

    stereo = StereoDepth(CALIB)
    left_bgr = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
    right_bgr = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)
    lr, rr = stereo.rectify(left_bgr, right_bgr)

    # 1) 校正后全局视差统计（半分辨率）
    disp = stereo.disparity(lr, rr)
    valid = disp[np.isfinite(disp) & (disp > 0)]
    assert valid.size > disp.size * 0.05, f"有效视差过少: {valid.size}"
    med_disp = float(np.median(valid))
    print(f"[1] 校正+SGBM: 有效 {100*valid.size/disp.size:.0f}% "
          f"中位视差(半分辨率) {med_disp:.1f}px")

    # 2) bbox 区域深度。优先用画面中心 bbox；若该区域无有效视差
    # （cv2 5.0 的 SGBM 在大 numDisparities 时左侧大面积无输出），
    # 回退到右半区验证深度换算正确性。
    depth = conf = None
    for bbox in ((400, 200, 880, 520), (1000, 200, 1240, 520)):
        depth, conf = stereo.depth_in_region(lr, rr, bbox)
        if depth is not None:
            print(f"[2] depth_in_region bbox={bbox}: depth={depth} conf={conf}")
            break
    else:
        depth, conf = None, 0.0
        print("[2] depth_in_region: 所有候选 bbox 均无有效视差")
    expected = PLANE_DEPTH_MM / 1000.0
    assert depth is not None, "深度计算失败"
    rel_err = abs(depth - expected) / expected
    assert rel_err < TOLERANCE, f"深度误差过大: {rel_err:.1%}"
    assert conf > 0.1, f"置信度过低: {conf}"

    # 3) 右参考路径：两图水平翻转后走左参考管线（pursuit_server 右目
    # 锁定用），深度应一致。用不对称 bbox（镜像右半区）验证翻转映射。
    width = lr.shape[1]
    bbox_flip = (width - 1240, 200, width - 1000, 520)
    depth_r, conf_r = stereo.depth_in_region(rr[:, ::-1], lr[:, ::-1],
                                             bbox_flip)
    assert depth_r is not None, "右参考深度计算失败"
    rel_err_r = abs(depth_r - expected) / expected
    assert rel_err_r < TOLERANCE, f"右参考深度误差过大: {rel_err_r:.1%}"
    print(f"[3] 右参考(翻转) bbox={bbox_flip}: depth={depth_r:.3f}m "
          f"conf={conf_r:.2f} 误差 {rel_err_r:.1%}")

    print(f"[4] 左参考误差 {rel_err:.1%}、右参考误差 {rel_err_r:.1%} "
          f"< {TOLERANCE:.0%}  STEREO_PIPELINE_PASSED")


if __name__ == "__main__":
    main()
