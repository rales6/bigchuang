"""分析示教样本的一致性: 每个样本与其 3 近邻的标签差异。"""
import json
import numpy as np

state = json.load(open("arm_teach_artifacts/model_state.json", encoding="utf-8"))
hull = np.asarray(state["hull"])
ids = state["sample_ids"]
labels = []
for sid in ids:
    path = "arm_teach_dataset/samples/" + sid + ".json"
    m = json.load(open(path, encoding="utf-8"))
    labels.append([m["forward_mm"], m["up_mm"], m["yaw_deg"]])
labels = np.asarray(labels)

print(f"{'样本时间':>8} {'近邻距':>6} {'Δfwd':>6} {'Δup':>6} {'Δyaw':>6}  标签(fwd,up,yaw) 与近邻标签")
for i in range(len(ids)):
    d = np.linalg.norm(hull - hull[i], axis=1)
    order = np.argsort(d)[1:4]  # 跳过自己
    nn_dist = d[order[0]] * 640
    diffs = np.abs(labels[order] - labels[i])
    own = labels[i].round(0).astype(int)
    neigh = [tuple(v) for v in labels[order].round(0).astype(int)]
    print(
        f"{ids[i][6:12]:>8} {nn_dist:>5.0f}px "
        f"{diffs[:, 0].mean():>6.1f} {diffs[:, 1].mean():>6.1f} {diffs[:, 2].mean():>6.1f}  "
        f"({own[0]},{own[1]},{own[2]}) vs {neigh}"
    )
