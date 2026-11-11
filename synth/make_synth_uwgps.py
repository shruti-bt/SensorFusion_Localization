#!/usr/bin/env python3
"""Add a simulated Water Linked UGPS to a dataset made by Mapping/synth/make_synth_fusion.py.

The true ship frame is placed arbitrarily relative to the simulated world, so the alignment is a real test.
Writes into DATA: uwgps.csv (receiver frame, noisy, with outliers and a dropout), localization.json (receiver mounting,
locator lever arm) and gt_ship_poses.txt (camera-to-ship ground truth).
usage: make_synth_uwgps.py DATA [--rate 4] [--noise 0.08] [--outliers 0.05] [--seed 1]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "Mapping"))
from colmap_rgbd_pose import R_to_quat_wxyz, quat_to_R
from geomlib import exp_so3

ap = argparse.ArgumentParser()
ap.add_argument("data")
ap.add_argument("--rate", type=float, default=4.0, help="fixes per second")
ap.add_argument("--noise", type=float, default=0.08, help="m, per axis")
ap.add_argument("--outliers", type=float, default=0.05, help="fraction of fixes jumped by 1-3 m (multipath)")
ap.add_argument("--dropout", default="3.0,4.5", help="time range a,b with no fixes ('' for none)")
ap.add_argument("--seed", type=int, default=1)
a = ap.parse_args()
d = Path(a.data)
rng = np.random.default_rng(a.seed)

gt = {}
for l in (d / "gt_poses.txt").read_text().splitlines():
    t = l.split()
    gt[t[0]] = (np.array(t[1:4], float), quat_to_R(*[float(x) for x in t[4:8]]))
times = {l.split()[0]: float(l.split()[1]) for l in (d / "times.txt").read_text().splitlines()}
fus = json.loads((d / "fusion.json").read_text())
Rbc, tbc = np.array(fus["R_bc"]), np.array(fus["t_bc"])
names = sorted(gt)
tq = np.array([times[n] for n in names])

# true transforms. sim world: down = +y. receiver frame: down = +z.
Rx90 = exp_so3(np.array([np.pi / 2, 0, 0]))              # +y -> +z
A = exp_so3(np.array([0, 0, 0.7])) @ Rx90                # world -> rx (yaw 0.7 rad about down)
ta = np.array([12.3, -4.5, 3.1])
R_ship_rx = exp_so3(np.array([0, 0, 0.3])); t_ship_rx = np.array([50.0, 2.0, -1.0])
t_loc = np.array([-0.3, 0.0, -0.1])                      # locator in the body frame

Rs, ts = R_ship_rx @ A, R_ship_rx @ ta + t_ship_rx
with open(d / "gt_ship_poses.txt", "w") as f:
    for n in names:
        C, R = gt[n]
        q = R_to_quat_wxyz(Rs @ R); Cs = Rs @ C + ts
        f.write(f"{n} {Cs[0]:.9g} {Cs[1]:.9g} {Cs[2]:.9g} {q[1]:.9g} {q[2]:.9g} {q[3]:.9g} {q[0]:.9g}\n")


def gt_pose(t):
    k = int(np.clip(np.searchsorted(tq, t) - 1, 0, len(tq) - 2)); w = (t - tq[k]) / (tq[k + 1] - tq[k])
    C = (1 - w) * gt[names[k]][0] + w * gt[names[k + 1]][0]
    R = gt[names[k if w < 0.5 else k + 1]][1]
    return C, R


drop = tuple(float(x) for x in a.dropout.split(",")) if a.dropout else None
rows = ["t,x,y,z,std"]
for t in np.arange(tq[0], tq[-1], 1 / a.rate):
    if drop and drop[0] <= t < drop[1]:
        continue
    C, R = gt_pose(t)
    Rwb = R @ Rbc.T
    p = A @ ((C - Rwb @ tbc) + Rwb @ t_loc) + ta
    p = p + rng.normal(0, a.noise, 3)
    if rng.random() < a.outliers:
        p = p + rng.normal(0, 1, 3) / 1.0 * rng.uniform(1, 3)
    rows.append(f"{t:.3f},{p[0]:.4f},{p[1]:.4f},{p[2]:.4f},{a.noise:.3f}")
(d / "uwgps.csv").write_text("\n".join(rows) + "\n")
(d / "localization.json").write_text(json.dumps({"R_ship_rx": R_ship_rx.tolist(), "t_ship_rx": t_ship_rx.tolist(),
                                                  "t_loc_b": t_loc.tolist(), "sigma_ugps": 0.1}))
print(f"wrote {len(rows) - 1} fixes, gt_ship_poses.txt, localization.json to {d}")
