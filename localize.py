#!/usr/bin/env python3
"""Localize the ROV in the SHIP frame: dead reckoning (ZED VIO + gravity + pressure + DVL) + Water Linked Underwater GPS.

Same sensors and the same factors as Mapping/fusion_graph.py, plus one absolute-position factor per acoustic fix.
The acoustic fixes are in the receiver frame (rx). The pose graph lives in the arbitrary map frame of the ZED
(origin = first image, gravity from the ROV IMU), so one alignment map -> rx is estimated at the same time:
    p_rx = A p_map + t_a,   A = Rg(gravity) * Rz(psi)   (yaw psi about gravity + translation = 4 unknowns)
and the ship frame follows from the fixed mounting of the receiver on the ship:
    p_ship = R_ship_rx p_rx + t_ship_rx

Files in --data (same as Mapping, plus):
    uwgps.csv        header row: t,x,y,z[,std]   acoustic locator position in the receiver frame, metres, same clock as
                     rov.csv / times.txt. std (optional) = standard deviation reported by the unit.
    localization.json  optional, see DEFAULT_LOC (receiver mounting on the ship, locator lever arm, sigmas)
    fusion.json      optional, as in Mapping (R_bc, t_bc, sigmas)

Writes to --out:
    poses_ship.txt   name tx ty tz qx qy qz qw   camera-to-ship
    rov_ship.csv     name,t,x,y,z,roll,pitch,yaw,std_x,std_y,std_z   ROV body (FRD) in the ship frame
    localization_info.json
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Mapping"))
from scipy.optimize import least_squares
from scipy.sparse import identity
from scipy.sparse.linalg import splu

from fusion_graph import build_measurements, load_cfg, load_rov, load_times, load_zed, write_poses
from geomlib import Pattern, exp_so3, huber, log_so3

DEFAULT_LOC = {
    # receiver mounting on the ship: p_ship = R_ship_rx p_rx + t_ship_rx. Default: ship frame == receiver frame.
    "R_ship_rx": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    "t_ship_rx": [0.0, 0.0, 0.0],
    "down_in_rx": [0.0, 0.0, 1.0],   # which way is "down" in the receiver frame (z = depth, positive down)
    "t_loc_b": [0.0, 0.0, 0.0],      # acoustic locator position in the ROV body frame (FRD), metres
    "sigma_ugps": 0.2,               # m, used when the unit reports no std (and as the floor when it does)
    "std_scale": 1.0,                # factor on the reported std
    "use_ugps": True,                # False = dead reckoning only (alignment from a best fit, for comparison)
    "max_nfev": 60,
}


def load_loc(path=None):
    loc = json.loads(json.dumps(DEFAULT_LOC))
    if path and Path(path).exists():
        loc.update(json.loads(Path(path).read_text()))
    return loc


def load_ugps(path):
    t, z, s = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                p = [float(row[k]) for k in ("t", "x", "y", "z")]
            except (KeyError, ValueError, TypeError):
                continue
            if not np.all(np.isfinite(p)):
                continue
            try:
                sd = float(row.get("std") or "nan")
            except ValueError:
                sd = np.nan
            t.append(p[0]); z.append(p[1:]); s.append(sd)
    o = np.argsort(t)
    return np.array(t)[o], np.array(z)[o].reshape(-1, 3), np.array(s)[o]


def rot_between(a, b):
    """rotation taking unit vector a to unit vector b."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    s, c = np.linalg.norm(v), float(a @ b)
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        k = np.cross(a, [1.0, 0, 0] if abs(a[0]) < 0.9 else [0, 1.0, 0]); k /= np.linalg.norm(k)
        return exp_so3(np.pi * k)
    return exp_so3(v / s * np.arctan2(s, c))


def initial_alignment(P, Z, Rg, d_w, u_d):
    """yaw psi about gravity + translation t_a that best map dead-reckoned positions P to fixes Z (trimmed Kabsch)."""
    e1 = np.cross(u_d, [1.0, 0, 0] if abs(u_d[0]) < 0.9 else [0, 1.0, 0]); e1 /= np.linalg.norm(e1)
    e2 = np.cross(u_d, e1)
    Q = P @ Rg.T
    keep = np.ones(len(P), bool)
    psi, ta = 0.0, Z.mean(0) - Q.mean(0)
    for _ in range(3):
        q = np.stack([Q[keep] @ e1, Q[keep] @ e2], 1); z = np.stack([Z[keep] @ e1, Z[keep] @ e2], 1)
        qc, zc = q - q.mean(0), z - z.mean(0)
        spread = np.linalg.svd(qc, compute_uv=False)[0] / np.sqrt(max(keep.sum(), 1))
        if spread < 0.05:
            print("warning: the trajectory has almost no horizontal extent; heading in the ship frame is not observable")
            psi = 0.0
        else:
            H = qc.T @ zc
            psi = np.arctan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1])   # angle about u_d (e1, e2, u_d right-handed)
        A = Rg @ exp_so3(psi * d_w)
        ta = Z[keep].mean(0) - (P[keep] @ A.T).mean(0)
        r = np.linalg.norm(P @ A.T + ta - Z, axis=1)
        keep = r <= max(3 * np.median(r), 0.5)
        if keep.sum() < 3:
            break
    return psi, ta


def localize(R0, C0, quality, tq, rov, ugps, cfg, loc, verbose=1):
    N = len(R0)
    S, use, hub = cfg["sigma"], cfg["use"], cfg["huber"]
    Rbc, tbc = np.array(cfg["R_bc"], float), np.array(cfg["t_bc"], float)
    t_loc = np.array(loc["t_loc_b"], float)
    m = build_measurements(R0, C0, tq, rov, cfg)
    d_w = m["d_w"]
    u_d = np.array(loc["down_in_rx"], float); u_d /= np.linalg.norm(u_d)
    Rg = rot_between(d_w, u_d)
    p_wb0 = C0 - np.einsum("nij,j->ni", R0 @ Rbc.T, tbc)
    b0 = np.nanmedian((m["depth"] - p_wb0 @ d_w)[m["depth_ok"]]) if m["depth_ok"].any() else 0.0

    # ---- acoustic fixes -> interpolation weights between the two neighbouring images
    ft, Z, fstd = ugps
    inside = (ft >= tq[0]) & (ft <= tq[-1])
    ft, Z, fstd = ft[inside], Z[inside], fstd[inside]
    nfix = len(ft)
    fk = np.clip(np.searchsorted(tq, ft) - 1, 0, N - 2)
    fw = (ft - tq[fk]) / (tq[fk + 1] - tq[fk])
    fn = fk + (fw > 0.5)                                    # image whose rotation sets the lever arm
    sig_f = np.maximum(loc["sigma_ugps"], np.nan_to_num(fstd, nan=0.0) * loc["std_scale"])
    if nfix < 3 and loc["use_ugps"]:
        raise SystemExit(f"only {nfix} acoustic fixes inside the image time span; need at least 3")

    p_wb_f = ((1 - fw)[:, None] * p_wb0[fk] + fw[:, None] * p_wb0[fk + 1]
              + np.einsum("nij,j->ni", (R0 @ Rbc.T)[fn], t_loc))
    psi0, ta0 = initial_alignment(p_wb_f, Z, Rg, d_w, u_d)

    i0 = np.arange(N - 1)
    zed_dR = np.einsum("nji,njk->nik", R0[:-1], R0[1:])
    zed_dt = np.einsum("nji,nj->ni", R0[:-1], C0[1:] - C0[:-1])
    q = np.clip(np.minimum(quality[:-1], quality[1:]), 0.01, 1.0)
    z_sr = S["zed_rot"] / q
    z_st = (S["zed_trans_abs"] + S["zed_trans_rel"] * np.linalg.norm(zed_dt, axis=1)) / q
    dvl_idx = i0[m["dvl_ok"]]
    dvl_d = m["dvl_d"][m["dvl_ok"]]
    dvl_s = np.sqrt((S["dvl_abs"] * m["dt"][m["dvl_ok"]]) ** 2 + (S["dvl_rel"] * np.linalg.norm(dvl_d, axis=1)) ** 2) + 1e-4
    g_idx = np.where(m["g_ok"])[0] if use["gravity"] else np.array([], int)
    z_idx = np.where(m["depth_ok"])[0] if use["depth"] else np.array([], int)
    use_f = loc["use_ugps"] and nfix > 0

    nvar = 6 * N + 5                                         # poses, depth offset b, psi, t_a
    ib, ipsi, ita = 6 * N, 6 * N + 1, 6 * N + 2
    col = lambda i: 6 * np.asarray(i)[:, None] + np.arange(6)[None]
    acol = np.tile(np.arange(ipsi, ipsi + 4), (nfix, 1))

    def unpack(x):
        X = x[:6 * N].reshape(N, 6)
        return exp_so3(X[:, :3]) @ R0, C0 + X[:, 3:], x[ib], x[ipsi], x[ita:ita + 3]

    def residuals(x, parts=False):
        R, C, b, psi, ta = unpack(x)
        out = [("anchor", (x[:6] / 1e-3)[None])]
        if use["zed"] and N > 1:
            dR = np.einsum("nji,njk->nik", R[:-1], R[1:])
            er = log_so3(np.einsum("nji,njk->nik", zed_dR, dR)) / z_sr[:, None]
            et = (np.einsum("nji,nj->ni", R[:-1], C[1:] - C[:-1]) - zed_dt) / z_st[:, None]
            out.append(("zed", huber(np.concatenate([er, et], 1), hub)))
        R_wb = R @ Rbc.T
        p_wb = C - np.einsum("nij,j->ni", R_wb, tbc)
        if len(g_idx):
            e = (np.einsum("nji,j->ni", R_wb[g_idx], d_w) - m["g_b"][g_idx]) / S["gravity"]
            out.append(("gravity", huber(e, hub)))
        if len(z_idx):
            e = ((p_wb[z_idx] @ d_w + b - m["depth"][z_idx]) / S["depth"])[:, None]
            out.append(("depth", huber(e, hub)))
        if use["dvl"] and len(dvl_idx):
            e = (np.einsum("nji,nj->ni", R_wb[dvl_idx], p_wb[dvl_idx + 1] - p_wb[dvl_idx]) - dvl_d) / dvl_s[:, None]
            out.append(("dvl", huber(e, hub)))
        if use_f:
            A = Rg @ exp_so3(psi * d_w)
            pm = ((1 - fw)[:, None] * p_wb[fk] + fw[:, None] * p_wb[fk + 1]
                  + np.einsum("nij,j->ni", R_wb[fn], t_loc))
            out.append(("uwgps", huber((pm @ A.T + ta - Z) / sig_f[:, None], hub)))
        out.append(("align", np.concatenate([[(psi - psi0) / 1.0], (ta - ta0) / 10.0])))   # weak: keeps the problem well posed
        out.append(("depth_b", np.array([b - b0])))
        if parts:
            return out
        return np.concatenate([o.ravel() for _, o in out])

    pat = Pattern()
    pat.add(6, col([0]))
    if use["zed"] and N > 1:
        pat.add(6, np.concatenate([col(i0), col(i0 + 1)], 1))
    if len(g_idx):
        pat.add(3, col(g_idx))
    if len(z_idx):
        pat.add(1, np.concatenate([col(z_idx), np.full((len(z_idx), 1), ib)], 1))
    if use["dvl"] and len(dvl_idx):
        pat.add(3, np.concatenate([col(dvl_idx), col(dvl_idx + 1)], 1))
    if use_f:
        pat.add(3, np.concatenate([col(fk), col(fk + 1), acol], 1))
    pat.add(1, np.array([[ipsi]]))
    pat.add(3, np.arange(ita, ita + 3)[None])
    pat.add(1, np.array([[ib]]))
    S_ = pat.matrix(nvar)

    x0 = np.zeros(nvar)
    x0[ib], x0[ipsi], x0[ita:ita + 3] = b0, psi0, ta0
    parts0 = {n: float(np.sqrt(np.mean(r ** 2))) for n, r in residuals(x0, parts=True)}
    sol = least_squares(residuals, x0, jac_sparsity=S_, method="trf", x_scale="jac", max_nfev=loc["max_nfev"])
    R, C, b, psi, ta = unpack(sol.x)
    parts1 = {n: float(np.sqrt(np.mean(r ** 2))) for n, r in residuals(sol.x, parts=True)}

    # ---- map -> receiver -> ship
    A = Rg @ exp_so3(psi * d_w)
    Rs_rx = np.array(loc["R_ship_rx"], float); ts_rx = np.array(loc["t_ship_rx"], float)
    Rs, ts = Rs_rx @ A, Rs_rx @ ta + ts_rx
    R_wb = R @ Rbc.T
    p_wb = C - np.einsum("nij,j->ni", R_wb, tbc)
    R_ship_body = np.einsum("ij,njk->nik", Rs, R_wb)
    p_ship_body = p_wb @ Rs.T + ts
    R_ship_cam = np.einsum("ij,njk->nik", Rs, R)
    C_ship = C @ Rs.T + ts

    # ---- position covariance of the body origin (map frame -> ship frame); optimistic, assumes the sigmas are right
    J = sol.jac.tocsr()
    H = (J.T @ J + 1e-9 * identity(nvar)).tocsc()
    lu = splu(H)
    # ship position of the body origin depends on its pose AND on the alignment (psi, t_a): propagate through both
    from geomlib import skew
    Rg_dw = Rg @ skew(d_w) @ exp_so3(psi * d_w)
    std = np.zeros((N, 3))
    for a in range(0, N, 100):
        idx = np.arange(a, min(a + 100, N))
        G = np.zeros((nvar, 3 * len(idx)))
        for j, i in enumerate(idx):
            v = R_wb[i] @ tbc
            G[6 * i:6 * i + 3, 3 * j:3 * j + 3] = (Rs @ skew(v)).T          # body rotation moves the lever arm
            G[6 * i + 3:6 * i + 6, 3 * j:3 * j + 3] = Rs.T
            G[ipsi, 3 * j:3 * j + 3] = Rs_rx @ Rg_dw @ p_wb[i]
            G[ita:ita + 3, 3 * j:3 * j + 3] = Rs_rx.T
        X = lu.solve(G)
        for j, i in enumerate(idx):
            c = G[:, 3 * j:3 * j + 3].T @ X[:, 3 * j:3 * j + 3]
            std[i] = np.sqrt(np.clip(np.diag(c), 0, None))

    res_f = None
    if use_f:
        pm = ((1 - fw)[:, None] * p_wb[fk] + fw[:, None] * p_wb[fk + 1] + np.einsum("nij,j->ni", R_wb[fn], t_loc))
        res_f = np.linalg.norm(pm @ A.T + ta - Z, axis=1)
    info = {
        "fixes_used": int(nfix) if use_f else 0,
        "fix_residual_median_m": float(np.median(res_f)) if res_f is not None else None,
        "fix_outliers_gt_3sigma": int((res_f > 3 * sig_f).sum()) if res_f is not None else None,
        "yaw_map_to_rx_rad": float(psi),
        "R_ship_map": Rs.tolist(), "t_ship_map": ts.tolist(),
        "rms_sigma_before": parts0, "rms_sigma_after": parts1,
        "nfev": int(sol.nfev),
    }
    if verbose:
        print("factors (rms in sigma units) before -> after:")
        for k in parts1:
            print(f"  {k:8s} {parts0[k]:8.2f} -> {parts1[k]:8.2f}")
        if res_f is not None:
            print(f"acoustic fixes: {nfix} used, median residual {np.median(res_f):.3f} m, "
                  f"{info['fix_outliers_gt_3sigma']} beyond 3 sigma")
    return dict(R_ship_cam=R_ship_cam, C_ship=C_ship, R_ship_body=R_ship_body, p_ship_body=p_ship_body, std=std, info=info)


def euler_zyx(R):
    """roll, pitch, yaw (rad) of body-to-ship rotation, ZYX convention (meaningful if the ship frame is FRD)."""
    yaw = np.arctan2(R[:, 1, 0], R[:, 0, 0])
    pitch = -np.arcsin(np.clip(R[:, 2, 0], -1, 1))
    roll = np.arctan2(R[:, 2, 1], R[:, 2, 2])
    return roll, pitch, yaw


def read_gt(path):
    d = {}
    for l in Path(path).read_text().splitlines():
        t = l.split()
        if t and not t[0].startswith("#"):
            d[t[0]] = np.array(t[1:4], float)
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ugps", default=None, help="default DATA/uwgps.csv")
    ap.add_argument("--config", default=None, help="localization config, default DATA/localization.json")
    ap.add_argument("--gt-ship", default=None, help="ground-truth camera-to-ship poses (poses.txt format): prints position error")
    ap.add_argument("--no-ugps", action="store_true", help="dead reckoning only (for comparison)")
    args = ap.parse_args()
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_cfg(data / "fusion.json")
    loc = load_loc(args.config or data / "localization.json")
    if args.no_ugps:
        loc["use_ugps"] = False
    names, R0, C0, qual = load_zed(data / "poses.txt")
    times = load_times(data / "times.txt")
    tq = np.array([times[n] for n in names])
    ugps = load_ugps(args.ugps or data / "uwgps.csv")
    res = localize(R0, C0, qual, tq, load_rov(data / "rov.csv"), ugps, cfg, loc)

    write_poses(out / "poses_ship.txt", names, res["R_ship_cam"], res["C_ship"])
    roll, pitch, yaw = euler_zyx(res["R_ship_body"])
    with open(out / "rov_ship.csv", "w") as f:
        f.write("name,t,x,y,z,roll,pitch,yaw,std_x,std_y,std_z\n")
        for k, n in enumerate(names):
            p, s = res["p_ship_body"][k], res["std"][k]
            f.write(f"{n},{tq[k]:.3f},{p[0]:.4f},{p[1]:.4f},{p[2]:.4f},{roll[k]:.5f},{pitch[k]:.5f},{yaw[k]:.5f},"
                    f"{s[0]:.4f},{s[1]:.4f},{s[2]:.4f}\n")
    info = res["info"]
    if args.gt_ship:
        gt = read_gt(args.gt_ship)
        err = np.array([np.linalg.norm(res["C_ship"][k] - gt[n]) for k, n in enumerate(names) if n in gt])
        info["gt_position_error_m"] = {"rmse": float(np.sqrt((err ** 2).mean())), "max": float(err.max()),
                                       "last": float(err[-1])}
        e = info["gt_position_error_m"]
        print(f"position error vs ground truth in the ship frame: rmse {e['rmse']:.3f} m, max {e['max']:.3f} m, last {e['last']:.3f} m")
    (out / "localization_info.json").write_text(json.dumps(info, indent=2))
    print("wrote", out / "rov_ship.csv", "and poses_ship.txt")


if __name__ == "__main__":
    main()
