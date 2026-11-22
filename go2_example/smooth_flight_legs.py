#!/usr/bin/env python3
"""Redraw the in-air leg motion of a planar trajectory (towr_to_planar.py output) in joint space.

    python3 smooth_flight_legs.py go2_jump_planar.csv [-o out.csv] [--blend 0.04]

TOWR plans massless legs, so while a leg pair is in the air its path does not affect the body
motion or the contact forces; only its lift-off and touch-down states matter. TOWR's swing feet,
seen through IK, can demand calf speeds above the motor limit. This keeps every lift-off and
touch-down joint state (angle and rate) and replaces each in-air stretch, per joint, with:
    brake to rest in `blend` s  ->  minimum-jerk move to the landing pose  ->  accelerate into the
    touch-down rate in the last `blend` s.
Rows on the ground, base motion, forces and torques are left untouched (in-air torque is 0).
"""
import argparse
import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd

# Planar mode -> which leg pair is in the air (towr_to_planar.MODES: key = (front down, hind down)).
AIRBORNE = {"F": ("F", "R"), "H": ("F", "Fr")}
JOINTS = ("thigh", "calf")
SPEED_LIMIT = {"thigh": 30.1, "calf": 15.7}                     # rad/s, go2_description.urdf
ANGLE_LIMIT = {"thigh_F": (-1.5708, 3.4907), "thigh_H": (-0.5236, 4.5379),
               "calf_F": (-2.7227, -0.83776), "calf_H": (-2.7227, -0.83776)}


def load_converter():
    """towr_to_planar.py from this folder: same leg geometry and frames as the CSV."""
    path = pathlib.Path(__file__).with_name("towr_to_planar.py")
    spec = importlib.util.spec_from_file_location("towr_to_planar", path)
    if spec is None or spec.loader is None:
        sys.exit(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def airborne_runs(mask):
    """(first, last) row index of every contiguous True stretch."""
    edges = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1))


def brake_move_accelerate(t, t0, q0, v0, t1, q1, v1, blend):
    """Joint angle and rate on [t0, t1]: constant deceleration from v0 to rest over `blend`,
    minimum-jerk (quintic) rest-to-rest move, constant acceleration into v1 over `blend`."""
    ta = min(blend, (t1 - t0) / 4)
    qa = q0 + v0 * ta / 2                # where braking ends
    qc = q1 - v1 * ta / 2                # where the final acceleration starts
    tb0, tb1 = t0 + ta, t1 - ta

    s = np.clip((t - tb0) / (tb1 - tb0), 0.0, 1.0)
    quintic = 10 * s**3 - 15 * s**4 + 6 * s**5
    dquintic = (30 * s**2 - 60 * s**3 + 30 * s**4) / (tb1 - tb0)
    q = qa + (qc - qa) * quintic
    dq = (qc - qa) * dquintic

    braking = t < tb0
    tau = t - t0
    q = np.where(braking, q0 + v0 * tau - v0 / (2 * ta) * tau**2, q)
    dq = np.where(braking, v0 * (1 - tau / ta), dq)

    landing = t > tb1
    tau = t1 - t
    q = np.where(landing, q1 - v1 * tau + v1 / (2 * ta) * tau**2, q)
    dq = np.where(landing, v1 * (1 - tau / ta), dq)
    return q, dq


def smooth(rows, blend):
    out = rows.copy()
    for side, modes in AIRBORNE.items():
        for i0, i1 in airborne_runs(rows["mode"].isin(modes).to_numpy()):
            if i0 == 0 or i1 == len(rows) - 1 or i1 - i0 < 2:
                continue  # needs ground states on both sides
            t = rows["time"].to_numpy()[i0:i1 + 1]
            for joint in JOINTS:
                col, rate = f"{joint}_{side}", f"d{joint}_{side}"
                q, dq = brake_move_accelerate(t, t[0], rows[col].iat[i0], rows[rate].iat[i0],
                                              t[-1], rows[col].iat[i1], rows[rate].iat[i1], blend)
                out.loc[out.index[i0 + 1:i1], col] = q[1:-1]
                out.loc[out.index[i0 + 1:i1], rate] = dq[1:-1]
    return out


def foot_heights(rows, conv, side):
    """World foot height from base pose and joint angles (planar FK matching the converter's IK)."""
    thigh, calf = rows[f"thigh_{side}"].to_numpy(), rows[f"calf_{side}"].to_numpy()
    in_hip = conv.LINK * np.stack([-(np.sin(thigh) + np.sin(thigh + calf)),
                                   -(np.cos(thigh) + np.cos(thigh + calf))], 1)
    in_body = in_hip + np.array([conv.HIP_X[side], 0.0])
    world = conv.apply(conv.rotation(rows["pitch"].to_numpy()), in_body)
    return rows["z"].to_numpy() + world[:, 1]


def report(label, rows, conv):
    air = {s: rows["mode"].isin(m) for s, m in AIRBORNE.items()}
    speed = {j: max(rows[f"d{j}_{s}"][air[s]].abs().max() for s in AIRBORNE) for j in JOINTS}
    over_angle = [c for c, (lo, hi) in ANGLE_LIMIT.items() if rows[c].min() < lo or rows[c].max() > hi]
    clearance = min(foot_heights(rows, conv, s)[air[s].to_numpy()].min() for s in AIRBORNE)
    print(f"{label}: in-air speed thigh {speed['thigh']:.1f}/{SPEED_LIMIT['thigh']} calf {speed['calf']:.1f}/"
          f"{SPEED_LIMIT['calf']} rad/s | angle limits {'OK' if not over_angle else 'EXCEEDED ' + ','.join(over_angle)}"
          f" | lowest in-air foot {clearance:+.3f} m")
    return speed, over_angle, clearance


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("planar", type=pathlib.Path)
    parser.add_argument("-o", "--output", type=pathlib.Path, help="default: overwrite the input")
    parser.add_argument("--blend", type=float, default=0.04, help="brake / accelerate time [s]")
    args = parser.parse_args()

    conv = load_converter()
    rows = pd.read_csv(args.planar, dtype={"mode": str})
    report("before", rows, conv)
    smoothed = smooth(rows, args.blend)
    speed, over_angle, clearance = report("after ", smoothed, conv)
    smoothed.to_csv(args.output or args.planar, index=False, float_format="%.9g")
    print(f"wrote {args.output or args.planar}")
    ok = all(speed[j] <= SPEED_LIMIT[j] for j in JOINTS) and not over_angle and clearance > -1e-3
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
