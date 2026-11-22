#!/usr/bin/env python3
"""Convert a go2_motion sample CSV (TOWR single rigid body) into the planar trajectory CSV that
forceStance's MdlTrajectoryTracker loads ([trajectory_tracking].csv_path).

    python3 towr_to_planar.py build/go2_bound.csv [-p build/go2_bound_phases.csv] [-o out.csv]

The motion must be planar: left/right legs of a pair share contact timing, and roll, yaw and y stay 0.
Output rows are on the 1 ms control grid, so the loader's interpolation reproduces them exactly; each
contact switch is written twice (end of the old phase, start of the new one), as the loader expects.
"""
import argparse
import pathlib
import sys

import numpy as np

TICK = 0.001                         # [s] forceStance control period
LINK = 0.213                         # [m] thigh and calf length
HIP_X = {"F": 0.1934, "H": -0.1934}  # [m] hip pitch axis along the trunk, from the base origin
# CoM in the base frame (x, z) at the plan's crouch (thigh 0.9, calf -1.8), from go2_srbd_from_urdf.py.
# TOWR's body is the CoM; forceStance tracks the trunk base origin.
DEFAULT_COM_IN_BASE = (-0.0022, -0.0214)
PAIR_LEGS = {"F": ("LF", "RF"), "H": ("LH", "RH")}
ALL_LEGS = ("LF", "RF", "LH", "RH")
MODES = {(True, True): "S", (False, True): "R", (True, False): "Fr", (False, False): "F"}
PLANAR_TOLERANCE = 1e-6

COLUMNS = ["time", "phase", "mode", "x", "z", "pitch", "thigh_F", "calf_F", "thigh_H", "calf_H",
           "dx", "dz", "dpitch", "dthigh_F", "dcalf_F", "dthigh_H", "dcalf_H",
           "tau_thigh_F", "tau_calf_F", "tau_thigh_H", "tau_calf_H", "fx_F", "fz_F", "fx_H", "fz_H"]


# ── Inputs ──

def read_samples(path):
    return np.genfromtxt(path, delimiter=",", names=True)


def read_stance_intervals(phases_path, leg):
    phases = np.genfromtxt(phases_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    rows = phases[(phases["leg"] == leg) & (phases["contact"] == 1)]
    return [(float(r["t_start"]), float(r["t_end"])) for r in rows]


def stance_intervals_from_samples(samples, leg):
    """Stance runs of the per-sample contact flag; switch times are good to one sample period."""
    in_contact = samples[f"{leg}_contact"] > 0.5
    switches = np.flatnonzero(np.diff(in_contact.astype(int))) + 1
    bounds = np.concatenate([[0], switches, [len(in_contact) - 1]])
    return [(float(samples["t"][a]), float(samples["t"][b])) for a, b in zip(bounds[:-1], bounds[1:]) if in_contact[a]]


def phases_end(phases_path):
    phases = np.genfromtxt(phases_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    return float(phases["t_end"].max())


def stance_intervals(samples, phases_path):
    """Per-foot stance timing from the phases file, unless it is missing or belongs to another run."""
    if phases_path.exists() and abs(phases_end(phases_path) - samples["t"][-1]) < 1e-6:
        return {leg: read_stance_intervals(phases_path, leg) for leg in ALL_LEGS}
    print(f"note: {phases_path.name} missing or not for these samples; stance timing from their contact flags")
    return {leg: stance_intervals_from_samples(samples, leg) for leg in ALL_LEGS}


def check_planar(samples, intervals):
    out_of_plane = max(np.abs(samples[c]).max() for c in ("com_y", "roll", "yaw", "wx", "wz"))
    if out_of_plane > PLANAR_TOLERANCE:
        sys.exit(f"not planar: |y, roll, yaw, wx, wz| reaches {out_of_plane:.3g}")
    for left, right in PAIR_LEGS.values():
        if intervals[left] != intervals[right]:
            sys.exit(f"{left} and {right} do not share contact timing; a planar pair needs it")


# ── Resampling onto the control grid ──

def hermite(t_knots, p, v, t):
    """Cubic Hermite through (p, v) at the knots: position and velocity at t."""
    k = np.clip(np.searchsorted(t_knots, t, side="right") - 1, 0, len(t_knots) - 2)
    h = t_knots[k + 1] - t_knots[k]
    s = (t - t_knots[k]) / h
    p0, p1, m0, m1 = p[k], p[k + 1], v[k] * h, v[k + 1] * h
    pos = (2 * s**3 - 3 * s**2 + 1) * p0 + (s**3 - 2 * s**2 + s) * m0 + (-2 * s**3 + 3 * s**2) * p1 + (s**3 - s**2) * m1
    vel = ((6 * s**2 - 6 * s) * p0 + (3 * s**2 - 4 * s + 1) * m0 + (-6 * s**2 + 6 * s) * p1 + (3 * s**2 - 2 * s) * m1) / h
    return pos, vel


def stance_mask(ticks, stance_intervals):
    """Contact at each tick; a switch tick belongs to the later phase, the final tick to the last one."""
    def in_stance(a, b):
        end = round(b / TICK)
        return (ticks >= round(a / TICK)) & ((ticks < end) | ((ticks == end) & (end == ticks[-1])))

    return np.logical_or.reduce([in_stance(a, b) for a, b in stance_intervals] + [np.zeros_like(ticks, dtype=bool)])


def stance_force(samples, legs, axis, t, ticks, stance_intervals):
    """Force of a virtual leg (sum of its pair), interpolated inside each stance phase only, zero elsewhere.
    Each stance phase covers its closed tick range, so a switch tick carries the stance-side force."""
    total = np.sum([samples[f"{leg}_f{axis}"] for leg in legs], axis=0)
    in_contact = samples[f"{legs[0]}_contact"] > 0.5

    def one_phase(interval):
        a, b = interval
        inside = (ticks >= round(a / TICK)) & (ticks <= round(b / TICK))
        knots = in_contact & (samples["t"] >= a - 1e-9) & (samples["t"] <= b + 1e-9)
        return np.where(inside, np.interp(t, samples["t"][knots], total[knots]), 0.0)

    return np.sum([one_phase(interval) for interval in stance_intervals], axis=0)


# ── Planar kinematics ──

def rotation(pitch):
    """Body (x, z) -> world (x, z) for a nose-up pitch; shape (n, 2, 2)."""
    c, s = np.cos(pitch), np.sin(pitch)
    return np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)


def rotation_rate(pitch, pitch_rate):
    c, s = np.cos(pitch), np.sin(pitch)
    return pitch_rate[:, None, None] * np.stack([np.stack([-s, -c], -1), np.stack([c, -s], -1)], -2)


def leg_ik(foot_in_hip):
    """Thigh and calf angles (URDF convention, knee bent backwards) placing the foot at (x, z) from the hip."""
    reach = np.linalg.norm(foot_in_hip, axis=1)
    if (reach > 2 * LINK).any():
        sys.exit(f"foot out of reach: {reach.max():.3f} m > {2 * LINK:.3f} m")
    calf = -np.arccos(np.clip(reach**2 / (2 * LINK**2) - 1.0, -1.0, 1.0))
    thigh = np.arctan2(-foot_in_hip[:, 0], -foot_in_hip[:, 1]) - calf / 2
    return thigh, calf


def leg_jacobian(thigh, calf):
    """d(foot x, z in hip frame)/d(thigh, calf); shape (n, 2, 2)."""
    c1, c12 = np.cos(thigh), np.cos(thigh + calf)
    s1, s12 = np.sin(thigh), np.sin(thigh + calf)
    return LINK * np.stack([np.stack([-c1 - c12, -c12], -1), np.stack([s1 + s12, s12], -1)], -2)


def apply(matrices, vectors):
    return np.einsum("nij,nj->ni", matrices, vectors)


def apply_transposed(matrices, vectors):
    return np.einsum("nji,nj->ni", matrices, vectors)


# ── Conversion ──

def convert(samples, intervals, com_in_base):
    ticks = np.arange(round(samples["t"][-1] / TICK) + 1)
    t = ticks * TICK

    com_x, com_vx = hermite(samples["t"], samples["com_x"], samples["com_vx"], t)
    com_z, com_vz = hermite(samples["t"], samples["com_z"], samples["com_vz"], t)
    # TOWR pitch is nose-down (ZYX about +y); the planar plan is nose-up
    pitch, pitch_rate = hermite(samples["t"], -samples["pitch"], -samples["wy"], t)

    R, dR = rotation(pitch), rotation_rate(pitch, pitch_rate)
    com_offset = np.broadcast_to(np.asarray(com_in_base, float), (len(t), 2))
    base = np.stack([com_x, com_z], 1) - apply(R, com_offset)
    base_vel = np.stack([com_vx, com_vz], 1) - apply(dR, com_offset)

    out = {"time": t, "x": base[:, 0], "z": base[:, 1], "pitch": pitch,
           "dx": base_vel[:, 0], "dz": base_vel[:, 1], "dpitch": pitch_rate}
    contact, jacobians = {}, {}
    for side, (leg, _) in PAIR_LEGS.items():
        foot_x, foot_vx = hermite(samples["t"], samples[f"{leg}_px"], samples[f"{leg}_vx"], t)
        foot_z, foot_vz = hermite(samples["t"], samples[f"{leg}_pz"], samples[f"{leg}_vz"], t)
        foot_from_base = np.stack([foot_x, foot_z], 1) - base
        foot_in_hip = apply_transposed(R, foot_from_base) - np.array([HIP_X[side], 0.0])
        foot_vel_in_body = apply_transposed(R, np.stack([foot_vx, foot_vz], 1) - base_vel) + \
            apply_transposed(dR, foot_from_base)

        thigh, calf = leg_ik(foot_in_hip)
        J = leg_jacobian(thigh, calf)
        joint_rate = np.linalg.solve(J, foot_vel_in_body[..., None])[..., 0]
        out.update({f"thigh_{side}": thigh, f"calf_{side}": calf,
                    f"dthigh_{side}": joint_rate[:, 0], f"dcalf_{side}": joint_rate[:, 1]})

        stance = intervals[leg]
        contact[side] = stance_mask(ticks, stance)
        out[f"fx_{side}"] = stance_force(samples, PAIR_LEGS[side], "x", t, ticks, stance)
        out[f"fz_{side}"] = stance_force(samples, PAIR_LEGS[side], "z", t, ticks, stance)
        jacobians[side] = (J, R)
    return out, contact, jacobians


def stance_torque(J, R, force_world):
    """Joint torque holding a planned ground force, legs massless: tau = -J^T R^T f."""
    return -apply_transposed(J, apply_transposed(R, force_world))


def rows_with_phases(out, contact, jacobians):
    """One row per tick; a tick where the contact mode changes is written for both phases."""
    modes = np.array([MODES[(f, h)] for f, h in zip(contact["F"], contact["H"])])
    switch = np.concatenate([[False], modes[1:] != modes[:-1]])
    phase = np.cumsum(switch)

    # The old-phase copy of a switch tick keeps the old mode and contact set
    tick_index = np.concatenate([np.arange(len(modes)), np.flatnonzero(switch)])
    is_old_copy = np.concatenate([np.zeros(len(modes), bool), np.ones(switch.sum(), bool)])
    order = np.lexsort((~is_old_copy, tick_index))
    tick_index, is_old_copy = tick_index[order], is_old_copy[order]

    row_mode = np.where(is_old_copy, modes[np.maximum(tick_index - 1, 0)], modes[tick_index])
    row_phase = phase[tick_index] - is_old_copy
    rows = {k: v[tick_index] for k, v in out.items()}
    rows["mode"], rows["phase"] = row_mode, row_phase

    for side, planted_modes in (("F", ("S", "Fr")), ("H", ("S", "R"))):
        planted = np.isin(row_mode, planted_modes)
        for axis in ("x", "z"):
            rows[f"f{axis}_{side}"] = np.where(planted, rows[f"f{axis}_{side}"], 0.0)
        J, R = (m[tick_index] for m in jacobians[side])
        tau = stance_torque(J, R, np.stack([rows[f"fx_{side}"], rows[f"fz_{side}"]], 1))
        rows[f"tau_thigh_{side}"], rows[f"tau_calf_{side}"] = tau[:, 0], tau[:, 1]
    return rows


def write_csv(rows, path):
    def cell(name, i):
        value = rows[name][i]
        return str(value) if name == "mode" else str(int(value)) if name == "phase" else f"{value:.9g}"

    lines = [",".join(COLUMNS)] + [",".join(cell(c, i) for c in COLUMNS) for i in range(len(rows["time"]))]
    pathlib.Path(path).write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("samples", type=pathlib.Path)
    parser.add_argument("-p", "--phases", type=pathlib.Path, help="default: <samples>_phases.csv")
    parser.add_argument("-o", "--output", type=pathlib.Path, help="default: <samples>_planar.csv")
    parser.add_argument("--com-in-base", type=float, nargs=2, default=DEFAULT_COM_IN_BASE, metavar=("X", "Z"))
    args = parser.parse_args()

    phases = args.phases or args.samples.with_name(args.samples.stem + "_phases.csv")
    output = args.output or args.samples.with_name(args.samples.stem + "_planar.csv")
    samples = read_samples(args.samples)
    intervals = stance_intervals(samples, phases)
    check_planar(samples, intervals)

    out, contact, jacobians = convert(samples, intervals, args.com_in_base)
    rows = rows_with_phases(out, contact, jacobians)
    write_csv(rows, output)

    sequence = [m for i, m in enumerate(rows["mode"]) if i == 0 or m != rows["mode"][i - 1]]
    print(f"wrote {output}: {len(rows['time'])} rows, {rows['time'][-1]:.3f} s, phases {' '.join(sequence)}")
    for side in ("F", "H"):
        print(f"  {side}: thigh [{out[f'thigh_{side}'].min():+.2f}, {out[f'thigh_{side}'].max():+.2f}]"
              f"  calf [{out[f'calf_{side}'].min():+.2f}, {out[f'calf_{side}'].max():+.2f}] rad")


if __name__ == "__main__":
    main()
