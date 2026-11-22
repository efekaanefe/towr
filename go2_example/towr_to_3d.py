#!/usr/bin/env python3
"""Convert a go2_motion sample CSV (TOWR single rigid body) into the 3D trajectory CSV that forceStance's
MdlTrajectoryTracker loads ([trajectory_tracking].csv_path). towr_to_planar.py does the same for planar plans;
the tracker reads both.

    python3 towr_to_3d.py go2_trot_3d.csv [-p go2_trot_3d_phases.csv] [-o out.csv]

Output: one row per 1 ms tick; a tick where any foot's contact changes is written twice (end of the old phase,
start of the new one), as in the planar file. Columns: time, phase, c_<leg>, base x y z roll pitch yaw (ZYX,
+pitch = nose down) and their rates, hip/thigh/calf per leg (RAW URDF convention) and their rates, planned joint
torque per leg, ground force on each foot (plan frame). Legs FL, FR, RL, RR.
"""
import argparse
import importlib.util
import pathlib
import sys

import numpy as np

LEGS = ("FL", "FR", "RL", "RR")
TOWR_LEGS = ("LF", "RF", "LH", "RH")  # same order as LEGS
JOINTS = ("hip", "thigh", "calf")

# Go2 leg geometry, as quadcontrol's createGo2Config (QuadrupedConfigs.cc)
HIP = np.array([[0.1934, 0.0465, 0.0], [0.1934, -0.0465, 0.0], [-0.1934, 0.0465, 0.0], [-0.1934, -0.0465, 0.0]])
HIP_OFFSET = np.array([0.0955, -0.0955, 0.0955, -0.0955])
THIGH = CALF = 0.213
# CoM in the base frame at the plan's crouch (go2_srbd_from_urdf.py); TOWR's body is the CoM, the tracker's the trunk
DEFAULT_COM_IN_BASE = (-0.0022, 0.0, -0.0214)

ANGLE_LIMITS = {"hip": (-1.0472, 1.0472), "thigh_front": (-1.5708, 3.4907), "thigh_rear": (-0.5236, 4.5379),
                "calf": (-2.7227, -0.83776)}
SPEED_LIMITS = {"hip": 30.1, "thigh": 30.1, "calf": 15.7}  # rad/s
TORQUE_LIMITS = {"hip": 23.7, "thigh": 23.7, "calf": 45.43}  # Nm


def load_planar_converter():
    """towr_to_planar.py from this folder, for its generic resampling and contact helpers."""
    path = pathlib.Path(__file__).with_name("towr_to_planar.py")
    spec = importlib.util.spec_from_file_location("towr_to_planar", path)
    if spec is None or spec.loader is None:
        sys.exit(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── Rotations ──

def rotation_zyx(rpy):
    """(n, 3) ZYX Euler angles -> (n, 3, 3) base-to-world rotations."""
    cr, sr = np.cos(rpy[:, 0]), np.sin(rpy[:, 0])
    cp, sp = np.cos(rpy[:, 1]), np.sin(rpy[:, 1])
    cy, sy = np.cos(rpy[:, 2]), np.sin(rpy[:, 2])
    return np.stack([np.stack([cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], -1),
                     np.stack([sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], -1),
                     np.stack([-sp, cp * sr, cp * cr], -1)], -2)


def euler_rate_map(rpy):
    """(n, 3, 3) matrices mapping ZYX Euler rates to the world angular velocity (TOWR's EulerConverter::GetM)."""
    cp, sp = np.cos(rpy[:, 1]), np.sin(rpy[:, 1])
    cy, sy = np.cos(rpy[:, 2]), np.sin(rpy[:, 2])
    zero, one = np.zeros_like(cp), np.ones_like(cp)
    return np.stack([np.stack([cp * cy, -sy, zero], -1),
                     np.stack([cp * sy, cy, zero], -1),
                     np.stack([-sp, zero, one], -1)], -2)


def cross(a, b):
    return np.cross(a, b)


def apply(matrices, vectors):
    return np.einsum("nij,nj->ni", matrices, vectors)


def apply_transposed(matrices, vectors):
    return np.einsum("nji,nj->ni", matrices, vectors)


# ── Leg kinematics (RAW URDF angles; a port of QuadrupedKinematics::computeFK / computeIK) ──

def leg_fk(q, leg):
    """(n, 3) joint angles -> (n, 3) foot positions in the trunk frame."""
    q1, q2, q3 = q.T
    reach = THIGH * np.cos(q2) + CALF * np.cos(q2 + q3)
    return np.stack([HIP[leg, 0] - THIGH * np.sin(q2) - CALF * np.sin(q2 + q3),
                     HIP[leg, 1] + HIP_OFFSET[leg] * np.cos(q1) + reach * np.sin(q1),
                     HIP[leg, 2] + HIP_OFFSET[leg] * np.sin(q1) - reach * np.cos(q1)], 1)


def leg_jacobian(q, leg):
    """(n, 3, 3) d(foot in trunk)/d(hip, thigh, calf)."""
    q1, q2, q3 = q.T
    reach = THIGH * np.cos(q2) + CALF * np.cos(q2 + q3)
    d_reach_thigh = -THIGH * np.sin(q2) - CALF * np.sin(q2 + q3)
    d_reach_calf = -CALF * np.sin(q2 + q3)
    zero = np.zeros_like(q1)
    return np.stack([np.stack([zero, -THIGH * np.cos(q2) - CALF * np.cos(q2 + q3), -CALF * np.cos(q2 + q3)], -1),
                     np.stack([-HIP_OFFSET[leg] * np.sin(q1) + reach * np.cos(q1), d_reach_thigh * np.sin(q1),
                               d_reach_calf * np.sin(q1)], -1),
                     np.stack([HIP_OFFSET[leg] * np.cos(q1) + reach * np.sin(q1), -d_reach_thigh * np.cos(q1),
                               -d_reach_calf * np.cos(q1)], -1)], -2)


def leg_ik(foot, leg):
    """(n, 3) foot positions in the trunk frame -> (n, 3) joint angles, knee bent backwards."""
    rel = foot - HIP[leg]
    lateral = np.hypot(rel[:, 1], rel[:, 2])
    if (lateral < abs(HIP_OFFSET[leg]) - 1e-6).any():
        sys.exit(f"{LEGS[leg]}: foot too close to the hip axis")
    alpha = np.arctan2(rel[:, 2], rel[:, 1])
    hip = np.arccos(HIP_OFFSET[leg] / lateral)
    hip = np.where(alpha < 0, hip + alpha, alpha - hip)

    flexion_joint = HIP[leg] + np.stack([np.zeros_like(hip), HIP_OFFSET[leg] * np.cos(hip), HIP_OFFSET[leg] * np.sin(hip)], 1)
    rel_flexion = foot - flexion_joint
    down = -rel_flexion[:, 2] / np.cos(hip)
    back = -rel_flexion[:, 0]
    cos_calf = (down**2 + back**2 - THIGH**2 - CALF**2) / (2 * THIGH * CALF)
    if (np.abs(cos_calf) > 1.0).any():
        sys.exit(f"{LEGS[leg]}: foot out of reach")
    calf = -np.arccos(cos_calf)
    thigh = np.arctan2(back, down) - np.arctan2(CALF * np.sin(calf), THIGH + CALF * np.cos(calf))
    return np.stack([hip, thigh, calf], 1)


# ── Conversion ──

def read_samples(path):
    return np.genfromtxt(path, delimiter=",", names=True)


def foot_force_in_stance(samples, towr_leg, axis, t, ticks, stance_intervals, tick):
    """Ground force on one foot, interpolated inside each stance phase only, zero elsewhere."""
    force = samples[f"{towr_leg}_f{axis}"]
    in_contact = samples[f"{towr_leg}_contact"] > 0.5

    def one_phase(interval):
        a, b = interval
        inside = (ticks >= round(a / tick)) & (ticks <= round(b / tick))
        knots = in_contact & (samples["t"] >= a - 1e-9) & (samples["t"] <= b + 1e-9)
        return np.where(inside, np.interp(t, samples["t"][knots], force[knots]), 0.0)

    return np.sum([one_phase(interval) for interval in stance_intervals] + [np.zeros_like(t)], axis=0)


def convert(samples, phases_path, com_in_base, planar):
    tick = planar.TICK
    ticks = np.arange(round(samples["t"][-1] / tick) + 1)
    t = ticks * tick

    rpy_knots = np.stack([np.unwrap(samples[c]) for c in ("roll", "pitch", "yaw")], 1)
    omega_knots = np.stack([samples[c] for c in ("wx", "wy", "wz")], 1)
    rpy_rate_knots = np.linalg.solve(euler_rate_map(rpy_knots), omega_knots[..., None])[..., 0]

    com, com_vel = np.zeros((len(t), 3)), np.zeros((len(t), 3))
    rpy, rpy_rate = np.zeros((len(t), 3)), np.zeros((len(t), 3))
    for i, axis in enumerate("xyz"):
        com[:, i], com_vel[:, i] = planar.hermite(samples["t"], samples[f"com_{axis}"], samples[f"com_v{axis}"], t)
        rpy[:, i], rpy_rate[:, i] = planar.hermite(samples["t"], rpy_knots[:, i], rpy_rate_knots[:, i], t)

    R = rotation_zyx(rpy)
    omega = apply(euler_rate_map(rpy), rpy_rate)
    com_offset = apply(R, np.broadcast_to(np.asarray(com_in_base, float), (len(t), 3)))
    base = com - com_offset
    base_vel = com_vel - cross(omega, com_offset)

    out = {"time": t}
    out.update({name: base[:, i] for i, name in enumerate(("x", "y", "z"))})
    out.update({name: rpy[:, i] for i, name in enumerate(("roll", "pitch", "yaw"))})
    out.update({"d" + name: base_vel[:, i] for i, name in enumerate(("x", "y", "z"))})
    out.update({"d" + name: rpy_rate[:, i] for i, name in enumerate(("roll", "pitch", "yaw"))})

    omega_body = apply_transposed(R, omega)
    contact, jacobians, forces = {}, {}, {}
    for leg, (name, towr_leg) in enumerate(zip(LEGS, TOWR_LEGS)):
        foot, foot_vel = np.zeros((len(t), 3)), np.zeros((len(t), 3))
        for i, axis in enumerate("xyz"):
            foot[:, i], foot_vel[:, i] = planar.hermite(samples["t"], samples[f"{towr_leg}_p{axis}"],
                                                        samples[f"{towr_leg}_v{axis}"], t)
        foot_in_trunk = apply_transposed(R, foot - base)
        foot_vel_in_trunk = apply_transposed(R, foot_vel - base_vel) - cross(omega_body, foot_in_trunk)

        q = leg_ik(foot_in_trunk, leg)
        J = leg_jacobian(q, leg)
        qd = np.linalg.solve(J, foot_vel_in_trunk[..., None])[..., 0]
        for j, joint in enumerate(JOINTS):
            out[f"{joint}_{name}"], out[f"d{joint}_{name}"] = q[:, j], qd[:, j]

        stance = planar.read_stance_intervals(phases_path, towr_leg)
        contact[name] = planar.stance_mask(ticks, stance)
        forces[name] = np.stack([foot_force_in_stance(samples, towr_leg, axis, t, ticks, stance, tick)
                                 for axis in "xyz"], 1)
        jacobians[name] = J
    return out, contact, jacobians, forces, R


def rows_with_phases(out, contact, jacobians, forces, R):
    """One row per tick; a tick where the contact set changes is written for both phases. Forces and torques are
    zero for a foot the row's contact set has in the air."""
    contact_set = np.stack([contact[name] for name in LEGS], 1)
    switch = np.concatenate([[False], (contact_set[1:] != contact_set[:-1]).any(axis=1)])
    phase = np.cumsum(switch)

    # The old-phase copy of a switch tick keeps the old contact set
    tick_index = np.concatenate([np.arange(len(contact_set)), np.flatnonzero(switch)])
    is_old_copy = np.concatenate([np.zeros(len(contact_set), bool), np.ones(switch.sum(), bool)])
    order = np.lexsort((~is_old_copy, tick_index))
    tick_index, is_old_copy = tick_index[order], is_old_copy[order]

    rows = {k: v[tick_index] for k, v in out.items()}
    rows["phase"] = phase[tick_index] - is_old_copy
    row_contact = np.where(is_old_copy[:, None], contact_set[np.maximum(tick_index - 1, 0)], contact_set[tick_index])
    for leg, name in enumerate(LEGS):
        planted = row_contact[:, leg]
        force_world = np.where(planted[:, None], forces[name][tick_index], 0.0)
        tau = -apply_transposed(jacobians[name][tick_index], apply_transposed(R[tick_index], force_world))
        rows[f"c_{name}"] = planted.astype(float)
        for i, axis in enumerate("xyz"):
            rows[f"f{axis}_{name}"] = force_world[:, i]
        for j, joint in enumerate(JOINTS):
            rows[f"tau_{joint}_{name}"] = tau[:, j]
    return rows


def columns():
    coords = ["x", "y", "z", "roll", "pitch", "yaw"] + [f"{joint}_{leg}" for leg in LEGS for joint in JOINTS]
    per_leg = [f"tau_{joint}_{leg}" for leg in LEGS for joint in JOINTS]
    per_leg += [f"f{axis}_{leg}" for leg in LEGS for axis in "xyz"]
    return ["time", "phase"] + [f"c_{leg}" for leg in LEGS] + coords + ["d" + c for c in coords] + per_leg


def write_csv(rows, path):
    names = columns()

    def cell(name, i):
        return str(int(rows[name][i])) if name == "phase" else f"{rows[name][i]:.9g}"

    lines = [",".join(names)] + [",".join(cell(n, i) for n in names) for i in range(len(rows["time"]))]
    pathlib.Path(path).write_text("\n".join(lines) + "\n")


def limit_report(rows):
    """Go2 joint angle, speed and torque limits over the whole plan; returns the broken ones."""
    broken = []
    for leg in LEGS:
        thigh_limits = ANGLE_LIMITS["thigh_front" if leg[0] == "F" else "thigh_rear"]
        for joint, (lo, hi) in (("hip", ANGLE_LIMITS["hip"]), ("thigh", thigh_limits), ("calf", ANGLE_LIMITS["calf"])):
            q = rows[f"{joint}_{leg}"]
            if q.min() < lo or q.max() > hi:
                broken.append(f"{joint}_{leg} angle [{q.min():+.2f}, {q.max():+.2f}] outside [{lo:+.2f}, {hi:+.2f}]")
    for joint in JOINTS:
        speed = max(np.abs(rows[f"d{joint}_{leg}"]).max() for leg in LEGS)
        torque = max(np.abs(rows[f"tau_{joint}_{leg}"]).max() for leg in LEGS)
        print(f"  {joint:5s}: speed {speed:5.1f}/{SPEED_LIMITS[joint]} rad/s, torque {torque:5.1f}/{TORQUE_LIMITS[joint]} Nm")
        if speed > SPEED_LIMITS[joint]:
            broken.append(f"{joint} speed {speed:.1f} rad/s")
        if torque > TORQUE_LIMITS[joint]:
            broken.append(f"{joint} torque {torque:.1f} Nm")
    return broken


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("samples", type=pathlib.Path)
    parser.add_argument("-p", "--phases", type=pathlib.Path, help="default: <samples>_phases.csv")
    parser.add_argument("-o", "--output", type=pathlib.Path, help="default: <samples stem without _3d>_3d.csv next to it")
    parser.add_argument("--com-in-base", type=float, nargs=3, default=DEFAULT_COM_IN_BASE, metavar=("X", "Y", "Z"))
    args = parser.parse_args()

    planar = load_planar_converter()
    phases = args.phases or args.samples.with_name(args.samples.stem + "_phases.csv")
    output = args.output or args.samples.with_name(args.samples.stem.removesuffix("_3d") + "_3d.csv")
    if output.resolve() == args.samples.resolve():
        sys.exit("output would overwrite the input; pass -o")

    out, contact, jacobians, forces, R = convert(read_samples(args.samples), phases, args.com_in_base, planar)
    rows = rows_with_phases(out, contact, jacobians, forces, R)
    write_csv(rows, output)

    print(f"wrote {output}: {len(rows['time'])} rows, {rows['time'][-1]:.3f} s, {int(rows['phase'].max()) + 1} contact phases")
    broken = limit_report(rows)
    print("  Go2 limits: " + ("OK" if not broken else "BROKEN: " + "; ".join(broken)))


if __name__ == "__main__":
    main()
