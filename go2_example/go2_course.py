#!/usr/bin/env python3
"""Long benchmark motions for the Go2, built as a chain of short TOWR plans ("moving joins").

TOWR solves one segment at a time (gait, move, time). Each segment starts at the exact state where the previous
one ended (go2_motion --start <prefix>_end.csv): base pose and velocity, Euler rates, foot positions. Two segments
join in a short four-feet stance while the body moves (--end-vel, --stand). Base position and velocity are
continuous at a join; base acceleration and foot forces are not (the join report prints the force step).

    python3 go2_course.py agility                  # solve, join, convert to 3D, limit check, plot
    python3 go2_course.py agility --from-segment 4 # keep segments 0-3 from the last run, solve the rest

Output in --out-dir (default trajectories/), prefix go2_course_<name>:
    .csv / _phases.csv   joined TOWR samples and phases (as go2_motion writes them; stance across a join = one phase)
    _segments.csv        segment table: name, gait, start/end time, move, end velocity
    _3d.csv              tracker CSV (towr_to_3d.py)
    .png                 plot (plot_trajectory.py)
Segment files go to --work-dir (default build/courses/<name>/).
"""
import argparse
import importlib.util
import math
import pathlib
import subprocess
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
LEGS = ("LF", "RF", "LH", "RH")

# Settings of the tested 3D plans (TOWR_GO2_GUIDE.md, "3D plans"): 10 ms physics grid, force limits every 10 ms.
# --end-feet: every segment ends with its feet near the nominal stance, so the next one starts from a neat stance
# (without it a trot leaves one diagonal pair ~7 cm behind, and a bound entered from there needs 40-60 Nm thigh).
SOLVER_FLAGS = ("--dense-force", "--mu", "0.8", "--dt-dyn", "0.01", "--base-poly", "0.01",
                "--max-cpu", "300", "--sample-dt", "0.002", "--end-feet", "0.03")

# Gaits: a box that keeps the knee bent. TOWR's default box reaches 0.357 m below the CoM and +-0.10 m fore/aft;
# its lower corners are past the leg's reach (calf straighter than its -0.84 rad limit).
GAIT_BOX = ("--foot-z", "0.22", "0.32")
# Jumps: deeper crouch for the push and the landing.
JUMP_BOX = ("--foot-z", "0.13", "0.31")


@dataclass(frozen=True)
class Segment:
    name: str
    gait: str          # go2_motion gait
    move: tuple        # (forward [m], left [m], turn [rad]) from the segment start, start heading frame
    time: float        # [s]
    end_vel: tuple = (0.0, 0.0, 0.0)  # (forward [m/s], left [m/s], yaw rate [rad/s]) at the end, end heading frame
    flags: tuple = ()  # more go2_motion flags: gait(...) or jump(...)
    apex: tuple = (0.03, 0.08)        # swing apex bounds [m]


def arc(length, turn):
    """Move along a circular arc of this length, turning by `turn`: (forward, left, turn) in the start heading frame."""
    if abs(turn) < 1e-9:
        return (length, 0.0, 0.0)
    radius = length / turn
    return (radius * math.sin(turn), radius * (1.0 - math.cos(turn)), turn)


def gait(strides, *more, stand=0.06, box=GAIT_BOX):
    """Flags for a walk/trot/bound/... segment. stand: the four-feet stance at each end [s]. Keep it short: at a
    moving join the body moves on over planted feet."""
    return ("--strides", str(strides), "--stand", str(stand)) + box + more


def jump(flight, *more):
    """Flags for one jump. With TOWR's 2-piece swing and a low apex (Segment.apex) the in-air legs stay slow; 4 pieces
    solved too, but needed ~25 rad/s calf speed in the air."""
    return ("--strides", "1", "--flight", str(flight)) + JUMP_BOX + more

COURSES = {
    # trot out, S-curve, bound, turning pronk hops, forward jump, turn on the spot
    "agility": [
        Segment("trot_out", "trot", (0.7, 0.0, 0.0), 2.0, (0.45, 0, 0), gait(4)),
        Segment("arc_left", "trot", arc(0.8, math.pi / 2), 2.0, (0.4, 0, 0), gait(4)),
        Segment("arc_right", "trot", arc(0.8, -math.pi / 2), 2.0, (0.5, 0, 0), gait(4)),
        Segment("bound", "bound", (1.2, 0.0, 0.0), 2.0, (0.5, 0, 0), gait(6, "--force-cost", "3e-6")),
        Segment("pronk_turn", "pronk", (0.5, 0.15, 0.5), 1.8, (0.2, 0, 0), gait(4)),
        Segment("jump", "jump", (0.5, 0.0, 0.0), 1.3, (0, 0, 0), jump(0.25, "--force-cost", "1e-6"), (0.02, 0.08)),
        Segment("spin", "trot", (0.0, 0.0, math.pi), 3.0, (0, 0, 0), gait(6)),
    ],
    # one straight line through the gaits, pronk to a stop. TOWR's gallop (Hop3) did not solve here in 300 s (moving
    # or from rest, 0.6-2.0 m, 2-3 strides, 10 or 20 ms grid; only a slow 0.6 m one from rest), so pace takes its place
    "gait_ladder": [
        Segment("walk", "walk", (0.4, 0.0, 0.0), 3.0, (0.2, 0, 0), gait(2)),
        Segment("trot", "trot", (0.7, 0.0, 0.0), 2.0, (0.45, 0, 0), gait(4)),
        Segment("flytrot", "flytrot", (1.0, 0.0, 0.0), 1.8, (0.6, 0, 0), gait(4)),
        Segment("bound", "bound", (1.4, 0.0, 0.0), 2.0, (0.75, 0, 0), gait(7, "--force-cost", "3e-6")),
        # slows to 0.5 m/s: a pronk's push stance is long, entered at 0.75 m/s the body runs 15 cm past the feet
        Segment("pace", "pace", (1.4, 0.0, 0.0), 2.0, (0.5, 0, 0), gait(5, box=("--foot-z", "0.22", "0.31"))),
        Segment("pronk_stop", "pronk", (0.7, 0.0, 0.0), 2.2, (0, 0, 0), gait(5)),
    ],
    # hops that turn, sideways hops, a diagonal jump, a turning pronk back
    "spin_hop": [
        Segment("pronk_spin", "pronk", (0.0, 0.0, math.pi), 2.8, (0, 0, 0), gait(6)),
        Segment("side_hops", "pronk", (0.0, 0.45, 0.0), 2.2, (0, 0, 0), gait(5)),
        Segment("diag_jump", "jump", (0.35, -0.25, -0.5), 1.3, (0, 0, 0), jump(0.25), (0.02, 0.08)),
        Segment("pronk_turn", "pronk", (0.4, 0.0, -math.pi / 2), 2.4, (0, 0, 0), gait(5)),
    ],
    # trot slalom: heading swings +-0.6 rad, joins at the heading extremes (yaw rate 0), speed up and down
    "slalom": [
        Segment("in", "trot", arc(0.6, 0.6), 1.8, (0.4, 0, 0), gait(4)),
        Segment("gate1", "trot", arc(1.0, -1.2), 2.0, (0.5, 0, 0), gait(4)),
        Segment("gate2", "flytrot", arc(1.2, 1.2), 2.0, (0.6, 0, 0), gait(6)),
        Segment("gate3", "flytrot", arc(1.2, -1.2), 2.0, (0.5, 0, 0), gait(6)),
        Segment("gate4", "flytrot", arc(1.0, 1.2), 2.0, (0.4, 0, 0), gait(6)),
        Segment("out", "trot", arc(0.6, -0.6), 2.0, (0, 0, 0), gait(4)),
    ],
}


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    if spec is None or spec.loader is None:
        sys.exit(f"cannot load {name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── Solving ──

def motion_command(binary, segment, prefix, start_state):
    command = [str(binary), segment.gait, *map(str, segment.move), str(segment.time), *map(str, segment.apex),
               *SOLVER_FLAGS, *segment.flags, "--end-vel", *map(str, segment.end_vel), "--out", str(prefix)]
    return command + (["--start", str(start_state)] if start_state else [])


def solve_chain(binary, segments, work_dir, from_segment):
    """Solve the segments in order, each from the previous end state; returns the output prefixes."""
    prefixes = [work_dir / f"{k:02d}_{segment.name}" for k, segment in enumerate(segments)]
    for k, (segment, prefix) in enumerate(zip(segments, prefixes)):
        if k < from_segment:
            continue
        start_state = pathlib.Path(f"{prefixes[k - 1]}_end.csv") if k > 0 else None
        if start_state and not start_state.exists():
            sys.exit(f"{start_state} missing; solve from an earlier segment")
        result = subprocess.run(motion_command(binary, segment, prefix, start_state), capture_output=True, text=True)
        status = next((line for line in result.stdout.splitlines() if line.startswith("IPOPT status")), "no status")
        print(f"segment {k} {segment.name:12s} {segment.gait:8s} {status}")
        if result.returncode != 0:
            sys.exit(f"segment {k} ({segment.name}) not solved; tune it and rerun with --from-segment {k}\n"
                     + result.stderr)
    return prefixes


# ── Joining ──

def merge_join_stances(phases):
    """A foot planted at the end of one segment and the start of the next is one stance phase."""
    merged = []
    for _, rows in phases.groupby("leg", sort=False):
        leg_phases = []
        for row in rows.to_dict("records"):
            previous = leg_phases[-1] if leg_phases else None
            if previous and previous["contact"] and row["contact"] and abs(previous["t_end"] - row["t_start"]) < 1e-9:
                previous.update(t_end=row["t_end"], end_x=row["end_x"], end_y=row["end_y"], end_z=row["end_z"])
            else:
                leg_phases.append(dict(row))
        for index, phase in enumerate(leg_phases):
            phase["phase"] = index
            phase["duration"] = phase["t_end"] - phase["t_start"]
        merged += leg_phases
    return pd.DataFrame(merged)[phases.columns]


def relabel_phases(samples, phases):
    """Per-sample phase number, progress and contact from the joined phase table (as go2_motion's PhaseAt)."""
    for leg, rows in phases.groupby("leg"):
        rows = rows.sort_values("t_start")
        index = np.minimum(np.searchsorted(rows.t_end.to_numpy(), samples.t.to_numpy(), side="right"), len(rows) - 1)
        t_start, duration = rows.t_start.to_numpy()[index], rows.duration.to_numpy()[index]
        samples[f"{leg}_phase"] = rows.phase.to_numpy()[index]
        samples[f"{leg}_phase_progress"] = np.clip((samples.t.to_numpy() - t_start) / duration, 0.0, 1.0)
        samples[f"{leg}_contact"] = rows.contact.to_numpy()[index]
    return samples


def join(prefixes, segments):
    """One time axis for all segments. The join instant is kept once (from the earlier segment)."""
    samples, phases, table, join_steps = [], [], [], []
    offset = 0.0
    for k, (prefix, segment) in enumerate(zip(prefixes, segments)):
        s = pd.read_csv(f"{prefix}.csv")
        p = pd.read_csv(f"{prefix}_phases.csv")
        duration = s.t.iloc[-1]
        s["t"] += offset
        p[["t_start", "t_end"]] += offset
        if k > 0:
            join_steps.append(force_step(samples[-1].iloc[-1], s.iloc[0]))
        samples.append(s if k == 0 else s.iloc[1:])
        phases.append(p)
        table.append({"segment": k, "name": segment.name, "gait": segment.gait, "t_start": offset,
                      "t_end": offset + duration, "move_forward": segment.move[0], "move_left": segment.move[1],
                      "move_turn": segment.move[2], "end_forward": segment.end_vel[0],
                      "end_left": segment.end_vel[1], "end_yaw_rate": segment.end_vel[2]})
        offset += duration
    phases = merge_join_stances(pd.concat(phases, ignore_index=True))
    samples = relabel_phases(pd.concat(samples, ignore_index=True), phases)
    return samples, phases, pd.DataFrame(table), join_steps


def force_step(before, after):
    """Largest change of one foot's force across a join [N] (forces are not continuous there)."""
    return max(float(np.linalg.norm([after[f"{leg}_f{a}"] - before[f"{leg}_f{a}"] for a in "xyz"])) for leg in LEGS)


# ── Checking ──

def convert_and_check(samples_csv, phases_csv, table, out_3d):
    """3D tracker CSV for the whole course, then the Go2 limit check per segment."""
    to3d = load_module("towr_to_3d")
    planar = to3d.load_planar_converter()
    out, contact, jacobians, forces, R = to3d.convert(to3d.read_samples(samples_csv), phases_csv,
                                                       to3d.DEFAULT_COM_IN_BASE, planar)
    rows = to3d.rows_with_phases(out, contact, jacobians, forces, R)
    to3d.write_csv(rows, out_3d)
    print(f"wrote {out_3d}: {len(rows['time'])} rows, {rows['time'][-1]:.3f} s")

    all_ok = True
    for segment in table.itertuples():
        inside = (rows["time"] >= segment.t_start - 1e-9) & (rows["time"] <= segment.t_end + 1e-9)
        print(f"segment {segment.segment} {segment.name} [{segment.t_start:.2f}, {segment.t_end:.2f}] s")
        broken = to3d.limit_report({name: values[inside] for name, values in rows.items()})
        print("  Go2 limits: " + ("OK" if not broken else "BROKEN: " + "; ".join(broken)))
        all_ok = all_ok and not broken
    return all_ok


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("course", choices=sorted(COURSES))
    parser.add_argument("--binary", type=pathlib.Path, default=HERE / "build" / "go2_motion")
    parser.add_argument("--out-dir", type=pathlib.Path, default=HERE / "trajectories")
    parser.add_argument("--work-dir", type=pathlib.Path, help="default: build/courses/<course>")
    parser.add_argument("--from-segment", type=int, default=0, help="reuse the solved segments before this one")
    args = parser.parse_args()

    segments = COURSES[args.course]
    work_dir = args.work_dir or HERE / "build" / "courses" / args.course
    work_dir.mkdir(parents=True, exist_ok=True)
    prefixes = solve_chain(args.binary, segments, work_dir, args.from_segment)

    samples, phases, table, join_steps = join(prefixes, segments)
    out = args.out_dir / f"go2_course_{args.course}"
    samples_csv, phases_csv = out.with_suffix(".csv"), out.with_name(out.name + "_phases.csv")
    samples.to_csv(samples_csv, index=False, float_format="%.9g")
    phases.to_csv(phases_csv, index=False, float_format="%.9g")
    table.to_csv(out.with_name(out.name + "_segments.csv"), index=False, float_format="%.9g")
    print(f"joined {len(segments)} segments, {samples.t.iloc[-1]:.2f} s; foot force step at the joins [N]: "
          + ", ".join(f"{step:.0f}" for step in join_steps))

    limits_ok = convert_and_check(samples_csv, phases_csv, table, out.with_name(out.name + "_3d.csv"))
    subprocess.run([sys.executable, str(HERE / "plot_trajectory.py"), str(samples_csv),
                    "--save", str(out.with_suffix(".png"))], check=True)
    print(f"wrote {out}.png; Go2 limits {'OK in every segment' if limits_ok else 'BROKEN (see above)'}")


if __name__ == "__main__":
    main()
