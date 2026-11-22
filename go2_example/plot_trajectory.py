"""Plot a go2_motion trajectory as one figure: base motion, each foot, gait timing.

usage:
    python3 plot_trajectory.py go2_trot.csv                       # open window
    python3 plot_trajectory.py go2_trot.csv --save plots/trot.png  # write PNG

Reads go2_<gait>.csv (10 ms samples) and go2_<gait>_phases.csv (one row per stance/swing phase).

Layout (rows):
    base     CoM position | CoM velocity | CoM acceleration
             orientation  | angular vel. | top view (footholds, CoM path)
    per leg  position xyz | velocity xyz | contact force xyz   (stance shaded)
    timing   contact schedule + phase progress of all legs
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

LEGS = ("LF", "RF", "LH", "RH")
GO2_NAME = {"LF": "FL", "RF": "FR", "LH": "RL", "RH": "RR"}
LEG_MARKER = {"LF": "o", "RF": "s", "LH": "^", "RH": "D"}

# One meaning per colour across the whole figure: x / y / z (roll / pitch / yaw about x / y / z).
# Categorical slots 1-3, validated for colour-vision deficiency; every plot has a legend.
COMPONENT_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
NEUTRAL_LINE = "#3d3d3a"
STANCE_SHADE = "#ecebe7"
STANCE_BAR = "#cfcec8"
FOOT_PATH = "#b9b8b2"

STYLE = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 10, "axes.labelsize": 9, "axes.grid": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": INK_2, "ytick.color": INK_2, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "lines.linewidth": 1.5, "legend.frameon": False, "legend.fontsize": 8, "text.color": INK,
}

# (title, {legend label: CSV column}, scale)
BASE_PANELS = (
    ("CoM position [m]", {"x": "com_x", "y": "com_y", "z": "com_z"}, 1.0),
    ("CoM velocity [m/s]", {"x": "com_vx", "y": "com_vy", "z": "com_vz"}, 1.0),
    ("CoM acceleration [m/s²]", {"x": "com_ax", "y": "com_ay", "z": "com_az"}, 1.0),
    ("orientation, ZYX Euler [deg]", {"roll": "roll", "pitch": "pitch", "yaw": "yaw"}, 180.0 / np.pi),
    ("angular velocity, world [rad/s]", {"ωx": "wx", "ωy": "wy", "ωz": "wz"}, 1.0),
)

# (title, CSV key: <leg>_<key><axis>)
FOOT_PANELS = (
    ("position [m]", "p"),
    ("velocity [m/s]", "v"),
    ("contact force, ground on foot [N]", "f"),
)


def draw_components(ax, t: pd.Series, columns: dict, d: pd.DataFrame, title: str, scale: float = 1.0):
    """Up to three components of one quantity in one plot, legend top-right."""
    for (label, col), color in zip(columns.items(), COMPONENT_COLORS):
        ax.plot(t, d[col] * scale, color=color, label=label)
    ax.set_title(title, loc="left")
    ax.legend(ncol=len(columns), loc="lower right", bbox_to_anchor=(1.0, 1.0),
              borderaxespad=0.2, handlelength=1.2, columnspacing=1.0)


def shade_stance(ax, stance_of_leg: pd.DataFrame):
    for _, phase in stance_of_leg.iterrows():
        ax.axvspan(phase.t_start, phase.t_end, color=STANCE_SHADE, linewidth=0, zorder=0)


def draw_contact_timing(ax, d: pd.DataFrame, stance: pd.DataFrame):
    """Gait diagram: stance bars per leg, phase progress (0 -> 1) drawn inside each row."""
    for row, leg in enumerate(LEGS):
        spans = stance.query("leg == @leg")
        ax.broken_barh(list(zip(spans.t_start, spans.duration)), (row - 0.4, 0.8),
                       facecolors=STANCE_BAR, edgecolor=SURFACE, linewidth=1.5)
        ax.plot(d.t, row + 0.4 - 0.8 * d[f"{leg}_phase_progress"], color=NEUTRAL_LINE, linewidth=1.2)
    ax.set_yticks(range(len(LEGS)), [f"{leg} ({GO2_NAME[leg]})" for leg in LEGS])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("t [s]")
    ax.set_title("contact schedule and phase timing", loc="left")
    handles = [Patch(color=STANCE_BAR, label="stance (gap = swing)"),
               Line2D([], [], color=NEUTRAL_LINE, linewidth=1.2, label="phase progress 0 → 1")]
    ax.legend(handles=handles, ncol=2, loc="lower right", bbox_to_anchor=(1.0, 1.0), borderaxespad=0.2)


def draw_top_view(ax, d: pd.DataFrame, stance: pd.DataFrame):
    """x-y plane: foot paths, footholds (marker per leg), CoM path with heading arrows."""
    for leg in LEGS:
        ax.plot(d[f"{leg}_px"], d[f"{leg}_py"], color=FOOT_PATH, linewidth=0.8, zorder=1)
        footholds = stance.query("leg == @leg")
        ax.scatter(footholds.start_x, footholds.start_y, marker=LEG_MARKER[leg], s=40,
                   color=INK_2, edgecolors=SURFACE, linewidths=1.5, zorder=3, label=leg)
        ax.annotate(leg, (footholds.start_x.iloc[0], footholds.start_y.iloc[0]),
                    textcoords="offset points", xytext=(-6, 7), fontsize=8, color=INK_2)
    ax.plot(d.com_x, d.com_y, color=NEUTRAL_LINE, zorder=2, label="CoM")
    heading = d.iloc[:: max(1, len(d) // 12)]
    ax.quiver(heading.com_x, heading.com_y, np.cos(heading.yaw), np.sin(heading.yaw),
              color=NEUTRAL_LINE, angles="xy", scale_units="xy", scale=1 / 0.06, width=0.004, zorder=2)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("top view (x-y)", loc="left")
    ax.legend(ncol=5, loc="lower right", bbox_to_anchor=(1.0, 1.0), borderaxespad=0.2,
              handlelength=1.2, columnspacing=0.8)


def plot_trajectory(d: pd.DataFrame, phases: pd.DataFrame, title: str):
    stance = phases.query("contact == 1")
    fig = plt.figure(figsize=(16, 23), layout="constrained")
    n_rows = 2 + len(LEGS) + 1
    grid = fig.add_gridspec(n_rows, 3, height_ratios=(1,) * (n_rows - 1) + (1.2,))

    first = fig.add_subplot(grid[0, 0])
    def time_axis(spec):
        return fig.add_subplot(spec, sharex=first)

    # base: first five cells of the 2x3 block, top view in the sixth
    base_axes = [first, time_axis(grid[0, 1]), time_axis(grid[0, 2]),
                 time_axis(grid[1, 0]), time_axis(grid[1, 1])]
    for ax, (panel_title, columns, scale) in zip(base_axes, BASE_PANELS):
        draw_components(ax, d.t, columns, d, panel_title, scale)
    draw_top_view(fig.add_subplot(grid[1, 2]), d, stance)

    # one row per leg: position | velocity | force, stance phases shaded
    for row, leg in enumerate(LEGS, start=2):
        for col, (panel_title, key) in enumerate(FOOT_PANELS):
            ax = time_axis(grid[row, col])
            shade_stance(ax, stance.query("leg == @leg"))
            columns = {axis: f"{leg}_{key}{axis}" for axis in "xyz"}
            draw_components(ax, d.t, columns, d, f"{leg} ({GO2_NAME[leg]}) foot {panel_title}")

    draw_contact_timing(time_axis(grid[-1, :]), d, stance)
    fig.suptitle(f"{title}  ·  world frame  ·  grey background = foot in stance", color=INK, fontsize=13)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=Path, help="samples CSV written by go2_motion, e.g. go2_trot.csv")
    parser.add_argument("--save", type=Path, metavar="PNG", help="write the figure to this file instead of showing it")
    args = parser.parse_args()

    samples = pd.read_csv(args.csv)
    phases = pd.read_csv(args.csv.with_name(args.csv.stem + "_phases.csv"))

    if args.save:
        plt.switch_backend("Agg")
    plt.rcParams.update(STYLE)
    fig = plot_trajectory(samples, phases, args.csv.stem)

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=100)
        print(f"wrote {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
