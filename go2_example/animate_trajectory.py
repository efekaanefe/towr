#!/usr/bin/env python3
"""3D animation of a go2_motion / go2_course sample CSV: the trunk as a box, each foot as a red sphere, and the ground
reaction force as an arrow on each foot in contact. Writes a GIF or an MP4 (chosen by the output extension; MP4
needs ffmpeg). The camera follows the trunk; the CoM path is drawn on the ground.

    python3 animate_trajectory.py trajectories/go2_course_agility.csv            # -> trajectories/go2_course_agility.gif
    python3 animate_trajectory.py trajectories/go2_bound_2m.csv -o bound.mp4 --speed 0.5
"""
import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.colors import LightSource  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

from towr_to_3d import DEFAULT_COM_IN_BASE  # noqa: E402

LEGS = ("LF", "RF", "LH", "RH")
TRUNK_SIZE = np.array([0.3762, 0.0935, 0.114])  # go2_description.urdf, base collision box [m]
FOOT_RADIUS = 0.022  # [m]
VIEW_HALF_WIDTH = 0.6  # [m] around the trunk, in x and y
VIEW_HEIGHT = 0.6  # [m]

TRUNK_FACE, TRUNK_EDGE = "#4a7bb7", "#1f3b5c"
FOOT_COLOR, FORCE_COLOR, PATH_COLOR = "#d62728", "#2a9d3c", "#9a9a9a"

# Corner signs of a box and its six faces as corner indices
BOX_CORNERS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) / 2.0
BOX_FACES = ((0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5))


def rotation_from_quaternion(w, x, y, z):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def trunk_faces(row):
    """The six faces of the trunk box in the world. TOWR's body is the CoM; the box sits on the trunk origin."""
    R = rotation_from_quaternion(row.qw, row.qx, row.qy, row.qz)
    base = np.array([row.com_x, row.com_y, row.com_z]) - R @ np.asarray(DEFAULT_COM_IN_BASE)
    corners = base + (BOX_CORNERS * TRUNK_SIZE) @ R.T
    return [corners[list(face)] for face in BOX_FACES], base


def unit_sphere(resolution=10):
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, resolution), np.linspace(0, np.pi, resolution // 2 + 1))
    return np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)


def frame_rows(t, fps, speed):
    """Sample row for each video frame: real time times `speed`, last frame on the last sample."""
    frame_times = np.append(np.arange(0.0, t[-1], speed / fps), t[-1])
    return np.minimum(np.searchsorted(t, frame_times), len(t) - 1)


def segment_lookup(csv):
    """t -> segment row, when go2_course wrote a segment table next to the CSV; otherwise None."""
    table_path = csv.with_name(csv.stem + "_segments.csv")
    if not table_path.exists():
        return None
    table = pd.read_csv(table_path)
    return lambda t: table.iloc[min(np.searchsorted(table.t_end.to_numpy(), t - 1e-9), len(table) - 1)]


class Animation:
    """Draws one frame at a time; the moving artists are removed and drawn again each frame."""

    def __init__(self, samples, segments, force_scale, elev, azim):
        self.samples = samples
        self.segments = segments
        self.force_scale = force_scale
        self.sphere = unit_sphere()
        self.figure = plt.figure(figsize=(8, 6), dpi=100)
        self.ax = self.figure.add_subplot(projection="3d")
        self.ax.view_init(elev=elev, azim=azim)
        self.ax.set_box_aspect((2 * VIEW_HALF_WIDTH, 2 * VIEW_HALF_WIDTH, VIEW_HEIGHT), zoom=1.15)
        self.figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.light = LightSource(azdeg=225, altdeg=60)  # the default light is low and turns the spheres black
        self.ax.set_xlabel("x [m]")
        self.ax.set_ylabel("y [m]")
        self.ax.set_zlabel("z [m]")
        self.moving = []

    def draw(self, row_index):
        for artist in self.moving:
            artist.remove()
        row = self.samples.iloc[row_index]
        faces, base = trunk_faces(row)
        self.moving = [self.ax.add_collection3d(
            Poly3DCollection(faces, facecolor=TRUNK_FACE, edgecolor=TRUNK_EDGE, alpha=0.8, linewidth=0.8))]
        self.moving += [self.draw_foot(row, leg) for leg in LEGS]
        self.moving += self.draw_forces(row)
        self.moving += self.draw_path_near(base)
        self.follow(base)
        self.ax.set_title(self.caption(row), loc="left", y=0.92)
        return self.moving

    def draw_foot(self, row, leg):
        x, y, z = (FOOT_RADIUS * axis for axis in self.sphere)
        return self.ax.plot_surface(x + row[f"{leg}_px"], y + row[f"{leg}_py"], z + row[f"{leg}_pz"] + FOOT_RADIUS,
                                    color=FOOT_COLOR, linewidth=0, shade=True, lightsource=self.light)

    def draw_forces(self, row):
        planted = [leg for leg in LEGS if row[f"{leg}_contact"] > 0.5]
        if not planted:
            return []
        start = np.array([[row[f"{leg}_p{a}"] for a in "xyz"] for leg in planted])
        force = np.array([[row[f"{leg}_f{a}"] for a in "xyz"] for leg in planted]) * self.force_scale
        return [self.ax.quiver(*start.T, *force.T, color=FORCE_COLOR, linewidth=2, arrow_length_ratio=0.15)]

    def draw_path_near(self, base):
        """CoM path on the ground, inside the view only (3D axes do not clip lines)."""
        x, y = self.samples.com_x.to_numpy(), self.samples.com_y.to_numpy()
        outside = (np.abs(x - base[0]) > VIEW_HALF_WIDTH) | (np.abs(y - base[1]) > VIEW_HALF_WIDTH)
        x, y = np.where(outside, np.nan, x), np.where(outside, np.nan, y)
        return self.ax.plot(x, y, np.zeros_like(x), color=PATH_COLOR, linewidth=1)

    def follow(self, base):
        self.ax.set_xlim(base[0] - VIEW_HALF_WIDTH, base[0] + VIEW_HALF_WIDTH)
        self.ax.set_ylim(base[1] - VIEW_HALF_WIDTH, base[1] + VIEW_HALF_WIDTH)
        self.ax.set_zlim(0.0, VIEW_HEIGHT)

    def caption(self, row):
        text = f"t = {row.t:5.2f} s"
        if self.segments:
            segment = self.segments(row.t)
            text += f"   segment {segment.segment}: {segment['name']} ({segment.gait})"
        return text


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=pathlib.Path, help="sample CSV written by go2_motion or go2_course")
    parser.add_argument("-o", "--output", type=pathlib.Path, help=".gif or .mp4 (default: <csv>.gif)")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed, 1 = real time")
    parser.add_argument("--force-scale", type=float, default=0.0015, help="arrow length per newton [m/N]")
    parser.add_argument("--elev", type=float, default=20.0, help="camera elevation [deg]")
    parser.add_argument("--azim", type=float, default=-60.0, help="camera azimuth [deg]")
    args = parser.parse_args()

    output = args.output or args.csv.with_suffix(".gif")
    if output.suffix not in (".gif", ".mp4"):
        parser.error("output must end in .gif or .mp4")
    samples = pd.read_csv(args.csv)
    animation = Animation(samples, segment_lookup(args.csv), args.force_scale, args.elev, args.azim)
    rows = frame_rows(samples.t.to_numpy(), args.fps, args.speed)

    movie = FuncAnimation(animation.figure, animation.draw, frames=rows, blit=False)
    writer = PillowWriter(fps=args.fps) if output.suffix == ".gif" else FFMpegWriter(fps=args.fps, bitrate=2400)
    movie.save(output, writer=writer)
    print(f"wrote {output}: {len(rows)} frames, {args.fps} fps, speed x{args.speed}")


if __name__ == "__main__":
    main()
