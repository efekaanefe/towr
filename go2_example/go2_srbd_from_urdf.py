"""Compute single-rigid-body params (mass, CoM, composite inertia, nominal feet) from a URDF.

usage: python3 go2_srbd_from_urdf.py <robot.urdf> <thigh_angle_rad> <calf_angle_rad>
"""
import sys
import xml.etree.ElementTree as ET
import numpy as np

def rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

def axis_angle(axis, q):
    a = np.asarray(axis, float); a /= np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * K @ K

def parse_vec(s, default="0 0 0"):
    return np.array([float(v) for v in (s or default).split()])

urdf = ET.parse(sys.argv[1]).getroot()
links = {l.get("name"): l for l in urdf.findall("link")}
joints = urdf.findall("joint")
child_joint = {j.find("child").get("link"): j for j in joints}

thigh_q, calf_q = float(sys.argv[2]), float(sys.argv[3])
def joint_q(name):
    if name.endswith("thigh_joint"): return thigh_q
    if name.endswith("calf_joint"): return calf_q
    return 0.0

def link_pose(name):
    """World pose of link frame, base at origin with identity rotation."""
    if name not in child_joint: return np.eye(3), np.zeros(3)
    j = child_joint[name]
    R_p, p_p = link_pose(j.find("parent").get("link"))
    o = j.find("origin")
    xyz = parse_vec(o.get("xyz") if o is not None else None)
    rpy = parse_vec(o.get("rpy") if o is not None else None)
    R_j = rpy_to_R(*rpy)
    if j.get("type") in ("revolute", "continuous"):
        ax = j.find("axis")
        R_j = R_j @ axis_angle(parse_vec(ax.get("xyz") if ax is not None else "1 0 0"), joint_q(j.get("name")))
    return R_p @ R_j, p_p + R_p @ xyz

bodies = []
for name, l in links.items():
    inert = l.find("inertial")
    if inert is None: continue
    m = float(inert.find("mass").get("value"))
    o = inert.find("origin")
    c = parse_vec(o.get("xyz") if o is not None else None)
    rc = parse_vec(o.get("rpy") if o is not None else None)
    I = inert.find("inertia")
    g = {k: float(I.get(k)) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")}
    I_l = np.array([[g["ixx"], g["ixy"], g["ixz"]], [g["ixy"], g["iyy"], g["iyz"]], [g["ixz"], g["iyz"], g["izz"]]])
    R, p = link_pose(name)
    Rc = R @ rpy_to_R(*rc)
    bodies.append((name, m, p + R @ c, Rc @ I_l @ Rc.T))

M = sum(b[1] for b in bodies)
com = sum(b[1] * b[2] for b in bodies) / M
I_com = sum(Ib + m * ((r - com) @ (r - com) * np.eye(3) - np.outer(r - com, r - com)) for _, m, r, Ib in bodies)

print(f"total mass       : {M:.3f} kg  ({len(bodies)} bodies)")
print(f"CoM in base frame: {com.round(4)}")
print("composite inertia about CoM (base axes):\n", I_com.round(4))
for leg in ("FL", "FR", "RL", "RR"):
    R, p = link_pose(f"{leg}_foot")
    print(f"{leg}_foot rel. base origin: {p.round(4)}   rel. CoM: {(p - com).round(4)}")
