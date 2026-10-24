"""MJCF builder for the approximate digital twin of the Tidy workspace.

Table frame = robot frame (see common/kinematics.py).  The scene contains:
  * SO-100-class 5-DoF arm + parallel gripper (position servos)
  * placement mat (light grey) with 4 dark corner markers (stand-in for ArUco markers)
  * fixtures (bins / shelf / tray) and free objects with distinct flat colours
  * optional human-hand proxy (mocap) with coloured thumb/index fingertip caps
  * fixed overhead robot camera + a movable "phone" camera
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass, field

import numpy as np

from common.kinematics import GEOM

# Mat region (reachable placements). Grid rows A..E along x (far from robot = E), cols 1..6 along y.
MAT_X = (0.10, 0.26)
MAT_Y = (-0.13, 0.13)
GRID_ROWS = "ABCDE"
GRID_COLS = 6
MARKER_POS = [(0.085, -0.145), (0.085, 0.145), (0.275, 0.145), (0.275, -0.145)]  # mat corners, table frame
MARKER_HALF = 0.008


def hsv(h_deg: float, s: float = 0.9, v: float = 0.85) -> tuple[float, float, float]:
    r, g, b = colorsys.hsv_to_rgb((h_deg % 360) / 360.0, s, v)
    return (r, g, b)


# Hue palette (deg). Fingertip caps use hues not used by any object.
HUES = {"red": 0, "orange": 30, "yellow": 55, "green": 130, "cyan": 180, "blue": 225, "purple": 270}
THUMB_HUE, INDEX_HUE = 90, 330  # lime, pink
SKIN_RGB = (0.80, 0.62, 0.50)


@dataclass
class ObjectSpec:
    name: str
    kind: str                      # box | cylinder | capsule
    size: tuple[float, ...]        # box: half extents (x,y,z); cylinder: (radius, half_height); capsule: (radius, half_len)
    color: str                     # key into HUES
    mass: float = 0.05
    category: str = ""             # pen | cup | tape | box | block | plate | can

    @property
    def height(self) -> float:
        if self.kind == "box":
            return 2 * self.size[2]
        if self.kind == "cylinder":
            return 2 * self.size[1]
        return 2 * self.size[0]  # capsule lying flat

    def half_extent_along(self, yaw_obj: float, direction_yaw: float) -> float:
        """Half-extent of the object's footprint along a horizontal direction (support function)."""
        d = direction_yaw - yaw_obj
        if self.kind == "box":
            return abs(self.size[0] * np.cos(d)) + abs(self.size[1] * np.sin(d))
        if self.kind == "cylinder":
            return self.size[0]
        # capsule lying along its local x axis
        return abs(self.size[1] * np.cos(d)) + self.size[0]


@dataclass
class FixtureSpec:
    name: str
    kind: str                       # bin | shelf | tray
    pos: tuple[float, float]        # centre xy
    half: tuple[float, float]       # inner half extents xy
    height: float                   # wall height (bin) / surface height (shelf) / thickness (tray)
    color: tuple[float, float, float] = (0.35, 0.35, 0.40)

    def contains_xy(self, xy: np.ndarray, margin: float = 0.0) -> bool:
        return bool(abs(xy[0] - self.pos[0]) <= self.half[0] - margin and abs(xy[1] - self.pos[1]) <= self.half[1] - margin)

    @property
    def surface_z(self) -> float:
        if self.kind == "bin":
            return 0.004
        return self.height


# ---- object library --------------------------------------------------------------------------
LIB: dict[str, ObjectSpec] = {
    "pen": ObjectSpec("pen", "capsule", (0.007, 0.050), "blue", 0.015, "pen"),
    "cup": ObjectSpec("cup", "cylinder", (0.026, 0.035), "red", 0.06, "cup"),
    "tape": ObjectSpec("tape", "cylinder", (0.033, 0.015), "orange", 0.05, "tape"),
    "box": ObjectSpec("box", "box", (0.040, 0.018, 0.022), "green", 0.05, "box"),
    "can": ObjectSpec("can", "cylinder", (0.024, 0.040), "cyan", 0.08, "can"),
    "block_red": ObjectSpec("block_red", "box", (0.018, 0.018, 0.018), "red", 0.03, "block"),
    "block_blue": ObjectSpec("block_blue", "box", (0.018, 0.018, 0.018), "blue", 0.03, "block"),
    "block_red2": ObjectSpec("block_red2", "box", (0.018, 0.018, 0.018), "red", 0.03, "block"),
    "block_blue2": ObjectSpec("block_blue2", "box", (0.018, 0.018, 0.018), "blue", 0.03, "block"),
    "plate_blue": ObjectSpec("plate_blue", "cylinder", (0.034, 0.010), "blue", 0.04, "plate"),
    "plate_red": ObjectSpec("plate_red", "cylinder", (0.030, 0.010), "red", 0.04, "plate"),
    "plate_yellow": ObjectSpec("plate_yellow", "cylinder", (0.026, 0.010), "yellow", 0.04, "plate"),
    "snack_box": ObjectSpec("snack_box", "box", (0.030, 0.016, 0.035), "purple", 0.05, "box"),
    "eraser": ObjectSpec("eraser", "box", (0.022, 0.012, 0.010), "yellow", 0.02, "box"),
}

FIXTURES: dict[str, FixtureSpec] = {
    "bin": FixtureSpec("bin", "bin", (0.06, 0.205), (0.055, 0.045), 0.045),
    "bin_left": FixtureSpec("bin_left", "bin", (0.06, 0.205), (0.055, 0.045), 0.045, (0.30, 0.30, 0.55)),
    "bin_right": FixtureSpec("bin_right", "bin", (0.06, -0.205), (0.055, 0.045), 0.045, (0.55, 0.30, 0.30)),
    "shelf": FixtureSpec("shelf", "shelf", (0.075, 0.205), (0.065, 0.050), 0.045, (0.45, 0.40, 0.35)),
    "tray": FixtureSpec("tray", "tray", (0.065, -0.195), (0.05, 0.045), 0.008, (0.25, 0.45, 0.30)),
}


@dataclass
class SceneSpec:
    objects: list[ObjectSpec]
    fixtures: list[FixtureSpec]
    include_hand: bool = False
    light_scale: float = 1.0
    light_tint: tuple[float, float, float] = (1.0, 1.0, 1.0)
    distractors: list[ObjectSpec] = field(default_factory=list)


def _rgba(rgb: tuple[float, float, float], a: float = 1.0) -> str:
    return f"{rgb[0]:.3f} {rgb[1]:.3f} {rgb[2]:.3f} {a:.2f}"


def _arm_xml() -> str:
    g = GEOM
    grey, dark = "0.92 0.92 0.90 1", "0.15 0.15 0.15 1"
    lift_off = g.shoulder_height - g.base_height
    return f"""
    <body name="base" pos="0 0 0">
      <geom name="base_g" type="cylinder" size="0.045 {g.base_height/2:.4f}" pos="0 0 {g.base_height/2:.4f}" rgba="{grey}" class="arm"/>
      <body name="pan_link" gravcomp="1" pos="0 0 {g.base_height:.4f}">
        <joint name="shoulder_pan" type="hinge" axis="0 0 1" range="{g.q_min[0]} {g.q_max[0]}" damping="0.5" armature="0.01"/>
        <geom type="cylinder" size="0.022 {lift_off/2:.4f}" pos="0 0 {lift_off/2:.4f}" rgba="{grey}" class="arm"/>
        <body name="upper_arm" gravcomp="1" pos="0 0 {lift_off:.4f}">
          <joint name="shoulder_lift" type="hinge" axis="0 1 0" range="{g.q_min[1]} {g.q_max[1]}" damping="0.5" armature="0.01"/>
          <geom type="capsule" fromto="0 0 0 0 0 {g.l1:.4f}" size="0.016" rgba="{grey}" class="arm"/>
          <body name="forearm" gravcomp="1" pos="0 0 {g.l1:.4f}">
            <joint name="elbow_flex" type="hinge" axis="0 1 0" range="{g.q_min[2]} {g.q_max[2]}" damping="0.4" armature="0.01"/>
            <geom type="capsule" fromto="0 0 0 0 0 {g.l2:.4f}" size="0.014" rgba="{grey}" class="arm"/>
            <body name="wrist" gravcomp="1" pos="0 0 {g.l2:.4f}">
              <joint name="wrist_flex" type="hinge" axis="0 1 0" range="{g.q_min[3]} {g.q_max[3]}" damping="0.2" armature="0.005"/>
              <geom type="sphere" size="0.014" rgba="{grey}" class="arm"/>
              <body name="hand" gravcomp="1" pos="0 0 0">
                <joint name="wrist_roll" type="hinge" axis="0 0 1" range="{g.q_min[4]} {g.q_max[4]}" damping="0.1" armature="0.002"/>
                <geom name="palm" type="box" size="0.012 0.045 0.012" pos="0 0 {g.l3-0.045:.4f}" rgba="{dark}" class="arm"/>
                <geom type="capsule" fromto="0 0 0 0 0 {g.l3-0.055:.4f}" size="0.012" rgba="{grey}" class="arm"/>
                <site name="tcp" pos="0 0 {g.l3:.4f}" size="0.004" rgba="1 0 0 0.6"/>
                <body name="finger_l" gravcomp="1" pos="0 0 {g.l3-0.012:.4f}">
                  <joint name="finger_l" type="slide" axis="0 1 0" range="0 {g.max_opening/2:.3f}" damping="2" armature="0.01"/>
                  <geom name="finger_l_g" type="box" size="0.010 0.004 0.024" pos="0 0.004 0" rgba="{dark}" class="finger"/>
                </body>
                <body name="finger_r" gravcomp="1" pos="0 0 {g.l3-0.012:.4f}">
                  <joint name="finger_r" type="slide" axis="0 -1 0" range="0 {g.max_opening/2:.3f}" damping="2" armature="0.01"/>
                  <geom name="finger_r_g" type="box" size="0.010 0.004 0.024" pos="0 -0.004 0" rgba="{dark}" class="finger"/>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>"""


def _fixture_xml(f: FixtureSpec) -> str:
    x, y = f.pos
    hx, hy = f.half
    c = _rgba(f.color)
    if f.kind == "bin":
        t, h = 0.004, f.height
        walls = [
            (x, y, h / 2, hx + t, t, h / 2, 0, hy + t),
            (x, y, h / 2, hx + t, t, h / 2, 0, -(hy + t)),
            (x, y, h / 2, t, hy + t, h / 2, hx + t, 0),
            (x, y, h / 2, t, hy + t, h / 2, -(hx + t), 0),
        ]
        s = f'<body name="{f.name}" pos="0 0 0"><geom type="box" size="{hx:.4f} {hy:.4f} 0.002" pos="{x} {y} 0.002" rgba="{c}" class="fixture"/>'
        for (cx, cy, cz, sx, sy, sz, ox, oy) in walls:
            s += f'<geom type="box" size="{sx:.4f} {sy:.4f} {sz:.4f}" pos="{cx+ox:.4f} {cy+oy:.4f} {cz:.4f}" rgba="{c}" class="fixture"/>'
        return s + "</body>"
    h = f.height
    return (f'<body name="{f.name}" pos="0 0 0"><geom type="box" size="{hx:.4f} {hy:.4f} {h/2:.4f}" '
            f'pos="{x} {y} {h/2:.4f}" rgba="{c}" class="fixture"/></body>')


def _object_xml(o: ObjectSpec, idx: int) -> str:
    rgb = hsv(HUES[o.color])
    if o.kind == "box":
        gs = f'type="box" size="{o.size[0]} {o.size[1]} {o.size[2]}"'
        z = o.size[2]
    elif o.kind == "cylinder":
        gs = f'type="cylinder" size="{o.size[0]} {o.size[1]}"'
        z = o.size[1]
    else:
        gs = f'type="capsule" size="{o.size[0]} {o.size[1]}" euler="0 1.5708 0"'
        z = o.size[0]
    return (f'<body name="obj_{o.name}" pos="0.18 {0.05*idx:.3f} {z+0.001:.4f}"><freejoint name="obj_{o.name}_jnt"/>'
            f'<geom name="obj_{o.name}_g" {gs} mass="{o.mass}" rgba="{_rgba(rgb)}" class="object"/></body>')


def _hand_xml() -> str:
    skin = _rgba(SKIN_RGB)
    th, ix = _rgba(hsv(THUMB_HUE, 0.9, 0.9)), _rgba(hsv(INDEX_HUE, 0.8, 0.95))
    # mocap bodies: wrist (with palm + forearm), thumb tip, index tip. No collisions (humans don't knock
    # things in curated demos; grasping is via kinematic attachment like the robot).
    return f"""
    <body name="h_wrist" mocap="true" pos="0.6 0 0.4">
      <geom type="sphere" size="0.022" rgba="{skin}" class="hand"/>
      <geom type="capsule" fromto="0 0 0 0.25 0 0.12" size="0.024" rgba="{skin}" class="hand"/>
      <site name="h_wrist_site" size="0.003"/>
    </body>
    <body name="h_thumb" mocap="true" pos="0.6 0.05 0.4"><geom type="sphere" size="0.009" rgba="{th}" class="hand"/></body>
    <body name="h_index" mocap="true" pos="0.6 -0.05 0.4"><geom type="sphere" size="0.008" rgba="{ix}" class="hand"/></body>
    <body name="h_thumb_seg" mocap="true" pos="0.6 0 0.4"><geom name="h_thumb_seg_g" type="capsule" size="0.009 0.02" rgba="{skin}" class="hand"/></body>
    <body name="h_index_seg" mocap="true" pos="0.6 0 0.4"><geom name="h_index_seg_g" type="capsule" size="0.008 0.025" rgba="{skin}" class="hand"/></body>
    """


def build_xml(spec: SceneSpec) -> str:
    ls, tint = spec.light_scale, spec.light_tint
    diff = " ".join(f"{0.7*ls*t:.3f}" for t in tint)
    amb = " ".join(f"{0.35*ls*t:.3f}" for t in tint)
    mat_cx, mat_cy = (MAT_X[0] + MAT_X[1]) / 2, (MAT_Y[0] + MAT_Y[1]) / 2
    mat_hx, mat_hy = (MAT_X[1] - MAT_X[0]) / 2 + 0.02, (MAT_Y[1] - MAT_Y[0]) / 2 + 0.02
    markers = "".join(
        f'<geom type="box" size="{MARKER_HALF} {MARKER_HALF} 0.0005" pos="{mx} {my} 0.0015" rgba="0.05 0.05 0.05 1" '
        f'contype="0" conaffinity="0"/>' for mx, my in MARKER_POS)
    objs = "".join(_object_xml(o, i) for i, o in enumerate(spec.objects + spec.distractors))
    fixtures = "".join(_fixture_xml(f) for f in spec.fixtures)
    n_act = ""
    for j, kp in [("shoulder_pan", 80), ("shoulder_lift", 120), ("elbow_flex", 100), ("wrist_flex", 50), ("wrist_roll", 15)]:
        n_act += f'<position name="a_{j}" joint="{j}" kp="{kp}" kv="{kp/12:.2f}" forcerange="-6 6"/>'
    n_act += '<position name="a_finger_l" joint="finger_l" kp="300" kv="8" forcerange="-8 8"/>'
    n_act += '<position name="a_finger_r" joint="finger_r" kp="300" kv="8" forcerange="-8 8"/>'
    hand = _hand_xml() if spec.include_hand else ""
    return f"""
<mujoco model="tidy_workspace">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast" cone="elliptic" impratio="5"/>
  <visual><global offwidth="640" offheight="480"/><quality shadowsize="1024"/>
    <headlight ambient="{amb}" diffuse="{diff}" specular="0 0 0"/></visual>
  <default>
    <geom solref="0.004 1" solimp="0.95 0.99 0.001" friction="1.0 0.02 0.002"/>
    <default class="arm"><geom contype="1" conaffinity="0" group="1" mass="0.05"/></default>
    <default class="finger"><geom contype="2" conaffinity="2" friction="1.5 0.05 0.002" mass="0.02"/></default>
    <default class="object"><geom contype="2" conaffinity="3" condim="4"/></default>
    <default class="fixture"><geom contype="2" conaffinity="3"/></default>
    <default class="hand"><geom contype="0" conaffinity="0" group="2"/></default>
  </default>
  <worldbody>
    <light pos="0.2 0 1.2" dir="0 0 -1" diffuse="{diff}" specular="0 0 0" castshadow="false"/>
    <geom name="table" type="plane" size="0.8 0.8 0.01" rgba="0.42 0.44 0.48 1" contype="2" conaffinity="3"/>
    <geom name="mat" type="box" size="{mat_hx:.3f} {mat_hy:.3f} 0.0005" pos="{mat_cx:.3f} {mat_cy:.3f} 0.0005" rgba="0.82 0.82 0.80 1" contype="0" conaffinity="0"/>
    {markers}
    <camera name="overhead" pos="0.17 0.0 0.62" xyaxes="0 -1 0 1 0 0" fovy="58"/>
    <camera name="phone" pos="0.62 0.05 0.42" xyaxes="0 1 0 -0.55 0 0.83" fovy="60"/>
    {_arm_xml()}
    {fixtures}
    {objs}
    {hand}
  </worldbody>
  <actuator>{n_act}</actuator>
</mujoco>"""
