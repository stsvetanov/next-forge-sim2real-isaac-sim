#!/usr/bin/env python3
r"""
Task board + Franka -- ST1 blue-button press, then ST2 slider movement,
using a CLOSED gripper holding a red pin/stylus, with collision/contact logging.

Updated from the original taskboard_franka_press.py so the same demo continues
from ST1 to ST2 instead of stopping after the button press.

What you should see in Isaac Sim:
  1) The Franka approaches the blue button with a closed gripper holding a pin.
  2) The pin presses the blue button.
  3) The arm retracts and moves to the ST2 slider.
  4) The pin touches the slider handle and drags it to a new position.
  5) All PhysX contact pairs are logged to a CSV file.

Run from Isaac Sim:
    ./python.sh /path/to/taskboard_franka_press.py

Useful environment variable:
    TASKBOARD_LOG_DIR=/data/isaac-sim/logs ./python.sh taskboard_franka_press.py

Important notes:
  - The pin is modeled as an attached tool under panda_hand. This is robust for
    skill/digital-twin demos. Learning the actual free-body grasp of the probe
    should be a separate skill.
  - The script first tries to find the real CAD slider by name. If it cannot find
    it, it creates a small visible proxy slider near the blue button so that ST2
    is still demonstrable. Tune SLIDER_PROXY_* offsets if your USD coordinate
    frame differs.
"""
import csv
import math
import os
from datetime import datetime

from isaacsim import SimulationApp

CONFIG = {"headless": False}
simulation_app = SimulationApp(CONFIG)

import numpy as np  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import RigidPrim  # noqa: E402
from isaacsim.core.utils.rotations import euler_angles_to_quat  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from isaacsim.robot.manipulators.examples.franka import Franka  # noqa: E402
from isaacsim.robot.manipulators.examples.franka.controllers.rmpflow_controller import (  # noqa: E402
    RMPFlowController,
)
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade  # noqa: E402

# Contact reporting imports vary slightly between Isaac Sim versions.
try:  # noqa: E402
    import omni.physx
    from pxr import PhysxSchema, PhysicsSchemaTools

    HAVE_CONTACT_REPORTS = True
except Exception as exc:  # noqa: E402
    omni = None
    PhysxSchema = None
    PhysicsSchemaTools = None
    HAVE_CONTACT_REPORTS = False
    CONTACT_IMPORT_ERROR = exc

# ============================ EDIT THIS =====================================
# You can override this without editing the file:
#   TASKBOARD_USD=/data/eurobin/Task_Board_physics.usd ./python.sh taskboard_franka_press.py
BOARD_USD = os.environ.get("TASKBOARD_USD", "/data/eurobin/Task_Board_physics.usd")
# ============================================================================

BOARD_POS = np.array([0.45, 0.0, 0.0795])
BOARD_PRIM = "/World/TaskBoard"
BOARD_GEOM = "/World/TaskBoard/Geom"

# ST1: blue button.
TARGET_BUTTON = "tn__BlueButton_kA"
BUTTON_TRAVEL = 0.0035   # button is only 3.2 mm tall -> keep travel small
PRESS_DEPTH = 0.003      # press ~3 mm (>2 mm success) without burying it
BUTTON_HOVER_H = 0.10
BUTTON_SUCCESS_SINK_M = 0.002

# ST2: slider. The script searches by these names/hints. If nothing is found,
# it creates a visible proxy slider.
SLIDER_EXACT_NAME_CANDIDATES = [
    # Try the environment variable first if you discover the exact prim name
    # from the terminal output. Example:
    #   TASKBOARD_SLIDER_NAME=Meter_Slider ./python.sh taskboard_franka_press.py
    os.environ.get("TASKBOARD_SLIDER_NAME", ""),
    "tn__Slider_u03",
    "tn__Slider",
    "MeterSlider",
    "Meter_Slider",
    "meter_slider",
    "Slider",
    "slider",
    "Slide",
    "slide",
    "SliderHandle",
    "slider_handle",
]
SLIDER_NAME_HINTS = [
    "slider",
    "slide",
    "meter",
    "m5stick",
    "screen",
    "fader",
    "potentiometer",
    "linear",
    "knob",
]
SLIDER_BAD_HINTS = [
    "button",
    "red",
    "blue",
    "cable",
    "probe",
    "post",
    "door",
    "base",
    "case",
    "board",
]

# If a real CAD slider is found, this vector is applied as an extra transform
# while the robot drags it. Try changing sign or axis if the slider moves the
# wrong way in your USD. In the screenshot the slider is horizontal on the board;
# most Onshape exports work with either X or Y. Start with Y, then try X below.
SLIDER_MOVE_VECTOR = np.array([0.0, 0.045, 0.0])
# Alternative to test manually:
# SLIDER_MOVE_VECTOR = np.array([0.045, 0.0, 0.0])

# If the real slider cannot be found, a proxy slider is created at:
#     proxy_start_top = blue_button_top + SLIDER_PROXY_START_OFFSET_FROM_BUTTON
# Then it moves by SLIDER_PROXY_MOVE_VECTOR.
# These offsets are deliberately easy to tune from the Isaac GUI.
SLIDER_PROXY_START_OFFSET_FROM_BUTTON = np.array([-0.060, 0.020, 0.012])
SLIDER_PROXY_MOVE_VECTOR = np.array([0.0, 0.050, 0.0])
SLIDER_PROXY_HANDLE_SIZE = np.array([0.018, 0.012, 0.018])
SLIDER_PROXY_RAIL_EXTRA = 0.025

# When dragging the slider, keep the pin tip this much above/into the top of
# the handle. A small positive value usually looks cleaner than penetrating it.
SLIDER_TIP_CLEARANCE = 0.002
SLIDER_HOVER_H = 0.085

# For a realistic ST2, the pin should push the slider from the SIDE of the knob,
# not press on top of it. The code computes the side automatically from the
# slider motion vector: if the slider moves +Y, the pin is placed on the -Y side.
SLIDER_SIDE_PUSH = True
SLIDER_SIDE_PUSH_PENETRATION = 0.0008  # small visual overlap so contact is clear
SLIDER_SIDE_Z_FRACTION = 0.55         # 0=bottom, 1=top of slider bbox

# Tool/pin geometry. In this Franka/USD setup, +Z of panda_hand points toward
# the fingertips/TCP, matching the original stylus script.
TCP_BELOW_HAND = 0.1034
PIN_LEN = 0.123
PIN_RAD = 0.003
TIP_BELOW_TCP = PIN_LEN - TCP_BELOW_HAND

# Close the Franka fingers around the pin. Each joint is half of the total
# opening. For a 6 mm diameter pin, 0.0035-0.005 m is reasonable.
GRIPPER_HALF_OPENING_FOR_PIN = PIN_RAD + 0.0015
GRIPPER_OPEN_HALF_OPENING = 0.035     # open enough before grasping the probe
GRIPPER_PROBE_HALF_OPENING = 0.0065   # closed around the probe body

# Rotate the held pin around the approach axis to test clearance. If the gripper
# hits CableWrapPostLeft or the M5Stick, try 90, -90, or 180 degrees.
TOOL_YAW_DEG = 0.0

# Collision/contact logging.
LOG_DIR = os.environ.get("TASKBOARD_LOG_DIR", os.getcwd())
CONTACT_CSV = os.path.join(
    LOG_DIR,
    "taskboard_contacts_press_then_slider_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv",
)
PRINT_EACH_UNIQUE_PAIR_ONCE = True
PRINT_EVERY_CONTACT = False

PLAN = [
    ("ST1", "Press Button", "ACTIVE: blue button"),
    ("ST2", "Match Slider to Screen", "ACTIVE: drag slider"),
    ("ST3", "Move Probe Plug", "later"),
    ("ST4", "Open Door / Probe Circuit", "later"),
    ("ST5", "Wrap Cable / Stow Probe", "later"),
    ("ST6", "Press Stop Button", "later: red button"),
]


def log(message):
    print(f"[task] {message}", flush=True)


def print_plan():
    print("\n" + "=" * 72, flush=True)
    print(" TASK BOARD MANIPULATION PLAN  (Robothon TBv2023)", flush=True)
    print("=" * 72, flush=True)
    for sid, name, status in PLAN:
        print(f"   {sid}  {name:<30s} [{status}]", flush=True)
    print("=" * 72 + "\n", flush=True)


def find_child(root, name):
    for prim in Usd.PrimRange(root):
        if prim.GetName() == name:
            return prim
    return None


def lower_name(prim):
    return prim.GetName().lower()


def prim_path(prim):
    if prim and prim.IsValid():
        return str(prim.GetPath())
    return ""


def bbox_range(stage, prim):
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    return bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()


def bbox_top_center(stage, prim):
    rng = bbox_range(stage, prim)
    mn, mx = rng.GetMin(), rng.GetMax()
    return np.array([(mn[0] + mx[0]) / 2.0, (mn[1] + mx[1]) / 2.0, mx[2]], dtype=float)


def bbox_center(stage, prim):
    rng = bbox_range(stage, prim)
    mn, mx = rng.GetMin(), rng.GetMax()
    return np.array([(mn[0] + mx[0]) / 2.0, (mn[1] + mx[1]) / 2.0, (mn[2] + mx[2]) / 2.0], dtype=float)


def normalized(v, fallback=None):
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return np.asarray(fallback if fallback is not None else [1.0, 0.0, 0.0], dtype=float)
    return v / n


def bbox_half_extent_along(stage, prim, unit):
    size = bbox_size(stage, prim)
    unit = np.abs(normalized(unit))
    return 0.5 * float(size[0] * unit[0] + size[1] * unit[1] + size[2] * unit[2])


def bbox_side_point_for_push(stage, prim, top_at_start, move_vector):
    """Return the pin-tip point on the side of a slider knob.

    The pin is put on the side opposite the desired travel direction.  This
    makes the vertical red pin visibly push the side wall of the slider instead
    of pressing down from above.
    """
    move_unit = normalized(move_vector, fallback=[0.0, 1.0, 0.0])
    side_unit = -move_unit
    rng = bbox_range(stage, prim)
    mn, mx = rng.GetMin(), rng.GetMax()
    center = np.array([(mn[0] + mx[0]) / 2.0, (mn[1] + mx[1]) / 2.0, (mn[2] + mx[2]) / 2.0], dtype=float)
    half = bbox_half_extent_along(stage, prim, move_unit)
    offset = half + PIN_RAD - SLIDER_SIDE_PUSH_PENETRATION
    z = float(mn[2] + SLIDER_SIDE_Z_FRACTION * (mx[2] - mn[2]))
    # Preserve the actuator's top_at() XY, because for a physics joint the bbox
    # may not update until the simulation has stepped.
    p = np.asarray(top_at_start, dtype=float).copy()
    p[0] = center[0]
    p[1] = center[1]
    p[2] = z
    return p + side_unit * offset


def bbox_size(stage, prim):
    rng = bbox_range(stage, prim)
    mn, mx = rng.GetMin(), rng.GetMax()
    return np.array([mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]], dtype=float)


def find_slider_candidate(stage, root_path):
    """Find the real ST2 slider/handle in the imported CAD.

    The Onshape export names can vary. This function first checks exact names,
    then prints and ranks fuzzy candidates. If it picks the wrong prim, rerun
    with TASKBOARD_SLIDER_NAME=<exact_prim_name>.
    """
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return None

    for name in SLIDER_EXACT_NAME_CANDIDATES:
        if not name:
            continue
        prim = find_child(root, name)
        if prim and prim.IsValid():
            log(f"ST2 exact slider name matched: {name}")
            return prim

    candidates = []
    debug_names = []
    for prim in Usd.PrimRange(root):
        lname = lower_name(prim)
        if any(h in lname for h in SLIDER_NAME_HINTS) and not any(b in lname for b in SLIDER_BAD_HINTS):
            try:
                size = bbox_size(stage, prim)
                center = bbox_top_center(stage, prim)
                # Prefer small/medium movable parts close to the board top.
                # Huge assemblies get a large penalty.
                diagonal = float(np.linalg.norm(size))
                volume = float(max(size[0], 1e-6) * max(size[1], 1e-6) * max(size[2], 1e-6))
                score = diagonal + 10.0 * volume
            except Exception:
                size = np.array([999.0, 999.0, 999.0])
                center = np.array([999.0, 999.0, 999.0])
                score = 999.0
            candidates.append((score, prim, size, center))
            debug_names.append(prim.GetName())

    if not candidates:
        log("ST2 no real slider-like CAD prim found; will create proxy slider")
        return None

    candidates.sort(key=lambda x: x[0])
    log("ST2 slider-like CAD candidates, best first:")
    for score, prim, size, center in candidates[:8]:
        log(
            f"   candidate name='{prim.GetName()}' path='{prim.GetPath()}' "
            f"size={np.round(size, 4)} top={np.round(center, 4)} score={score:.4f}"
        )
    return candidates[0][1]

def get_or_add_translate_op(prim, suffix="drive"):
    xformable = UsdGeom.Xformable(prim)
    wanted = f"xformOp:translate:{suffix}"
    for op in xformable.GetOrderedXformOps():
        if op.GetOpName() == wanted:
            return op
    try:
        return xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble, suffix)
    except Exception:
        # Fallback for older bindings; this may fail if there is already an
        # unsuffixed translate op, but works for many simple prims.
        return xformable.AddTranslateOp()


def apply_collision_to_meshes(root_prim, approximation="none"):
    count = 0
    for prim in Usd.PrimRange(root_prim):
        if prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Cube) or prim.IsA(UsdGeom.Cylinder) or prim.IsA(UsdGeom.Sphere):
            UsdPhysics.CollisionAPI.Apply(prim)
            if prim.IsA(UsdGeom.Mesh):
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set(approximation)
            count += 1
    return count


def add_static_collision(stage, root_path, skip_paths):
    skip_paths = tuple(skip_paths)
    count = 0
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path)):
        p = str(prim.GetPath())
        if prim.IsA(UsdGeom.Mesh) and not any(p.startswith(s) for s in skip_paths):
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set("none")
            count += 1
    return count


def enable_contact_report_on_prim(prim, threshold=0.0):
    if not HAVE_CONTACT_REPORTS or not prim or not prim.IsValid():
        return False
    try:
        api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
        api.CreateThresholdAttr().Set(float(threshold))
        return True
    except Exception:
        return False


def enable_contact_reports(stage, root_paths):
    if not HAVE_CONTACT_REPORTS:
        log(f"contact reports unavailable in this Isaac/Python environment: {CONTACT_IMPORT_ERROR}")
        return 0

    count = 0
    for root_path in root_paths:
        root = stage.GetPrimAtPath(root_path)
        if not root or not root.IsValid():
            continue
        if enable_contact_report_on_prim(root):
            count += 1
        for prim in Usd.PrimRange(root):
            if prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Cube) or prim.IsA(UsdGeom.Cylinder) or prim.IsA(UsdGeom.Sphere) or prim.HasAPI(UsdPhysics.CollisionAPI):
                if enable_contact_report_on_prim(prim):
                    count += 1
    return count


def make_button_spring(stage, button_prim, world_origin):
    UsdPhysics.RigidBodyAPI.Apply(button_prim)
    UsdPhysics.MassAPI.Apply(button_prim).CreateMassAttr().Set(0.02)
    # We filter out button<->board collision below, so the plate no longer holds
    # the button up; without this it would sag onto its lower limit and look
    # pressed/black before the pin ever touches it. Disable gravity so the spring
    # holds it exactly at rest (it's a spring-return button -- gravity is moot).
    try:
        from pxr import PhysxSchema
        PhysxSchema.PhysxRigidBodyAPI.Apply(button_prim).CreateDisableGravityAttr().Set(True)
    except Exception as exc:
        log(f"(could not disable gravity on button: {exc})")
    for sub in Usd.PrimRange(button_prim):
        if sub.IsA(UsdGeom.Mesh):
            UsdPhysics.CollisionAPI.Apply(sub)
            UsdPhysics.MeshCollisionAPI.Apply(sub).CreateApproximationAttr().Set("convexHull")

    # CRITICAL: the button sits ON the static plate, so pressing it down drives
    # it into the board collider -> it jams (won't spring back) and dips below
    # the black surface (looks black). Filter out button<->board collision so it
    # slides freely on its joint, pushed by the pin and returned by the spring.
    fp = UsdPhysics.FilteredPairsAPI.Apply(button_prim)
    fp.CreateFilteredPairsRel().AddTarget(Sdf.Path(BOARD_GEOM))

    joint = UsdPhysics.PrismaticJoint.Define(stage, f"{button_prim.GetPath()}/btn_slide")
    joint.CreateBody1Rel().SetTargets([button_prim.GetPath()])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*world_origin))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1, 0, 0, 0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))
    joint.CreateAxisAttr().Set("Z")
    joint.CreateLowerLimitAttr().Set(-BUTTON_TRAVEL)
    joint.CreateUpperLimitAttr().Set(0.0)

    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
    drive.CreateTypeAttr().Set("force")
    drive.CreateStiffnessAttr().Set(250.0)   # stiff enough to spring back crisply
    drive.CreateDampingAttr().Set(15.0)
    drive.CreateTargetPositionAttr().Set(0.0)


def make_slider_joint(stage, slider_prim, axis="Y", travel=0.012):
    """Give the REAL slider part a bounded, damped prismatic joint.

    Unlike the button there is no return spring -- a slider stays where it is
    set. Bounded limits keep it on the board; heavy damping removes any inertia;
    gravity is disabled and board collision is filtered so it cannot jam.
    Returns a world-frame move vector for the actuator/pin to follow.
    """
    UsdPhysics.RigidBodyAPI.Apply(slider_prim)
    UsdPhysics.MassAPI.Apply(slider_prim).CreateMassAttr().Set(0.03)
    try:
        from pxr import PhysxSchema
        PhysxSchema.PhysxRigidBodyAPI.Apply(slider_prim).CreateDisableGravityAttr().Set(True)
    except Exception as exc:
        log(f"(could not disable gravity on slider: {exc})")
    for sub in Usd.PrimRange(slider_prim):
        if sub.IsA(UsdGeom.Mesh):
            UsdPhysics.CollisionAPI.Apply(sub)
            UsdPhysics.MeshCollisionAPI.Apply(sub).CreateApproximationAttr().Set("convexHull")
    fp = UsdPhysics.FilteredPairsAPI.Apply(slider_prim)
    fp.CreateFilteredPairsRel().AddTarget(Sdf.Path(BOARD_GEOM))

    origin = UsdGeom.XformCache().GetLocalToWorldTransform(slider_prim).ExtractTranslation()
    joint = UsdPhysics.PrismaticJoint.Define(stage, f"{slider_prim.GetPath()}/slide_joint")
    joint.CreateBody1Rel().SetTargets([slider_prim.GetPath()])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*origin))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1, 0, 0, 0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))
    joint.CreateAxisAttr().Set(axis)
    joint.CreateLowerLimitAttr().Set(0.0)
    joint.CreateUpperLimitAttr().Set(float(travel))
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
    drive.CreateTypeAttr().Set("force")
    drive.CreateStiffnessAttr().Set(600.0)   # tracks the commanded position
    drive.CreateDampingAttr().Set(60.0)      # heavily damped -> no overshoot
    drive.CreateTargetPositionAttr().Set(0.0)

    unit = {"X": np.array([1.0, 0, 0]), "Y": np.array([0, 1.0, 0]),
            "Z": np.array([0, 0, 1.0])}[axis]
    return unit * float(travel)


# ---- ST3 (probe) + ST4 (door) ----------------------------------------------
PROBE_PRIM_NAME = "Probe"
DOOR_PRIM_NAME = "Door"
DOOR_HANDLE_NAME = "tn__DoorHandleNut_nDB"

# ST3 probe handling.  The probe is not lifted vertically out of the clips.
# It is first grasped with an aligned gripper, then pulled along its own axis to
# release it from the holder, and only then lifted.  If it pulls the wrong way in
# your USD, change the sign of PROBE_PULL_VECTOR.
PROBE_GRASP_YAW_DEG = 90.0
PROBE_PULL_VECTOR = np.array([0.0, 0.075, 0.0])
PROBE_LIFT_VECTOR = np.array([0.0, 0.0, 0.105])

# The door hinges along the board X axis at its -Y edge and swings UP. If it
# opens the wrong way in your scene, negate DOOR_OPEN_DEG.
DOOR_OPEN_DEG = 70.0
DOOR_TIP_LIFT = 0.055   # how far the probe tip rises as the door swings open
DOOR_CONTACT_DELAY = 0.18  # first part of ST4_OPEN keeps the door closed while the probe seats


def make_door_hinge(stage, door_prim, hinge_world, axis="X", open_deg=70.0):
    """Bounded, damped revolute hinge on the real Door (push-to-open, no explode).

    Uses the corrected frame convention: localRot0 = identity (joint axis in
    world), localRot1 = inverse(door world rotation) so the door's own 180-deg
    CAD rotation does not fight the joint (that mismatch is what exploded before).
    The door is DRIVEN open by a damped angular drive, so it is reliable.
    """
    UsdPhysics.RigidBodyAPI.Apply(door_prim)
    UsdPhysics.MassAPI.Apply(door_prim).CreateMassAttr().Set(0.05)
    try:
        from pxr import PhysxSchema
        PhysxSchema.PhysxRigidBodyAPI.Apply(door_prim).CreateDisableGravityAttr().Set(True)
    except Exception as exc:
        log(f"(could not disable gravity on door: {exc})")
    for sub in Usd.PrimRange(door_prim):
        if sub.IsA(UsdGeom.Mesh):
            UsdPhysics.CollisionAPI.Apply(sub)
            UsdPhysics.MeshCollisionAPI.Apply(sub).CreateApproximationAttr().Set("convexHull")
    fp = UsdPhysics.FilteredPairsAPI.Apply(door_prim)
    fp.CreateFilteredPairsRel().AddTarget(Sdf.Path(BOARD_GEOM))

    M = UsdGeom.XformCache().GetLocalToWorldTransform(door_prim)
    hinge_local = M.GetInverse().Transform(Gf.Vec3d(*hinge_world))
    door_rot_inv = Gf.Quatf(M.ExtractRotationQuat().GetInverse())

    joint = UsdPhysics.RevoluteJoint.Define(stage, f"{door_prim.GetPath()}/door_hinge")
    joint.CreateBody1Rel().SetTargets([door_prim.GetPath()])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*hinge_world))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1, 0, 0, 0))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(*hinge_local))
    joint.CreateLocalRot1Attr().Set(door_rot_inv)
    joint.CreateAxisAttr().Set(axis)
    lo, hi = (0.0, float(open_deg)) if open_deg >= 0 else (float(open_deg), 0.0)
    joint.CreateLowerLimitAttr().Set(lo)
    joint.CreateUpperLimitAttr().Set(hi)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    drive.CreateTypeAttr().Set("force")
    drive.CreateStiffnessAttr().Set(800.0)
    drive.CreateDampingAttr().Set(80.0)
    drive.CreateTargetPositionAttr().Set(0.0)
    return joint.GetPrim()


class DoorActuator:
    """Drives the door hinge open via its angular drive target (degrees)."""

    def __init__(self, joint_prim, open_deg, label="Door"):
        self.drive = UsdPhysics.DriveAPI.Get(joint_prim, "angular")
        self.open_deg = float(open_deg)
        self.label = label
        self.progress = 0.0
        self.set_progress(0.0)

    def set_progress(self, p):
        self.progress = float(np.clip(p, 0.0, 1.0))
        if self.drive is not None:
            self.drive.GetTargetPositionAttr().Set(self.open_deg * self.progress)


class ProbeTool:
    """Scripted rigid 'grasp': after grab(), the probe pose tracks the gripper.

    The probe carries no rigid body -- it is positioned purely by overwriting a
    single transform op each frame. That is reliable (no friction grasp to tune)
    and is enough here because the door is opened by its own driven joint; the
    probe just needs to be carried to and held against the door.
    """

    def __init__(self, stage, probe_prim, hand_prim):
        self.stage = stage
        self.probe = probe_prim
        self.hand = hand_prim
        self.parent = probe_prim.GetParent()
        self.grabbed = False
        self.C = None
        # Convert the probe to a single controllable transform op up-front (before
        # play) so it stays exactly in its holder until grabbed; no mid-sim op
        # reshuffle.
        xc = UsdGeom.XformCache()
        Wp0 = xc.GetLocalToWorldTransform(probe_prim)
        Mp0 = xc.GetLocalToWorldTransform(self.parent)
        L0 = Wp0 * Mp0.GetInverse()
        xf = UsdGeom.Xformable(probe_prim)
        xf.ClearXformOpOrder()
        self.op = xf.AddTransformOp()
        self.op.Set(L0)
        self._Mparent = Mp0  # holder/board is static -> constant

    def grab(self):
        xc = UsdGeom.XformCache()
        Wp = xc.GetLocalToWorldTransform(self.probe)
        Wh = xc.GetLocalToWorldTransform(self.hand)
        self.C = Wp * Wh.GetInverse()   # constant probe-in-hand relation
        self.grabbed = True
        self.update()

    def update(self):
        if not self.grabbed:
            return
        Wh = UsdGeom.XformCache().GetLocalToWorldTransform(self.hand)
        Wp = self.C * Wh
        self.op.Set(Wp * self._Mparent.GetInverse())


def make_material(stage, path, color, roughness=0.35):
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/PBR")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def attach_held_pin(stage, hand_path):
    """Create a visible collision pin held in the closed Franka gripper."""
    cyl = UsdGeom.Cylinder.Define(stage, hand_path + "/HeldPin")
    cyl.CreateRadiusAttr(PIN_RAD)
    cyl.CreateHeightAttr(PIN_LEN)
    cyl.CreateAxisAttr("Z")
    UsdGeom.Xformable(cyl.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0, 0, PIN_LEN / 2.0))
    UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())

    # Rounded tip gives more stable visual/contact behavior than a sharp flat end.
    tip = UsdGeom.Sphere.Define(stage, hand_path + "/HeldPinTip")
    tip.CreateRadiusAttr(PIN_RAD * 1.05)
    UsdGeom.Xformable(tip.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0, 0, PIN_LEN))
    UsdPhysics.CollisionAPI.Apply(tip.GetPrim())

    mat = make_material(stage, hand_path + "/HeldPinMat", (0.8, 0.05, 0.05))
    UsdShade.MaterialBindingAPI(cyl.GetPrim()).Bind(mat)
    UsdShade.MaterialBindingAPI(tip.GetPrim()).Bind(mat)
    return [str(cyl.GetPath()), str(tip.GetPath())]


# The slider is a real, SEPARABLE part: the knob is named "Slider" (5x8x8 mm),
# riding on the track "tn__SliderUnit_mA" (24x65 mm, long axis Y). So the robot
# moves just the knob along the track -- the faithful behaviour.
# NOTE: "tn__PotentiometerUnit_VI" is NOT this -- that's the door's angle-sensor
# potentiometer next to the hinge, which is why targeting it hit the hinge.
FORCE_PROXY_SLIDER = False
SLIDER_PRIM_NAME = "Slider"   # the movable knob riding on tn__SliderUnit_mA
SLIDER_AXIS = "Y"             # track long axis (slot runs along board Y)
SLIDER_TRAVEL = 0.020         # bounded knob travel (m) along the track

# Proxy fallback only (used if the real slider prim can't be found):

# Proxy slider geometry/placement (world frame). Travel is along board +X and is
# bounded by the joint limits below, so the knob always stays on the board.
SLIDER_PROXY_TRAVEL = 0.040
SLIDER_PROXY_HANDLE = np.array([0.016, 0.016, 0.012], dtype=float)


def create_proxy_slider(stage, button_top_world):
    """A bounded, damped physics slider knob (no real knob exists in the CAD)."""
    root_path = "/World/ST2ProxySlider"
    UsdGeom.Xform.Define(stage, root_path)

    travel = float(SLIDER_PROXY_TRAVEL)
    handle_size = SLIDER_PROXY_HANDLE.copy()
    move_vec = np.array([travel, 0.0, 0.0], dtype=float)  # slide along board +X

    # Place on the board top surface, centre-left, clear of the two buttons.
    knob_top_z = float(button_top_world[2]) + 0.004
    start_top = np.array([BOARD_POS[0] - 0.035, 0.015, knob_top_z], dtype=float)
    end_top = start_top + move_vec

    # Rail: static visual + collision, running along the travel.
    rail_size = np.array([travel + 0.025, 0.012, 0.004], dtype=float)
    rail_center = (start_top + end_top) / 2.0
    rail_center[2] = start_top[2] - handle_size[2] - 0.001
    rail = UsdGeom.Cube.Define(stage, root_path + "/Rail")
    rail.CreateSizeAttr(1.0)
    UsdGeom.Xformable(rail.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*rail_center.tolist()))
    UsdGeom.Xformable(rail.GetPrim()).AddScaleOp().Set(Gf.Vec3f(*rail_size.tolist()))
    UsdShade.MaterialBindingAPI(rail.GetPrim()).Bind(
        make_material(stage, root_path + "/RailMat", (0.08, 0.08, 0.08), roughness=0.6))
    UsdPhysics.CollisionAPI.Apply(rail.GetPrim())

    # Knob: dynamic rigid body.
    handle_center = start_top.copy()
    handle_center[2] = start_top[2] - handle_size[2] / 2.0
    handle = UsdGeom.Cube.Define(stage, root_path + "/Handle")
    handle.CreateSizeAttr(1.0)
    UsdGeom.Xformable(handle.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*handle_center.tolist()))
    UsdGeom.Xformable(handle.GetPrim()).AddScaleOp().Set(Gf.Vec3f(*handle_size.tolist()))
    UsdShade.MaterialBindingAPI(handle.GetPrim()).Bind(
        make_material(stage, root_path + "/HandleMat", (0.85, 0.85, 0.85), roughness=0.4))
    UsdPhysics.CollisionAPI.Apply(handle.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(handle.GetPrim())
    UsdPhysics.MassAPI.Apply(handle.GetPrim()).CreateMassAttr().Set(0.02)
    try:
        from pxr import PhysxSchema
        PhysxSchema.PhysxRigidBodyAPI.Apply(handle.GetPrim()).CreateDisableGravityAttr().Set(True)
    except Exception:
        pass
    # Knob must not fight the board/rail; the joint governs it entirely.
    fp = UsdPhysics.FilteredPairsAPI.Apply(handle.GetPrim())
    fp.CreateFilteredPairsRel().SetTargets([Sdf.Path(BOARD_GEOM), rail.GetPath()])

    # Prismatic joint: world -> knob, bounded by limits + damped (no inertia).
    handle_origin = UsdGeom.XformCache().GetLocalToWorldTransform(
        handle.GetPrim()).ExtractTranslation()
    joint = UsdPhysics.PrismaticJoint.Define(stage, root_path + "/Handle/slide_joint")
    joint.CreateBody1Rel().SetTargets([handle.GetPath()])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*handle_origin))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1, 0, 0, 0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))
    joint.CreateAxisAttr().Set("X")
    joint.CreateLowerLimitAttr().Set(0.0)
    joint.CreateUpperLimitAttr().Set(travel)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
    drive.CreateTypeAttr().Set("force")
    drive.CreateStiffnessAttr().Set(400.0)   # tracks the target position
    drive.CreateDampingAttr().Set(40.0)      # heavily damped -> no overshoot
    drive.CreateTargetPositionAttr().Set(0.0)

    log(f"ST2 proxy slider (bounded physics joint): start_top={np.round(start_top, 3)} "
        f"travel={travel*1000:.0f}mm axis=X")
    return handle.GetPrim(), start_top, move_vec, [root_path]


class SliderActuator:
    """Drives the slider via its bounded, damped prismatic joint (no inertia)."""

    def __init__(self, prim, start_top, move_vector, label):
        self.prim = prim
        self.start_top = np.asarray(start_top, dtype=float)
        self.move_vector = np.asarray(move_vector, dtype=float)
        self.label = label
        self.travel = float(np.linalg.norm(self.move_vector))
        self.progress = 0.0
        self.drive = None
        self.drive_op = None
        # Prefer the physics prismatic joint created by create_proxy_slider.
        stage = prim.GetStage()
        jprim = stage.GetPrimAtPath(prim.GetPath().AppendChild("slide_joint"))
        if jprim and jprim.IsValid():
            self.drive = UsdPhysics.DriveAPI.Get(jprim, "linear")
        if self.drive is None:
            # Fallback (only if a CAD slider is forced): scripted xform.
            self.drive_op = get_or_add_translate_op(prim, suffix="st2_drive")
        self.set_progress(0.0)

    @property
    def end_top(self):
        return self.start_top + self.move_vector

    def set_progress(self, progress):
        self.progress = float(np.clip(progress, 0.0, 1.0))
        if self.drive is not None:
            self.drive.GetTargetPositionAttr().Set(self.travel * self.progress)
        elif self.drive_op is not None:
            offset = self.move_vector * self.progress
            self.drive_op.Set(Gf.Vec3d(*offset.tolist()))

    def top_at(self, progress):
        return self.start_top + self.move_vector * float(np.clip(progress, 0.0, 1.0))


def path_from_actor(actor_id):
    if not HAVE_CONTACT_REPORTS:
        return str(actor_id)
    try:
        return str(PhysicsSchemaTools.intToSdfPath(actor_id))
    except Exception:
        return str(actor_id)


def v3_to_tuple(v):
    if v is None:
        return ""
    try:
        return f"({float(v[0]):.5f},{float(v[1]):.5f},{float(v[2]):.5f})"
    except Exception:
        return str(v)


def install_contact_logger(state):
    """Subscribe to PhysX contact reports and stream contacts to CSV."""
    if not HAVE_CONTACT_REPORTS:
        return None, None

    os.makedirs(LOG_DIR, exist_ok=True)
    contact_file = open(CONTACT_CSV, "w", newline="")
    writer = csv.writer(contact_file)
    writer.writerow([
        "step",
        "phase",
        "actor0",
        "actor1",
        "position",
        "normal",
        "impulse",
        "separation",
    ])
    seen_pairs = set()

    def on_contact_report_event(contact_headers, contact_data):
        for header in contact_headers:
            actor0 = path_from_actor(header.actor0)
            actor1 = path_from_actor(header.actor1)
            pair = tuple(sorted((actor0, actor1)))
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                if PRINT_EACH_UNIQUE_PAIR_ONCE:
                    log(
                        f"COLLISION pair first seen at step={state['step']} "
                        f"phase={state['phase']}: {actor0}  <->  {actor1}"
                    )

            num_contacts = int(getattr(header, "num_contact_data", 0))
            offset = int(getattr(header, "contact_data_offset", 0))
            if num_contacts <= 0:
                writer.writerow([state["step"], state["phase"], actor0, actor1, "", "", "", ""])
                continue

            for i in range(num_contacts):
                contact = contact_data[offset + i]
                pos = v3_to_tuple(getattr(contact, "position", None))
                normal = v3_to_tuple(getattr(contact, "normal", None))
                impulse = getattr(contact, "impulse", "")
                separation = getattr(contact, "separation", "")
                writer.writerow([
                    state["step"],
                    state["phase"],
                    actor0,
                    actor1,
                    pos,
                    normal,
                    impulse,
                    separation,
                ])
                if PRINT_EVERY_CONTACT:
                    log(
                        f"CONTACT step={state['step']} phase={state['phase']} "
                        f"{actor0} <-> {actor1} pos={pos} normal={normal} "
                        f"impulse={impulse} separation={separation}"
                    )
        contact_file.flush()

    sub = omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(
        on_contact_report_event
    )
    return sub, contact_file


def set_gripper_half_opening(franka, half_opening):
    try:
        franka.gripper.set_joint_positions(
            np.array([half_opening, half_opening], dtype=np.float32)
        )
    except Exception as exc:
        log(f"could not set gripper opening to {half_opening:.4f} m: {exc}")


def command_gripper_to_hold_pin(franka):
    """Close the Franka gripper to a small opening suitable for the attached pin."""
    set_gripper_half_opening(franka, GRIPPER_HALF_OPENING_FOR_PIN)
    log(
        "gripper set to hold pin: "
        f"finger joints {GRIPPER_HALF_OPENING_FOR_PIN * 1000:.1f} mm each"
    )


def gripper_opening_for_phase(phase):
    if phase in ("ST3_TRANSIT", "ST3_DESCEND"):
        return GRIPPER_OPEN_HALF_OPENING
    if phase in ("ST3_GRASP", "ST3_PULL_RELEASE", "ST3_LIFT",
                 "ST4_TRANSIT", "ST4_DESCEND", "ST4_OPEN", "ST4_RETRACT", "DONE"):
        return GRIPPER_PROBE_HALF_OPENING
    return GRIPPER_HALF_OPENING_FOR_PIN


def tcp_for_tip_position(tip_world):
    """TCP target so the held pin tip reaches tip_world."""
    tip_world = np.asarray(tip_world, dtype=float)
    return np.array([tip_world[0], tip_world[1], tip_world[2] + TIP_BELOW_TCP], dtype=float)


def lerp(a, b, t):
    t = float(np.clip(t, 0.0, 1.0))
    return np.asarray(a, dtype=float) * (1.0 - t) + np.asarray(b, dtype=float) * t


def resolve_board_usd():
    """Use the configured USD path, or the uploaded sandbox copy if present."""
    if os.path.exists(BOARD_USD):
        return BOARD_USD
    uploaded_copy = "/mnt/data/Task_Board_physics.usd"
    if os.path.exists(uploaded_copy):
        log(f"BOARD_USD not found at {BOARD_USD}; using uploaded copy {uploaded_copy}")
        return uploaded_copy
    return BOARD_USD


def main():
    print_plan()

    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    stage = world.stage

    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr(1000.0)
    UsdLux.DistantLight.Define(stage, "/World/KeyLight").CreateIntensityAttr(300.0)

    wrapper = UsdGeom.Xform.Define(stage, BOARD_PRIM)
    UsdGeom.Xformable(wrapper.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*BOARD_POS.tolist()))
    add_reference_to_stage(usd_path=resolve_board_usd(), prim_path=BOARD_GEOM)

    button = find_child(stage.GetPrimAtPath(BOARD_GEOM), TARGET_BUTTON)
    if button is None:
        raise RuntimeError(f"Could not find target button prim named {TARGET_BUTTON!r} under {BOARD_GEOM}")
    button_path = str(button.GetPath())
    button_top = bbox_top_center(stage, button)
    log(f"ST1 blue button '{TARGET_BUTTON}' localized at top={np.round(button_top, 4)}")

    # Prefer the REAL slider part (tn__PotentiometerUnit_VI) by exact name.
    slider_prim = None
    if not FORCE_PROXY_SLIDER:
        named = find_child(stage.GetPrimAtPath(BOARD_GEOM), SLIDER_PRIM_NAME)
        slider_prim = named if (named and named.IsValid()) else find_slider_candidate(stage, BOARD_GEOM)
    slider_extra_contact_roots = []
    slider_is_real = False
    if slider_prim and slider_prim.IsValid():
        slider_path = str(slider_prim.GetPath())
        slider_top = bbox_top_center(stage, slider_prim)
        slider_label = slider_prim.GetName()
        slider_is_real = True
        log(f"ST2 real slider '{slider_label}' at {slider_path}, top={np.round(slider_top, 4)}, "
            f"axis={SLIDER_AXIS}, travel={SLIDER_TRAVEL*1000:.0f}mm")
    else:
        slider_prim, slider_top, slider_move_vector, slider_extra_contact_roots = create_proxy_slider(
            stage, button_top
        )
        slider_path = str(slider_prim.GetPath())
        slider_label = "ST2ProxySlider"

    # ST3/ST4 parts. Keep all moving parts out of the fixed-board collider pass.
    board_root = stage.GetPrimAtPath(BOARD_GEOM)
    door_prim = find_child(board_root, DOOR_PRIM_NAME)
    probe_prim = find_child(board_root, PROBE_PRIM_NAME)
    handle_prim = find_child(board_root, DOOR_HANDLE_NAME)
    door_path = str(door_prim.GetPath()) if (door_prim and door_prim.IsValid()) else ""
    probe_path = str(probe_prim.GetPath()) if (probe_prim and probe_prim.IsValid()) else ""

    skip_paths = [button_path, slider_path]
    if door_path:
        skip_paths.append(door_path)
    if probe_path:
        skip_paths.append(probe_path)
    n_static = add_static_collision(stage, BOARD_GEOM, skip_paths)
    log(f"board prepared: {n_static} static collision meshes")

    make_button_spring(
        stage,
        button,
        UsdGeom.XformCache().GetLocalToWorldTransform(button).ExtractTranslation(),
    )

    # The real slider gets a bounded, damped prismatic joint on the part itself.
    # (The proxy already built its own joint inside create_proxy_slider.)
    if slider_is_real:
        slider_move_vector = make_slider_joint(
            stage, slider_prim, axis=SLIDER_AXIS, travel=SLIDER_TRAVEL)
    slider = SliderActuator(slider_prim, slider_top, slider_move_vector, slider_label)

    # ST4 door: bounded, damped hinge on the real Door (push-to-open).
    door_actuator = None
    if door_prim and door_prim.IsValid():
        dr = bbox_range(stage, door_prim)
        dmn, dmx = dr.GetMin(), dr.GetMax()
        hinge_world = np.array([(dmn[0] + dmx[0]) / 2.0, dmn[1], (dmn[2] + dmx[2]) / 2.0])
        jp = make_door_hinge(stage, door_prim, hinge_world, axis="X", open_deg=DOOR_OPEN_DEG)
        door_actuator = DoorActuator(jp, DOOR_OPEN_DEG, label="Door")
        log(f"ST4 door hinge at world {np.round(hinge_world, 4)}, open={DOOR_OPEN_DEG:.0f}deg")
    else:
        log("ST4 door: 'Door' prim not found; ST4 will hold instead")

    franka = world.scene.add(
        Franka(
            prim_path="/World/Franka",
            name="franka",
            position=np.array([0.0, 0.0, 0.0]),
        )
    )
    pin_paths = attach_held_pin(stage, "/World/Franka/panda_hand")
    log("attached a thin red pin/stylus to panda_hand; the gripper will close around it")

    # ST3 probe: prepare a scripted-grasp tool on the real Probe part (set up its
    # single controllable transform op BEFORE play so it stays in its holder).
    probe_tool = None
    if probe_prim and probe_prim.IsValid():
        probe_tool = ProbeTool(
            stage, probe_prim, stage.GetPrimAtPath("/World/Franka/panda_hand"))
        log("ST3 probe tool ready (scripted rigid grasp)")
    else:
        log("ST3 probe: 'Probe' prim not found; ST3 will hold instead")

    cr_count = enable_contact_reports(
        stage,
        [
            BOARD_GEOM,
            slider_path,
            *slider_extra_contact_roots,
            "/World/Franka",
            "/World/Franka/panda_hand",
            *pin_paths,
            *( [door_path] if door_path else [] ),
            *( [probe_path] if probe_path else [] ),
        ],
    )
    log(f"contact reports enabled on {cr_count} prims; CSV: {CONTACT_CSV}")

    world.reset()
    controller = RMPFlowController(name="rmpflow", robot_articulation=franka)
    controller.reset()
    command_gripper_to_hold_pin(franka)

    # Let the gripper pose settle visually.
    for _ in range(30):
        world.step(render=not CONFIG["headless"])

    btn_view = RigidPrim(prim_paths_expr=button_path, name="btn_view")
    try:
        hand_view = RigidPrim(prim_paths_expr="/World/Franka/panda_hand", name="hand_view")
    except Exception:
        hand_view = None

    # ST1 button targets.
    button_hover_tip = button_top + np.array([0.0, 0.0, BUTTON_HOVER_H])
    button_press_tip = button_top + np.array([0.0, 0.0, -PRESS_DEPTH])
    button_hover_tcp = tcp_for_tip_position(button_hover_tip)
    button_press_tcp = tcp_for_tip_position(button_press_tip)

    # ST2 slider targets.  Push the side wall of the slider knob rather than
    # pressing the top.  This also makes the motion read as "pushing", not
    # "pulling from above".
    if SLIDER_SIDE_PUSH:
        slider_start_tip = bbox_side_point_for_push(
            stage, slider.prim, slider.top_at(0.0), slider.move_vector)
        slider_end_tip = slider_start_tip + slider.move_vector
        log(f"ST2 side-push point: start={np.round(slider_start_tip, 4)} "
            f"end={np.round(slider_end_tip, 4)}")
    else:
        slider_start_tip = slider.top_at(0.0) + np.array([0.0, 0.0, SLIDER_TIP_CLEARANCE])
        slider_end_tip = slider.top_at(1.0) + np.array([0.0, 0.0, SLIDER_TIP_CLEARANCE])
    slider_start_hover_tcp = tcp_for_tip_position(slider_start_tip + np.array([0.0, 0.0, SLIDER_HOVER_H]))
    slider_start_touch_tcp = tcp_for_tip_position(slider_start_tip)
    slider_end_touch_tcp = tcp_for_tip_position(slider_end_tip)
    slider_end_hover_tcp = tcp_for_tip_position(slider_end_tip + np.array([0.0, 0.0, SLIDER_HOVER_H]))

    down_q = euler_angles_to_quat(np.array([0.0, math.pi, math.radians(TOOL_YAW_DEG)]))
    probe_q = euler_angles_to_quat(np.array([0.0, math.pi, math.radians(PROBE_GRASP_YAW_DEG)]))

    def orientation_for_phase(phase):
        if phase.startswith("ST3") or phase.startswith("ST4"):
            return probe_q
        return down_q

    # ST3 probe targets: grasp the probe body, pull it along its axis to release
    # from the clips, then lift. The gripper yaw is changed to PROBE_GRASP_YAW_DEG
    # before descending so the fingers are aligned with the probe.
    if probe_tool is not None:
        pr = bbox_range(stage, probe_prim)
        pmn, pmx = pr.GetMin(), pr.GetMax()
        cx = (pmn[0] + pmx[0]) / 2.0
        probe_tip_w = np.array([cx, pmn[1], (pmn[2] + pmx[2]) / 2.0])
        probe_grasp_w = np.array([cx, pmn[1] + 0.030, pmx[2] + 0.004])
        probe_grasp_hover = probe_grasp_w + np.array([0.0, 0.0, 0.10])
        probe_pull_w = probe_grasp_w + PROBE_PULL_VECTOR
        probe_grasp_lift = probe_pull_w + PROBE_LIFT_VECTOR
        tip_offset = probe_tip_w - probe_grasp_w   # tip vs TCP, fixed orientation while carried
        log(f"ST3 probe grasp~{np.round(probe_grasp_w, 3)} "
            f"pull_to~{np.round(probe_pull_w, 3)} tip~{np.round(probe_tip_w, 3)} "
            f"yaw={PROBE_GRASP_YAW_DEG:.0f}deg")
    else:
        probe_grasp_w = probe_grasp_hover = probe_pull_w = probe_grasp_lift = None
        tip_offset = np.zeros(3)

    # ST4 door push target: place the probe TIP just above the door handle/edge.
    if door_actuator is not None and handle_prim and handle_prim.IsValid():
        hr = bbox_range(stage, handle_prim)
        hmn, hmx = hr.GetMin(), hr.GetMax()
        door_push_tip = np.array([(hmn[0] + hmx[0]) / 2.0, (hmn[1] + hmx[1]) / 2.0, hmx[2] + 0.004])
    elif door_actuator is not None:
        dr2 = bbox_range(stage, door_prim)
        dmn2, dmx2 = dr2.GetMin(), dr2.GetMax()
        door_push_tip = np.array([(dmn2[0] + dmx2[0]) / 2.0, dmx2[1] - 0.008, dmx2[2] + 0.004])
    else:
        door_push_tip = None

    def tcp_for_probe_tip(tip_world):
        """TCP target so the carried probe TIP reaches tip_world (orientation fixed)."""
        return np.asarray(tip_world, float) - tip_offset

    for _ in range(40):
        world.step(render=not CONFIG["headless"])

    try:
        button_rest_z = float(btn_view.get_world_poses()[0][0][2])
        measure_button = True
    except Exception as exc:
        button_rest_z = 0.0
        measure_button = False
        log(f"button readout unavailable: {exc}")

    def diag_button(tag):
        try:
            hand_z = float(hand_view.get_world_poses()[0][0][2]) if hand_view else float("nan")
            button_z = float(btn_view.get_world_poses()[0][0][2])
            log(
                f"     [diag {tag}] held_pin_tip_est={hand_z - PIN_LEN:.3f}  "
                f"button_top={button_top[2]:.3f}  button_z={button_z:.3f}  "
                f"sink={(button_rest_z - button_z) * 1000:.1f}mm"
            )
        except Exception as exc:
            log(f"     [diag {tag}] {exc}")

    state = {"step": 0, "phase": "INIT"}
    contact_sub, contact_file = install_contact_logger(state)

    # Timeline in simulation steps.
    D_ST1_APPROACH = 180
    D_ST1_PRESS = 300
    D_ST1_RELEASE = 160
    D_ST2_TRANSIT = 180
    D_ST2_DESCEND = 100
    D_ST2_DRAG = 360
    D_ST2_RELEASE = 160
    D_ST3_TRANSIT = 220   # slider -> above probe, while rotating/opening gripper
    D_ST3_DESCEND = 130   # lower onto probe body
    D_ST3_GRASP = 90      # close fingers before scripted attach
    D_ST3_PULL = 180      # pull along probe axis to release from the holder
    D_ST3_LIFT = 140      # lift the released probe clear of the board
    D_ST4_TRANSIT = 240   # carry probe to the door
    D_ST4_DESCEND = 130   # lower tip onto door edge
    D_ST4_OPEN = 340      # keep contact, then pull/lift and open door
    D_ST4_RETRACT = 150

    t0 = 0
    t1 = t0 + D_ST1_APPROACH
    t2 = t1 + D_ST1_PRESS
    t3 = t2 + D_ST1_RELEASE
    t4 = t3 + D_ST2_TRANSIT
    t5 = t4 + D_ST2_DESCEND
    t6 = t5 + D_ST2_DRAG
    t7 = t6 + D_ST2_RELEASE
    t8 = t7 + D_ST3_TRANSIT
    t9 = t8 + D_ST3_DESCEND
    t10 = t9 + D_ST3_GRASP
    t11 = t10 + D_ST3_PULL
    t12 = t11 + D_ST3_LIFT
    t13 = t12 + D_ST4_TRANSIT
    t14 = t13 + D_ST4_DESCEND
    t15 = t14 + D_ST4_OPEN
    t16 = t15 + D_ST4_RETRACT

    log("--------------------------------------------------------")
    log("ST1  Press Button : START  (held pin, gripper closed)")
    log("ST1  -> APPROACH : moving the held pin above the blue button")

    last_phase = None
    max_button_sink = 0.0
    last_target_tcp = button_hover_tcp
    probe_grabbed = False
    step = 0

    try:
        while simulation_app.is_running():
            if step < t1:
                phase = "ST1_APPROACH"
                target_tcp = button_hover_tcp
            elif step < t2:
                phase = "ST1_PRESS"
                target_tcp = button_press_tcp
            elif step < t3:
                phase = "ST1_RELEASE"
                target_tcp = button_hover_tcp
            elif step < t4:
                phase = "ST2_TRANSIT"
                u = (step - t3) / max(1, D_ST2_TRANSIT)
                target_tcp = lerp(button_hover_tcp, slider_start_hover_tcp, u)
            elif step < t5:
                phase = "ST2_DESCEND"
                u = (step - t4) / max(1, D_ST2_DESCEND)
                target_tcp = lerp(slider_start_hover_tcp, slider_start_touch_tcp, u)
            elif step < t6:
                phase = "ST2_DRAG"
                u = (step - t5) / max(1, D_ST2_DRAG)
                slider.set_progress(u)
                target_tcp = lerp(slider_start_touch_tcp, slider_end_touch_tcp, u)
            elif step < t7:
                phase = "ST2_RELEASE"
                slider.set_progress(1.0)
                u = (step - t6) / max(1, D_ST2_RELEASE)
                target_tcp = lerp(slider_end_touch_tcp, slider_end_hover_tcp, u)
            elif step < t8:
                phase = "ST3_TRANSIT"
                u = (step - t7) / max(1, D_ST3_TRANSIT)
                b = probe_grasp_hover if probe_grasp_hover is not None else slider_end_hover_tcp
                target_tcp = lerp(slider_end_hover_tcp, b, u)
            elif step < t9:
                phase = "ST3_DESCEND"
                u = (step - t8) / max(1, D_ST3_DESCEND)
                if probe_grasp_w is not None:
                    target_tcp = lerp(probe_grasp_hover, probe_grasp_w, u)
                else:
                    target_tcp = last_target_tcp
            elif step < t10:
                phase = "ST3_GRASP"
                target_tcp = probe_grasp_w if probe_grasp_w is not None else last_target_tcp
            elif step < t11:
                phase = "ST3_PULL_RELEASE"
                u = (step - t10) / max(1, D_ST3_PULL)
                if probe_pull_w is not None:
                    target_tcp = lerp(probe_grasp_w, probe_pull_w, u)
                else:
                    target_tcp = last_target_tcp
            elif step < t12:
                phase = "ST3_LIFT"
                u = (step - t11) / max(1, D_ST3_LIFT)
                if probe_grasp_lift is not None:
                    target_tcp = lerp(probe_pull_w, probe_grasp_lift, u)
                else:
                    target_tcp = last_target_tcp
            elif step < t13:
                phase = "ST4_TRANSIT"
                u = (step - t12) / max(1, D_ST4_TRANSIT)
                if door_push_tip is not None and probe_grasp_lift is not None:
                    b = tcp_for_probe_tip(door_push_tip + np.array([0.0, 0.0, 0.10]))
                    target_tcp = lerp(probe_grasp_lift, b, u)
                else:
                    target_tcp = last_target_tcp
            elif step < t14:
                phase = "ST4_DESCEND"
                u = (step - t13) / max(1, D_ST4_DESCEND)
                if door_push_tip is not None:
                    a = tcp_for_probe_tip(door_push_tip + np.array([0.0, 0.0, 0.10]))
                    b = tcp_for_probe_tip(door_push_tip)
                    target_tcp = lerp(a, b, u)
                else:
                    target_tcp = last_target_tcp
            elif step < t15:
                phase = "ST4_OPEN"
                u = (step - t14) / max(1, D_ST4_OPEN)
                # The door stays closed until the probe has visibly seated on the edge.
                door_u = 0.0 if u < DOOR_CONTACT_DELAY else (u - DOOR_CONTACT_DELAY) / (1.0 - DOOR_CONTACT_DELAY)
                door_u = float(np.clip(door_u, 0.0, 1.0))
                if door_actuator is not None:
                    door_actuator.set_progress(door_u)
                if door_push_tip is not None:
                    target_tcp = tcp_for_probe_tip(
                        door_push_tip + np.array([0.0, 0.0, DOOR_TIP_LIFT * door_u]))
                else:
                    target_tcp = last_target_tcp
            elif step < t16:
                phase = "ST4_RETRACT"
                u = (step - t15) / max(1, D_ST4_RETRACT)
                if door_actuator is not None:
                    door_actuator.set_progress(1.0)
                if door_push_tip is not None:
                    a = tcp_for_probe_tip(door_push_tip + np.array([0.0, 0.0, DOOR_TIP_LIFT]))
                    b = tcp_for_probe_tip(door_push_tip + np.array([0.0, 0.0, 0.13]))
                    target_tcp = lerp(a, b, u)
                else:
                    target_tcp = last_target_tcp
            else:
                phase = "DONE"
                slider.set_progress(1.0)
                if door_actuator is not None:
                    door_actuator.set_progress(1.0)
                target_tcp = last_target_tcp

            state["step"] = step
            state["phase"] = phase

            if phase != last_phase:
                if phase == "ST1_PRESS":
                    diag_button("after-ST1-approach")
                    log("ST1  -> PRESS : lowering the held pin onto the blue button")
                elif phase == "ST1_RELEASE":
                    diag_button("end-of-ST1-press")
                    if measure_button:
                        ok = max_button_sink >= BUTTON_SUCCESS_SINK_M
                        log(
                            f"ST1  RESULT: {'SUCCESS' if ok else 'FAIL'} "
                            f"(max depression {max_button_sink * 1000:.1f} mm / "
                            f"{BUTTON_SUCCESS_SINK_M * 1000:.1f} mm)"
                        )
                    log("ST1  -> RELEASE : retracting from the blue button")
                elif phase == "ST2_TRANSIT":
                    log("ST2  Match Slider to Screen : START")
                    log(
                        f"ST2  -> TRANSIT : moving from button to slider '{slider.label}' "
                        f"at {np.round(slider.start_top, 4)}"
                    )
                elif phase == "ST2_DESCEND":
                    log("ST2  -> DESCEND : lowering the pin onto the slider handle")
                elif phase == "ST2_DRAG":
                    log(
                        f"ST2  -> DRAG : moving slider by {np.round(slider.move_vector, 4)} m "
                        "while keeping pin contact"
                    )
                elif phase == "ST2_RELEASE":
                    log("ST2  RESULT: slider moved to target position")
                    log("ST2  -> RELEASE : retracting from slider")
                elif phase == "ST3_TRANSIT":
                    log("ST3  Acquire Probe : START")
                    for pp in pin_paths:
                        pr2 = stage.GetPrimAtPath(pp)
                        if pr2 and pr2.IsValid():
                            UsdGeom.Imageable(pr2).MakeInvisible()
                    log("ST3  -> TRANSIT : dropped the blunt pin; moving to the probe")
                elif phase == "ST3_DESCEND":
                    log("ST3  -> DESCEND : lowering the open, rotated gripper onto the probe body")
                elif phase == "ST3_GRASP":
                    if probe_tool is not None:
                        log("ST3  -> GRASP : closing the gripper around the probe")
                    else:
                        log("ST3  -> GRASP : (no probe prim found; nothing to grab)")
                elif phase == "ST3_PULL_RELEASE":
                    log(f"ST3  -> PULL : pulling the probe by {np.round(PROBE_PULL_VECTOR, 3)} m to release it")
                elif phase == "ST3_LIFT":
                    log("ST3  RESULT: probe released from holder")
                    log("ST3  -> LIFT : lifting the released probe clear of the board")
                elif phase == "ST4_TRANSIT":
                    log("ST4  Open Door : START")
                    log("ST4  -> TRANSIT : carrying the probe to the door")
                elif phase == "ST4_DESCEND":
                    log("ST4  -> DESCEND : placing the probe tip on the door edge")
                elif phase == "ST4_OPEN":
                    log(f"ST4  -> OPEN : probe remains on the door edge; door drive starts after contact delay and opens to {DOOR_OPEN_DEG:.0f} deg")
                elif phase == "ST4_RETRACT":
                    log("ST4  RESULT: door opened")
                    log("ST4  -> RETRACT : backing the probe away")
                elif phase == "DONE":
                    log("ST1-ST4 DONE. Holding final pose. Inspect collision CSV, then Ctrl-C to exit.")
                last_phase = phase

            franka.apply_action(
                controller.forward(
                    target_end_effector_position=target_tcp,
                    target_end_effector_orientation=orientation_for_phase(phase),
                )
            )
            last_target_tcp = target_tcp

            # Phase-dependent gripper command: closed on the pin for ST1/ST2,
            # open while approaching the probe, then closed around the probe.
            set_gripper_half_opening(franka, gripper_opening_for_phase(phase))

            # Attach the probe only after the gripper has had time to close.
            if (phase == "ST3_GRASP" and probe_tool is not None and not probe_grabbed
                    and (step - t9) > D_ST3_GRASP * 0.55):
                probe_tool.grab()
                probe_grabbed = True
                log("ST3  -> GRASP : probe attached after finger closure")

            # Carry the probe with the gripper once grabbed (scripted rigid grasp).
            if probe_tool is not None:
                probe_tool.update()

            world.step(render=not CONFIG["headless"])

            if measure_button and phase == "ST1_PRESS":
                sink = button_rest_z - float(btn_view.get_world_poses()[0][0][2])
                max_button_sink = max(max_button_sink, sink)
                if (step - t1) % 100 == 0:
                    diag_button("ST1-press")

            step += 1

    finally:
        if contact_file is not None:
            contact_file.flush()
            contact_file.close()
            log(f"collision log saved: {CONTACT_CSV}")
        # Keep subscription referenced until shutdown.
        _ = contact_sub
        simulation_app.close()


if __name__ == "__main__":
    main()
