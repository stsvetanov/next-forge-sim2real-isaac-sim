#!/usr/bin/env python3
r"""
Task Board DIGITAL TWIN -- keyboard-driven, mirrors the physical board.

This is a minimal stand-in for the ROS 2 / micro-ROS Task Board Recorder
(hpcbg/ERTiRoB). The physical M5StickC PLUS2 board publishes sensor events
(button pressed, slider moved, door opened). Here those events are produced by
the keyboard instead, and the SIMULATED board reflects them in real time -- so a
press on the physical port shows up as a press in Isaac Sim. No robot is loaded;
the board is the mirror, not the manipulator.

Every event is logged as  timestamp -> event  (console + CSV), exactly the shape
a recorder needs. When the real integration lands, only `emit()` changes: a
micro-ROS subscription calls emit(sensor, value) instead of the keyboard.

KEYS
    1            red button   (hold = pressed, release = up)
    2            blue button  (hold = pressed, release = up)
    Left / Right slide the slider  (-Y / +Y)
    Up / Down    open / close the door
    R            reset all sensors to rest
    H            print this help
    Esc          quit

Run from Isaac Sim:
    ./python.sh /path/to/taskboard_twin_keyboard.py

Env vars:
    TASKBOARD_USD=/data/eurobin/Task_Board_physics.usd   (baked board asset)
    TASKBOARD_LOG_DIR=/data/isaac-sim/logs               (where the CSV goes)
"""
import csv
import os
import threading
from datetime import datetime, timezone

from isaacsim import SimulationApp

CONFIG = {"headless": False}
simulation_app = SimulationApp(CONFIG)

# --------------------------------------------------------------------------- #
# ROS 2 must use Isaac's INTERNAL Jazzy libraries (built for Python 3.11), not
# a system /opt/ros/jazzy install (built for 3.12) -- a 3.12 rclpy C-extension
# cannot load into Isaac's 3.11 interpreter. So:
#   1) enable the ros2 bridge extension (it puts the internal libs on the path),
#   2) import rclpy AFTER that.
# Run this script in a terminal where system ROS is NOT sourced; run external
# ROS nodes from a separate, ROS-sourced terminal (DDS bridges the two).
# --------------------------------------------------------------------------- #
os.environ.setdefault("ROS_DISTRO", "jazzy")
os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")

_HAVE_ROS2 = False
try:
    from isaacsim.core.utils.extensions import enable_extension
    enable_extension("isaacsim.ros2.bridge")
    simulation_app.update()  # let the extension load its internal ROS 2 libs
    import rclpy
    from std_msgs.msg import Bool, Float32
    _HAVE_ROS2 = True
    print("[twin] ROS 2 enabled via Isaac internal libraries")
except Exception as exc:
    print(f"[twin] ROS 2 unavailable ({type(exc).__name__}: {exc}); "
          f"keyboard -> emit() directly. "
          f"If you expected ROS: launch in a terminal where /opt/ros is NOT "
          f"sourced (check $ROS_DISTRO and that /opt/ros is off PYTHONPATH).")

import numpy as np  # noqa: E402
from pxr import Usd, UsdGeom, UsdLux, Gf  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402

try:
    import carb
    import carb.input
    import omni.appwindow
    _HAVE_INPUT = True
except Exception as exc:  # pragma: no cover
    _HAVE_INPUT = False
    print(f"[twin] keyboard interface unavailable: {exc}")

# --------------------------------------------------------------------------- #
# Config (matches the baked asset used by the robot demo)
# --------------------------------------------------------------------------- #
BOARD_USD = os.environ.get("TASKBOARD_USD", "/data/eurobin/Task_Board_physics.usd")
BOARD_PRIM = "/World/TaskBoard"
BOARD_GEOM = "/World/TaskBoard/Geom"
BOARD_POS = np.array([0.45, 0.0, 0.0795])

RED_BUTTON_NAME = "tn__RedButton_i9"
BLUE_BUTTON_NAME = "tn__BlueButton_kA"
SLIDER_NAME = "Slider"
DOOR_NAME = "Door"

PRESS_DEPTH = 0.003     # how far a button visibly sinks (m)
SLIDER_TRAVEL = 0.020   # full slider travel along board Y (m)
SLIDER_STEP = 0.12      # slider target change per Left/Right key event (fraction)
SLIDER_RAMP = 0.05      # slider smoothing per frame (fraction)
DOOR_OPEN_DEG = 70.0    # door open angle; negate if it opens the wrong way
DOOR_RAMP = 0.04        # door smoothing per frame (fraction)

LOG_DIR = os.environ.get("TASKBOARD_LOG_DIR", os.getcwd())
EVENT_CSV = os.path.join(LOG_DIR, "taskboard_twin_events.csv")


def now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


def resolve_board_usd():
    if os.path.exists(BOARD_USD):
        return BOARD_USD
    for alt in ("/mnt/data/Task_Board_physics.usd", "Task_Board_physics.usd"):
        if os.path.exists(alt):
            print(f"[twin] BOARD_USD not at {BOARD_USD}; using {alt}")
            return alt
    return BOARD_USD


def find_child(root, name):
    for prim in Usd.PrimRange(root):
        if prim.GetName() == name:
            return prim
    return None


def bbox_range(prim):
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    return cache.ComputeWorldBound(prim).ComputeAlignedRange()


# --------------------------------------------------------------------------- #
# Kinematic part: mirror one board part by overwriting one transform op/frame.
# A digital twin reflects external truth, so the parts are posed directly (no
# physics) from the incoming sensor state -- deterministic and never explodes.
# --------------------------------------------------------------------------- #
class KinematicPart:
    def __init__(self, prim, label):
        self.prim = prim
        self.label = label
        xc = UsdGeom.XformCache()
        self.W0 = xc.GetLocalToWorldTransform(prim)             # rest world pose
        self.Mp_inv = xc.GetLocalToWorldTransform(prim.GetParent()).GetInverse()
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        self.op = xf.AddTransformOp()
        self._set_world(self.W0)

    def _set_world(self, W):
        self.op.Set(W * self.Mp_inv)

    def translate(self, t):
        T = Gf.Matrix4d(1.0)
        T.SetTranslate(Gf.Vec3d(float(t[0]), float(t[1]), float(t[2])))
        self._set_world(self.W0 * T)

    def rotate_about(self, hinge, axis, deg):
        h = Gf.Vec3d(float(hinge[0]), float(hinge[1]), float(hinge[2]))
        Tn = Gf.Matrix4d(1.0); Tn.SetTranslate(-h)
        R = Gf.Matrix4d(1.0); R.SetRotate(Gf.Rotation(Gf.Vec3d(*axis), float(deg)))
        Tp = Gf.Matrix4d(1.0); Tp.SetTranslate(h)
        self._set_world(self.W0 * (Tn * R * Tp))


# --------------------------------------------------------------------------- #
def main():
    print("=" * 64)
    print("Task Board DIGITAL TWIN (keyboard -> board, micro-ROS style)")
    print("=" * 64)

    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    stage = world.stage

    UsdLux.DomeLight.Define(stage, "/World/DomeLight").CreateIntensityAttr(1000.0)
    UsdLux.DistantLight.Define(stage, "/World/KeyLight").CreateIntensityAttr(300.0)

    wrapper = UsdGeom.Xform.Define(stage, BOARD_PRIM)
    UsdGeom.Xformable(wrapper.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*BOARD_POS.tolist()))
    add_reference_to_stage(usd_path=resolve_board_usd(), prim_path=BOARD_GEOM)

    root = stage.GetPrimAtPath(BOARD_GEOM)

    def need(name):
        p = find_child(root, name)
        if not (p and p.IsValid()):
            raise RuntimeError(f"part '{name}' not found under {BOARD_GEOM}")
        return p

    red = KinematicPart(need(RED_BUTTON_NAME), "red_button")
    blue = KinematicPart(need(BLUE_BUTTON_NAME), "blue_button")
    slider = KinematicPart(need(SLIDER_NAME), "slider")
    door_prim = need(DOOR_NAME)
    door = KinematicPart(door_prim, "door")

    # Door hinge: board X axis at the door's -Y edge (the door swings up).
    dr = bbox_range(door_prim)
    dmn, dmx = dr.GetMin(), dr.GetMax()
    hinge_world = np.array([(dmn[0] + dmx[0]) / 2.0, dmn[1], (dmn[2] + dmx[2]) / 2.0])
    print(f"[twin] door hinge at world {np.round(hinge_world, 4)}")

    world.reset()

    # ---- sensor state ---------------------------------------------------- #
    # 'target' is what the (physical) board reports; 'actual' is the smoothed
    # value we render. Buttons are momentary so they snap.
    state = {
        "running": True,
        "red_button": 0, "blue_button": 0,
        "slider_target": 0.5, "slider_actual": 0.5,   # 0..1, 0.5 = centred
        "door_target": 0.0, "door_actual": 0.0,        # 0 closed .. 1 open
    }

    # ---- event log ------------------------------------------------------- #
    try:
        csv_file = open(EVENT_CSV, "w", newline="")
        writer = csv.writer(csv_file)
        writer.writerow(["timestamp", "sensor", "value", "note"])
        csv_file.flush()
    except Exception as exc:
        csv_file, writer = None, None
        print(f"[twin] could not open {EVENT_CSV}: {exc}")

    def log_event(sensor, value, note=""):
        stamp = now_iso()
        print(f"[event] {stamp}  {sensor:12s} -> {value}   {note}")
        if writer is not None:
            writer.writerow([stamp, sensor, value, note])
            csv_file.flush()

    def emit(sensor, value, note=""):
        """Single ingestion point for board events.

        Keyboard calls this now; a micro-ROS subscription calls the very same
        function later (e.g. on /task_board/red_button -> emit('red_button', msg.data)).
        """
        if sensor in ("red_button", "blue_button"):
            if state[sensor] == value:
                return
            state[sensor] = value
            log_event(sensor, "PRESSED" if value else "RELEASED")
        elif sensor == "slider":
            state["slider_target"] = float(np.clip(value, 0.0, 1.0))
            log_event("slider", round(state["slider_target"], 3))
        elif sensor == "door":
            state["door_target"] = float(np.clip(value, 0.0, 1.0))
            log_event("door", "OPEN" if value >= 0.5 else "CLOSED")

    def reset_all():
        state.update(red_button=0, blue_button=0,
                     slider_target=0.5, door_target=0.0)
        log_event("system", "RESET")

    def print_help():
        print(__doc__[__doc__.index("KEYS"):__doc__.index("Run from")])

    # ---- ROS 2 integration ----------------------------------------------- #
    # Key presses publish to /task_board/<sensor>; the subscriber callbacks
    # call emit() so the model update flows through a single path regardless
    # of whether the source is the keyboard or real hardware.
    _ros_node = _ros_thread = None
    _pub_red = _pub_blue = _pub_slider = _pub_door = None

    if _HAVE_ROS2:
        if not rclpy.ok():
            rclpy.init()
        _ros_node = rclpy.create_node("taskboard_twin")

        _pub_red    = _ros_node.create_publisher(Bool,    "/task_board/red_button",  10)
        _pub_blue   = _ros_node.create_publisher(Bool,    "/task_board/blue_button", 10)
        _pub_slider = _ros_node.create_publisher(Float32, "/task_board/slider",      10)
        _pub_door   = _ros_node.create_publisher(Float32, "/task_board/door",        10)

        _ros_node.create_subscription(
            Bool,    "/task_board/red_button",  lambda m: emit("red_button",  int(m.data)), 10)
        _ros_node.create_subscription(
            Bool,    "/task_board/blue_button", lambda m: emit("blue_button", int(m.data)), 10)
        _ros_node.create_subscription(
            Float32, "/task_board/slider",      lambda m: emit("slider",      m.data),      10)
        _ros_node.create_subscription(
            Float32, "/task_board/door",        lambda m: emit("door",        m.data),      10)

        _ros_thread = threading.Thread(target=rclpy.spin, args=(_ros_node,), daemon=True)
        _ros_thread.start()
        print("[twin] ROS 2 node 'taskboard_twin' active on /task_board/{red_button,blue_button,slider,door}")
    else:
        print("[twin] running without ROS 2.")

    def publish_sensor(sensor, value):
        """Publish a board-sensor event to ROS 2; subscriber calls emit() to update model.

        Falls back to calling emit() directly when rclpy is unavailable.
        """
        if _ros_node is None:
            emit(sensor, value)
            return
        if sensor == "red_button":
            m = Bool(); m.data = bool(value); _pub_red.publish(m)
        elif sensor == "blue_button":
            m = Bool(); m.data = bool(value); _pub_blue.publish(m)
        elif sensor == "slider":
            m = Float32(); m.data = float(value); _pub_slider.publish(m)
        elif sensor == "door":
            m = Float32(); m.data = float(value); _pub_door.publish(m)

    # ---- keyboard -------------------------------------------------------- #
    kb_sub = keyboard = input_iface = None
    if _HAVE_INPUT:
        app_window = omni.appwindow.get_default_app_window()
        keyboard = app_window.get_keyboard()
        input_iface = carb.input.acquire_input_interface()
        ET = carb.input.KeyboardEventType
        K = carb.input.KeyboardInput

        def on_kb(event, *args):
            et, k = event.type, event.input
            if et == ET.KEY_PRESS:
                if k == K.KEY_1:
                    publish_sensor("red_button", 1)
                elif k == K.KEY_2:
                    publish_sensor("blue_button", 1)
                elif k == K.LEFT:
                    publish_sensor("slider", state["slider_target"] - SLIDER_STEP)
                elif k == K.RIGHT:
                    publish_sensor("slider", state["slider_target"] + SLIDER_STEP)
                elif k == K.UP:
                    publish_sensor("door", 1.0)
                elif k == K.DOWN:
                    publish_sensor("door", 0.0)
                elif k == K.R:
                    reset_all()
                elif k == K.H:
                    print_help()
                elif k == K.ESCAPE:
                    state["running"] = False
            elif et == ET.KEY_REPEAT:
                if k == K.LEFT:
                    publish_sensor("slider", state["slider_target"] - SLIDER_STEP)
                elif k == K.RIGHT:
                    publish_sensor("slider", state["slider_target"] + SLIDER_STEP)
            elif et == ET.KEY_RELEASE:
                if k == K.KEY_1:
                    publish_sensor("red_button", 0)
                elif k == K.KEY_2:
                    publish_sensor("blue_button", 0)
            return True

        kb_sub = input_iface.subscribe_to_keyboard_events(keyboard, on_kb)
        print_help()
    else:
        print("[twin] running without keyboard input (no carb.input).")

    print(f"[twin] event log -> {EVENT_CSV}")
    print("[twin] ready. Click the viewport so it has focus, then press keys.")

    # ---- main loop: ingest -> mirror ------------------------------------ #
    try:
        while simulation_app.is_running() and state["running"]:
            # Smooth the slowly-moving sensors toward their reported value.
            for key in ("slider", "door"):
                a, t = state[f"{key}_actual"], state[f"{key}_target"]
                ramp = SLIDER_RAMP if key == "slider" else DOOR_RAMP
                state[f"{key}_actual"] = a + float(np.clip(t - a, -ramp, ramp))

            # Pose every part from the current sensor state.
            red.translate((0.0, 0.0, -PRESS_DEPTH * state["red_button"]))
            blue.translate((0.0, 0.0, -PRESS_DEPTH * state["blue_button"]))
            slider.translate((0.0, (state["slider_actual"] - 0.5) * SLIDER_TRAVEL, 0.0))
            door.rotate_about(hinge_world, (1.0, 0.0, 0.0),
                              state["door_actual"] * DOOR_OPEN_DEG)

            world.step(render=not CONFIG["headless"])
    finally:
        if kb_sub is not None and input_iface is not None:
            try:
                input_iface.unsubscribe_to_keyboard_events(keyboard, kb_sub)
            except Exception:
                pass
        if _ros_node is not None:
            _ros_node.destroy_node()
        if _HAVE_ROS2 and rclpy.ok():
            rclpy.shutdown()
        if _ros_thread is not None:
            _ros_thread.join(timeout=2.0)
        if csv_file is not None:
            csv_file.flush()
            csv_file.close()
            print(f"[twin] event log saved: {EVENT_CSV}")
        simulation_app.close()


if __name__ == "__main__":
    main()
