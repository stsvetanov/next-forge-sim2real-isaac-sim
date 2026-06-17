# next-forge-sim2real-isaac-sim

Isaac Sim digital-twin and manipulation demos for the **Robothon TBv2023 task
board**, part of the NEXT-FORGE sim-to-real work. The board CAD (Onshape →
baked USD) is driven three ways:

1. a **Franka robot** that performs the task-board sequence (press button → move
   slider → grab probe → open door),
2. a **keyboard digital twin** that mirrors the *physical* board's sensor events
   in the simulator, and
3. a **ROS 2-enabled digital twin** that publishes/subscribes task-board sensor
   states through Isaac Sim's internal ROS 2 Jazzy bridge over DDS.

## Contents

| File | What it is |
| --- | --- |
| `Task_Board_physics.usd` | The baked task-board asset (flattened, in **meters**, Z-up, ~290×184×108 mm, materials baked in). Both scripts reference it. |
| `taskboard_franka_press_c3_fixed.py` | Robot demo: a Franka manipulates the board through ST1–ST4. (Latest of the `taskboard_franka_press_*.py` iterations.) |
| `taskboard_twin_keyboard.py` | Keyboard/ROS 2-driven digital twin: the board reflects external sensor events and logs them. No robot. |

## Requirements

- **NVIDIA Isaac Sim 5.1** (standalone Python).
- An NVIDIA RTX GPU.
- The `Task_Board_physics.usd` asset from this repo.
- Optional, for external topic inspection/control: **ROS 2 Jazzy** in a separate terminal.

Both scripts are standalone Isaac apps, so they are launched with Isaac's
bundled Python, not the system one:

```bash
cd /path/to/isaac-sim        # the folder containing python.sh
/python.sh /path/to/script.py
```

To install Isaac Sim.
Download Isaac Sim 5.1.0 for your platform to the Downloads folder and Unzip it. (/data/isaac-sim-5.1/ in this example)

```bash
cd /data/isaac-sim-5.1
and run
./post_install.sh
```

## Setup

Place the USD where the scripts expect it, or point them at it with an env var:

```bash
# default location the scripts look for:
#   /data/eurobin/Task_Board_physics.usd
# or override:
export TASKBOARD_USD=/path/to/Task_Board_physics.usd
export TASKBOARD_LOG_DIR=/path/to/logs   # where CSV logs are written
```

`TASKBOARD_LOG_DIR` defaults to the current working directory.

## 1. Robot demo — `taskboard_franka_press_c3_fixed.py`

```bash
/data/isaac-sim-5.1/python.sh taskboard_franka_press.py
```

A Franka with a closed gripper holding a red pin runs the sequence:

- **ST1** – press the blue button.
- **ST2** – drag the slider to a target position.
- **ST3** – drop the pin and grasp the real `Probe` part out of its holder.
- **ST4** – carry the probe to the door and push it open (damped, bounded hinge).

The board parts move through real PhysX joints (spring-loaded button, prismatic
slider, revolute door), and every PhysX contact pair is logged to
`taskboard_contacts_press_then_slider_<timestamp>.csv` in `TASKBOARD_LOG_DIR`.

Common things to tune (top of the file): `BOARD_POS` (board placement),
`DOOR_OPEN_DEG` (negate it if the door opens the wrong way), and the `D_ST*`
phase durations that set the timeline pacing.

## 2. Keyboard digital twin — `taskboard_twin_keyboard.py`

```bash
/data/isaac-sim-5.1/python.sh taskboard_twin_keyboard.py
```

Loads the board **without a robot** and turns the red button, blue button,
slider, and door into parts you drive from the keyboard. Click the viewport so
it has focus, then:

| Key | Action |
| --- | --- |
| `1` / `2` | red / blue button (hold = pressed, release = up) |
| `Left` / `Right` | slide the slider (−Y / +Y) |
| `Up` / `Down` | open / close the door |
| `R` | reset all sensors to rest |
| `H` | print the key map |
| `Esc` | quit |

Every sensor change is printed and written to `taskboard_twin_events.csv`
(`timestamp, sensor, value, note`) in `TASKBOARD_LOG_DIR` — the `timestamp → event`
shape a recorder needs.

Tunables (top of the file): `DOOR_OPEN_DEG` (negate to flip door direction),
`SLIDER_STEP` (jump per key press), and `SLIDER_RAMP` / `DOOR_RAMP` (motion
smoothing).

## 3. ROS 2 integration — Isaac Sim internal Jazzy bridge

`taskboard_twin_keyboard.py` also starts a ROS 2 node named `taskboard_twin` when
Isaac's ROS 2 bridge can be loaded. The node publishes and subscribes to the
following task-board topics:

| Topic | Type | Meaning |
| --- | --- | --- |
| `/task_board/red_button` | `std_msgs/msg/Bool` | red button pressed/released |
| `/task_board/blue_button` | `std_msgs/msg/Bool` | blue button pressed/released |
| `/task_board/slider` | `std_msgs/msg/Float32` | slider position, normalized `0.0..1.0` |
| `/task_board/door` | `std_msgs/msg/Float32` | door state, normalized `0.0..1.0` |

The keyboard publishes to these topics, and the same script subscribes to them.
This keeps the digital-twin update path identical for keyboard events, ROS 2 test
messages, and later micro-ROS board messages.

### Terminal 1 — run Isaac Sim digital twin

Run Isaac Sim in a terminal where **ROS 2 Jazzy is not sourced**:

```bash
cd /data/eurobin/next-forge-sim2real-isaac-sim/
export LD_LIBRARY_PATH=/data/isaac-sim-5.1/exts/isaacsim.ros2.bridge/jazzy/lib:$LD_LIBRARY_PATH
/data/isaac-sim-5.1/python.sh taskboard_twin_keyboard.py
```

Do **not** run `source /opt/ros/jazzy/setup.bash` in this terminal. Isaac Sim uses
its own Python and its own Jazzy bridge libraries. The external ROS 2 tools can
run in another terminal; communication is established through DDS.

Expected startup lines include:

```text
[ext: isaacsim.ros2.bridge-4.12.4] startup
Attempting to load internal rclpy for ROS Distro: jazzy
rclpy loaded
[twin] ROS 2 enabled via Isaac internal libraries
[twin] ROS 2 node 'taskboard_twin' active on /task_board/{red_button,blue_button,slider,door}
```

### Terminal 2 — inspect ROS 2 topics

Use a normal ROS 2 Jazzy terminal for ROS tools:

```bash
source /opt/ros/jazzy/setup.bash
ros2 topic list
```

Expected topics:

```text
/parameter_events
/rosout
/task_board/blue_button
/task_board/door
/task_board/red_button
/task_board/slider
```

Echo a button topic:

```bash
ros2 topic echo /task_board/blue_button
```

When pressing/releasing key `2` in Isaac Sim, you should see:

```text
data: true
---
data: false
---
```

You can also publish test messages from Terminal 2:

```bash
ros2 topic pub --once /task_board/blue_button std_msgs/msg/Bool "{data: true}"
ros2 topic pub --once /task_board/blue_button std_msgs/msg/Bool "{data: false}"
ros2 topic pub --once /task_board/slider std_msgs/msg/Float32 "{data: 0.9}"
ros2 topic pub --once /task_board/door std_msgs/msg/Float32 "{data: 1.0}"
```

## Notes

- The USD is already scaled to meters with `metersPerUnit = 1.0`; don't re-scale it.
- If a script can't find the asset it prints the path it tried — check
  `TASKBOARD_USD`.
- For ROS 2, keep the Isaac Sim terminal and the external ROS 2 terminal separate:
  Isaac Sim uses internal Jazzy libraries, while the external terminal uses the
  system `/opt/ros/jazzy` installation.
