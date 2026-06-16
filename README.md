# next-forge-sim2real-isaac-sim

Isaac Sim digital-twin and manipulation demos for the **Robothon TBv2023 task
board**, part of the NEXT-FORGE sim-to-real work. The board CAD (Onshape →
baked USD) is driven two ways:

1. a **Franka robot** that performs the task-board sequence (press button → move
   slider → grab probe → open door), and
2. a **keyboard digital twin** that mirrors the *physical* board's sensor events
   in the simulator (a stand-in for the micro-ROS Task Board Recorder).

## Contents

| File | What it is |
| --- | --- |
| `Task_Board_physics.usd` | The baked task-board asset (flattened, in **meters**, Z-up, ~290×184×108 mm, materials baked in). Both scripts reference it. |
| `taskboard_franka_press_c3_fixed.py` | Robot demo: a Franka manipulates the board through ST1–ST4. (Latest of the `taskboard_franka_press_*.py` iterations.) |
| `taskboard_twin_keyboard.py` | Keyboard-driven digital twin: the board reflects external sensor events and logs them. No robot. |

## Requirements

- **NVIDIA Isaac Sim 5.1** (standalone Python).
- An NVIDIA RTX GPU (developed on an RTX 4090).
- The `Task_Board_physics.usd` asset from this repo.

Both scripts are standalone Isaac apps, so they are launched with Isaac's
bundled Python, not the system one:

```bash
cd /path/to/isaac-sim        # the folder containing python.sh
./python.sh /path/to/script.py
```

## Setup

Place the USD where the scripts expect it, or point them at it with an env var:

```bash
# default location the scripts look for:
#   /data/eurobin/Task_Board_physics.usd
# or override:
export TASKBOARD_USD=/path/to/Task_Board_physics.usd
export TASKBOARD_LOG_DIR=/data/isaac-sim/logs   # where CSV logs are written
```

`TASKBOARD_LOG_DIR` defaults to the current working directory.

## 1. Robot demo — `taskboard_franka_press_c3_fixed.py`

```bash
./python.sh taskboard_franka_press_c3_fixed.py
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
./python.sh taskboard_twin_keyboard.py
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

**Going to micro-ROS later:** all events funnel through a single `emit(sensor,
value)` function. The keyboard calls it now; a micro-ROS subscription calls the
same function later (e.g. on `/task_board/red_button` → `emit("red_button",
msg.data)`), and nothing else changes.

Tunables (top of the file): `DOOR_OPEN_DEG` (negate to flip door direction),
`SLIDER_STEP` (jump per key press), and `SLIDER_RAMP` / `DOOR_RAMP` (motion
smoothing).

## Notes

- The USD is already scaled to meters with `metersPerUnit = 1.0`; don't re-scale it.
- If a script can't find the asset it prints the path it tried — check
  `TASKBOARD_USD`.

