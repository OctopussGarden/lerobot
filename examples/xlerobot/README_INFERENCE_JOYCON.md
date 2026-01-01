# OpenPi Inference with Joycon Control

This script `xlerobot_inference_joycon.py` integrates OpenPi inference capabilities with Joycon teleoperation for the XLeRobot. It allows you to run a trained policy (served by OpenPi) while maintaining Joycon control for safety and episode management.

## Features

- **Joycon Integration**: Uses Joycon controllers to manage the inference loop (Exit, Reset/Re-record).
- **OpenPi Inference**: Connects to an OpenPi websocket server to receive actions based on robot observations.
- **Safety**: Allows immediate exit or reset using Joycon buttons.
- **Automatic Start Pose**: Automatically moves the robot to the standard start position before starting inference.

## Prerequisites

1.  **OpenPi Server**: An OpenPi server must be running and accessible (default: `0.0.0.0:8000`).
2.  **Joycon Controllers**: Paired and connected to the system.
3.  **XLeRobot**: Connected via USB.
4.  **Dependencies**: `lerobot` (installed or in path), `openpi-client`, `numpy`, `scipy`.

## Usage

Run the script from the `lerobot/examples/xlerobot/` directory:

```bash
python3 xlerobot_inference_joycon.py
```

## Joycon Controls

- **Left Joycon (-)**: Exit the current episode and stop the script.
- **Right Joycon (+) / Home**: Reset the robot to the start position and restart the episode (or re-record).
- **Joystick**: (Optional) Can be used for manual intervention if configured, but primarily used for safety stops in this script.

## Configuration

You can modify the following constants in the script:

- `OPENPI_SERVER_IP`: IP address of the OpenPi server.
- `OPENPI_SERVER_PORT`: Port of the OpenPi server.
- `NUM_EPISODES`: Number of episodes to run.
- `EPISODE_TIME_SEC`: Maximum duration of an episode.
