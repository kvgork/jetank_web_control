# jetank_web_control

Browser-based remote control for the JeTank robot. Everything runs on the Jetson Orin Nano — no ROS2 installation is needed on the controlling device.

## What it does

- Streams the **left camera** feed (MJPEG) to any browser on the same network.
- Accepts **movement commands** via WebSocket and publishes them to `/cmd_vel`.
- Serves a single-page controller app that auto-detects the input device:
  - **Phone / tablet** → virtual joystick (touch)
  - **Desktop / laptop** → keyboard (WASD / arrow keys), D-pad buttons, or gamepad
- Stops the robot automatically if the browser tab closes or the connection drops.

## Architecture

```
Jetson Orin Nano
├── stereo_camera_node  ──► /stereo_camera/left/image_raw/compressed
├── robot_controller    ◄── /cmd_vel
└── web_control_node
    ├── GET  /           → controller page (HTML/JS)
    ├── GET  /stream.mjpg → MJPEG camera stream
    └── WS   /ws         → JSON cmd_vel channel

Any browser (same network)
└── http://<jetson-ip>:8080
```

## Quick start

### 1. Install the Python dependency

```bash
pip3 install aiohttp
```

### 2. Build

```bash
cd ~/workspaces/ros2_ws
colcon build --packages-select jetank_web_control
source install/setup.bash
```

### 3. Launch

**Standalone** (web control only, robot and camera must already be running):

```bash
ros2 launch jetank_web_control web_control.launch.py
```

**As part of the full system** (included by default):

```bash
ros2 launch jetank_ros_main unified.launch.py
```

**Disable web control** in unified launch:

```bash
ros2 launch jetank_ros_main unified.launch.py enable_web_control:=false
```

### 4. Open the controller

Find the Jetson's IP address:

```bash
hostname -I
```

Then open **`http://<jetson-ip>:8080`** in any browser on the same Wi-Fi network.

## Controls

### Phone / tablet (auto-detected)

The page shows a **virtual joystick**:
- Drag toward the top arrow → move forward
- Drag toward the bottom arrow → move backward
- Drag left/right → turn
- Release → stop

Use the **Speed** slider to limit maximum velocity.

### Desktop / laptop

| Input | Action |
|-------|--------|
| `W` / `↑` | Forward |
| `S` / `↓` | Backward |
| `A` / `←` | Turn left |
| `D` / `→` | Turn right |
| `Space` | Stop |
| D-pad buttons | Same as keys |
| Gamepad left stick | Full analog control |

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `web_port` | `8080` | HTTP server port |
| `image_topic` | `/stereo_camera/left/image_raw/compressed` | Compressed image topic to stream |
| `cmd_vel_topic` | `/cmd_vel` | Twist topic for movement commands |
| `max_linear_speed` | `0.5` | Maximum linear speed (m/s) — UI scale maps to this |
| `max_angular_speed` | `1.0` | Maximum angular speed (rad/s) |
| `cmd_timeout_sec` | `0.5` | Stop robot if no WebSocket message received for this long |

Override at launch:

```bash
ros2 launch jetank_web_control web_control.launch.py \
  web_port:=9090 \
  max_linear_speed:=0.3
```

## Safety

- **Auto-stop watchdog**: if the browser disconnects or no command arrives for 0.5 s, the node publishes a zero Twist to stop the robot.
- **Speed scale**: the UI slider (5–100 %) multiplies the raw joystick/key value before publishing, so you can limit max speed without changing parameters.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Page unreachable | Check Jetson firewall: `sudo ufw allow 8080` |
| "No stream" badge | Confirm camera node is running and topic exists: `ros2 topic list \| grep image_raw` |
| Robot doesn't move | Confirm motor controller is running: `ros2 topic echo /cmd_vel` while moving the joystick |
| `ModuleNotFoundError: aiohttp` | `pip3 install aiohttp` |
