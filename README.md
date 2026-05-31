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

### In simulation (Gazebo)

The web control also drives the **Gazebo** robot. Two mismatches are handled by
`sim:=true`:

- Gazebo's `diff_drive_controller` wants `TwistStamped` on
  `/diff_drive_controller/cmd_vel`, but the web UI publishes `Twist` on
  `/cmd_vel`. `sim:=true` starts a `cmd_vel_bridge` node that relays
  `Twist → TwistStamped`.
- Gazebo publishes a **raw** camera Image (no compressed transport plugin), so
  `sim:=true` switches the node to subscribe to the raw `Image` topic and
  JPEG-encodes it locally.

```bash
# Standalone (a Gazebo sim must already be running)
ros2 launch jetank_web_control web_control.launch.py sim:=true

# Or fold into the unified sim in one command:
ros2 launch jetank_ros_main sim_demo.launch.py web:=true
```

Then open `http://localhost:8080`. Verified: driving from the browser moves the
robot in Gazebo, and the camera feed streams.

### Mapping & navigation from the browser (Nav2)

The desktop layout shows a **live map directly under the camera feed**. It drives
the full SLAM + Nav2 stack and supports click-to-navigate:

1. **Start Mapping** — launches `jetank_navigation/slam_nav2.launch.py`
   (slam_toolbox online mapping + Nav2 navigation-only via `navigation_only.launch.py`;
   slam supplies `/map` and `map -> odom`, so no AMCL/map_server). Live map appears.
2. Drive around (WASD / joystick) to build the map.
3. **Click anywhere on the map** → sends a `NavigateToPose` goal at that point
   (click pixel → map-frame metres via origin + resolution). The robot drives there.
4. **Save Map** — writes a canonical `~/maps/sim_map.{yaml,pgm}`
   (map_saver runs with `use_sim_time`).
5. **Navigate (saved map)** — enabled once a saved map exists; launches
   `nav2_bringup.launch.py map:=~/maps/sim_map.yaml` (map_server + AMCL + Nav2).
   It first **localizes**: a "Determining robot position…" loader shows while AMCL
   converges, then a **red pose arrow** is drawn on the map at the robot's
   estimated position + heading (updated live).
6. **Stop** — shuts the nav stack down.

Endpoints: `POST /start_mapping`, `POST /start_navigation`, `POST /stop_nav`,
`GET /nav_status`, `GET /robot_pose`, `POST /navigate {x,y}` (map-image pixel).

### Capturing images for a detection dataset

A **Capture** button sits directly under the camera feed. Each press saves the
current full-resolution frame to a **persistent directory on the robot**
(`capture_dir`, default `~/datasets/detection/`, created if absent — never
`/tmp`). Filenames are timestamp-only and flat: `<YYYYMMDDTHHMMSS>_<seq>.jpg`.
The button shows a running saved-count. These images are intended as raw input
for later detection-model training (no labelling/annotation is done here).

- `POST /capture` → `{ok, filename, count, dir}` on success; `503 {ok:false, error}`
  if no camera frame has arrived yet.
- The saved frame is the same source as the MJPEG stream: on the robot the
  `CompressedImage` JPEG bytes are written through unchanged; in sim the raw
  `Image` is JPEG-encoded. Both are full camera resolution.
- Writes use exclusive-create, so a capture never overwrites an existing file —
  even after a node restart (the in-process sequence resets but same-second
  collisions bump the sequence and retry).
- **No disk-space cap.** Captures accumulate in `capture_dir` indefinitely; on a
  Jetson with limited eMMC/SD, prune the directory periodically so it can't fill
  the rootfs.

**Velocity muxing:** the web teleop publishes `/cmd_vel_teleop` and Nav2 publishes
`/cmd_vel`; `cmd_vel_bridge` muxes them (teleop wins only while actively non-zero)
and republishes `TwistStamped` to `/diff_drive_controller/cmd_vel`. This stops the
teleop watchdog's idle zeros and Nav2 from fighting over one topic.

**Notes / known limits:**
- **Saved-map mode needs a map that matches the live world** and the robot to start
  at the seeded pose (`initial_pose_*` params, default origin). AMCL is seeded via
  `/initialpose` (re-published until it converges). If the robot starts elsewhere,
  localization will be wrong — a "click to set pose" control is a future addition.
- The **Start Mapping** path is the most robust for navigation (slam_toolbox gives
  continuous localization, no AMCL init/lifecycle fragility).
- Costmap `inflation_radius` (0.18) / `robot_radius` (0.12) are tuned for the small
  JeTank; adjust in `jetank_navigation/config/nav2/nav2_params.yaml`.

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
| `image_compressed` | `true` | `true` = subscribe `CompressedImage`; `false` = raw `Image` + local JPEG encode (sim) |
| `capture_dir` | `~/datasets/detection` | Persistent dir where `POST /capture` saves full-res JPEGs for dataset building (created if absent) |

Launch-only args: `sim` (`false`) enables sim mode (bridge + raw image + use_sim_time);
`output_cmd_vel` (`/diff_drive_controller/cmd_vel`) is the bridge's TwistStamped output topic.

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
