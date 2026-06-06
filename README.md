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

#### Labelling captures in the browser

A **Label** button next to Capture opens a desktop-only panel that lets you
annotate bounding boxes on any captured image and save them in **YOLO detection
format** — ready for `yolov8 train` or any YOLO-compatible pipeline.

**How to use:**
1. Click **Label** — the panel slides in and lists all captured images (newest
   first) with a badge showing the box count (or "n" for unlabelled).
2. Click an image in the list to load it on the right.
3. **Drag** on the image to draw a box; the current class from the dropdown is
   assigned.  A plain **click** on a box selects it (highlighted in orange).
4. Use **Delete box** to remove the selected box.
5. Click **Save** to persist the labels.  The badge updates immediately **and the
   panel auto-advances to the next image** in the list (stops on the last one).
6. Add new class names with the text input + **+** button.

**Auto-detect (rough pre-labelling):** the **✨ Auto-detect** button calls
`POST /captures/autolabel/{name}` and proposes rough boxes for review. It is a
**CV colour-blob detector** (HSV saturation/value threshold → contours), **not a
trained model** — it has no `sock`/object concept, just "saturated blob on a
plain floor". It bootstraps the *first* dataset before any model exists. Known
limit: it **misses white / low-saturation objects** (they don't clear the
saturation threshold) — draw those by hand. The **Auto (rough)** checkbox runs it
automatically on each unlabelled image as you open it. (Model-assisted
pre-labelling with a trained YOLO `.pt` is a planned upgrade, not wired yet.)

**Hotkeys** (active only while the Label panel is open; ignored while typing in a
text field). Drive keys (WASD/arrows) are disabled while the panel is open so you
can't move the robot mid-label:

| Key | Action |
|-----|--------|
| `Enter` | Save labels (then auto-advance to next image) |
| `E` / `]` | Next image |
| `Q` / `[` | Previous image |
| `R` | Auto-detect (rough colour-blob boxes) |
| `X` / `Del` / `Backspace` | Delete the selected box |
| `Esc` | Deselect the current box |
| `1`–`9`, `0` | Pick class index (`0` = 10th class) |

**Label format (YOLO detection — `*.txt` sidecars):**

One `.txt` file per image, same basename, stored alongside the JPEG in
`capture_dir`.  Each line: `<cls> <cx> <cy> <w> <h>` where `cls` is an integer
class index and `cx`, `cy`, `w`, `h` are floats normalized to [0, 1] (center +
size, 6 decimal places).  An empty or absent sidecar means the image is
unlabelled.

**`classes.txt`** in `capture_dir` lists one class name per line; the line
index is the class id.  It is created on first node startup (seeded from the
`capture_classes` parameter).  Once present it is the source of truth — the
parameter is ignored.

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/captures` | List captured images + label status; returns `{images:[{name,labelled,n_boxes},...], classes:[...]}` |
| `GET`  | `/captures/img/{name}` | Raw JPEG bytes (`image/jpeg`) or `404 {ok:false}` |
| `GET`  | `/captures/labels/{name}` | `{ok:true, boxes:[{cls,cx,cy,w,h},...], classes:[...]}` or `404` |
| `POST` | `/captures/labels/{name}` | Body `{boxes:[...]}` — write YOLO sidecar; `{ok:true}` or `400 {ok:false,error}` |
| `POST` | `/captures/autolabel/{name}` | Propose rough boxes via **CV colour-blob** (not a model); `{ok:true, boxes:[...]}` |
| `POST` | `/captures/classes` | Body `{name}` — append class; returns `{ok:true,classes:[...],index:n}` |
| `GET`  | `/detections/latest` | Latest live detections: `{ok:true, boxes:[{cx,cy,w,h,label,score},...], age}` (px coords, `age` = seconds since last message or `null`) |

#### Live detection overlay

A **👁 Detections** toggle in the capture bar draws the sock detector's
bounding boxes live over the left camera stream. The node subscribes to
`Detection2DArray` on `detections_topic` (default `/detections/socks`); the
browser polls `/detections/latest` at 10 Hz and overlays the boxes (label +
score) on a canvas above the stream. Toggle off to hide and stop polling.

This is a **display** toggle — it shows whatever the detector publishes. You
still need the detector **running with a trained model**:

```bash
# sim: starts the detector (continuous) alongside Gazebo + web UI
ros2 launch jetank_ros_main sim_demo.launch.py world:=sock_arena detect:=true \
    slam:=false web:=true        # then pass model_path_sim:=/path/to/sock_sim.pt
ros2 lifecycle set /sock_detector configure && ros2 lifecycle set /sock_detector activate
```

With no detector running, the overlay stays empty and the status reads
`no detector`. Box coords are pixels in the detector's input image; the browser
normalizes against the camera frame's natural size, so detector input topic and
the streamed camera should share resolution (they do in sim — both
`/stereo_camera/left/image_raw`).

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
| `capture_classes` | `['object']` | Initial class list written to `classes.txt` when it doesn't exist yet |

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

---

## ROS 2 API

`jetank_web_control` (ament_python) runs a browser-based teleop/streaming server and a sim cmd_vel mux. The browser-facing HTTP/WebSocket endpoints (e.g. `/`, `/stream.mjpg`, `/ws`, `/map.png`, `/capture`, `/captures/*`, `/navigate`, `/grab`) are **not** ROS interfaces and are not listed here; only the ROS 2 wire API is documented below.

### Nodes

| Node | Executable | Role |
|------|-----------|------|
| `web_control_node` | `web_control_node` | HTTP/WebSocket server: streams the left camera, publishes teleop `cmd_vel`, drives Nav2 (NavigateToPose) + optional grasp (GraspObject), serves the live map and capture/label dataset tools. |
| `cmd_vel_bridge` | `cmd_vel_bridge` | Sim-only mux: combines web-teleop + Nav2 `Twist` streams (teleop priority) and republishes `TwistStamped` for Gazebo's diff_drive_controller. Launched only when `sim:=true`. |

### Published topics

| Topic | Type | Node |
|-------|------|------|
| `/cmd_vel` (param `cmd_vel_topic`; remapped to `/cmd_vel_teleop` when `sim:=true`) | `geometry_msgs/Twist` | `web_control_node` |
| `/initialpose` | `geometry_msgs/PoseWithCovarianceStamped` | `web_control_node` |
| `/diff_drive_controller/cmd_vel` (param `output_topic`) | `geometry_msgs/TwistStamped` | `cmd_vel_bridge` |

### Subscribed topics

| Topic | Type | Node |
|-------|------|------|
| `/stereo_camera/left/image_raw/compressed` (param `image_topic`, when `image_compressed=true`) | `sensor_msgs/CompressedImage` | `web_control_node` |
| `/stereo_camera/left/image_raw` (param `image_topic`, when `image_compressed=false` / sim) | `sensor_msgs/Image` | `web_control_node` |
| `/map` | `nav_msgs/OccupancyGrid` | `web_control_node` |
| `/amcl_pose` | `geometry_msgs/PoseWithCovarianceStamped` | `web_control_node` |
| `/detections/socks` (param `detections_topic`) | `vision_msgs/Detection2DArray` | `web_control_node` |
| `/cmd_vel_teleop` (param `teleop_topic`) | `geometry_msgs/Twist` | `cmd_vel_bridge` |
| `/cmd_vel` (param `nav_topic`) | `geometry_msgs/Twist` | `cmd_vel_bridge` |

### Actions (clients)

| Action | Type | Role |
|--------|------|------|
| `/navigate_to_pose` | `nav2_msgs/action/NavigateToPose` | `web_control_node` client — click-to-navigate goals. |
| `/grasp_object` | `jetank_manipulation/action/GraspObject` | `web_control_node` client — Grab button. Optional: created only if `jetank_manipulation` is importable; otherwise the Grab button is disabled. |

This package defines no `action/`, `srv/`, or `msg/` interfaces of its own.

### Key parameters

#### `web_control_node`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `web_port` | `8080` | HTTP server port. |
| `image_topic` | `/stereo_camera/left/image_raw/compressed` | Camera topic streamed (launch sets raw topic in sim). |
| `image_compressed` | `true` | `true` → subscribe `CompressedImage`; `false` → raw `Image` + local JPEG encode. |
| `cmd_vel_topic` | `/cmd_vel` | Teleop `Twist` output topic (launch sets `/cmd_vel_teleop` in sim). |
| `max_linear_speed` | `0.5` | Max linear speed (m/s). |
| `max_angular_speed` | `1.0` | Max angular speed (rad/s). |
| `cmd_timeout_sec` | `0.5` | Watchdog: publish zero `Twist` if no command for this long. |
| `sim` | `false` | Sim mode flag. |
| `map_dir` | `~/maps` | Map save/load directory. |
| `sim_map_name` | `sim_map` | Saved-map basename in sim. |
| `initial_pose_x` / `initial_pose_y` / `initial_pose_yaw` | `0.0` | Seed pose republished to `/initialpose` for AMCL. |
| `capture_dir` | `~/datasets/detection` | Directory for captured JPEGs + YOLO labels. |
| `capture_classes` | `['object']` | Initial class list seeded into `classes.txt`. |
| `detections_topic` | `/detections/socks` | `Detection2DArray` topic for the live overlay. |

#### `cmd_vel_bridge`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `teleop_topic` | `/cmd_vel_teleop` | Web-teleop `Twist` input. |
| `nav_topic` | `/cmd_vel` | Nav2 `Twist` input. |
| `output_topic` | `/diff_drive_controller/cmd_vel` | Muxed `TwistStamped` output. |
| `frame_id` | `base_link` | Header frame for output `TwistStamped`. |
| `rate_hz` | `20.0` | Output publish rate. |
| `teleop_timeout` | `0.4` | Seconds before a teleop command is considered stale. |
| `nav_timeout` | `0.5` | Seconds before a Nav2 command is considered stale. |

### Launch

`web_control.launch.py` — args: `web_port` (8080), `image_topic` (auto), `cmd_vel_topic` (/cmd_vel), `max_linear` (0.5), `max_angular` (1.0), `sim` (false), `output_cmd_vel` (/diff_drive_controller/cmd_vel), `nav_cmd_vel` (/cmd_vel). With `sim:=true` it also starts `cmd_vel_bridge` and enables `use_sim_time`.
