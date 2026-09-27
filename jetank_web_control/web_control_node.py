#!/usr/bin/env python3
"""
JeTank web control node.

Runs an HTTP/WebSocket server on the Jetson that lets a browser on any
laptop drive the robot and watch the left camera stream without installing
any ROS tooling on the laptop.

Endpoints:
  GET  /            - control page (HTML)
  GET  /static/*    - packaged page assets (app.js, style.css)
  GET  /stream.mjpg - MJPEG camera stream
  WS   /ws          - JSON command channel  {"linear_x": float, "angular_z": float}
  GET  /map.png     - current Nav2 occupancy map as PNG (404 when no map yet)
  GET  /map_meta    - map metadata JSON  {"width", "height", "resolution"}
  POST /save_map    - save map via map_saver_cli to ~/maps/jetank_map_<ts>
  GET  /captures                    - list captured images + label status
  GET  /captures/img/{name}         - raw JPEG bytes for a capture
  GET  /captures/labels/{name}      - YOLO label boxes for a capture
  POST /captures/labels/{name}      - write YOLO label boxes for a capture
  POST /captures/autolabel/{name}   - propose rough boxes via CV colour-blob
  POST /captures/classes            - add a new detection class
  POST /grab                        - trigger GraspObject action (503 when unavailable)
  GET  /grab/status                 - grasp state JSON {available,running,stage,...}
  POST /mission/goal                - {ix,iy} pixel -> send a RunMission goal (fetch)
  POST /mission/deposit             - {ix,iy} pixel -> store + persist deposit pose
  GET  /mission/deposit             - deposit pose {x,y} or {"set":false}
  GET  /mission/status              - latest mission status JSON {status,active}
  POST /mission/cancel              - cancel the active RunMission goal (no-op if none)
  POST /start_mapping               - start slam_toolbox (mapping only; teleop to drive)
  POST /start_navigation            - start nav2 + AMCL on saved map
  POST /stop_nav                    - stop whichever nav stack is running
  GET  /nav_status                  - {running, has_map, ...}
"""

import asyncio
import io
import json
import math
import os
import re
import signal
import subprocess
import threading
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import CompressedImage, Image
from nav2_msgs.action import NavigateToPose
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import String

try:
    from jetank_mission.action import RunMission as _RunMissionAction
    _MISSION_AVAILABLE = True
except ImportError:
    _RunMissionAction = None
    _MISSION_AVAILABLE = False

try:
    import numpy as np
    from PIL import Image as _PILImage
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

try:
    from jetank_manipulation.action import GraspObject as _GraspObjectAction
    _GRASP_AVAILABLE = True
except ImportError:
    _GraspObjectAction = None
    _GRASP_AVAILABLE = False

try:
    from aiohttp import web
    import aiohttp
except ImportError:
    raise SystemExit(
        "aiohttp is required: pip install aiohttp"
    )

# ---------------------------------------------------------------------------
# Pure module-level helpers (no ROS / aiohttp deps — directly unit-testable)
# ---------------------------------------------------------------------------

_SAFE_NAME_RE = re.compile(r'^[A-Za-z0-9._-]+\.[Jj][Pp][Gg]$')


def _safe_capture_name(name: str) -> Optional[str]:
    r"""Return *name* if it is a safe bare filename (no path separators, no ..).

    Accepts filenames matching ``^[A-Za-z0-9._-]+\.jpg$`` (case-insensitive
    extension).  Returns ``None`` for anything else.
    """
    if not name or '/' in name or '\\' in name or name == '..':
        return None
    if '..' in name.split('.'):
        return None
    if not _SAFE_NAME_RE.match(name):
        return None
    return name


def _yolo_parse(text: str, n_classes: int) -> list:
    """Parse a YOLO-format sidecar text into a list of box dicts.

    Each box is ``{'cls': int, 'cx': float, 'cy': float, 'w': float,
    'h': float}``.  Malformed lines, out-of-range class indices, and
    coords outside ``[0, 1]`` are silently skipped.  Never raises.
    """
    boxes = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            cls = int(parts[0])
            cx = float(parts[1])
            cy = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
        except (ValueError, TypeError):
            continue
        if cls not in range(n_classes):
            continue
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0
                and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
            continue
        boxes.append({'cls': cls, 'cx': cx, 'cy': cy, 'w': w, 'h': h})
    return boxes


def _yolo_serialize(boxes: list) -> str:
    r"""Serialize a list of box dicts to YOLO-format text.

    Returns an empty string for an empty list.  Each line ends with ``\n``.
    Coords are formatted to 6 decimal places.
    """
    return ''.join(
        f"{b['cls']} {b['cx']:.6f} {b['cy']:.6f} {b['w']:.6f} {b['h']:.6f}\n"
        for b in boxes
    )


def map_pixel_to_world(ix, iy, resolution, width, height,
                       origin_x, origin_y) -> tuple:
    """Convert a ``/map.png`` pixel ``(ix, iy)`` to a map-frame point.

    The served PNG is vertically flipped (``np.flipud`` in ``_on_map``), so the
    occupancy-grid row is recovered as ``grid_row = (height - 1) - iy`` before
    applying the standard cell-centre conversion::

        wx = origin_x + (ix + 0.5) * resolution
        wy = origin_y + (grid_row + 0.5) * resolution

    ``ix``/``iy`` are coerced to ``int`` and clamped to ``[0, width-1]`` /
    ``[0, height-1]`` so an out-of-range click maps to the nearest edge cell.
    Pure: no ROS / aiohttp / numpy. Returns ``(wx, wy)`` floats.
    """
    width = int(width)
    height = int(height)
    ix = max(0, min(width - 1, int(ix)))
    iy = max(0, min(height - 1, int(iy)))
    grid_row = (height - 1) - iy   # the PNG was flipud'd before serving
    wx = origin_x + (ix + 0.5) * resolution
    wy = origin_y + (grid_row + 0.5) * resolution
    return wx, wy


def deposit_serialize(x, y) -> str:
    """Serialize a deposit pose ``(x, y)`` to a JSON string for persistence.

    Stores ``{"x": float, "y": float}``.  Pure / unit-testable: round-trips
    with :func:`deposit_parse`.
    """
    return json.dumps({'x': float(x), 'y': float(y)})


def deposit_parse(text: str) -> Optional[tuple]:
    """Parse persisted deposit-pose JSON back into ``(x, y)`` floats.

    Returns ``None`` for empty/blank input, malformed JSON, a non-object, or a
    payload missing numeric ``x``/``y``.  Never raises.
    """
    if not text or not text.strip():
        return None
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    try:
        return float(data['x']), float(data['y'])
    except (KeyError, TypeError, ValueError):
        return None


# Terminal mission states: once the cached /mission/status reaches one of these
# the UI stops polling. DONE/FAILED/CANCELLED come from the coordinator's
# feedback/status; IDLE is the coordinator's resting state (also when no mission
# is active). Matching is case-insensitive and tolerant of trailing punctuation
# (e.g. "DONE ✓" / "FAILED — reason").
_TERMINAL_MISSION_STATES = frozenset({'DONE', 'FAILED', 'CANCELLED', 'IDLE'})


def is_terminal_mission_status(status) -> bool:
    """Return True if *status* is a terminal mission state (stop polling).

    Terminal states are DONE / FAILED / CANCELLED / IDLE. The first whitespace-
    delimited token of *status* is matched case-insensitively so suffixes added
    by the UI/coordinator (e.g. ``"FAILED — no sock found"``) still count.
    Empty / None / non-string input is treated as terminal (nothing to poll).
    Pure / unit-testable: no ROS / aiohttp deps. Never raises.
    """
    if not isinstance(status, str):
        return True
    stripped = status.strip()
    if not stripped:                       # empty / whitespace -> nothing to poll
        return True
    return stripped.split()[0].upper() in _TERMINAL_MISSION_STATES


def rough_boxes_from_bgr(img, sat_min=70, val_min=40, min_area_frac=0.0006,
                         max_area_frac=0.20, max_boxes=20) -> list:
    """Propose rough bounding boxes from a BGR image via colour-blob detection.

    Intended for *rough* auto-annotation of brightly coloured objects (e.g. the
    sim socks) lying on a plain, low-saturation floor: it thresholds the HSV
    saturation/value channels, cleans the mask with morphology, and returns one
    box per surviving contour. Output is a list of YOLO-style boxes
    ``{'cx', 'cy', 'w', 'h'}`` normalised to ``[0, 1]`` (largest first); class
    assignment is left to the caller. Heuristic only — boxes need human review,
    and it will miss low-saturation objects (e.g. a white sock). Never raises.

    ``cv2`` is imported lazily so the module stays importable without it.
    """
    if not _PIL_AVAILABLE:  # numpy shares the same guarded import block
        return []
    try:
        import cv2
    except ImportError:
        return []
    if img is None or getattr(img, 'size', 0) == 0:
        return []
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return []
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    mask = ((sat >= sat_min) & (val >= val_min)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # findContours returns (contours, hierarchy) on cv2>=4 and
    # (image, contours, hierarchy) on cv2 3.x — take the second-to-last item.
    found = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = found[-2]
    img_area = float(w * h)
    boxes = []
    for contour in contours:
        bx, by, bw, bh = cv2.boundingRect(contour)
        frac = (bw * bh) / img_area
        if frac < min_area_frac or frac > max_area_frac:
            continue
        boxes.append({
            'cx': (bx + bw / 2.0) / w,
            'cy': (by + bh / 2.0) / h,
            'w': bw / float(w),
            'h': bh / float(h),
            '_frac': frac,
        })
    boxes.sort(key=lambda b: b['_frac'], reverse=True)
    boxes = boxes[:max_boxes]
    for box in boxes:
        del box['_frac']
    return boxes


# ---------------------------------------------------------------------------
# Static web assets (the control page formerly lived here as an inline _HTML
# literal; it is now packaged as static/index.html + app.js + style.css)
# ---------------------------------------------------------------------------

def resolve_static_dir() -> str:
    """Return the directory holding index.html / app.js / style.css.

    Prefers the installed share path — setup.py data_files install the assets
    under ``share/jetank_web_control/static`` and ament_index resolves it, so
    the node works from the install space (including --symlink-install).
    Falls back to the source-tree ``static/`` beside the package for
    uninstalled runs (e.g. pytest). Raises ``FileNotFoundError`` when neither
    candidate contains ``index.html``.
    """
    candidates = []
    try:
        from ament_index_python.packages import get_package_share_directory
        candidates.append(os.path.join(
            get_package_share_directory('jetank_web_control'), 'static'))
    except Exception:  # noqa: BLE001 — ament index missing or pkg unregistered
        pass
    candidates.append(os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), os.pardir, 'static')))
    for cand in candidates:
        if os.path.isfile(os.path.join(cand, 'index.html')):
            return cand
    raise FileNotFoundError(
        'jetank_web_control static assets not found; looked in: '
        + ', '.join(candidates))


# ---------------------------------------------------------------------------
# ROS2 node
# ---------------------------------------------------------------------------

class WebControlNode(Node):
    def __init__(self):
        super().__init__('web_control_node')

        self.declare_parameter('web_port', 8080)
        self.declare_parameter('image_topic', '/stereo_camera/left/image_raw/compressed')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('max_linear_speed', 0.5)
        self.declare_parameter('max_angular_speed', 1.0)
        self.declare_parameter('cmd_timeout_sec', 0.5)
        # When false, subscribe to a raw sensor_msgs/Image and JPEG-encode it
        # locally (used in simulation, where Gazebo publishes raw Image and the
        # compressed_image_transport plugin is not available).
        self.declare_parameter('image_compressed', True)
        # Simulation mode: nav launches use the sim clock and maps live under a
        # canonical name so a previously-made sim map can be reused.
        self.declare_parameter('sim', False)
        self.declare_parameter('map_dir', os.path.expanduser('~/maps'))
        self.declare_parameter('sim_map_name', 'sim_map')
        # Robot spawn pose in the map frame, used to seed AMCL when navigating on
        # a saved map (otherwise AMCL never publishes map->odom and nav is dead).
        self.declare_parameter('initial_pose_x', 0.0)
        self.declare_parameter('initial_pose_y', 0.0)
        self.declare_parameter('initial_pose_yaw', 0.0)
        # Persistent dir on the robot for captured training images (NOT /tmp).
        self.declare_parameter('capture_dir', os.path.expanduser('~/datasets/detection'))
        # Default class names written to classes.txt if it doesn't exist yet.
        self.declare_parameter('capture_classes', ['object'])
        # Live detection overlay: topic carrying Detection2DArray from the sock
        # detector (jetank_detection). The web UI "Detections" toggle draws these
        # boxes over the camera stream. Stays empty when no detector is running.
        self.declare_parameter('detections_topic', '/detections/socks')
        # Persisted deposit pose for the web map-click mission ("Set deposit
        # area" mode). Stored as JSON {"x","y"} so it survives node restarts.
        self.declare_parameter(
            'deposit_file', os.path.expanduser('~/.jetank/deposit_pose.json'))

        self._port = self.get_parameter('web_port').value
        image_topic = self.get_parameter('image_topic').value
        cmd_topic = self.get_parameter('cmd_vel_topic').value
        self._max_linear = self.get_parameter('max_linear_speed').value
        self._max_angular = self.get_parameter('max_angular_speed').value
        self._cmd_timeout = self.get_parameter('cmd_timeout_sec').value
        image_compressed = self.get_parameter('image_compressed').value
        self._sim = bool(self.get_parameter('sim').value)
        self._map_dir = os.path.expanduser(self.get_parameter('map_dir').value)
        self._sim_map_name = self.get_parameter('sim_map_name').value

        self._frame_lock = threading.Lock()
        self._latest_jpeg: Optional[bytes] = None
        # Increments on every new frame so the MJPEG loop can skip re-sending
        # the identical cached JPEG when the camera runs below the stream rate.
        self._frame_seq = 0

        # Image capture (persistent, for detection-model training datasets).
        self._capture_dir = os.path.expanduser(
            self.get_parameter('capture_dir').value)
        os.makedirs(self._capture_dir, exist_ok=True)
        self._capture_lock = threading.Lock()
        self._capture_seq = 0
        # Seed the saved-count once so each capture doesn't re-scan the dir
        # (it is expected to grow large). Counter is maintained in memory after.
        try:
            self._capture_count = sum(
                1 for n in os.listdir(self._capture_dir)
                if n.lower().endswith('.jpg'))
        except OSError:
            self._capture_count = 0

        # YOLO class list — loaded/created in _load_or_init_classes().
        self._classes_path = os.path.join(self._capture_dir, 'classes.txt')
        self._classes_lock = threading.Lock()
        self._classes: list = []
        self._load_or_init_classes()

        self._cmd_lock = threading.Lock()
        self._last_cmd_time = 0.0
        # On command timeout, publish a short zero burst to brake and then go
        # SILENT instead of flooding zeros at 10 Hz forever (same idle-silence
        # pattern as cmd_vel_bridge in this package: a permanent zero stream on
        # cmd_vel interleaves with any other publisher on the topic, e.g.
        # base_approach driving the base during a grasp APPROACH).
        self._stop_burst_ticks = 3   # 0.3 s at the 10 Hz watchdog rate
        self._stop_burst = 0         # armed while commands are fresh

        self._map_lock = threading.Lock()
        self._latest_map_png: Optional[bytes] = None
        self._map_meta: dict = {}
        self._map_origin = (0.0, 0.0)   # (x, y) of map cell (0,0) in the map frame

        # Navigation stack lifecycle (a launched ros2 process) + goal action client.
        self._nav_lock = threading.Lock()
        self._nav_proc: Optional[subprocess.Popen] = None
        self._nav_mode: Optional[str] = None   # 'mapping' | 'navigation' | None
        self._nav_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        self._initpose_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)

        # Web map-click mission (M6): a "fetch" click sends a RunMission goal to
        # the mission_coordinator action server and tracks live status.
        self._mission_available = _MISSION_AVAILABLE
        self._mission_lock = threading.Lock()
        self._mission_status: str = 'IDLE'   # cached /mission/status + feedback/result
        self._mission_goal_handle = None     # active goal handle (for cancel)
        self._mission_active = False         # a goal is in flight
        if self._mission_available:
            self._mission_client = ActionClient(
                self, _RunMissionAction, '/mission_coordinator/run_mission')
            # The coordinator publishes /mission/status latched (transient-local),
            # so subscribe transient-local to catch the last value on connect.
            status_qos = QoSProfile(
                depth=1,
                history=HistoryPolicy.KEEP_LAST,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            self.create_subscription(
                String, '/mission/status', self._on_mission_status, status_qos)
            self.get_logger().info(
                'RunMission action client created on '
                '/mission_coordinator/run_mission')
        else:
            self._mission_client = None
            self.get_logger().warn(
                'jetank_mission not found — Fetch mission will be disabled')
        self._deposit_file = os.path.expanduser(
            self.get_parameter('deposit_file').value)
        self._deposit_lock = threading.Lock()
        self._deposit_pose: Optional[tuple] = None   # (x, y) in the map frame
        self._load_deposit_pose()
        # AMCL's estimated robot pose (for the web map arrow + localization status).
        self._amcl_lock = threading.Lock()
        self._amcl_pose = None
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose',
                                 self._on_amcl_pose, 10)

        # Latest detections (from jetank_detection) for the live overlay.
        # Boxes are stored in source-image pixel coords; the browser normalizes
        # against the camera frame's natural size.
        self._det_lock = threading.Lock()
        self._latest_dets: list = []
        self._latest_dets_mono = 0.0

        # GraspObject action client — guarded so the node starts if jetank_manipulation absent.
        self._grasp_available = _GRASP_AVAILABLE
        self._grasp_lock = threading.Lock()
        self._grasp_running = False
        self._grasp_stage: str = ''
        self._grasp_last_success: Optional[bool] = None
        self._grasp_last_message: str = ''
        self._grasp_started_at: float = 0.0  # monotonic; watchdog clears stuck runs
        if self._grasp_available:
            self._grasp_client = ActionClient(
                self, _GraspObjectAction, '/grasp_object')
            self.get_logger().info(
                'GraspObject action client created on /grasp_object')
        else:
            self._grasp_client = None
            self.get_logger().warn(
                'jetank_manipulation not found — Grab button will be disabled')

        self._cmd_vel_pub = self.create_publisher(Twist, cmd_topic, 10)
        # Camera subscription is created lazily, gated by the number of
        # active stream viewers (see _add_stream_viewer / _remove_stream_
        # viewer). With 0 browser viewers the always-on subscription used to
        # deserialize ~30 fps of frames purely to discard them, burning
        # ~5-6% of an Orin Nano core 24/7 in the common (no-viewer) case.
        self._image_msg_type = CompressedImage if image_compressed else Image
        self._image_cb = self._on_image if image_compressed else self._on_raw_image
        self._image_topic = image_topic
        self._image_sub_lock = threading.Lock()
        self._image_sub = None
        self._image_viewers = 0
        self.create_subscription(OccupancyGrid, '/map', self._on_map, 1)
        detections_topic = self.get_parameter('detections_topic').value
        self.create_subscription(Detection2DArray, detections_topic,
                                 self._on_detections, 10)

        # Watchdog: stop robot if commands stop arriving
        self.create_timer(0.1, self._watchdog_cb)

        self.get_logger().info(
            f'Web control: http://<jetson-ip>:{self._port}  '
            f'| camera: {image_topic}  | cmd_vel: {cmd_topic}'
        )

    # ---- classes.txt helpers ---------------------------------------------

    def _load_or_init_classes(self) -> None:
        """Load classes.txt if present; otherwise seed from the param and write it."""
        if os.path.isfile(self._classes_path):
            try:
                with open(self._classes_path, 'r', encoding='utf-8') as f:
                    self._classes = [ln.strip() for ln in f if ln.strip()]
                return
            except OSError:
                pass
        # Seed from param
        param_val = self.get_parameter('capture_classes').value
        self._classes = list(param_val) if param_val else ['object']
        self._write_classes_file()

    def _write_classes_file(self) -> None:
        """Persist self._classes to classes.txt (caller holds _classes_lock or is __init__)."""
        try:
            with open(self._classes_path, 'w', encoding='utf-8') as f:
                for name in self._classes:
                    f.write(name + '\n')
        except OSError as exc:
            try:
                self.get_logger().warn(f'Could not write classes.txt: {exc}')
            except Exception:
                pass

    # ---- callbacks --------------------------------------------------------

    def _on_image(self, msg: CompressedImage):
        with self._frame_lock:
            self._latest_jpeg = bytes(msg.data)
            self._frame_seq += 1

    def _on_raw_image(self, msg: Image):
        # Encode a raw sensor_msgs/Image to JPEG (simulation path). Supports the
        # common rgb8/bgr8 encodings; falls back to mono treatment otherwise.
        if not _PIL_AVAILABLE:
            return
        h, w = msg.height, msg.width
        if h == 0 or w == 0:
            return
        # msg.data supports the buffer protocol, so frombuffer wraps it
        # zero-copy — no bytes() materialisation of the full raw frame. The
        # array is only read within this callback (PIL copies on save).
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        enc = msg.encoding.lower()
        try:
            if enc in ('rgb8', 'bgr8'):
                arr = arr.reshape((h, w, 3))
                if enc == 'bgr8':
                    arr = arr[:, :, ::-1]
                img = _PILImage.fromarray(arr, 'RGB')
            elif enc in ('mono8', '8uc1'):
                img = _PILImage.fromarray(arr.reshape((h, w)), 'L')
            elif enc in ('rgba8', 'bgra8'):
                arr = arr.reshape((h, w, 4))[:, :, :3]
                if enc == 'bgra8':
                    arr = arr[:, :, ::-1]
                img = _PILImage.fromarray(np.ascontiguousarray(arr), 'RGB')
            else:  # best-effort: assume 3-channel
                img = _PILImage.fromarray(arr.reshape((h, w, 3)), 'RGB')
        except ValueError:
            return
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=80)
        with self._frame_lock:
            self._latest_jpeg = buf.getvalue()
            self._frame_seq += 1

    def _on_map(self, msg: OccupancyGrid):
        if not _PIL_AVAILABLE:
            return
        w, h = msg.info.width, msg.info.height
        if w == 0 or h == 0:
            return
        data = np.frombuffer(msg.data, dtype=np.int8).reshape((h, w))
        gray = np.full((h, w), 128, dtype=np.uint8)      # unknown = mid-gray
        gray[data == 0] = 220                             # free = light
        gray[data > 0] = 20                                # occupied = dark
        img = _PILImage.fromarray(np.flipud(gray), 'L')
        buf = io.BytesIO()
        img.save(buf, format='PNG', optimize=False, compress_level=1)
        ox = float(msg.info.origin.position.x)
        oy = float(msg.info.origin.position.y)
        with self._map_lock:
            self._latest_map_png = buf.getvalue()
            self._map_origin = (ox, oy)
            self._map_meta = {
                'resolution': round(float(msg.info.resolution), 4),
                'width': w,
                'height': h,
                'origin_x': round(ox, 4),
                'origin_y': round(oy, 4),
            }

    def _watchdog_cb(self):
        with self._cmd_lock:
            age = time.monotonic() - self._last_cmd_time
        if age <= self._cmd_timeout:
            self._stop_burst = self._stop_burst_ticks   # re-arm while driving
            return
        # Command stream stopped: brake with a brief zero burst, then stay
        # silent until a new command arrives (see __init__).
        if self._stop_burst > 0:
            self._stop_burst -= 1
            self._publish_twist(0.0, 0.0)

    # ---- public API for web handlers -------------------------------------

    def get_frame(self) -> Optional[bytes]:
        with self._frame_lock:
            return self._latest_jpeg

    def get_frame_and_seq(self) -> tuple:
        """Return ``(jpeg_or_None, seq)``; ``seq`` changes only on new frames."""
        with self._frame_lock:
            return self._latest_jpeg, self._frame_seq

    # ---- lazy camera subscription (stream-client refcount) ---------------
    #
    # The CompressedImage/Image subscription is only alive while at least one
    # consumer (an MJPEG stream client, or a one-shot /capture) needs it.
    # create_subscription()/destroy_subscription() are safe to call from this
    # (aiohttp) thread while rclpy.spin() runs on the background ROS thread:
    # they mutate the node's own subscription list and wake the executor's
    # wait set via its guard condition rather than touching it directly.

    def _add_stream_viewer(self) -> None:
        """Subscribe to the camera topic on the 0->1 viewer transition."""
        with self._image_sub_lock:
            self._image_viewers += 1
            if self._image_sub is None:
                self._image_sub = self.create_subscription(
                    self._image_msg_type, self._image_topic, self._image_cb, 10)

    def _remove_stream_viewer(self) -> None:
        """Unsubscribe on the 1->0 viewer transition.

        Also drops the cached frame: otherwise the next 0->1 viewer (a
        fresh MJPEG client, or a /capture that borrows the subscription)
        would see the stale frame from the previous session before the
        new subscription delivers anything.
        """
        with self._image_sub_lock:
            self._image_viewers = max(0, self._image_viewers - 1)
            if self._image_viewers == 0 and self._image_sub is not None:
                sub = self._image_sub
                self._image_sub = None
                self.destroy_subscription(sub)
                with self._frame_lock:
                    self._latest_jpeg = None

    async def save_capture_async(self, timeout: float = 1.0):
        """Like ``save_capture()``, but subscribes briefly first if no
        stream client is already open, so a capture works even with 0
        MJPEG viewers connected (the camera subscription is otherwise
        gated off in that case). Awaits without blocking the event loop.

        When borrowing the subscription, waits for a genuinely fresh
        frame (a new seq, which only ever accompanies a non-None frame)
        rather than falling back to whatever stale frame happened to be
        cached from a previous viewer session. If the deadline passes
        with no fresh frame, returns an explicit error instead of saving
        the stale one.
        """
        borrowed = False
        with self._image_sub_lock:
            if self._image_viewers == 0:
                borrowed = True
                self._image_viewers += 1
                if self._image_sub is None:
                    self._image_sub = self.create_subscription(
                        self._image_msg_type, self._image_topic, self._image_cb, 10)
        try:
            if borrowed:
                _, seq_before = self.get_frame_and_seq()
                deadline = time.monotonic() + timeout
                got_fresh = False
                while time.monotonic() < deadline:
                    frame_now, seq_now = self.get_frame_and_seq()
                    if frame_now is not None and seq_now != seq_before:
                        got_fresh = True
                        break
                    await asyncio.sleep(0.02)
                if not got_fresh:
                    return False, 'no camera frame received before capture timeout'
            return self.save_capture()
        finally:
            if borrowed:
                self._remove_stream_viewer()

    def save_capture(self):
        """Persist the current full-res frame as a JPEG in capture_dir.

        Reuses the streamed frame (already JPEG: passthrough on the robot's
        CompressedImage path, re-encoded on the sim raw-Image path), so no
        decode/re-encode is needed here. Returns (ok, info|error_str).
        """
        frame = self.get_frame()
        if frame is None:
            return False, 'no camera frame available yet'
        with self._capture_lock:
            # Timestamp-only flat filename; seq disambiguates same-second bursts.
            # Exclusive-create ('xb') guarantees we never overwrite an existing
            # file even across process restarts (where _capture_seq resets to 0):
            # on collision we bump the seq and retry rather than truncate data.
            ts = time.strftime('%Y%m%dT%H%M%S')
            for _ in range(100000):
                self._capture_seq += 1
                fname = f'{ts}_{self._capture_seq:04d}.jpg'
                path = os.path.join(self._capture_dir, fname)
                try:
                    with open(path, 'xb') as f:
                        f.write(frame)
                    break
                except FileExistsError:
                    continue
                except OSError as e:
                    return False, f'write failed: {e}'
            else:
                return False, 'could not allocate a unique capture filename'
            self._capture_count += 1
            count = self._capture_count
        self.get_logger().info(
            f'captured {fname} ({len(frame)} bytes) -> {self._capture_dir}')
        return True, {'filename': fname, 'count': count, 'dir': self._capture_dir}

    # ---- label / capture listing API ------------------------------------

    def list_captures(self) -> dict:
        """List *.jpg files in capture_dir (newest first) with label status."""
        try:
            names = sorted(
                [n for n in os.listdir(self._capture_dir) if n.lower().endswith('.jpg')],
                reverse=True,
            )
        except OSError:
            names = []
        with self._classes_lock:
            n_classes = len(self._classes)
            classes = list(self._classes)
        images = []
        for name in names:
            txt_path = os.path.join(
                self._capture_dir, os.path.splitext(name)[0] + '.txt')
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    txt = f.read()
            except OSError:
                txt = ''
            boxes = _yolo_parse(txt, n_classes)
            images.append({
                'name': name,
                'labelled': bool(txt.strip()),
                'n_boxes': len(boxes),
            })
        return {'images': images, 'classes': classes}

    def read_capture_image(self, name: str) -> Optional[bytes]:
        """Return raw JPEG bytes for *name*, or None if invalid/missing."""
        safe = _safe_capture_name(name)
        if safe is None:
            return None
        path = os.path.join(self._capture_dir, safe)
        try:
            with open(path, 'rb') as f:
                return f.read()
        except OSError:
            return None

    def read_labels(self, name: str) -> Optional[list]:
        """Return parsed boxes for *name*, or None if the .jpg doesn't exist."""
        safe = _safe_capture_name(name)
        if safe is None:
            return None
        jpg_path = os.path.join(self._capture_dir, safe)
        if not os.path.isfile(jpg_path):
            return None
        txt_path = os.path.join(
            self._capture_dir, os.path.splitext(safe)[0] + '.txt')
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                txt = f.read()
        except OSError:
            txt = ''
        with self._classes_lock:
            n_classes = len(self._classes)
        return _yolo_parse(txt, n_classes)

    def autolabel(self, name: str) -> tuple:
        """Propose rough boxes for capture *name* via CV colour-blob detection.

        Returns ``(True, {'boxes': [...], 'count': n})`` where each box is
        ``{'cls', 'cx', 'cy', 'w', 'h'}`` with ``cls`` set to the index of the
        ``sock`` class if it exists, else ``0``. Returns ``(False, reason_str)``
        on a bad name, missing image, or decode failure. Boxes are rough and
        meant for human review before saving.
        """
        safe = _safe_capture_name(name)
        if safe is None:
            return False, 'invalid filename'
        data = self.read_capture_image(safe)
        if data is None:
            return False, 'image not found'
        try:
            import cv2
            arr = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except ImportError:
            return False, 'cv2 not available for auto-annotation'
        if img is None:
            return False, 'image decode failed'
        raw = rough_boxes_from_bgr(img)
        with self._classes_lock:
            cls = self._classes.index('sock') if 'sock' in self._classes else 0
        boxes = [{'cls': cls, **b} for b in raw]
        return True, {'boxes': boxes, 'count': len(boxes)}

    def write_labels(self, name: str, boxes: list) -> tuple:
        """Validate and write a YOLO sidecar for *name*.

        *boxes* must be a list of dicts with keys ``cls`` (int, in class range)
        and ``cx``, ``cy``, ``w``, ``h`` (float, in [0, 1]).
        Returns ``(True, 'ok')`` or ``(False, reason_str)``.
        """
        safe = _safe_capture_name(name)
        if safe is None:
            return False, 'invalid filename'
        jpg_path = os.path.join(self._capture_dir, safe)
        if not os.path.isfile(jpg_path):
            return False, 'image not found'
        if not isinstance(boxes, list):
            return False, 'boxes must be a list'
        with self._classes_lock:
            n_classes = len(self._classes)
        validated = []
        for i, b in enumerate(boxes):
            if not isinstance(b, dict):
                return False, f'box {i} is not a dict'
            try:
                cls = int(b['cls'])
                cx = float(b['cx'])
                cy = float(b['cy'])
                w = float(b['w'])
                h = float(b['h'])
            except (KeyError, TypeError, ValueError) as exc:
                return False, f'box {i} missing/invalid field: {exc}'
            if cls not in range(n_classes):
                return False, f'box {i}: cls {cls} out of range(0, {n_classes})'
            if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0
                    and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
                return False, f'box {i}: coords outside [0, 1]'
            validated.append({'cls': cls, 'cx': cx, 'cy': cy, 'w': w, 'h': h})
        txt_path = os.path.join(
            self._capture_dir, os.path.splitext(safe)[0] + '.txt')
        try:
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(_yolo_serialize(validated))
        except OSError as exc:
            return False, f'write failed: {exc}'
        return True, 'ok'

    def add_class(self, name: str) -> tuple:
        """Add a new detection class (thread-safe).

        Returns ``(True, {'classes': list, 'index': int})`` or
        ``(False, reason_str)``.
        """
        name = name.strip() if name else ''
        if not name:
            return False, 'class name must not be empty'
        if '\n' in name or '\r' in name:
            return False, 'class name must not contain newlines'
        with self._classes_lock:
            if name in self._classes:
                return True, {'classes': list(self._classes),
                              'index': self._classes.index(name)}
            self._classes.append(name)
            idx = len(self._classes) - 1
            classes_copy = list(self._classes)
            self._write_classes_file()
        return True, {'classes': classes_copy, 'index': idx}

    def get_map_png(self) -> Optional[bytes]:
        with self._map_lock:
            return self._latest_map_png

    def get_map_meta(self) -> dict:
        with self._map_lock:
            return dict(self._map_meta)

    def _on_detections(self, msg: Detection2DArray):
        """Cache the latest sock detections for the live web overlay.

        Boxes are kept in source-image pixel coords (bbox center + size); the
        browser normalizes them against the camera frame's natural size.
        """
        boxes = []
        for det in msg.detections:
            label, score = '', 0.0
            if det.results:
                hyp = det.results[0].hypothesis
                label, score = hyp.class_id, float(hyp.score)
            boxes.append({
                'cx': round(float(det.bbox.center.position.x), 2),
                'cy': round(float(det.bbox.center.position.y), 2),
                'w': round(float(det.bbox.size_x), 2),
                'h': round(float(det.bbox.size_y), 2),
                'label': label,
                'score': round(score, 3),
            })
        with self._det_lock:
            self._latest_dets = boxes
            self._latest_dets_mono = time.monotonic()

    # Detections older than this are considered stale (detector stopped /
    # on-demand finished) and are NOT served — otherwise the last box would
    # freeze on the overlay forever.
    DET_STALE_SEC = 1.0

    def get_detections(self) -> dict:
        """Latest detections + age in seconds.

        Returns empty boxes when no detector has ever published (`age=None`) or
        when the last message is stale (`fresh=False`), so the overlay clears
        instead of freezing on an old box.
        """
        with self._det_lock:
            boxes = list(self._latest_dets)
            stamp = self._latest_dets_mono
        if not stamp:
            return {'ok': True, 'boxes': [], 'age': None, 'fresh': False}
        age = time.monotonic() - stamp
        fresh = age <= self.DET_STALE_SEC
        return {'ok': True, 'boxes': boxes if fresh else [],
                'age': round(age, 3), 'fresh': fresh}

    # ---- grasp action client -----------------------------------------------

    def start_grasp(self, object_hint: str = '') -> tuple:
        """Send a GraspObject goal.  Returns (True, 'ok') or (False, reason_str)."""
        if not self._grasp_available or self._grasp_client is None:
            return False, 'jetank_manipulation not available'
        with self._grasp_lock:
            if self._grasp_running:
                return False, 'grasp already running'
            # Non-blocking readiness check only — wait_for_server() would block
            # the aiohttp event loop. The client discovers the server in the
            # background ROS executor thread, so server_is_ready() flips on its own.
            if not self._grasp_client.server_is_ready():
                return False, 'grasp action server not ready'
            goal = _GraspObjectAction.Goal()
            goal.object_hint = str(object_hint)
            self._grasp_running = True
            self._grasp_started_at = time.monotonic()
            self._grasp_stage = 'sending goal'
            self._grasp_last_success = None
            self._grasp_last_message = ''
        future = self._grasp_client.send_goal_async(
            goal, feedback_callback=self._on_grasp_feedback)
        future.add_done_callback(self._on_grasp_goal_response)
        self.get_logger().info('GraspObject goal sent')
        return True, 'ok'

    def _on_grasp_goal_response(self, future):
        try:
            handle = future.result()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f'GraspObject send failed: {exc}')
            with self._grasp_lock:
                self._grasp_running = False
                self._grasp_stage = ''
                self._grasp_last_success = False
                self._grasp_last_message = f'send failed: {exc}'
            return
        if not handle.accepted:
            self.get_logger().warn('GraspObject goal REJECTED')
            with self._grasp_lock:
                self._grasp_running = False
                self._grasp_stage = ''
                self._grasp_last_success = False
                self._grasp_last_message = 'goal rejected'
            return
        self.get_logger().info('GraspObject goal accepted')
        with self._grasp_lock:
            self._grasp_stage = 'accepted'
        result_future = handle.get_result_async()
        result_future.add_done_callback(self._on_grasp_result)

    def _on_grasp_feedback(self, feedback_msg):
        stage = feedback_msg.feedback.stage
        with self._grasp_lock:
            self._grasp_stage = stage
        self.get_logger().info(f'GraspObject feedback: {stage}')

    def _on_grasp_result(self, future):
        try:
            result = future.result().result
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f'GraspObject result failed: {exc}')
            with self._grasp_lock:
                self._grasp_running = False
                self._grasp_stage = ''
                self._grasp_last_success = False
                self._grasp_last_message = f'result error: {exc}'
            return
        with self._grasp_lock:
            self._grasp_running = False
            self._grasp_stage = ''
            self._grasp_last_success = bool(result.success)
            self._grasp_last_message = str(result.message)
        self.get_logger().info(
            f'GraspObject result: success={result.success} msg={result.message}')

    GRASP_MAX_RUN_SEC = 60.0  # watchdog: clear a run whose callbacks never fired

    def grasp_status(self) -> dict:
        """Return a thread-safe snapshot of grasp state for the web handler."""
        with self._grasp_lock:
            # Watchdog: if a run exceeds the cap (e.g. the action server died
            # mid-goal and no result callback ever fires), clear it so the UI
            # button doesn't stay stuck-disabled forever.
            if (self._grasp_running
                    and time.monotonic() - self._grasp_started_at > self.GRASP_MAX_RUN_SEC):
                self._grasp_running = False
                self._grasp_stage = ''
                self._grasp_last_success = False
                self._grasp_last_message = 'timed out (no result from action server)'
            return {
                'available': self._grasp_available,
                'running': self._grasp_running,
                'stage': self._grasp_stage,
                'last_success': self._grasp_last_success,
                'last_message': self._grasp_last_message,
            }

    def _on_amcl_pose(self, msg: PoseWithCovarianceStamped):
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        cov = msg.pose.covariance
        with self._amcl_lock:
            self._amcl_pose = {
                'x': round(float(msg.pose.pose.position.x), 4),
                'y': round(float(msg.pose.pose.position.y), 4),
                'yaw': round(float(yaw), 4),
                # max of x/y position variance — small => localized/converged
                'cov': round(float(max(cov[0], cov[7])), 4),
            }

    def get_robot_pose(self) -> dict:
        with self._amcl_lock:
            if self._amcl_pose is None:
                return {'available': False}
            p = dict(self._amcl_pose)
        p['available'] = True
        p['converged'] = p['cov'] < 0.5
        return p

    # ---- navigation backend ----------------------------------------------

    def saved_map_yaml(self) -> str:
        return os.path.join(self._map_dir, self._sim_map_name + '.yaml')

    def has_saved_map(self) -> bool:
        return os.path.isfile(self.saved_map_yaml())

    def nav_status(self) -> dict:
        with self._nav_lock:
            proc, mode = self._nav_proc, self._nav_mode
        running = mode if (proc is not None and proc.poll() is None) else None
        return {
            'sim': self._sim,
            'running': running,
            'has_map': self.has_saved_map(),
            'have_live_map': bool(self.get_map_meta()),
        }

    # Nav node processes that must be gone before a new stack starts. Lingering
    # ones (from a previous mapping/navigation run) collide by name and make the
    # new bt_navigator flap active->inactive -> intermittent "robot won't move".
    _NAV_PROC_PATTERNS = (
        'controller_server', 'planner_server', 'bt_navigator', 'behavior_server',
        'smoother_server', 'velocity_smoother', 'waypoint_follower',
        'lifecycle_manager_navigation', 'async_slam_toolbox_node', 'amcl', 'map_server',
    )

    def _launch_nav(self, mode: str, launch_file: str, extra: list) -> None:
        self.stop_nav()
        # Belt-and-suspenders clean slate: kill any nav nodes the group-kill missed.
        for pat in self._NAV_PROC_PATTERNS:
            subprocess.run(['pkill', '-9', '-f', pat], capture_output=True)
        time.sleep(1.0)
        ust = 'true' if self._sim else 'false'
        cmd = ['ros2', 'launch', 'jetank_navigation', launch_file,
               f'use_sim_time:={ust}'] + extra
        logf = open(os.path.join('/tmp', f'jetank_nav_{mode}.log'), 'wb')
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                                start_new_session=True, env=os.environ)
        with self._nav_lock:
            self._nav_proc, self._nav_mode = proc, mode
        self.get_logger().info(f'nav stack started [{mode}]: {" ".join(cmd)}')

    def start_mapping(self) -> tuple:
        self._launch_nav('mapping', 'slam.launch.py', [])
        return True, 'mapping started (slam_toolbox only — drive with the joystick, then Save Map)'

    def start_navigation(self) -> tuple:
        if not self.has_saved_map():
            return False, 'no saved map — run mapping and Save Map first'
        self._launch_nav('navigation', 'nav2_bringup.launch.py',
                         [f'map:={self.saved_map_yaml()}'])
        # AMCL needs an initial pose or it never publishes map->odom (nav is then
        # dead). Seed /initialpose a few times once AMCL has come up.
        threading.Thread(target=self._seed_initial_pose, daemon=True).start()
        return True, f'navigation started on {self.saved_map_yaml()}'

    def _seed_initial_pose(self):
        x = float(self.get_parameter('initial_pose_x').value)
        y = float(self.get_parameter('initial_pose_y').value)
        yaw = float(self.get_parameter('initial_pose_yaw').value)
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'map'
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.pose.orientation.w = math.cos(yaw / 2.0)
        cov = [0.0] * 36
        # Phase 2 (localization tuning): start pose is well-known (robot spawns
        # at the seeded pose), so seed a tight covariance — a loose seed let the
        # filter wander into a wrong-heading basin.
        cov[0] = cov[7] = 0.04       # x, y variance
        cov[35] = 0.02               # yaw variance
        msg.pose.covariance = cov
        # Keep re-publishing until AMCL actually converges near the seed. A single
        # early publish is lost because AMCL is not subscribed yet during bringup
        # under load; this loop guarantees one lands once AMCL is ready.
        for _ in range(25):
            msg.header.stamp = self.get_clock().now().to_msg()
            self._initpose_pub.publish(msg)
            time.sleep(1.2)
            with self._amcl_lock:
                ap = dict(self._amcl_pose) if self._amcl_pose else None
            if ap and abs(ap['x'] - x) < 0.5 and abs(ap['y'] - y) < 0.5:
                self.get_logger().info(
                    f'AMCL accepted initial pose ({x:.2f}, {y:.2f}, {yaw:.2f})')
                return
        self.get_logger().warn('AMCL did not converge to the seeded initial pose')

    def stop_nav(self) -> Optional[str]:
        with self._amcl_lock:
            self._amcl_pose = None    # re-determine pose on the next navigation
        with self._nav_lock:
            proc, mode = self._nav_proc, self._nav_mode
            self._nav_proc, self._nav_mode = None, None
        if proc is not None and proc.poll() is None:
            # Kill the whole launch process group and WAIT for it to die, so a
            # following start_* gets a clean slate (otherwise the old
            # lifecycle_manager/amcl linger and fight the new stack).
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGINT)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                proc.wait(timeout=8.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    proc.wait(timeout=3.0)
                except (ProcessLookupError, PermissionError, subprocess.TimeoutExpired):
                    pass
        return mode

    def save_map(self) -> tuple:
        os.makedirs(self._map_dir, exist_ok=True)
        path = os.path.join(self._map_dir, self._sim_map_name)
        # In sim, map_saver_cli must use the sim clock or it times out waiting on
        # /map ("Failed to spin map subscription"). Also give it a longer window.
        cmd = ['ros2', 'run', 'nav2_map_server', 'map_saver_cli', '-f', path,
               '--ros-args',
               '-p', f'use_sim_time:={str(self._sim).lower()}',
               '-p', 'save_map_timeout:=10.0']
        try:
            res = subprocess.run(
                cmd, capture_output=True, text=True, timeout=20, env=os.environ)
        except subprocess.TimeoutExpired:
            return False, 'map_saver_cli timed out'
        except FileNotFoundError:
            return False, 'ros2 command not found'
        if res.returncode == 0:
            return True, path + '.yaml'
        return False, (res.stderr.strip() or res.stdout.strip() or 'map_saver failed')

    def navigate_to_pixel(self, ix: int, iy: int) -> tuple:
        """Convert a /map.png pixel click to a map pose and send a goal.

        The rendered PNG is vertically flipped, so the row is un-flipped before
        converting to a map-frame pose for the NavigateToPose action.
        """
        with self._map_lock:
            meta = dict(self._map_meta)
            ox, oy = self._map_origin
        if not meta:
            return False, 'no map yet'
        wx, wy = map_pixel_to_world(
            ix, iy, meta['resolution'], meta['width'], meta['height'], ox, oy)

        if not self._nav_client.server_is_ready():
            if not self._nav_client.wait_for_server(timeout_sec=2.0):
                return False, 'nav2 not ready — start mapping or navigation first'

        ps = PoseStamped()
        ps.header.frame_id = 'map'
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = wx
        ps.pose.position.y = wy
        ps.pose.orientation.w = 1.0
        goal = NavigateToPose.Goal()
        goal.pose = ps
        self._nav_client.send_goal_async(goal).add_done_callback(self._on_goal_response)
        self.get_logger().info(f'NavigateToPose goal -> map ({wx:.2f}, {wy:.2f})')
        return True, {'x': round(wx, 3), 'y': round(wy, 3)}

    def _on_goal_response(self, future):
        try:
            handle = future.result()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f'NavigateToPose send failed: {exc}')
            return
        if not handle.accepted:
            self.get_logger().warn('NavigateToPose goal REJECTED')
            return
        self.get_logger().info('NavigateToPose goal accepted')

    # ---- web map-click mission (M1) ---------------------------------------
    def _pixel_to_world_locked(self, ix: int, iy: int):
        """Convert a pixel to a map-frame ``(wx, wy)`` or ``None`` if no map.

        Reads ``_map_meta``/``_map_origin`` under ``_map_lock`` and delegates to
        the pure :func:`map_pixel_to_world` helper (same math as
        :meth:`navigate_to_pixel`).
        """
        with self._map_lock:
            meta = dict(self._map_meta)
            ox, oy = self._map_origin
        if not meta:
            return None
        return map_pixel_to_world(
            ix, iy, meta['resolution'], meta['width'], meta['height'], ox, oy)

    def publish_mission_goal_from_pixel(self, ix: int, iy: int) -> tuple:
        """Convert a map pixel to a world point and START a fetch mission there.

        Builds the ``site`` PoseStamped (map frame) and sends a RunMission
        goal to the mission_coordinator action server (search_timeout=0 ->
        coordinator default). Tracks the goal handle so it can be cancelled.

        Returns ``(True, {'x','y','status'})`` where ``status`` is
        ``'mission_started'`` (goal sent) or ``'mission_unavailable'`` (the
        action server isn't up / jetank_mission absent). ``(False, reason)`` if
        no map is available yet.
        """
        world = self._pixel_to_world_locked(ix, iy)
        if world is None:
            return False, 'no map yet'
        wx, wy = world
        ps = PoseStamped()
        ps.header.frame_id = 'map'
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = float(wx)
        ps.pose.position.y = float(wy)
        ps.pose.orientation.w = 1.0
        self.get_logger().info(f'mission goal -> map ({wx:.2f}, {wy:.2f})')

        info = {'x': round(wx, 3), 'y': round(wy, 3)}
        # Non-blocking readiness check only — wait_for_server() would block the
        # ROS executor thread. The client discovers the server in the background.
        if (not self._mission_available or self._mission_client is None
                or not self._mission_client.server_is_ready()):
            info['status'] = 'mission_unavailable'
            return True, info

        goal = _RunMissionAction.Goal()
        goal.site = ps
        goal.search_timeout = 0.0   # 0 -> coordinator default
        with self._mission_lock:
            self._mission_active = True
            self._mission_status = 'NAVIGATE_TO_SITE'
        future = self._mission_client.send_goal_async(
            goal, feedback_callback=self._on_mission_feedback)
        future.add_done_callback(self._on_mission_goal_response)
        self.get_logger().info('RunMission goal sent')
        info['status'] = 'mission_started'
        return True, info

    # ---- RunMission action client callbacks -------------------------------

    def _on_mission_status(self, msg: String):
        """Cache the latest latched /mission/status from the coordinator."""
        with self._mission_lock:
            self._mission_status = str(msg.data)

    def _on_mission_feedback(self, feedback_msg):
        state = feedback_msg.feedback.state
        with self._mission_lock:
            self._mission_status = str(state)
        self.get_logger().info(f'RunMission feedback: {state}')

    def _on_mission_goal_response(self, future):
        try:
            handle = future.result()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f'RunMission send failed: {exc}')
            with self._mission_lock:
                self._mission_active = False
                self._mission_goal_handle = None
                self._mission_status = f'FAILED — send failed: {exc}'
            return
        if not handle.accepted:
            self.get_logger().warn('RunMission goal REJECTED')
            with self._mission_lock:
                self._mission_active = False
                self._mission_goal_handle = None
                self._mission_status = 'FAILED — goal rejected'
            return
        self.get_logger().info('RunMission goal accepted')
        with self._mission_lock:
            self._mission_goal_handle = handle
        result_future = handle.get_result_async()
        result_future.add_done_callback(self._on_mission_result)

    def _on_mission_result(self, future):
        try:
            result = future.result().result
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f'RunMission result failed: {exc}')
            with self._mission_lock:
                self._mission_active = False
                self._mission_goal_handle = None
                self._mission_status = f'FAILED — result error: {exc}'
            return
        # The coordinator publishes DONE/FAILED via /mission/status + feedback;
        # only fold the result in if the cache hasn't already gone terminal so we
        # keep the coordinator's richer text (e.g. the FAILED reason).
        with self._mission_lock:
            self._mission_active = False
            self._mission_goal_handle = None
            if not is_terminal_mission_status(self._mission_status):
                self._mission_status = (
                    'DONE' if result.success
                    else f'FAILED — {result.outcome}')
        self.get_logger().info(
            f'RunMission result: success={result.success} '
            f'outcome={result.outcome}')

    def mission_status(self) -> dict:
        """Thread-safe snapshot of the mission status for the web handler."""
        with self._mission_lock:
            return {'status': self._mission_status,
                    'active': self._mission_active}

    def cancel_mission(self) -> tuple:
        """Cancel the active RunMission goal (no-op if none). (ok, message)."""
        with self._mission_lock:
            handle = self._mission_goal_handle
        if handle is None:
            return True, 'no active mission'
        try:
            handle.cancel_goal_async()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f'RunMission cancel failed: {exc}')
            return False, f'cancel failed: {exc}'
        self.get_logger().info('RunMission cancel requested')
        return True, 'cancel requested'

    def set_deposit_from_pixel(self, ix: int, iy: int) -> tuple:
        """Store + persist a deposit pose from a map pixel click.

        Returns ``(True, {'x','y'})`` on success, ``(False, reason)`` if no map.
        """
        world = self._pixel_to_world_locked(ix, iy)
        if world is None:
            return False, 'no map yet'
        wx, wy = world
        with self._deposit_lock:
            self._deposit_pose = (float(wx), float(wy))
            self._save_deposit_pose_locked()
        self.get_logger().info(f'deposit pose set -> map ({wx:.2f}, {wy:.2f})')
        return True, {'x': round(wx, 3), 'y': round(wy, 3)}

    def get_deposit_pose(self) -> Optional[dict]:
        """Return the stored deposit pose as ``{'x','y'}`` or ``None`` if unset."""
        with self._deposit_lock:
            if self._deposit_pose is None:
                return None
            x, y = self._deposit_pose
        return {'x': round(x, 3), 'y': round(y, 3)}

    def _load_deposit_pose(self):
        """Load the persisted deposit pose on start (best-effort, never raises)."""
        try:
            with open(self._deposit_file, 'r', encoding='utf-8') as fh:
                parsed = deposit_parse(fh.read())
        except (OSError, ValueError):
            parsed = None
        if parsed is not None:
            self._deposit_pose = parsed
            self.get_logger().info(
                f'loaded deposit pose ({parsed[0]:.2f}, {parsed[1]:.2f}) '
                f'from {self._deposit_file}')

    def _save_deposit_pose_locked(self):
        """Persist the current deposit pose to disk (call under _deposit_lock)."""
        if self._deposit_pose is None:
            return
        x, y = self._deposit_pose
        try:
            os.makedirs(os.path.dirname(self._deposit_file) or '.', exist_ok=True)
            with open(self._deposit_file, 'w', encoding='utf-8') as fh:
                fh.write(deposit_serialize(x, y))
        except OSError as exc:
            self.get_logger().warn(
                f'failed to persist deposit pose to {self._deposit_file}: {exc}')

    def apply_cmd(self, linear_x: float, angular_z: float):
        lx = max(-1.0, min(1.0, linear_x)) * self._max_linear
        az = max(-1.0, min(1.0, angular_z)) * self._max_angular
        self._publish_twist(lx, az)
        with self._cmd_lock:
            self._last_cmd_time = time.monotonic()

    def _publish_twist(self, linear_x: float, angular_z: float):
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.angular.z = float(angular_z)
        self._cmd_vel_pub.publish(msg)

    @property
    def port(self) -> int:
        return self._port


# ---------------------------------------------------------------------------
# aiohttp web handlers
# ---------------------------------------------------------------------------

async def handle_index(request: web.Request) -> web.Response:
    return web.Response(text=request.app['index_html'],
                        content_type='text/html')


async def handle_mjpeg(request: web.Request) -> web.StreamResponse:
    node: WebControlNode = request.app['node']
    boundary = b'--mjpegboundary'
    response = web.StreamResponse(headers={
        'Content-Type': 'multipart/x-mixed-replace; boundary=mjpegboundary',
        'Cache-Control': 'no-cache',
        'Connection': 'close',
    })
    await response.prepare(request)

    # Subscribe to the camera topic only while >=1 stream client is
    # connected (0->1 viewer transition); unsubscribe again in `finally`
    # so the node stops deserializing frames the moment the last viewer
    # disconnects, rather than running the subscription 24/7.
    node._add_stream_viewer()

    # Only send when a NEW frame has arrived: when the camera publishes below
    # the ~30 Hz poll rate (or stalls) the loop would otherwise re-transmit the
    # identical cached JPEG to every client, wasting CPU and WiFi bandwidth.
    last_seq = None
    try:
        while True:
            frame, seq = node.get_frame_and_seq()
            if frame is not None and seq != last_seq:
                last_seq = seq
                # Separate writes avoid a header+frame bytes concat (a full
                # frame copy) per part.
                await response.write(
                    boundary + b'\r\n'
                    b'Content-Type: image/jpeg\r\n'
                    b'Content-Length: ' + str(len(frame)).encode() + b'\r\n\r\n'
                )
                await response.write(frame)
                await response.write(b'\r\n')
            await asyncio.sleep(0.033)  # ~30 fps cap
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    finally:
        node._remove_stream_viewer()

    return response


async def handle_websocket(request: web.Request) -> web.WebSocketResponse:
    node: WebControlNode = request.app['node']
    ws = web.WebSocketResponse(heartbeat=5.0)
    await ws.prepare(request)

    node.get_logger().info(f'WebSocket client connected: {request.remote}')
    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    node.apply_cmd(
                        float(data.get('linear_x', 0.0)),
                        float(data.get('angular_z', 0.0)),
                    )
                except (json.JSONDecodeError, ValueError, TypeError):
                    pass
            elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSE):
                break
    finally:
        # Safety: stop robot when client disconnects
        node.apply_cmd(0.0, 0.0)
        node.get_logger().info(f'WebSocket client disconnected: {request.remote}')

    return ws


async def handle_detections(request: web.Request) -> web.Response:
    node: WebControlNode = request.app['node']
    return web.json_response(node.get_detections())


async def handle_map_png(request: web.Request) -> web.Response:
    node: WebControlNode = request.app['node']
    data = node.get_map_png()
    if data is None:
        return web.Response(status=404)
    return web.Response(body=data, content_type='image/png',
                        headers={'Cache-Control': 'no-cache, no-store'})


async def handle_map_meta(request: web.Request) -> web.Response:
    node: WebControlNode = request.app['node']
    return web.json_response(node.get_map_meta())


# The nav-lifecycle node methods below block for seconds (subprocess.run,
# proc.wait, pkill + time.sleep, wait_for_server) — run them in a worker
# thread via asyncio.to_thread so the single asyncio loop keeps servicing
# /ws teleop and /stream.mjpg while a nav stack starts/stops.
#
# Moving them off the loop also removed the loop's implicit serialization:
# two concurrent POSTs to /nav lifecycle endpoints could interleave (the
# second request's pkill killing the first's half-started stack, and the
# Popen handle being overwritten). app['nav_lock'] is therefore held across
# each FULL stop→pkill→sleep→Popen sequence to restore that guarantee.

async def handle_save_map(request: web.Request) -> web.Response:
    node: WebControlNode = request.app['node']
    ok, info = await asyncio.to_thread(node.save_map)
    if ok:
        return web.json_response({'status': 'ok', 'path': info})
    return web.json_response({'status': 'error', 'msg': info}, status=500)


async def handle_start_mapping(request: web.Request) -> web.Response:
    node: WebControlNode = request.app['node']
    async with request.app['nav_lock']:
        ok, msg = await asyncio.to_thread(node.start_mapping)
    return web.json_response({'status': 'ok' if ok else 'error', 'msg': msg},
                             status=200 if ok else 400)


async def handle_start_navigation(request: web.Request) -> web.Response:
    node: WebControlNode = request.app['node']
    async with request.app['nav_lock']:
        ok, msg = await asyncio.to_thread(node.start_navigation)
    return web.json_response({'status': 'ok' if ok else 'error', 'msg': msg},
                             status=200 if ok else 400)


async def handle_stop_nav(request: web.Request) -> web.Response:
    node: WebControlNode = request.app['node']
    async with request.app['nav_lock']:
        mode = await asyncio.to_thread(node.stop_nav)
    return web.json_response({'status': 'ok', 'stopped': mode})


async def handle_nav_status(request: web.Request) -> web.Response:
    node: WebControlNode = request.app['node']
    return web.json_response(node.nav_status())


async def handle_robot_pose(request: web.Request) -> web.Response:
    node: WebControlNode = request.app['node']
    return web.json_response(node.get_robot_pose())


async def handle_navigate(request: web.Request) -> web.Response:
    node: WebControlNode = request.app['node']
    try:
        data = await request.json()
        # navigate_to_pixel may block up to 2 s in wait_for_server(); hold the
        # nav lock so it never races a concurrent nav start/stop sequence.
        async with request.app['nav_lock']:
            ok, info = await asyncio.to_thread(
                node.navigate_to_pixel, int(data['x']), int(data['y']))
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
        return web.json_response({'status': 'error', 'msg': f'bad request: {exc}'}, status=400)
    if ok:
        return web.json_response({'status': 'ok', 'goal': info})
    return web.json_response({'status': 'error', 'msg': info}, status=409)


async def handle_mission_goal(request: web.Request) -> web.Response:
    node: WebControlNode = request.app['node']
    try:
        data = await request.json()
        ok, info = node.publish_mission_goal_from_pixel(
            int(data['ix']), int(data['iy']))
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
        return web.json_response({'status': 'error', 'msg': f'bad request: {exc}'}, status=400)
    if ok:
        # info: {'x','y','status': 'mission_started'|'mission_unavailable'}
        return web.json_response(info)
    return web.json_response({'status': 'error', 'msg': info}, status=404)


async def handle_mission_status(request: web.Request) -> web.Response:
    node: WebControlNode = request.app['node']
    return web.json_response(node.mission_status())


async def handle_mission_cancel(request: web.Request) -> web.Response:
    node: WebControlNode = request.app['node']
    ok, msg = node.cancel_mission()
    return web.json_response({'status': 'ok' if ok else 'error', 'msg': msg},
                             status=200 if ok else 500)


async def handle_set_deposit(request: web.Request) -> web.Response:
    node: WebControlNode = request.app['node']
    try:
        data = await request.json()
        ok, info = node.set_deposit_from_pixel(int(data['ix']), int(data['iy']))
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
        return web.json_response({'status': 'error', 'msg': f'bad request: {exc}'}, status=400)
    if ok:
        return web.json_response(info)
    return web.json_response({'status': 'error', 'msg': info}, status=404)


async def handle_get_deposit(request: web.Request) -> web.Response:
    node: WebControlNode = request.app['node']
    pose = node.get_deposit_pose()
    if pose is None:
        return web.json_response({'set': False})
    return web.json_response(pose)


async def handle_capture(request: web.Request) -> web.Response:
    node: WebControlNode = request.app['node']
    # Same viewer-gated subscription as the MJPEG stream: save_capture_async
    # subscribes briefly on its own when no stream client is already open,
    # so a capture still works even though the camera topic is otherwise
    # not subscribed with 0 viewers connected.
    ok, info = await node.save_capture_async()
    if ok:
        return web.json_response({'ok': True, **info})
    return web.json_response({'ok': False, 'error': info}, status=503)


async def handle_list_captures(request: web.Request) -> web.Response:
    node: WebControlNode = request.app['node']
    return web.json_response(node.list_captures())


async def handle_capture_img(request: web.Request) -> web.Response:
    node: WebControlNode = request.app['node']
    name = request.match_info['name']
    data = node.read_capture_image(name)
    if data is None:
        return web.json_response({'ok': False, 'error': 'not found'}, status=404)
    return web.Response(body=data, content_type='image/jpeg',
                        headers={'Cache-Control': 'no-cache, no-store'})


async def handle_get_labels(request: web.Request) -> web.Response:
    node: WebControlNode = request.app['node']
    name = request.match_info['name']
    boxes = node.read_labels(name)
    if boxes is None:
        return web.json_response({'ok': False, 'error': 'image not found'}, status=404)
    with node._classes_lock:
        classes = list(node._classes)
    return web.json_response({'ok': True, 'boxes': boxes, 'classes': classes})


async def handle_post_labels(request: web.Request) -> web.Response:
    node: WebControlNode = request.app['node']
    name = request.match_info['name']
    try:
        body = await request.json()
        boxes = body['boxes']
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        return web.json_response({'ok': False, 'error': f'bad request: {exc}'}, status=400)
    ok, info = node.write_labels(name, boxes)
    if ok:
        return web.json_response({'ok': True})
    return web.json_response({'ok': False, 'error': info}, status=400)


async def handle_autolabel(request: web.Request) -> web.Response:
    node: WebControlNode = request.app['node']
    name = request.match_info['name']
    ok, info = node.autolabel(name)
    if ok:
        return web.json_response({'ok': True, **info})
    status = 400
    if info == 'image not found':
        status = 404
    elif info.startswith('cv2 not available'):
        status = 503
    return web.json_response({'ok': False, 'error': info}, status=status)


async def handle_add_class(request: web.Request) -> web.Response:
    node: WebControlNode = request.app['node']
    try:
        body = await request.json()
        cls_name = body['name']
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        return web.json_response({'ok': False, 'error': f'bad request: {exc}'}, status=400)
    if not isinstance(cls_name, str):
        return web.json_response({'ok': False, 'error': 'name must be a string'}, status=400)
    ok, info = node.add_class(cls_name)
    if ok:
        return web.json_response({'ok': True, **info})
    return web.json_response({'ok': False, 'error': info}, status=400)


async def handle_grab(request: web.Request) -> web.Response:
    node: WebControlNode = request.app['node']
    try:
        body = await request.json()
        hint = body.get('object_hint', '') if isinstance(body, dict) else ''
    except Exception:  # noqa: BLE001
        hint = ''
    ok, reason = node.start_grasp(hint)
    if ok:
        return web.json_response({'ok': True, 'status': 'goal sent'})
    status = 503 if ('not available' in reason or 'not ready' in reason) else 409
    return web.json_response({'ok': False, 'status': reason}, status=status)


async def handle_grab_status(request: web.Request) -> web.Response:
    node: WebControlNode = request.app['node']
    return web.json_response(node.grasp_status())


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

def build_app(node: WebControlNode) -> web.Application:
    app = web.Application()
    app['node'] = node
    # Control page read once at startup; it is immutable for the server's
    # lifetime. app.js / style.css are served straight from the static dir.
    static_dir = resolve_static_dir()
    with open(os.path.join(static_dir, 'index.html'), encoding='utf-8') as fh:
        app['index_html'] = fh.read()
    # Serializes the blocking nav-lifecycle sequences (see comment above the
    # nav handlers) now that asyncio.to_thread runs them off the event loop.
    app['nav_lock'] = asyncio.Lock()
    app.router.add_get('/', handle_index)
    # follow_symlinks: colcon --symlink-install makes every installed asset a
    # symlink; aiohttp's default (False) would 404 all of them from install space.
    app.router.add_static('/static', static_dir, follow_symlinks=True)
    app.router.add_get('/stream.mjpg', handle_mjpeg)
    app.router.add_get('/ws', handle_websocket)
    app.router.add_get('/detections/latest', handle_detections)
    app.router.add_get('/map.png', handle_map_png)
    app.router.add_get('/map_meta', handle_map_meta)
    app.router.add_post('/save_map', handle_save_map)
    app.router.add_post('/start_mapping', handle_start_mapping)
    app.router.add_post('/start_navigation', handle_start_navigation)
    app.router.add_post('/stop_nav', handle_stop_nav)
    app.router.add_get('/nav_status', handle_nav_status)
    app.router.add_get('/robot_pose', handle_robot_pose)
    app.router.add_post('/navigate', handle_navigate)
    app.router.add_post('/mission/goal', handle_mission_goal)
    app.router.add_get('/mission/status', handle_mission_status)
    app.router.add_post('/mission/cancel', handle_mission_cancel)
    app.router.add_post('/mission/deposit', handle_set_deposit)
    app.router.add_get('/mission/deposit', handle_get_deposit)
    app.router.add_post('/capture', handle_capture)
    # Label / capture endpoints
    app.router.add_get('/captures', handle_list_captures)
    app.router.add_get('/captures/img/{name}', handle_capture_img)
    app.router.add_get('/captures/labels/{name}', handle_get_labels)
    app.router.add_post('/captures/labels/{name}', handle_post_labels)
    app.router.add_post('/captures/autolabel/{name}', handle_autolabel)
    app.router.add_post('/captures/classes', handle_add_class)
    app.router.add_post('/grab', handle_grab)
    app.router.add_get('/grab/status', handle_grab_status)
    return app


async def run_server(node: WebControlNode):
    app = build_app(node)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', node.port)
    await site.start()
    node.get_logger().info(f'Web server running on port {node.port}')
    try:
        await asyncio.Event().wait()  # run forever
    finally:
        await runner.cleanup()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = WebControlNode()

    # Spin ROS2 in a background thread so aiohttp owns the main event loop
    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    try:
        asyncio.run(run_server(node))
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_nav()
        node.destroy_node()
        rclpy.shutdown()
        ros_thread.join(timeout=2.0)


if __name__ == '__main__':
    main()
