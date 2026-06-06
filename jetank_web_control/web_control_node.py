#!/usr/bin/env python3
"""
JeTank web control node.

Runs an HTTP/WebSocket server on the Jetson that lets a browser on any
laptop drive the robot and watch the left camera stream without installing
any ROS tooling on the laptop.

Endpoints:
  GET  /            - control page (HTML)
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
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import CompressedImage, Image
from nav2_msgs.action import NavigateToPose
from vision_msgs.msg import Detection2DArray

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
    """Return *name* if it is a safe bare filename (no path separators, no ..).

    Accepts filenames matching ``^[A-Za-z0-9._-]+\\.jpg$`` (case-insensitive
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
    """Serialize a list of box dicts to YOLO-format text.

    Returns an empty string for an empty list.  Each line ends with ``\\n``.
    Coords are formatted to 6 decimal places.
    """
    lines = []
    for b in boxes:
        lines.append(
            f"{b['cls']} {b['cx']:.6f} {b['cy']:.6f} {b['w']:.6f} {b['h']:.6f}\n"
        )
    return ''.join(lines)


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
# HTML/JS controller page (served inline, no separate static files needed)
# ---------------------------------------------------------------------------
_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,user-scalable=no">
<title>JeTank Controller</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0;-webkit-tap-highlight-color:transparent}
  body{background:#0d1117;color:#e0e0e0;font-family:monospace;
       height:100dvh;display:flex;flex-direction:column;overflow:hidden;
       touch-action:none}
  header{background:#161b22;padding:8px 16px;display:flex;align-items:center;
         gap:12px;border-bottom:1px solid #30363d;flex-shrink:0}
  header h1{font-size:1rem;color:#58a6ff}
  .dot-ok{color:#3fb950}.dot-err{color:#f85149}
  .badge{font-size:.72rem;padding:2px 7px;border-radius:10px;background:#21262d;white-space:nowrap}
  #ctrl-mode{margin-left:auto;font-size:.7rem;color:#8b949e}

  /* ---- desktop layout ---------------------------------------- */
  main{display:flex;flex:1;overflow:hidden}
  .cam{flex:1;display:flex;align-items:center;justify-content:center;
       background:#010409;position:relative;overflow:hidden}
  #cam-img{max-width:100%;max-height:100%;object-fit:contain;display:block}
  .cam-det{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
  .cam-overlay{position:absolute;top:8px;left:8px;font-size:.68rem;color:#58a6ff;
               background:rgba(0,0,0,.55);padding:2px 6px;border-radius:4px;
               pointer-events:none}
  .sidebar{width:250px;background:#161b22;border-left:1px solid #30363d;
           padding:14px;display:flex;flex-direction:column;gap:12px;overflow-y:auto}
  .section{display:flex;flex-direction:column;gap:5px}
  .stitle{font-size:.62rem;text-transform:uppercase;letter-spacing:1px;
          color:#8b949e;border-bottom:1px solid #30363d;padding-bottom:3px}
  .vel-row{display:flex;justify-content:space-between;font-size:.72rem;color:#8b949e}
  .bar-track{background:#21262d;border-radius:3px;height:5px;margin-top:2px}
  .bar-fill{height:100%;border-radius:3px;transition:width .1s,background .1s}
  .bar-fwd{background:#3fb950}.bar-bwd{background:#f85149}
  .bar-cw{background:#d29922}.bar-ccw{background:#58a6ff}
  .slider-row{display:flex;justify-content:space-between;font-size:.72rem;color:#8b949e}
  input[type=range]{width:100%;accent-color:#58a6ff;margin-top:4px}
  /* d-pad */
  .dpad{display:grid;grid-template-columns:repeat(3,52px);
        grid-template-rows:repeat(3,52px);gap:4px;place-content:center}
  .dpad button{background:#21262d;border:1px solid #30363d;border-radius:6px;
               color:#c9d1d9;font-size:1.1rem;cursor:pointer;
               display:flex;align-items:center;justify-content:center;
               user-select:none;transition:background .1s}
  .dpad button:active{background:#58a6ff;color:#0d1117}
  .dpad .empty{background:none;border:none;pointer-events:none}
  .key-help{font-size:.7rem;color:#8b949e;line-height:1.8}
  kbd{background:#21262d;border:1px solid #30363d;border-radius:3px;padding:1px 5px;font-size:.7rem}
  .gp-row{font-size:.7rem;color:#8b949e}

  /* ---- phone layout (touch device detected by JS adding .is-touch to body) */
  /* portrait */
  body.is-touch main{flex-direction:column}
  body.is-touch .sidebar{display:none}           /* hide desktop sidebar   */
  body.is-touch .cam{flex:0 0 42%;max-height:42%}

  /* phone controls panel */
  .phone-panel{display:none;flex-direction:column;align-items:center;
               gap:10px;padding:10px;background:#161b22;
               border-top:1px solid #30363d;flex:1;overflow:hidden}
  body.is-touch .phone-panel{display:flex}

  /* speed strip */
  .phone-speed{display:flex;align-items:center;gap:10px;width:100%;max-width:360px}
  .phone-speed label{font-size:.7rem;color:#8b949e;white-space:nowrap}
  .phone-speed input{flex:1;accent-color:#58a6ff}
  .phone-speed span{font-size:.7rem;color:#8b949e;min-width:34px;text-align:right}

  /* velocity chips */
  .vel-chips{display:flex;gap:12px;font-size:.72rem;color:#8b949e}
  .vel-chip span{color:#e0e0e0}

  /* joystick */
  .joystick-wrap{flex:1;display:flex;align-items:center;justify-content:center;
                 width:100%;position:relative}
  #joystick{width:160px;height:160px;border-radius:50%;
            background:#21262d;border:2px solid #30363d;
            position:relative;touch-action:none;cursor:grab;flex-shrink:0}
  #joystick-thumb{width:56px;height:56px;border-radius:50%;
                  background:#58a6ff;position:absolute;
                  top:50%;left:50%;
                  transform:translate(-50%,-50%);
                  pointer-events:none;transition:background .15s}
  #joystick.active #joystick-thumb{background:#3fb950}
  /* axis labels around joystick */
  .jlabel{position:absolute;font-size:.65rem;color:#8b949e;pointer-events:none}
  .jlabel.top{top:6px;left:50%;transform:translateX(-50%)}
  .jlabel.bot{bottom:6px;left:50%;transform:translateX(-50%)}
  .jlabel.lft{left:6px;top:50%;transform:translateY(-50%)}
  .jlabel.rgt{right:6px;top:50%;transform:translateY(-50%)}

  /* landscape phone: side-by-side */
  @media (orientation:landscape) and (max-height:500px) {
    body.is-touch main{flex-direction:row}
    body.is-touch .cam{flex:1;max-height:100%}
    body.is-touch .phone-panel{border-top:none;border-left:1px solid #30363d;
                               flex:0 0 200px;padding:8px}
    #joystick{width:130px;height:130px}
  }

  /* ---- mapping panel (desktop only) --------------------------------- */
  .cam-col{flex:1;display:flex;flex-direction:column;overflow:hidden;min-width:0}
  .map-panel{display:flex;background:#010409;
             border-top:1px solid #30363d;flex-direction:column;
             overflow:hidden;flex:0 0 44%;max-height:44%}
  .map-header{padding:10px 12px;background:#161b22;border-bottom:1px solid #30363d;
              display:flex;flex-direction:column;gap:4px;flex-shrink:0}
  .map-title{font-size:.72rem;text-transform:uppercase;letter-spacing:1px;color:#58a6ff}
  .map-meta-txt{font-size:.65rem;color:#8b949e}
  .map-canvas-wrap{position:relative;flex:1;overflow:hidden;display:flex;
                   align-items:center;justify-content:center;padding:4px}
  #map-img{image-rendering:pixelated;width:100%;height:100%;
           object-fit:contain;display:block;opacity:.3}
  #map-img.loaded{opacity:1}
  #map-overlay{position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none}
  #map-loading{position:absolute;inset:0;display:none;flex-direction:column;
               gap:10px;align-items:center;justify-content:center;
               background:rgba(1,4,9,.66);color:#e6edf3;font-size:.78rem}
  .spinner{width:34px;height:34px;border:3px solid #30363d;border-top-color:#58a6ff;
           border-radius:50%;animation:spin 0.9s linear infinite}
  @keyframes spin{to{transform:rotate(360deg)}}
  .map-footer{padding:8px 12px;background:#161b22;border-top:1px solid #30363d;
              display:flex;gap:8px;align-items:center;flex-shrink:0}
  .mbtn{padding:5px 12px;border-radius:5px;border:1px solid #30363d;
        background:#21262d;color:#c9d1d9;font-size:.72rem;cursor:pointer;
        font-family:monospace;transition:background .1s;white-space:nowrap}
  .mbtn:hover{background:#30363d}.mbtn:active{background:#58a6ff22}
  .mbtn:disabled{opacity:.5;cursor:default}
  .map-save-sts{font-size:.65rem;flex:1}
  .capture-bar{display:flex;align-items:center;gap:8px;padding:6px 8px;
        background:#161b22;border-top:1px solid #30363d}
  .capture-sts{font-size:.7rem;color:#8b949e;font-family:monospace}
  /* mapping toggle at bottom of sidebar */
  .sidebar-footer{margin-top:auto;padding-top:10px;border-top:1px solid #30363d}
  /* never show map panel on touch */
  body.is-touch .map-panel{display:none!important}
  body.is-touch .sidebar-footer{display:none}
  /* mapping-mode toggle button states */
  .mbtn-on{background:#238636;border-color:#2ea043;color:#fff}
  .mbtn-on:hover{background:#2ea043}
  /* grab button status text */
  .grab-sts{font-size:.68rem;color:#8b949e;margin-top:4px;min-height:1.2em}
  .grab-sts.ok{color:#3fb950}.grab-sts.err{color:#f85149}

  /* ---- label panel (desktop only) ----------------------------------- */
  .label-panel{display:none;position:fixed;top:0;right:0;bottom:0;
               width:min(820px,90vw);background:#161b22;
               border-left:1px solid #30363d;z-index:100;
               flex-direction:column;overflow:hidden}
  .label-panel.open{display:flex}
  /* never show label panel on touch */
  body.is-touch .label-panel{display:none!important}
  .lbl-header{padding:10px 14px;background:#0d1117;border-bottom:1px solid #30363d;
              display:flex;align-items:center;gap:8px;flex-shrink:0}
  .lbl-title{font-size:.78rem;text-transform:uppercase;letter-spacing:1px;color:#58a6ff;flex:1}
  .lbl-body{display:flex;flex:1;overflow:hidden;min-height:0}
  .lbl-list{width:190px;flex-shrink:0;overflow-y:auto;border-right:1px solid #30363d;
            padding:4px 0;background:#0d1117}
  .lbl-row{padding:5px 10px;font-size:.7rem;cursor:pointer;color:#c9d1d9;
           display:flex;justify-content:space-between;align-items:center;gap:6px;
           border-left:2px solid transparent}
  .lbl-row:hover{background:#21262d}
  .lbl-row.active{background:#21262d;border-left-color:#58a6ff;color:#e0e0e0}
  .lbl-badge{font-size:.65rem;padding:1px 5px;border-radius:8px;
             background:#21262d;color:#8b949e;white-space:nowrap;flex-shrink:0}
  .lbl-badge.labelled{background:#238636;color:#fff}
  .lbl-canvas-wrap{flex:1;position:relative;overflow:hidden;display:flex;
                   align-items:center;justify-content:center;background:#010409}
  #lbl-img{max-width:100%;max-height:100%;object-fit:contain;display:block;
           user-select:none;-webkit-user-drag:none}
  #lbl-overlay{position:absolute;top:0;left:0;width:100%;height:100%;cursor:crosshair}
  .lbl-footer{padding:8px 10px;background:#0d1117;border-top:1px solid #30363d;
              display:flex;flex-wrap:wrap;gap:6px;align-items:center;flex-shrink:0}
  .lbl-footer select{background:#21262d;color:#c9d1d9;border:1px solid #30363d;
                     border-radius:4px;padding:3px 6px;font-size:.72rem;font-family:monospace}
  .lbl-footer input[type=text]{background:#21262d;color:#c9d1d9;border:1px solid #30363d;
                               border-radius:4px;padding:3px 6px;font-size:.72rem;
                               font-family:monospace;width:110px}
  #lbl-sts{font-size:.68rem;flex:1;text-align:right}
  .lbl-hint{font-size:.65rem;color:#8b949e;width:100%;padding-top:2px}
  .lbl-auto-lbl{font-size:.72rem;color:#c9d1d9;display:flex;align-items:center;
                gap:4px;cursor:pointer;user-select:none}
  /* ---- header tabs (desktop only) ----------------------------------- */
  .tabs{display:flex;gap:4px;margin-left:12px}
  .tab{background:#21262d;border:1px solid #30363d;border-radius:6px 6px 0 0;
       color:#8b949e;font-family:monospace;font-size:.72rem;padding:4px 12px;
       cursor:pointer}
  .tab:hover{color:#c9d1d9}
  .tab.active{background:#0d1117;color:#58a6ff;border-bottom-color:#0d1117}
  body.is-touch .tabs{display:none}   /* annotation is desktop-only */
</style>
</head>
<body>
<header>
  <h1>&#x1F916; JeTank</h1>
  <nav class="tabs">
    <button class="tab active" id="tab-drive" onclick="showTab('drive')">Drive</button>
    <button class="tab" id="tab-annot" onclick="showTab('annotate')">&#x1F3F7; Annotate</button>
  </nav>
  <span id="ws-label" class="badge dot-err">&#x25CF; Disconnected</span>
  <span id="cam-label" class="badge" style="color:#8b949e">&#x25CF; Camera</span>
  <span id="ctrl-mode"></span>
</header>

<main>
  <!-- left column: camera on top, live map underneath -->
  <div class="cam-col">
  <!-- camera -->
  <div class="cam">
    <img id="cam-img" src="/stream.mjpg" alt="camera"
         onerror="scheduleReconnectStream()" onload="onImgLoad()">
    <canvas id="det-overlay" class="cam-det"></canvas>
    <div class="cam-overlay">left camera</div>
  </div>

  <!-- capture bar: save the current frame to the robot for dataset building -->
  <div class="capture-bar">
    <button class="mbtn" id="capture-btn" onclick="capture()">&#x1F4F7; Capture</button>
    <button class="mbtn" id="label-btn" onclick="openLabeler()">&#x1F3F7; Label</button>
    <button class="mbtn" id="det-btn" onclick="toggleDetections()">&#x1F441; Detections</button>
    <span class="capture-sts" id="capture-sts">0 saved</span>
    <span class="capture-sts" id="det-sts"></span>
  </div>

  <!-- mapping panel: desktop only, shown when mapping mode is active -->
  <div class="map-panel" id="map-panel">
    <div class="map-header">
      <span class="map-title">&#x1F5FA; Live Map</span>
      <span class="map-meta-txt" id="map-meta-txt">Waiting for /map topic&#x2026;</span>
    </div>
    <div class="map-canvas-wrap">
      <img id="map-img" src="" alt="" style="cursor:crosshair"
           onclick="mapClick(event)"
           onload="this.classList.add('loaded')"
           onerror="this.classList.remove('loaded');document.getElementById('map-meta-txt').textContent='Waiting for /map topic&#x2026;'">
      <canvas id="map-overlay"></canvas>
      <div id="map-loading">
        <div class="spinner"></div>
        <div id="map-loading-txt">Determining robot position&#x2026;</div>
      </div>
    </div>
    <div class="map-footer">
      <button class="mbtn" id="start-map-btn" onclick="startMapping()">Start Mapping</button>
      <button class="mbtn" id="start-nav-btn" onclick="startNavigation()" disabled>Navigate (saved map)</button>
      <button class="mbtn" id="save-map-btn" onclick="saveMap()">Save Map</button>
      <button class="mbtn" id="stop-nav-btn" onclick="stopNav()">Stop</button>
      <span class="map-save-sts" id="map-save-sts"></span>
      <div class="map-hint" id="nav-hint">Start mapping (or load a saved map), then click the map to send the robot there.</div>
    </div>
  </div>
  </div><!-- /cam-col -->

  <!-- desktop sidebar -->
  <div class="sidebar">
    <div class="section">
      <div class="stitle">Velocity</div>
      <div class="vel-row"><span>Linear X</span><span id="lv-d">0.00 m/s</span></div>
      <div class="bar-track"><div id="lbar-d" class="bar-fill bar-fwd" style="width:0%"></div></div>
      <div class="vel-row" style="margin-top:6px"><span>Angular Z</span><span id="av-d">0.00 rad/s</span></div>
      <div class="bar-track"><div id="abar-d" class="bar-fill bar-cw" style="width:0%"></div></div>
    </div>
    <div class="section">
      <div class="stitle">Speed Scale</div>
      <div class="slider-row"><span>Scale</span><span id="spd-label-d">50 %</span></div>
      <input type="range" id="spd-d" min="5" max="100" value="50">
    </div>
    <div class="section">
      <div class="stitle">D-Pad</div>
      <div class="dpad">
        <div class="empty"></div>
        <button onpointerdown="hold(1,0)"  onpointerup="release()" onpointercancel="release()">&#x25B2;</button>
        <div class="empty"></div>
        <button onpointerdown="hold(0,1)"  onpointerup="release()" onpointercancel="release()">&#x25C4;</button>
        <button onpointerdown="release()"  onpointerup="release()">&#x25FC;</button>
        <button onpointerdown="hold(0,-1)" onpointerup="release()" onpointercancel="release()">&#x25BA;</button>
        <div class="empty"></div>
        <button onpointerdown="hold(-1,0)" onpointerup="release()" onpointercancel="release()">&#x25BC;</button>
        <div class="empty"></div>
      </div>
    </div>
    <div class="section">
      <div class="stitle">Keyboard</div>
      <div class="key-help">
        <kbd>W</kbd>/<kbd>&#x2191;</kbd> Forward &nbsp;
        <kbd>S</kbd>/<kbd>&#x2193;</kbd> Back<br>
        <kbd>A</kbd>/<kbd>&#x2190;</kbd> Left &nbsp;&nbsp;&nbsp;
        <kbd>D</kbd>/<kbd>&#x2192;</kbd> Right<br>
        <kbd>Space</kbd> Stop
      </div>
    </div>
    <div class="section">
      <div class="stitle">Gamepad</div>
      <div class="gp-row" id="gp-status-d">Not detected</div>
    </div>
    <div class="section sidebar-footer">
      <div class="stitle">Navigation</div>
      <button class="mbtn" id="map-toggle-btn" onclick="toggleMappingMode()">
        &#x1F5FA; Mapping Mode
      </button>
    </div>
    <div class="section">
      <div class="stitle">Arm</div>
      <button class="mbtn" id="grab-btn-d" onclick="grab()">&#x1F9B2; Grab</button>
      <div class="grab-sts" id="grab-sts-d"></div>
    </div>
  </div>

  <!-- phone panel -->
  <div class="phone-panel">
    <div class="vel-chips">
      <div class="vel-chip">Lin: <span id="lv-p">0.00</span> m/s</div>
      <div class="vel-chip">Ang: <span id="av-p">0.00</span> rad/s</div>
    </div>
    <div class="phone-speed">
      <label>Speed</label>
      <input type="range" id="spd-p" min="5" max="100" value="50">
      <span id="spd-label-p">50 %</span>
    </div>
    <button class="mbtn" id="grab-btn-p" onclick="grab()" style="width:100%;max-width:360px">&#x1F9B2; Grab</button>
    <div class="grab-sts" id="grab-sts-p" style="text-align:center"></div>
    <div class="joystick-wrap">
      <div id="joystick">
        <span class="jlabel top">&#x25B2;</span>
        <span class="jlabel bot">&#x25BC;</span>
        <span class="jlabel lft">&#x25C4;</span>
        <span class="jlabel rgt">&#x25BA;</span>
        <div id="joystick-thumb"></div>
      </div>
    </div>
  </div>
</main>

<!-- label panel (desktop only, toggled by openLabeler()) -->
<div class="label-panel" id="label-panel">
  <div class="lbl-header">
    <span class="lbl-title">&#x1F3F7; Label captures</span>
    <button class="mbtn" onclick="closeLabeler()">&#x2715; Close</button>
  </div>
  <div class="lbl-body">
    <div class="lbl-list" id="lbl-list"></div>
    <div class="lbl-canvas-wrap">
      <img id="lbl-img" src="" alt="">
      <canvas id="lbl-overlay"></canvas>
    </div>
  </div>
  <div class="lbl-footer">
    <select id="lbl-class"></select>
    <input type="text" id="lbl-newclass" placeholder="new class">
    <button class="mbtn" onclick="lblAddClass()">+</button>
    <button class="mbtn" onclick="lblSaveLabels()">Save</button>
    <button class="mbtn" onclick="lblDeleteBox()">Delete box</button>
    <button class="mbtn" onclick="lblAutoDetect()">&#x2728; Auto-detect</button>
    <label class="lbl-auto-lbl"><input type="checkbox" id="lbl-auto"> Auto (rough)</label>
    <span id="lbl-sts"></span>
    <div class="lbl-hint">Drag on the image to draw a box; click a box to select it.
      &#x2728; Auto proposes rough boxes (review &amp; save); the toggle auto-runs on unlabelled images.<br>
      Keys: <b>Enter</b> save+next &middot; <b>E/]</b> next &middot; <b>Q/[</b> prev &middot;
      <b>R</b> auto-detect &middot; <b>X/Del</b> delete box &middot; <b>Esc</b> deselect &middot; <b>1-9,0</b> class</div>
  </div>
</div>

<script>
// ===========================================================================
// State
// ===========================================================================
let ws = null;
let linearX = 0, angularZ = 0;
let speedScale = 0.5;
let sendTimer = null, reconnectTimer = null;
const keysDown = new Set();
let isTouch = false;

// ===========================================================================
// Touch / desktop detection
// ===========================================================================
function detectInputMode() {
  isTouch = ('ontouchstart' in window) || navigator.maxTouchPoints > 0;
  if (isTouch) {
    document.body.classList.add('is-touch');
    document.getElementById('ctrl-mode').textContent = '\\u1F4F1 Touch mode';
  } else {
    document.getElementById('ctrl-mode').textContent = '\\u1F5A5 Desktop mode';
  }
}
detectInputMode();

// ===========================================================================
// DOM refs
// ===========================================================================
const wsLabel   = document.getElementById('ws-label');
const camLabel  = document.getElementById('cam-label');
const camImg    = document.getElementById('cam-img');

// desktop
const lvD = document.getElementById('lv-d'), avD = document.getElementById('av-d');
const lbarD = document.getElementById('lbar-d'), abarD = document.getElementById('abar-d');
const spdD = document.getElementById('spd-d'), spdLabelD = document.getElementById('spd-label-d');
const gpStatusD = document.getElementById('gp-status-d');

// phone
const lvP = document.getElementById('lv-p'), avP = document.getElementById('av-p');
const spdP = document.getElementById('spd-p'), spdLabelP = document.getElementById('spd-label-p');

// ===========================================================================
// Speed slider (both panels stay in sync)
// ===========================================================================
function onSpdChange(val) {
  speedScale = val / 100;
  spdLabelD.textContent = val + ' %';
  spdLabelP.textContent = val + ' %';
  spdD.value = val;
  spdP.value = val;
}
spdD.addEventListener('input', () => onSpdChange(spdD.value));
spdP.addEventListener('input', () => onSpdChange(spdP.value));

// ===========================================================================
// D-Pad (desktop)
// ===========================================================================
function hold(l, a) { linearX = l; angularZ = a; }
function release()  { linearX = 0; angularZ = 0; }

// ===========================================================================
// Keyboard
// ===========================================================================
const KEY_MAP = {
  ArrowUp:[1,0], KeyW:[1,0],
  ArrowDown:[-1,0], KeyS:[-1,0],
  ArrowLeft:[0,1], KeyA:[0,1],
  ArrowRight:[0,-1], KeyD:[0,-1],
  Space:[0,0],
};
document.addEventListener('keydown', e => {
  // Don't drive the robot while the annotation panel is open.
  if (document.getElementById('label-panel').classList.contains('open')) return;
  if (KEY_MAP[e.code] !== undefined) { keysDown.add(e.code); e.preventDefault(); }
});
document.addEventListener('keyup', e => {
  if (KEY_MAP[e.code] !== undefined) keysDown.delete(e.code);
});
function updateFromKeys() {
  if (keysDown.size === 0) { linearX = 0; angularZ = 0; return; }
  let l = 0, a = 0;
  keysDown.forEach(k => { if (KEY_MAP[k]) { l += KEY_MAP[k][0]; a += KEY_MAP[k][1]; } });
  linearX  = Math.max(-1, Math.min(1, l));
  angularZ = Math.max(-1, Math.min(1, a));
}

// ===========================================================================
// Virtual Joystick (phone)
// ===========================================================================
const joystickEl = document.getElementById('joystick');
const thumbEl    = document.getElementById('joystick-thumb');
const JOYSTICK_R = 52;   // max thumb travel radius in px
let joystickActive = false;
let joyOriginX = 0, joyOriginY = 0;
let joyActiveTouchId = null;

function joystickStart(cx, cy) {
  const rect = joystickEl.getBoundingClientRect();
  joyOriginX = rect.left + rect.width  / 2;
  joyOriginY = rect.top  + rect.height / 2;
  joystickActive = true;
  joystickEl.classList.add('active');
  joystickMove(cx, cy);
}
function joystickMove(cx, cy) {
  if (!joystickActive) return;
  const dx = cx - joyOriginX;
  const dy = cy - joyOriginY;
  const dist = Math.hypot(dx, dy);
  const clamped = Math.min(dist, JOYSTICK_R);
  const angle   = Math.atan2(dy, dx);
  const tx = Math.cos(angle) * clamped;
  const ty = Math.sin(angle) * clamped;
  thumbEl.style.transform = `translate(calc(-50% + ${tx}px), calc(-50% + ${ty}px))`;
  linearX  = -(ty / JOYSTICK_R);   // up   = positive linear
  angularZ = -(tx / JOYSTICK_R);   // left = positive angular
}
function joystickEnd() {
  joystickActive = false;
  joyActiveTouchId = null;
  joystickEl.classList.remove('active');
  thumbEl.style.transform = 'translate(-50%, -50%)';
  linearX = 0; angularZ = 0;
}

// Touch events on joystick
joystickEl.addEventListener('touchstart', e => {
  e.preventDefault();
  if (joyActiveTouchId !== null) return;
  const t = e.changedTouches[0];
  joyActiveTouchId = t.identifier;
  joystickStart(t.clientX, t.clientY);
}, {passive: false});

joystickEl.addEventListener('touchmove', e => {
  e.preventDefault();
  for (const t of e.changedTouches) {
    if (t.identifier === joyActiveTouchId) { joystickMove(t.clientX, t.clientY); break; }
  }
}, {passive: false});

joystickEl.addEventListener('touchend', e => {
  e.preventDefault();
  for (const t of e.changedTouches) {
    if (t.identifier === joyActiveTouchId) { joystickEnd(); break; }
  }
}, {passive: false});
joystickEl.addEventListener('touchcancel', e => { e.preventDefault(); joystickEnd(); }, {passive: false});

// Mouse fallback for testing joystick on desktop
joystickEl.addEventListener('mousedown', e => {
  joystickStart(e.clientX, e.clientY);
  const mm = ev => joystickMove(ev.clientX, ev.clientY);
  const mu = () => { joystickEnd(); document.removeEventListener('mousemove', mm); document.removeEventListener('mouseup', mu); };
  document.addEventListener('mousemove', mm);
  document.addEventListener('mouseup', mu);
});

// ===========================================================================
// Gamepad
// ===========================================================================
let gpConnected = false;
window.addEventListener('gamepadconnected', () => {
  gpConnected = true;
  gpStatusD.textContent = 'Connected';
});
window.addEventListener('gamepaddisconnected', () => {
  gpConnected = false;
  gpStatusD.textContent = 'Disconnected';
});
function updateFromGamepad() {
  if (!gpConnected) return;
  for (const gp of navigator.getGamepads()) {
    if (!gp) continue;
    const dead = 0.12;
    const rawL = -gp.axes[1], rawA = -gp.axes[0];
    linearX  = Math.abs(rawL) > dead ? rawL : 0;
    angularZ = Math.abs(rawA) > dead ? rawA : 0;
    break;
  }
}

// ===========================================================================
// WebSocket + send loop
// ===========================================================================
function connect() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${proto}//${location.host}/ws`);
  ws.onopen = () => {
    wsLabel.className = 'badge dot-ok';
    wsLabel.textContent = '\\u25CF Connected';
    startLoop();
  };
  ws.onclose = () => {
    wsLabel.className = 'badge dot-err';
    wsLabel.textContent = '\\u25CF Disconnected';
    stopLoop();
    setTimeout(connect, 2000);
  };
  ws.onerror = () => ws.close();
}

function tick() {
  if (!isTouch) {
    updateFromKeys();
    updateFromGamepad();
  }
  const l = linearX  * speedScale;
  const a = angularZ * speedScale;
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({linear_x: l, angular_z: a}));
  }
  updateUI(l, a);
}
function startLoop() { if (!sendTimer) sendTimer = setInterval(tick, 100); }
function stopLoop()  { if (sendTimer) { clearInterval(sendTimer); sendTimer = null; } }

// ===========================================================================
// UI update
// ===========================================================================
function updateUI(l, a) {
  const ls = l.toFixed(2), as_ = a.toFixed(2);
  lvD.textContent = ls + ' m/s'; avD.textContent = as_ + ' rad/s';
  lvP.textContent = ls;          avP.textContent = as_;
  lbarD.style.width = Math.abs(l) * 100 + '%';
  abarD.style.width = Math.abs(a) * 100 + '%';
  lbarD.className = 'bar-fill ' + (l >= 0 ? 'bar-fwd' : 'bar-bwd');
  abarD.className = 'bar-fill ' + (a >= 0 ? 'bar-cw'  : 'bar-ccw');
}

// ===========================================================================
// Camera stream
// ===========================================================================
function onImgLoad() {
  camLabel.className = 'badge dot-ok';
  camLabel.textContent = '\\u25CF Camera';
}
function scheduleReconnectStream() {
  camLabel.className = 'badge dot-err';
  camLabel.textContent = '\\u25CF No stream';
  clearTimeout(reconnectTimer);
  reconnectTimer = setTimeout(() => {
    camImg.src = '/stream.mjpg?' + Date.now();
  }, 2000);
}

// ===========================================================================
// Live detection overlay (toggle). Polls /detections/latest and draws the
// sock detector's boxes over the camera stream. Requires the detector to be
// running (e.g. sim_demo.launch.py detect:=true with a trained model); when no
// detector publishes, the overlay just stays empty.
// ===========================================================================
let detOn = false;
let detTimer = null;
const DET_POLL_MS = 100;   // 10 Hz

function toggleDetections() {
  detOn = !detOn;
  const btn = document.getElementById('det-btn');
  const sts = document.getElementById('det-sts');
  if (detOn) {
    btn.classList.add('mbtn-on');
    detTimer = setInterval(detPoll, DET_POLL_MS);
    detPoll();
  } else {
    btn.classList.remove('mbtn-on');
    clearInterval(detTimer); detTimer = null;
    sts.textContent = '';
    const cv = document.getElementById('det-overlay');
    const ctx = cv.getContext('2d');
    ctx.clearRect(0, 0, cv.width, cv.height);
  }
}

function detPoll() {
  fetch('/detections/latest')
    .then(r => r.json())
    .then(d => { if (detOn && d.ok) detDraw(d.boxes || [], d.age); })
    .catch(() => {});
}

// Boxes are in source-image pixel coords; normalize against the camera frame's
// natural size, then map onto the contained (letterboxed) image rect. The
// server already drops stale detections, so an empty list clears the overlay.
function detDraw(boxes, age) {
  const cv = document.getElementById('det-overlay');
  cv.width = cv.clientWidth;
  cv.height = cv.clientHeight;
  const ctx = cv.getContext('2d');
  ctx.clearRect(0, 0, cv.width, cv.height);   // always clear first — no frozen box

  const sts = document.getElementById('det-sts');
  sts.textContent = boxes.length ? (boxes.length + ' det')
                                  : (age === null ? 'no detector' : 'searching\\u2026');

  const natW = camImg.naturalWidth, natH = camImg.naturalHeight;
  if (!natW || !natH || !boxes.length) return;
  const er = camImg.getBoundingClientRect();
  const cr = cv.getBoundingClientRect();
  const scale = Math.min(er.width / natW, er.height / natH);
  const dispW = natW * scale, dispH = natH * scale;
  const ox = (er.left + (er.width - dispW) / 2) - cr.left;
  const oy = (er.top  + (er.height - dispH) / 2) - cr.top;

  ctx.lineWidth = 2;
  ctx.strokeStyle = '#3fb950';
  ctx.fillStyle = '#3fb950';
  ctx.font = '13px monospace';
  boxes.forEach(b => {
    const x = ox + ((b.cx - b.w / 2) / natW) * dispW;
    const y = oy + ((b.cy - b.h / 2) / natH) * dispH;
    const w = (b.w / natW) * dispW;
    const h = (b.h / natH) * dispH;
    ctx.strokeRect(x, y, w, h);
    const tag = b.label ? (b.label + ' ' + b.score.toFixed(2)) : b.score.toFixed(2);
    const ty = y - 4 > 10 ? y - 4 : y + 14;
    ctx.fillText(tag, x + 2, ty);
  });
}

// ===========================================================================
// Mapping Mode (desktop only)
// ===========================================================================
let mappingMode = false;
let mapRefreshTimer = null;

// The map panel is always visible under the camera. The sidebar button now
// just starts/stops the SLAM+Nav2 stack (the map itself refreshes on its own).
function toggleMappingMode() {
  if (isTouch) return;
  mappingMode = !mappingMode;
  const btn = document.getElementById('map-toggle-btn');
  if (mappingMode) {
    btn.classList.add('mbtn-on');
    btn.innerHTML = '&#x1F5FA; Stop Nav';
    startMapping();
  } else {
    btn.classList.remove('mbtn-on');
    btn.innerHTML = '&#x1F5FA; Start Mapping';
    stopNav();
  }
}

let lastMeta = null;
let localizing = false;
let localizeStart = 0;

function startMapRefresh() {
  refreshMap();
  refreshNavStatus();
  refreshRobotPose();
  mapRefreshTimer = setInterval(() => {
    refreshMap(); refreshNavStatus(); refreshRobotPose();
  }, 1000);
}

function refreshRobotPose() {
  fetch('/robot_pose')
    .then(r => r.ok ? r.json() : null)
    .then(p => {
      const load = document.getElementById('map-loading');
      if (!p || !p.available) {
        if (localizing) load.style.display = 'flex';  // still determining
        return;
      }
      drawRobotArrow(p);
      if (localizing) {
        if (p.converged) {
          localizing = false;
          load.style.display = 'none';
          navMsg('\\u2713 localized (' + p.x + ', ' + p.y + ')', true);
        } else if (Date.now() - localizeStart > 25000) {
          localizing = false;
          load.style.display = 'none';
          navMsg('localization uncertain \\u2014 drive a little or re-try', false);
        }
      }
    })
    .catch(() => {});
}

function drawRobotArrow(p) {
  const img = document.getElementById('map-img');
  const cv = document.getElementById('map-overlay');
  if (!lastMeta || !lastMeta.resolution || !img.naturalWidth) return;
  cv.width = cv.clientWidth; cv.height = cv.clientHeight;
  if (!cv.width || !cv.height) return;
  const er = img.getBoundingClientRect();   // img ELEMENT box (fills the wrap)
  const cr = cv.getBoundingClientRect();
  const natW = img.naturalWidth, natH = img.naturalHeight;
  // The image is letterboxed inside the element by object-fit:contain — compute
  // the actual rendered image rect, not the element box.
  const scale = Math.min(er.width / natW, er.height / natH);
  const cw = natW * scale, ch = natH * scale;
  const contentLeft = er.left + (er.width - cw) / 2;
  const contentTop = er.top + (er.height - ch) / 2;
  const col = (p.x - lastMeta.origin_x) / lastMeta.resolution;
  const gridRow = (p.y - lastMeta.origin_y) / lastMeta.resolution;
  const ix = col, iy = (lastMeta.height - 1) - gridRow;  // PNG is vertically flipped
  const px = (contentLeft - cr.left) + ix * scale;
  const py = (contentTop - cr.top) + iy * scale;
  const ctx = cv.getContext('2d');
  ctx.clearRect(0, 0, cv.width, cv.height);
  ctx.save();
  ctx.translate(px, py);
  ctx.rotate(-p.yaw);                  // image y is down => screen angle = -yaw
  ctx.fillStyle = '#ff3b30'; ctx.strokeStyle = '#fff'; ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(16, 0); ctx.lineTo(-10, -9); ctx.lineTo(-4, 0); ctx.lineTo(-10, 9);
  ctx.closePath(); ctx.fill(); ctx.stroke();
  ctx.restore();
}

function stopMapRefresh() {
  clearInterval(mapRefreshTimer);
  mapRefreshTimer = null;
}

function refreshMap() {
  fetch('/map_meta')
    .then(r => r.ok ? r.json() : null)
    .then(meta => {
      if (!meta || !meta.width) return;
      lastMeta = meta;
      const w = (meta.width  * meta.resolution).toFixed(1);
      const h = (meta.height * meta.resolution).toFixed(1);
      document.getElementById('map-meta-txt').textContent =
        meta.width + '\\u00D7' + meta.height + 'px \\u00B7 ' +
        w + '\\u00D7' + h + 'm \\u00B7 ' + meta.resolution + 'm/px';
    })
    .catch(() => {});
  document.getElementById('map-img').src = '/map.png?t=' + Date.now();
}

function saveMap() {
  const btn = document.getElementById('save-map-btn');
  const sts = document.getElementById('map-save-sts');
  btn.disabled = true;
  btn.textContent = 'Saving\\u2026';
  sts.textContent = '';
  fetch('/save_map', {method: 'POST'})
    .then(r => r.json())
    .then(d => {
      if (d.status === 'ok') {
        sts.style.color = '#3fb950';
        sts.textContent = '\\u2713 ' + d.path;
      } else {
        sts.style.color = '#f85149';
        sts.textContent = d.msg || 'Error';
      }
    })
    .catch(() => { sts.style.color = '#f85149'; sts.textContent = 'Request failed'; })
    .finally(() => { btn.disabled = false; btn.textContent = 'Save Map'; });
}

function navMsg(text, ok) {
  const sts = document.getElementById('map-save-sts');
  sts.style.color = ok ? '#3fb950' : '#f85149';
  sts.textContent = text;
}

function capture() {
  const btn = document.getElementById('capture-btn');
  const sts = document.getElementById('capture-sts');
  btn.disabled = true;
  fetch('/capture', {method: 'POST'})
    .then(r => r.json())
    .then(d => {
      if (d.ok) {
        sts.style.color = '#3fb950';
        sts.textContent = d.count + ' saved \\u00B7 ' + d.filename;
      } else {
        sts.style.color = '#f85149';
        sts.textContent = d.error || 'capture failed';
      }
    })
    .catch(() => { sts.style.color = '#f85149'; sts.textContent = 'request failed'; })
    .finally(() => { btn.disabled = false; });
}

function refreshNavStatus() {
  fetch('/nav_status')
    .then(r => r.ok ? r.json() : null)
    .then(s => {
      if (!s) return;
      document.getElementById('start-nav-btn').disabled = !s.has_map;
      const hint = document.getElementById('nav-hint');
      if (s.running === 'mapping')      hint.textContent = 'Mapping active \\u00B7 drive to build the map, then click it to navigate.';
      else if (s.running === 'navigation') hint.textContent = 'Navigation active (saved map) \\u00B7 click the map to send the robot.';
      else hint.textContent = s.has_map ? 'Saved map available \\u00B7 Start Mapping or Navigate (saved map).'
                                        : 'Start Mapping, drive around, then Save Map.';
    })
    .catch(() => {});
}

function startMapping() {
  navMsg('Starting mapping\\u2026', true);
  fetch('/start_mapping', {method: 'POST'})
    .then(r => r.json())
    .then(d => navMsg(d.status === 'ok' ? '\\u2713 mapping started' : (d.msg || 'error'), d.status === 'ok'))
    .catch(() => navMsg('Request failed', false))
    .finally(refreshNavStatus);
}

function startNavigation() {
  navMsg('Starting navigation\\u2026', true);
  fetch('/start_navigation', {method: 'POST'})
    .then(r => r.json())
    .then(d => {
      navMsg(d.status === 'ok' ? 'determining position\\u2026' : (d.msg || 'error'),
             d.status === 'ok');
      if (d.status === 'ok') {
        // Show the "determining robot position" loader until AMCL converges
        // (refreshRobotPose hides it and draws the pose arrow).
        localizing = true;
        localizeStart = Date.now();
        document.getElementById('map-loading-txt').textContent =
          'Determining robot position\\u2026';
        document.getElementById('map-loading').style.display = 'flex';
      }
    })
    .catch(() => navMsg('Request failed', false))
    .finally(refreshNavStatus);
}

function stopNav() {
  fetch('/stop_nav', {method: 'POST'})
    .then(r => r.json())
    .then(d => navMsg('Stopped ' + (d.stopped || 'nav'), true))
    .catch(() => navMsg('Request failed', false))
    .finally(refreshNavStatus);
}

function mapClick(ev) {
  const img = document.getElementById('map-img');
  if (!img.naturalWidth) return;
  const r = img.getBoundingClientRect();
  const x = Math.round((ev.clientX - r.left) / r.width  * img.naturalWidth);
  const y = Math.round((ev.clientY - r.top)  / r.height * img.naturalHeight);
  navMsg('Sending goal\\u2026', true);
  fetch('/navigate', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({x: x, y: y}),
  })
    .then(r => r.json())
    .then(d => {
      if (d.status === 'ok') navMsg('\\u2713 goal (' + d.goal.x + ', ' + d.goal.y + ') m', true);
      else navMsg(d.msg || 'goal rejected', false);
    })
    .catch(() => navMsg('Request failed', false));
}

// ===========================================================================
// Labeller
// ===========================================================================
let lblImages = [];
let lblClasses = [];
let lblCurrent = null;
let lblBoxes = [];
let lblSel = -1;

// Drag state
let lblDragging = false;
let lblDragStartNx = 0, lblDragStartNy = 0;
let lblDragCurNx = 0, lblDragCurNy = 0;
const LBL_MIN_PX = 5;  // ignore box smaller than this many canvas pixels

function openLabeler() {
  if (isTouch) return;
  fetch('/captures')
    .then(r => r.json())
    .then(d => {
      lblImages = d.images || [];
      lblClasses = d.classes || [];
      lblPopulateClassSelect();
      lblPopulateList();
      document.getElementById('label-panel').classList.add('open');
      if (lblImages.length === 0) {
        document.getElementById('lbl-list').textContent = 'No captures yet.';
      }
    })
    .catch(() => {});
}

function closeLabeler() {
  document.getElementById('label-panel').classList.remove('open');
  const td = document.getElementById('tab-drive');
  const ta = document.getElementById('tab-annot');
  if (td && ta) { td.classList.add('active'); ta.classList.remove('active'); }
}

// Header tab switching: Drive (live control) vs Annotate (labeller panel).
function showTab(which) {
  const td = document.getElementById('tab-drive');
  const ta = document.getElementById('tab-annot');
  if (which === 'annotate') {
    td.classList.remove('active'); ta.classList.add('active');
    openLabeler();
  } else {
    td.classList.add('active'); ta.classList.remove('active');
    document.getElementById('label-panel').classList.remove('open');
  }
}

function lblPopulateClassSelect() {
  const sel = document.getElementById('lbl-class');
  const prev = sel.value;
  sel.innerHTML = '';
  lblClasses.forEach((name, idx) => {
    const opt = document.createElement('option');
    opt.value = idx;
    opt.textContent = idx + ': ' + name;
    sel.appendChild(opt);
  });
  if (prev !== '' && Number(prev) < lblClasses.length) sel.value = prev;
}

function lblPopulateList() {
  const list = document.getElementById('lbl-list');
  list.innerHTML = '';
  lblImages.forEach(img => {
    const row = document.createElement('div');
    row.className = 'lbl-row' + (img.name === lblCurrent ? ' active' : '');
    row.dataset.name = img.name;
    row.onclick = () => lblSelectImage(img.name);
    const nameSpan = document.createElement('span');
    nameSpan.textContent = img.name;
    nameSpan.style.cssText = 'overflow:hidden;text-overflow:ellipsis;white-space:nowrap;min-width:0';
    const badge = document.createElement('span');
    badge.className = 'lbl-badge' + (img.labelled ? ' labelled' : '');
    badge.textContent = img.labelled ? ('\\u2713 ' + img.n_boxes) : 'n';
    badge.dataset.name = img.name;
    row.appendChild(nameSpan);
    row.appendChild(badge);
    list.appendChild(row);
  });
}

function lblUpdateBadge(name, boxes) {
  const badge = document.querySelector('.lbl-badge[data-name="' + CSS.escape(name) + '"]');
  if (!badge) return;
  if (boxes.length > 0) {
    badge.className = 'lbl-badge labelled';
    badge.textContent = '\\u2713 ' + boxes.length;
  } else {
    badge.className = 'lbl-badge';
    badge.textContent = 'n';
  }
}

function lblSelectImage(name) {
  lblCurrent = name;
  lblBoxes = [];
  lblSel = -1;
  // Highlight active row
  document.querySelectorAll('.lbl-row').forEach(r => {
    r.classList.toggle('active', r.dataset.name === name);
  });
  fetch('/captures/labels/' + encodeURIComponent(name))
    .then(r => r.json())
    .then(d => {
      if (d.ok) {
        lblBoxes = d.boxes || [];
        if (d.classes && d.classes.length) {
          lblClasses = d.classes;
          lblPopulateClassSelect();
        }
      }
      const img = document.getElementById('lbl-img');
      img.onload = () => {
        lblRedraw();
        const auto = document.getElementById('lbl-auto');
        if (auto && auto.checked && lblBoxes.length === 0) lblAutoDetect();
      };
      img.src = '/captures/img/' + encodeURIComponent(name) + '?t=' + Date.now();
    })
    .catch(() => {});
}

// Convert normalized [0,1] coords to canvas pixel coords using content-rect math.
function lblNormToCanvas(nx, ny) {
  const img = document.getElementById('lbl-img');
  const cv = document.getElementById('lbl-overlay');
  if (!img.naturalWidth) return [0, 0];
  const er = img.getBoundingClientRect();
  const cr = cv.getBoundingClientRect();
  const natW = img.naturalWidth, natH = img.naturalHeight;
  const scale = Math.min(er.width / natW, er.height / natH);
  const cw = natW * scale, ch = natH * scale;
  const contentLeft = er.left + (er.width - cw) / 2;
  const contentTop  = er.top  + (er.height - ch) / 2;
  const px = (contentLeft - cr.left) + nx * cw;
  const py = (contentTop  - cr.top)  + ny * ch;
  return [px, py];
}

// Convert canvas pixel coords to normalized [0,1] (clamped).
function lblCanvasToNorm(px, py) {
  const img = document.getElementById('lbl-img');
  const cv = document.getElementById('lbl-overlay');
  if (!img.naturalWidth) return [0, 0];
  const er = img.getBoundingClientRect();
  const cr = cv.getBoundingClientRect();
  const natW = img.naturalWidth, natH = img.naturalHeight;
  const scale = Math.min(er.width / natW, er.height / natH);
  const cw = natW * scale, ch = natH * scale;
  const contentLeft = er.left + (er.width - cw) / 2;
  const contentTop  = er.top  + (er.height - ch) / 2;
  const nx = (px - (contentLeft - cr.left)) / cw;
  const ny = (py - (contentTop  - cr.top))  / ch;
  return [Math.max(0, Math.min(1, nx)), Math.max(0, Math.min(1, ny))];
}

function lblRedraw(inProgressBox) {
  const cv = document.getElementById('lbl-overlay');
  cv.width = cv.clientWidth;
  cv.height = cv.clientHeight;
  if (!cv.width || !cv.height) return;
  const ctx = cv.getContext('2d');
  ctx.clearRect(0, 0, cv.width, cv.height);

  // Draw committed boxes
  lblBoxes.forEach((b, idx) => {
    const [x1, y1] = lblNormToCanvas(b.cx - b.w / 2, b.cy - b.h / 2);
    const [x2, y2] = lblNormToCanvas(b.cx + b.w / 2, b.cy + b.h / 2);
    const sel = idx === lblSel;
    ctx.strokeStyle = sel ? '#ff9500' : '#58a6ff';
    ctx.lineWidth = sel ? 2.5 : 1.5;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    const label = (lblClasses[b.cls] || ('cls' + b.cls));
    ctx.fillStyle = sel ? '#ff9500' : '#58a6ff';
    ctx.font = '11px monospace';
    ctx.fillText(label, x1 + 2, y1 - 3 > 0 ? y1 - 3 : y1 + 12);
  });

  // Draw in-progress rubber-band box
  if (inProgressBox) {
    const [ix1, iy1] = lblNormToCanvas(inProgressBox.x0, inProgressBox.y0);
    const [ix2, iy2] = lblNormToCanvas(inProgressBox.x1, inProgressBox.y1);
    ctx.strokeStyle = '#3fb950';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 3]);
    ctx.strokeRect(ix1, iy1, ix2 - ix1, iy2 - iy1);
    ctx.setLineDash([]);
  }
}

// Mouse handlers on lbl-overlay
(function() {
  const cv = document.getElementById('lbl-overlay');
  let dragThreshold = false;  // becomes true once we've dragged LBL_MIN_PX

  cv.addEventListener('mousedown', e => {
    if (!lblCurrent) return;
    const r = cv.getBoundingClientRect();
    const [nx, ny] = lblCanvasToNorm(e.clientX - r.left, e.clientY - r.top);
    lblDragStartNx = nx; lblDragStartNy = ny;
    lblDragCurNx = nx;   lblDragCurNy = ny;
    lblDragging = true;
    dragThreshold = false;
  });

  document.addEventListener('mousemove', e => {
    if (!lblDragging) return;
    const cv2 = document.getElementById('lbl-overlay');
    const r = cv2.getBoundingClientRect();
    const [nx, ny] = lblCanvasToNorm(e.clientX - r.left, e.clientY - r.top);
    lblDragCurNx = nx; lblDragCurNy = ny;
    // Check pixel distance to detect real drag vs. click
    const [sx, sy] = lblNormToCanvas(lblDragStartNx, lblDragStartNy);
    const [ex, ey] = lblNormToCanvas(nx, ny);
    if (Math.hypot(ex - sx, ey - sy) > LBL_MIN_PX) dragThreshold = true;
    if (dragThreshold) {
      lblRedraw({x0: lblDragStartNx, y0: lblDragStartNy, x1: nx, y1: ny});
    }
  });

  document.addEventListener('mouseup', e => {
    if (!lblDragging) return;
    lblDragging = false;
    const cv2 = document.getElementById('lbl-overlay');
    const r = cv2.getBoundingClientRect();
    const [nx, ny] = lblCanvasToNorm(e.clientX - r.left, e.clientY - r.top);

    if (dragThreshold) {
      // Commit new box
      const x0 = Math.min(lblDragStartNx, nx);
      const x1 = Math.max(lblDragStartNx, nx);
      const y0 = Math.min(lblDragStartNy, ny);
      const y1 = Math.max(lblDragStartNy, ny);
      const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
      const w  = x1 - x0, h = y1 - y0;
      const clsIdx = parseInt(document.getElementById('lbl-class').value) || 0;
      lblBoxes.push({cls: clsIdx, cx, cy, w, h});
      lblSel = lblBoxes.length - 1;
    } else {
      // Plain click: select topmost box containing the point, or deselect
      let hit = -1;
      for (let i = lblBoxes.length - 1; i >= 0; i--) {
        const b = lblBoxes[i];
        if (nx >= b.cx - b.w/2 && nx <= b.cx + b.w/2 &&
            ny >= b.cy - b.h/2 && ny <= b.cy + b.h/2) {
          hit = i; break;
        }
      }
      lblSel = (hit === lblSel) ? -1 : hit;
    }
    lblRedraw();
  });
})();

function lblDeleteBox() {
  if (lblSel < 0 || lblSel >= lblBoxes.length) return;
  lblBoxes.splice(lblSel, 1);
  lblSel = -1;
  lblRedraw();
}

function lblSaveLabels() {
  if (!lblCurrent) return;
  const sts = document.getElementById('lbl-sts');
  sts.style.color = '#8b949e';
  sts.textContent = 'Saving\\u2026';
  fetch('/captures/labels/' + encodeURIComponent(lblCurrent), {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({boxes: lblBoxes}),
  })
    .then(r => r.json())
    .then(d => {
      if (d.ok) {
        sts.style.color = '#3fb950';
        sts.textContent = '\\u2713 saved';
        lblUpdateBadge(lblCurrent, lblBoxes);
        // Reflect new state in the in-memory list so it survives re-render.
        const cur = lblImages.find(im => im.name === lblCurrent);
        if (cur) { cur.labelled = lblBoxes.length > 0; cur.n_boxes = lblBoxes.length; }
        lblSelectNext();
      } else {
        sts.style.color = '#f85149';
        sts.textContent = d.error || 'save failed';
      }
    })
    .catch(() => { sts.style.color = '#f85149'; sts.textContent = 'request failed'; });
}

// Advance to the next image in the list after a save. Stops at the last image.
function lblSelectNext() {
  if (!lblImages.length) return;
  const idx = lblImages.findIndex(im => im.name === lblCurrent);
  if (idx < 0 || idx + 1 >= lblImages.length) {
    const sts = document.getElementById('lbl-sts');
    sts.style.color = '#8b949e';
    sts.textContent = '\\u2713 saved \\u2014 last image';
    return;
  }
  const next = lblImages[idx + 1];
  lblSelectImage(next.name);
  // Keep the newly-selected row visible in the scrolling list.
  const row = document.querySelector('.lbl-row[data-name="' + CSS.escape(next.name) + '"]');
  if (row && row.scrollIntoView) row.scrollIntoView({block: 'nearest'});
}

// Step back to the previous image in the list. Stops at the first image.
function lblSelectPrev() {
  if (!lblImages.length) return;
  const idx = lblImages.findIndex(im => im.name === lblCurrent);
  if (idx <= 0) return;
  const prev = lblImages[idx - 1];
  lblSelectImage(prev.name);
  const row = document.querySelector('.lbl-row[data-name="' + CSS.escape(prev.name) + '"]');
  if (row && row.scrollIntoView) row.scrollIntoView({block: 'nearest'});
}

// Annotation hotkeys. Active ONLY while the labeller panel is open, and ignored
// while typing in a text field / select. Non-WASD keys to avoid the drive map.
document.addEventListener('keydown', e => {
  const panel = document.getElementById('label-panel');
  if (!panel || !panel.classList.contains('open')) return;
  const t = e.target;
  if (t && (t.tagName === 'INPUT' || t.tagName === 'SELECT' || t.tagName === 'TEXTAREA')) return;

  switch (e.code) {
    case 'Enter':        lblSaveLabels(); break;                 // save (auto-advances to next)
    case 'KeyE':
    case 'BracketRight': lblSelectNext(); break;                 // next image
    case 'KeyQ':
    case 'BracketLeft':  lblSelectPrev(); break;                 // previous image
    case 'KeyR':         lblAutoDetect(); break;                 // rough auto-detect
    case 'KeyX':
    case 'Delete':
    case 'Backspace':    lblDeleteBox(); break;                  // delete selected box
    case 'Escape':       lblSel = -1; lblRedraw(); break;        // deselect
    default: {
      // Digits 1-9 / 0 select the class index (0 = 10th class).
      const m = /^Digit([0-9])$/.exec(e.code);
      if (!m) return;                                            // unhandled: leave default
      const n = parseInt(m[1], 10);
      const idx = (n === 0) ? 9 : n - 1;
      const sel = document.getElementById('lbl-class');
      if (sel && idx < sel.options.length) sel.value = idx;
    }
  }
  e.preventDefault();
});

function lblAddClass() {
  const inp = document.getElementById('lbl-newclass');
  const name = inp.value.trim();
  if (!name) return;
  fetch('/captures/classes', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({name}),
  })
    .then(r => r.json())
    .then(d => {
      if (d.ok) {
        lblClasses = d.classes;
        lblPopulateClassSelect();
        document.getElementById('lbl-class').value = d.index;
        inp.value = '';
      } else {
        const sts = document.getElementById('lbl-sts');
        sts.style.color = '#f85149';
        sts.textContent = d.error || 'add class failed';
      }
    })
    .catch(() => {});
}

// Rough auto-annotation: ask the server for CV-proposed boxes and append them
// to the current image's boxes for the user to review, correct, and save.
let lblAutoBusy = false;
function lblAutoDetect() {
  if (!lblCurrent || lblAutoBusy) return;   // guard against double-run duplicates
  lblAutoBusy = true;
  const sts = document.getElementById('lbl-sts');
  sts.style.color = '#8b949e';
  sts.textContent = 'Auto-detecting\\u2026';
  fetch('/captures/autolabel/' + encodeURIComponent(lblCurrent), {method: 'POST'})
    .then(r => r.json())
    .then(d => {
      if (d.ok) {
        (d.boxes || []).forEach(b => lblBoxes.push(b));
        lblSel = -1;
        lblRedraw();
        sts.style.color = (d.count > 0) ? '#3fb950' : '#8b949e';
        sts.textContent = (d.count > 0)
          ? ('\\u2713 +' + d.count + ' rough \\u2014 review & save')
          : 'no objects found';
      } else {
        sts.style.color = '#f85149';
        sts.textContent = d.error || 'auto-detect failed';
      }
    })
    .catch(() => { sts.style.color = '#f85149'; sts.textContent = 'request failed'; })
    .finally(() => { lblAutoBusy = false; });
}

// Redraw on window resize (boxes stay in normalized coords)
window.addEventListener('resize', () => { if (lblCurrent) lblRedraw(); });

// ===========================================================================
// Grab (arm action)
// ===========================================================================
let grabPolling = false;
let grabPollTimer = null;

function grabSetStatus(text, cls) {
  ['grab-sts-d', 'grab-sts-p'].forEach(id => {
    const el = document.getElementById(id);
    if (!el) return;
    el.textContent = text;
    el.className = 'grab-sts' + (cls ? ' ' + cls : '');
  });
}

function grabSetBtnsDisabled(disabled) {
  ['grab-btn-d', 'grab-btn-p'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.disabled = disabled;
  });
}

function grabStartPoll() {
  if (grabPolling) return;
  grabPolling = true;
  grabPollTimer = setInterval(grabPoll, 500);
}

function grabStopPoll() {
  grabPolling = false;
  clearInterval(grabPollTimer);
  grabPollTimer = null;
}

function grabPoll() {
  fetch('/grab/status')
    .then(r => r.json())
    .then(d => {
      if (!d.available) {
        grabSetStatus('unavailable', '');
        grabSetBtnsDisabled(true);
        grabStopPoll();
        return;
      }
      if (d.running) {
        grabSetStatus(d.stage || 'running\u2026', '');
        grabSetBtnsDisabled(true);
      } else {
        grabStopPoll();
        grabSetBtnsDisabled(false);
        if (d.last_success === true) {
          grabSetStatus('\u2713 ' + (d.last_message || 'success'), 'ok');
        } else if (d.last_success === false) {
          grabSetStatus('\u2717 ' + (d.last_message || 'failed'), 'err');
        } else {
          grabSetStatus('', '');
        }
      }
    })
    .catch(() => {});
}

function grab() {
  grabSetBtnsDisabled(true);
  grabSetStatus('sending\u2026', '');
  fetch('/grab', {method: 'POST', headers: {'Content-Type': 'application/json'},
                  body: JSON.stringify({object_hint: ''})})
    .then(r => r.json())
    .then(d => {
      if (d.ok) {
        grabSetStatus('goal sent', '');
        grabStartPoll();
      } else {
        grabSetStatus('\u2717 ' + (d.status || 'error'), 'err');
        grabSetBtnsDisabled(false);
      }
    })
    .catch(() => {
      grabSetStatus('\u2717 request failed', 'err');
      grabSetBtnsDisabled(false);
    });
}

function grabCheckAvailability() {
  fetch('/grab/status')
    .then(r => r.json())
    .then(d => {
      if (!d.available) {
        grabSetStatus('unavailable', '');
        grabSetBtnsDisabled(true);
      }
    })
    .catch(() => {});
}

// ===========================================================================
// Boot
// ===========================================================================
connect();
if (!isTouch) startMapRefresh();   // map panel is always shown under the camera
grabCheckAvailability();
</script>
</body>
</html>
"""


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
        if image_compressed:
            self.create_subscription(CompressedImage, image_topic, self._on_image, 10)
        else:
            self.create_subscription(Image, image_topic, self._on_raw_image, 10)
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
                    lines = [l.strip() for l in f if l.strip()]
                self._classes = lines
                return
            except OSError:
                pass
        # Seed from param
        param_val = self.get_parameter('capture_classes').value
        self._classes = list(param_val) if param_val else ['object']
        self._write_classes_file()

    def _write_classes_file(self) -> None:
        """Persist self._classes to classes.txt (caller must hold _classes_lock or be in __init__)."""
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

    def _on_raw_image(self, msg: Image):
        # Encode a raw sensor_msgs/Image to JPEG (simulation path). Supports the
        # common rgb8/bgr8 encodings; falls back to mono treatment otherwise.
        if not _PIL_AVAILABLE:
            return
        h, w = msg.height, msg.width
        if h == 0 or w == 0:
            return
        arr = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        enc = msg.encoding.lower()
        try:
            if enc in ('rgb8', 'bgr8'):
                arr = arr.reshape((h, w, 3))
                if enc == 'bgr8':
                    arr = arr[:, :, ::-1]
                img = _PILImage.fromarray(arr, 'RGB')
            elif enc in ('mono8', '8uc1'):
                img = _PILImage.fromarray(arr.reshape((h, w)), 'L')
            else:  # best-effort: assume 3-channel
                img = _PILImage.fromarray(arr.reshape((h, w, 3)), 'RGB')
        except ValueError:
            return
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=80)
        with self._frame_lock:
            self._latest_jpeg = buf.getvalue()

    def _on_map(self, msg: OccupancyGrid):
        if not _PIL_AVAILABLE:
            return
        w, h = msg.info.width, msg.info.height
        if w == 0 or h == 0:
            return
        data = np.frombuffer(bytes(msg.data), dtype=np.int8).reshape((h, w))
        rgb = np.full((h, w, 3), 128, dtype=np.uint8)   # unknown = mid-gray
        rgb[data == 0] = [220, 220, 220]                 # free = light
        rgb[data > 0]  = [20,  20,  20]                  # occupied = dark
        img = _PILImage.fromarray(np.flipud(rgb), 'RGB')
        buf = io.BytesIO()
        img.save(buf, format='PNG', optimize=True)
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
        if age > self._cmd_timeout:
            self._publish_twist(0.0, 0.0)

    # ---- public API for web handlers -------------------------------------

    def get_frame(self) -> Optional[bytes]:
        with self._frame_lock:
            return self._latest_jpeg

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
            if not self._grasp_client.server_is_ready():
                if not self._grasp_client.wait_for_server(timeout_sec=1.0):
                    return False, 'grasp action server not ready'
            goal = _GraspObjectAction.Goal()
            goal.object_hint = str(object_hint)
            self._grasp_running = True
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

    def grasp_status(self) -> dict:
        """Return a thread-safe snapshot of grasp state for the web handler."""
        with self._grasp_lock:
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
        self._launch_nav('mapping', 'slam_nav2.launch.py', [])
        return True, 'mapping started (slam_toolbox + nav2)'

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
        cov[0] = cov[7] = 0.25       # x, y variance
        cov[35] = 0.0685             # yaw variance
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
        """Convert a click on the rendered /map.png (which is vertically
        flipped) to a map-frame pose and send a NavigateToPose goal."""
        with self._map_lock:
            meta = dict(self._map_meta)
            ox, oy = self._map_origin
        if not meta:
            return False, 'no map yet'
        res, w, h = meta['resolution'], meta['width'], meta['height']
        ix = max(0, min(w - 1, int(ix)))
        iy = max(0, min(h - 1, int(iy)))
        grid_row = (h - 1) - iy   # the PNG was flipud'd before serving
        wx = ox + (ix + 0.5) * res
        wy = oy + (grid_row + 0.5) * res

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

    def apply_cmd(self, linear_x: float, angular_z: float):
        lx = max(-1.0, min(1.0, linear_x))  * self._max_linear
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
    return web.Response(text=_HTML, content_type='text/html')


async def handle_mjpeg(request: web.Request) -> web.StreamResponse:
    node: WebControlNode = request.app['node']
    boundary = b'--mjpegboundary'
    response = web.StreamResponse(headers={
        'Content-Type': 'multipart/x-mixed-replace; boundary=mjpegboundary',
        'Cache-Control': 'no-cache',
        'Connection': 'close',
    })
    await response.prepare(request)

    try:
        while True:
            frame = node.get_frame()
            if frame is not None:
                header = (
                    boundary + b'\r\n'
                    b'Content-Type: image/jpeg\r\n'
                    b'Content-Length: ' + str(len(frame)).encode() + b'\r\n\r\n'
                )
                await response.write(header + frame + b'\r\n')
            await asyncio.sleep(0.033)  # ~30 fps cap
    except (ConnectionResetError, asyncio.CancelledError):
        pass

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


async def handle_save_map(request: web.Request) -> web.Response:
    node: WebControlNode = request.app['node']
    ok, info = node.save_map()
    if ok:
        return web.json_response({'status': 'ok', 'path': info})
    return web.json_response({'status': 'error', 'msg': info}, status=500)


async def handle_start_mapping(request: web.Request) -> web.Response:
    node: WebControlNode = request.app['node']
    ok, msg = node.start_mapping()
    return web.json_response({'status': 'ok' if ok else 'error', 'msg': msg},
                             status=200 if ok else 400)


async def handle_start_navigation(request: web.Request) -> web.Response:
    node: WebControlNode = request.app['node']
    ok, msg = node.start_navigation()
    return web.json_response({'status': 'ok' if ok else 'error', 'msg': msg},
                             status=200 if ok else 400)


async def handle_stop_nav(request: web.Request) -> web.Response:
    node: WebControlNode = request.app['node']
    mode = node.stop_nav()
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
        ok, info = node.navigate_to_pixel(int(data['x']), int(data['y']))
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
        return web.json_response({'status': 'error', 'msg': f'bad request: {exc}'}, status=400)
    if ok:
        return web.json_response({'status': 'ok', 'goal': info})
    return web.json_response({'status': 'error', 'msg': info}, status=409)


async def handle_capture(request: web.Request) -> web.Response:
    node: WebControlNode = request.app['node']
    ok, info = node.save_capture()
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
    app.router.add_get('/', handle_index)
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
