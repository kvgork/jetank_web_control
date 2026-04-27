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
"""

import asyncio
import io
import json
import os
import subprocess
import threading
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import CompressedImage

try:
    import numpy as np
    from PIL import Image as _PILImage
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

try:
    from aiohttp import web
    import aiohttp
except ImportError:
    raise SystemExit(
        "aiohttp is required: pip install aiohttp"
    )

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
  .map-panel{display:none;width:320px;background:#010409;
             border-left:1px solid #30363d;flex-direction:column;
             overflow:hidden;flex-shrink:0}
  .map-header{padding:10px 12px;background:#161b22;border-bottom:1px solid #30363d;
              display:flex;flex-direction:column;gap:4px;flex-shrink:0}
  .map-title{font-size:.72rem;text-transform:uppercase;letter-spacing:1px;color:#58a6ff}
  .map-meta-txt{font-size:.65rem;color:#8b949e}
  .map-canvas-wrap{flex:1;overflow:hidden;display:flex;align-items:center;
                   justify-content:center;padding:4px}
  #map-img{image-rendering:pixelated;max-width:100%;max-height:100%;
           object-fit:contain;display:block;opacity:.3}
  #map-img.loaded{opacity:1}
  .map-footer{padding:8px 12px;background:#161b22;border-top:1px solid #30363d;
              display:flex;gap:8px;align-items:center;flex-shrink:0}
  .mbtn{padding:5px 12px;border-radius:5px;border:1px solid #30363d;
        background:#21262d;color:#c9d1d9;font-size:.72rem;cursor:pointer;
        font-family:monospace;transition:background .1s;white-space:nowrap}
  .mbtn:hover{background:#30363d}.mbtn:active{background:#58a6ff22}
  .mbtn:disabled{opacity:.5;cursor:default}
  .map-save-sts{font-size:.65rem;flex:1}
  /* mapping toggle at bottom of sidebar */
  .sidebar-footer{margin-top:auto;padding-top:10px;border-top:1px solid #30363d}
  /* never show map panel on touch */
  body.is-touch .map-panel{display:none!important}
  body.is-touch .sidebar-footer{display:none}
  /* mapping-mode toggle button states */
  .mbtn-on{background:#238636;border-color:#2ea043;color:#fff}
  .mbtn-on:hover{background:#2ea043}
</style>
</head>
<body>
<header>
  <h1>&#x1F916; JeTank</h1>
  <span id="ws-label" class="badge dot-err">&#x25CF; Disconnected</span>
  <span id="cam-label" class="badge" style="color:#8b949e">&#x25CF; Camera</span>
  <span id="ctrl-mode"></span>
</header>

<main>
  <!-- camera -->
  <div class="cam">
    <img id="cam-img" src="/stream.mjpg" alt="camera"
         onerror="scheduleReconnectStream()" onload="onImgLoad()">
    <div class="cam-overlay">left camera</div>
  </div>

  <!-- mapping panel: desktop only, shown when mapping mode is active -->
  <div class="map-panel" id="map-panel">
    <div class="map-header">
      <span class="map-title">&#x1F5FA; Live Map</span>
      <span class="map-meta-txt" id="map-meta-txt">Waiting for /map topic&#x2026;</span>
    </div>
    <div class="map-canvas-wrap">
      <img id="map-img" src="" alt=""
           onload="this.classList.add('loaded')"
           onerror="this.classList.remove('loaded');document.getElementById('map-meta-txt').textContent='Waiting for /map topic&#x2026;'">
    </div>
    <div class="map-footer">
      <button class="mbtn" id="save-map-btn" onclick="saveMap()">Save Map</button>
      <span class="map-save-sts" id="map-save-sts"></span>
    </div>
  </div>

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
    document.getElementById('ctrl-mode').textContent = '&#x1F4F1; Touch mode';
  } else {
    document.getElementById('ctrl-mode').textContent = '&#x1F5A5; Desktop mode';
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
// Mapping Mode (desktop only)
// ===========================================================================
let mappingMode = false;
let mapRefreshTimer = null;

function toggleMappingMode() {
  if (isTouch) return;
  mappingMode = !mappingMode;
  const panel = document.getElementById('map-panel');
  const btn   = document.getElementById('map-toggle-btn');
  if (mappingMode) {
    panel.style.display = 'flex';
    btn.classList.add('mbtn-on');
    btn.innerHTML = '&#x1F5FA; Stop Mapping';
    startMapRefresh();
  } else {
    panel.style.display = 'none';
    btn.classList.remove('mbtn-on');
    btn.innerHTML = '&#x1F5FA; Mapping Mode';
    stopMapRefresh();
  }
}

function startMapRefresh() {
  refreshMap();
  mapRefreshTimer = setInterval(refreshMap, 1500);
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

// ===========================================================================
// Boot
// ===========================================================================
connect();
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

        self._port = self.get_parameter('web_port').value
        image_topic = self.get_parameter('image_topic').value
        cmd_topic = self.get_parameter('cmd_vel_topic').value
        self._max_linear = self.get_parameter('max_linear_speed').value
        self._max_angular = self.get_parameter('max_angular_speed').value
        self._cmd_timeout = self.get_parameter('cmd_timeout_sec').value

        self._frame_lock = threading.Lock()
        self._latest_jpeg: Optional[bytes] = None

        self._cmd_lock = threading.Lock()
        self._last_cmd_time = 0.0

        self._map_lock = threading.Lock()
        self._latest_map_png: Optional[bytes] = None
        self._map_meta: dict = {}

        self._cmd_vel_pub = self.create_publisher(Twist, cmd_topic, 10)
        self.create_subscription(CompressedImage, image_topic, self._on_image, 10)
        self.create_subscription(OccupancyGrid, '/map', self._on_map, 1)

        # Watchdog: stop robot if commands stop arriving
        self.create_timer(0.1, self._watchdog_cb)

        self.get_logger().info(
            f'Web control: http://<jetson-ip>:{self._port}  '
            f'| camera: {image_topic}  | cmd_vel: {cmd_topic}'
        )

    # ---- callbacks --------------------------------------------------------

    def _on_image(self, msg: CompressedImage):
        with self._frame_lock:
            self._latest_jpeg = bytes(msg.data)

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
        with self._map_lock:
            self._latest_map_png = buf.getvalue()
            self._map_meta = {
                'resolution': round(float(msg.info.resolution), 4),
                'width': w,
                'height': h,
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

    def get_map_png(self) -> Optional[bytes]:
        with self._map_lock:
            return self._latest_map_png

    def get_map_meta(self) -> dict:
        with self._map_lock:
            return dict(self._map_meta)

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
    ts = time.strftime('%Y%m%d_%H%M%S')
    map_dir = os.path.expanduser('~/maps')
    os.makedirs(map_dir, exist_ok=True)
    map_path = os.path.join(map_dir, f'jetank_map_{ts}')
    try:
        result = subprocess.run(
            ['ros2', 'run', 'nav2_map_server', 'map_saver_cli', '-f', map_path],
            capture_output=True, text=True, timeout=15, env=os.environ,
        )
        if result.returncode == 0:
            return web.json_response({'status': 'ok', 'path': map_path})
        return web.json_response(
            {'status': 'error', 'msg': result.stderr.strip() or result.stdout.strip()},
            status=500,
        )
    except subprocess.TimeoutExpired:
        return web.json_response({'status': 'error', 'msg': 'map_saver_cli timed out'}, status=500)
    except FileNotFoundError:
        return web.json_response({'status': 'error', 'msg': 'ros2 command not found'}, status=500)


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

def build_app(node: WebControlNode) -> web.Application:
    app = web.Application()
    app['node'] = node
    app.router.add_get('/', handle_index)
    app.router.add_get('/stream.mjpg', handle_mjpeg)
    app.router.add_get('/ws', handle_websocket)
    app.router.add_get('/map.png', handle_map_png)
    app.router.add_get('/map_meta', handle_map_meta)
    app.router.add_post('/save_map', handle_save_map)
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
        node.destroy_node()
        rclpy.shutdown()
        ros_thread.join(timeout=2.0)


if __name__ == '__main__':
    main()
