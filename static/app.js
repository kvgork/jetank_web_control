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
    document.getElementById('ctrl-mode').textContent = '\u1F4F1 Touch mode';
  } else {
    document.getElementById('ctrl-mode').textContent = '\u1F5A5 Desktop mode';
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
    wsLabel.textContent = '\u25CF Connected';
    startLoop();
  };
  ws.onclose = () => {
    wsLabel.className = 'badge dot-err';
    wsLabel.textContent = '\u25CF Disconnected';
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
  camLabel.textContent = '\u25CF Camera';
}
function scheduleReconnectStream() {
  camLabel.className = 'badge dot-err';
  camLabel.textContent = '\u25CF No stream';
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

// Shared object-fit:contain letterbox geometry. The rendered content is
// scaled by min(box/nat) and centred inside the element box; returns
// {scale, cw, ch, left, top} (left/top in viewport coords) or null when the
// image has no natural size yet. ALL overlay/click mappings must go through
// this helper — a diverged copy of this math once caused a real
// click-mapping bug (see clickToPixel).
function containRect(img) {
  const natW = img.naturalWidth, natH = img.naturalHeight;
  if (!natW || !natH) return null;
  const r = img.getBoundingClientRect();
  const scale = Math.min(r.width / natW, r.height / natH);
  const cw = natW * scale, ch = natH * scale;
  return {
    scale: scale,
    cw: cw, ch: ch,
    left: r.left + (r.width - cw) / 2,
    top:  r.top  + (r.height - ch) / 2,
  };
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
                                  : (age === null ? 'no detector' : 'searching\u2026');

  const natW = camImg.naturalWidth, natH = camImg.naturalHeight;
  const rect = containRect(camImg);
  if (!rect || !boxes.length) return;
  const cr = cv.getBoundingClientRect();
  const dispW = rect.cw, dispH = rect.ch;
  const ox = rect.left - cr.left;
  const oy = rect.top  - cr.top;

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
    btn.innerHTML = '&#x23F9; Stop Mapping';
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
  loadDeposit();
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
          navMsg('\u2713 localized (' + p.x + ', ' + p.y + ')', true);
        } else if (Date.now() - localizeStart > 25000) {
          localizing = false;
          load.style.display = 'none';
          navMsg('localization uncertain \u2014 drive a little or re-try', false);
        }
      }
    })
    .catch(() => {});
}

// Last AMCL pose + mission markers — all drawn together onto the single
// overlay canvas (redrawOverlay clears once, then draws every layer).
let lastPose = null;       // {x,y,yaw,...} from /robot_pose
let fetchMarker = null;    // {x,y} world point of the last fetch goal
let depositMarker = null;  // {x,y} world point of the stored deposit area

// Map a map-frame world point to overlay-canvas pixel coords, accounting for
// the object-fit:contain letterboxing and the vertically-flipped PNG. Returns
// {px,py,scale} or null when the geometry isn't ready yet.
function worldToCanvas(wx, wy) {
  const img = document.getElementById('map-img');
  const cv = document.getElementById('map-overlay');
  if (!lastMeta || !lastMeta.resolution) return null;
  if (!cv.width || !cv.height) return null;
  const rect = containRect(img);
  if (!rect) return null;
  const cr = cv.getBoundingClientRect();
  const col = (wx - lastMeta.origin_x) / lastMeta.resolution;
  const gridRow = (wy - lastMeta.origin_y) / lastMeta.resolution;
  const ix = col, iy = (lastMeta.height - 1) - gridRow;  // PNG is flipped
  return {
    px: (rect.left - cr.left) + ix * rect.scale,
    py: (rect.top - cr.top) + iy * rect.scale,
    scale: rect.scale,
  };
}

function drawMissionMarker(ctx, pt, color, label) {
  ctx.save();
  ctx.translate(pt.px, pt.py);
  ctx.fillStyle = color; ctx.strokeStyle = '#fff'; ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(0, 0, 6, 0, Math.PI * 2);
  ctx.fill(); ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(-9, 0); ctx.lineTo(9, 0); ctx.moveTo(0, -9); ctx.lineTo(0, 9);
  ctx.stroke();
  if (label) {
    ctx.font = '10px monospace'; ctx.fillStyle = '#fff';
    ctx.strokeStyle = 'rgba(0,0,0,.7)'; ctx.lineWidth = 3;
    ctx.strokeText(label, 9, -8); ctx.fillText(label, 9, -8);
  }
  ctx.restore();
}

function redrawOverlay() {
  const cv = document.getElementById('map-overlay');
  cv.width = cv.clientWidth; cv.height = cv.clientHeight;
  if (!cv.width || !cv.height) return;
  const ctx = cv.getContext('2d');
  ctx.clearRect(0, 0, cv.width, cv.height);
  // deposit area (persisted) — purple
  if (depositMarker) {
    const d = worldToCanvas(depositMarker.x, depositMarker.y);
    if (d) drawMissionMarker(ctx, d, '#a371f7', 'deposit');
  }
  // last fetch goal — blue
  if (fetchMarker) {
    const f = worldToCanvas(fetchMarker.x, fetchMarker.y);
    if (f) drawMissionMarker(ctx, f, '#388bfd', 'fetch');
  }
  // robot pose arrow — red
  if (lastPose) {
    const r = worldToCanvas(lastPose.x, lastPose.y);
    if (r) {
      ctx.save();
      ctx.translate(r.px, r.py);
      ctx.rotate(-lastPose.yaw);          // image y is down => screen angle = -yaw
      ctx.fillStyle = '#ff3b30'; ctx.strokeStyle = '#fff'; ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(16, 0); ctx.lineTo(-10, -9); ctx.lineTo(-4, 0); ctx.lineTo(-10, 9);
      ctx.closePath(); ctx.fill(); ctx.stroke();
      ctx.restore();
    }
  }
}

function drawRobotArrow(p) {
  lastPose = p;
  redrawOverlay();
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
        meta.width + '\u00D7' + meta.height + 'px \u00B7 ' +
        w + '\u00D7' + h + 'm \u00B7 ' + meta.resolution + 'm/px';
    })
    .catch(() => {});
  document.getElementById('map-img').src = '/map.png?t=' + Date.now();
  // keep mission markers visible even when no AMCL pose is being drawn
  redrawOverlay();
}

function saveMap() {
  const btn = document.getElementById('save-map-btn');
  const sts = document.getElementById('map-save-sts');
  btn.disabled = true;
  btn.textContent = 'Saving\u2026';
  sts.textContent = '';
  fetch('/save_map', {method: 'POST'})
    .then(r => r.json())
    .then(d => {
      if (d.status === 'ok') {
        sts.style.color = '#3fb950';
        sts.textContent = '\u2713 ' + d.path;
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
        sts.textContent = d.count + ' saved \u00B7 ' + d.filename;
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
      navRunning = s.running;
      document.getElementById('start-nav-btn').disabled = !s.has_map;
      const hint = document.getElementById('nav-hint');
      if (s.running === 'mapping')      hint.textContent = 'Mapping active \u00B7 drive with the joystick to build the map, then Save Map.';
      else if (s.running === 'navigation') hint.textContent = 'Navigation active (saved map) \u00B7 click the map to send the robot.';
      else hint.textContent = s.has_map ? 'Saved map available \u00B7 Start Mapping or Navigate (saved map).'
                                        : 'Start Mapping, drive around, then Save Map.';
    })
    .catch(() => {});
}

function startMapping() {
  navMsg('Starting mapping\u2026', true);
  fetch('/start_mapping', {method: 'POST'})
    .then(r => r.json())
    .then(d => navMsg(d.status === 'ok' ? '\u2713 mapping started' : (d.msg || 'error'), d.status === 'ok'))
    .catch(() => navMsg('Request failed', false))
    .finally(refreshNavStatus);
}

function startNavigation() {
  navMsg('Starting navigation\u2026', true);
  fetch('/start_navigation', {method: 'POST'})
    .then(r => r.json())
    .then(d => {
      navMsg(d.status === 'ok' ? 'determining position\u2026' : (d.msg || 'error'),
             d.status === 'ok');
      if (d.status === 'ok') {
        // Show the "determining robot position" loader until AMCL converges
        // (refreshRobotPose hides it and draws the pose arrow).
        localizing = true;
        localizeStart = Date.now();
        document.getElementById('map-loading-txt').textContent =
          'Determining robot position\u2026';
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

// Map click-mode: 'navigate' (existing NavigateToPose), 'fetch' (mission goal),
// or 'deposit' (store deposit area). Default keeps the original behaviour.
let mapMode = 'navigate';
let navRunning = null;

function setMapMode(mode) {
  mapMode = mode;
  ['navigate', 'fetch', 'deposit'].forEach(m => {
    const btn = document.getElementById('mode-' + m);
    if (btn) btn.classList.toggle('mbtn-on', m === mode);
  });
}

function missionMsg(text, ok) {
  const sts = document.getElementById('mission-sts');
  if (!sts) return;
  sts.style.color = ok ? '#3fb950' : '#f85149';
  sts.textContent = text;
}

// Compute the /map.png pixel under a click.
// #map-img is object-fit:contain, so the rendered map is letterboxed inside the
// element box (black bars on the axis where the panel is bigger than the map's
// aspect). Map the click through the SAME containRect geometry worldToCanvas
// uses, or only the middle band is reachable on the letterboxed axis. (The old
// code divided by the full element box, so a panel wider than the ~101x89 map
// made only the middle COLUMNS selectable: lots of range top-to-bottom, almost
// none left-to-right.)
function clickToPixel(ev) {
  const img = document.getElementById('map-img');
  const rect = containRect(img);
  if (!rect) return null;
  const lx = ev.clientX - rect.left;
  const ly = ev.clientY - rect.top;
  if (lx < 0 || ly < 0 || lx > rect.cw || ly > rect.ch) return null;  // clicked the letterbox
  // Clamp to valid PNG indices: the rightmost/bottom edge band would otherwise
  // round up to natW/natH (one past the last cell).
  return {
    x: Math.min(img.naturalWidth - 1, Math.max(0, Math.round(lx / rect.scale))),
    y: Math.min(img.naturalHeight - 1, Math.max(0, Math.round(ly / rect.scale))),
  };
}

function mapClick(ev) {
  const px = clickToPixel(ev);
  if (!px) return;
  if (navRunning === 'mapping') { navMsg('Driving mode \u2014 use the joystick to build the map, then Save Map.', true); return; }
  if (mapMode === 'fetch')   { sendMissionGoal(px.x, px.y); return; }
  if (mapMode === 'deposit') { setDeposit(px.x, px.y); return; }
  // ---- navigate mode (unchanged behaviour) ----
  navMsg('Sending goal\u2026', true);
  fetch('/navigate', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({x: px.x, y: px.y}),
  })
    .then(r => r.json())
    .then(d => {
      if (d.status === 'ok') navMsg('\u2713 goal (' + d.goal.x + ', ' + d.goal.y + ') m', true);
      else navMsg(d.msg || 'goal rejected', false);
    })
    .catch(() => navMsg('Request failed', false));
}

function sendMissionGoal(ix, iy) {
  missionMsg('Sending fetch goal\u2026', true);
  fetch('/mission/goal', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ix: ix, iy: iy}),
  })
    .then(r => r.json().then(d => ({ok: r.ok, d: d})))
    .then(({ok, d}) => {
      if (ok && d.x !== undefined) {
        fetchMarker = {x: d.x, y: d.y};
        redrawOverlay();
        if (d.status === 'mission_unavailable') {
          missionMsg('Mission: server unavailable', false);
        } else {
          missionMsg('Mission: started\u2026', true);
          startMissionPoll();   // live status until terminal
        }
      } else {
        missionMsg(d.msg || 'no map', false);
      }
    })
    .catch(() => missionMsg('Request failed', false));
}

// ---- Live mission status polling (M6) -------------------------------------
let missionPollTimer = null;
const MISSION_POLL_MS = 1000;

// Map a coordinator status token to a friendly UI line. Terminal states get a
// check / cross; in-progress states get an ellipsis.
function missionLine(status) {
  const tok = (status || '').trim().split(/\s+/)[0].toUpperCase();
  if (tok === 'DONE')               return ['Mission: DONE \u2713', true];
  if (tok === 'FAILED')             return ['Mission: FAILED \u2014 ' + status.replace(/^FAILED\s*[\u2014-]*\s*/i, ''), false];
  if (tok === 'CANCELLED')          return ['Mission: cancelled', false];
  if (tok === 'IDLE' || tok === '') return ['', true];
  return ['Mission: ' + tok + '\u2026', true];
}

function startMissionPoll() {
  if (missionPollTimer) return;
  missionPoll();
  missionPollTimer = setInterval(missionPoll, MISSION_POLL_MS);
}

function stopMissionPoll() {
  if (missionPollTimer) { clearInterval(missionPollTimer); missionPollTimer = null; }
}

function missionPoll() {
  fetch('/mission/status')
    .then(r => r.ok ? r.json() : null)
    .then(s => {
      if (!s) return;
      const [text, ok] = missionLine(s.status);
      if (text) missionMsg(text, ok);
      // Stop once the coordinator reports a terminal state AND nothing is active.
      if (isTerminalStatus(s.status) && !s.active) stopMissionPoll();
    })
    .catch(() => {});
}

// Mirrors the server-side is_terminal_mission_status helper (kept in sync).
function isTerminalStatus(status) {
  if (!status) return true;
  const tok = status.trim().split(/\s+/)[0].toUpperCase();
  return ['DONE', 'FAILED', 'CANCELLED', 'IDLE'].includes(tok);
}

function cancelMission() {
  missionMsg('Cancelling\u2026', false);
  fetch('/mission/cancel', {method: 'POST'})
    .then(r => r.json())
    .then(() => { /* status poll will reflect CANCELLED */ })
    .catch(() => missionMsg('Cancel request failed', false));
}

function setDeposit(ix, iy) {
  missionMsg('Setting deposit area\u2026', true);
  fetch('/mission/deposit', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ix: ix, iy: iy}),
  })
    .then(r => r.json().then(d => ({ok: r.ok, d: d})))
    .then(({ok, d}) => {
      if (ok && d.x !== undefined) {
        depositMarker = {x: d.x, y: d.y};
        redrawOverlay();
        missionMsg('\u2713 Deposit set (' + d.x + ', ' + d.y + ') m', true);
      } else {
        missionMsg(d.msg || 'no map', false);
      }
    })
    .catch(() => missionMsg('Request failed', false));
}

function loadDeposit() {
  fetch('/mission/deposit')
    .then(r => r.ok ? r.json() : null)
    .then(d => {
      if (d && d.x !== undefined) { depositMarker = {x: d.x, y: d.y}; redrawOverlay(); }
    })
    .catch(() => {});
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
    badge.textContent = img.labelled ? ('\u2713 ' + img.n_boxes) : 'n';
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
    badge.textContent = '\u2713 ' + boxes.length;
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
  const rect = containRect(img);
  if (!rect) return [0, 0];
  const cr = cv.getBoundingClientRect();
  return [(rect.left - cr.left) + nx * rect.cw,
          (rect.top  - cr.top)  + ny * rect.ch];
}

// Convert canvas pixel coords to normalized [0,1] (clamped).
function lblCanvasToNorm(px, py) {
  const img = document.getElementById('lbl-img');
  const cv = document.getElementById('lbl-overlay');
  const rect = containRect(img);
  if (!rect) return [0, 0];
  const cr = cv.getBoundingClientRect();
  const nx = (px - (rect.left - cr.left)) / rect.cw;
  const ny = (py - (rect.top  - cr.top))  / rect.ch;
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
  sts.textContent = 'Saving\u2026';
  fetch('/captures/labels/' + encodeURIComponent(lblCurrent), {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({boxes: lblBoxes}),
  })
    .then(r => r.json())
    .then(d => {
      if (d.ok) {
        sts.style.color = '#3fb950';
        sts.textContent = '\u2713 saved';
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
    sts.textContent = '\u2713 saved \u2014 last image';
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
  sts.textContent = 'Auto-detecting\u2026';
  fetch('/captures/autolabel/' + encodeURIComponent(lblCurrent), {method: 'POST'})
    .then(r => r.json())
    .then(d => {
      if (d.ok) {
        (d.boxes || []).forEach(b => lblBoxes.push(b));
        lblSel = -1;
        lblRedraw();
        sts.style.color = (d.count > 0) ? '#3fb950' : '#8b949e';
        sts.textContent = (d.count > 0)
          ? ('\u2713 +' + d.count + ' rough \u2014 review & save')
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
        grabSetStatus(d.stage || 'running…', '');
        grabSetBtnsDisabled(true);
      } else {
        grabStopPoll();
        grabSetBtnsDisabled(false);
        if (d.last_success === true) {
          grabSetStatus('✓ ' + (d.last_message || 'success'), 'ok');
        } else if (d.last_success === false) {
          grabSetStatus('✗ ' + (d.last_message || 'failed'), 'err');
        } else {
          grabSetStatus('', '');
        }
      }
    })
    .catch(() => {});
}

function grab() {
  grabSetBtnsDisabled(true);
  grabSetStatus('sending…', '');
  fetch('/grab', {method: 'POST', headers: {'Content-Type': 'application/json'},
                  body: JSON.stringify({object_hint: ''})})
    .then(r => r.json())
    .then(d => {
      if (d.ok) {
        grabSetStatus('goal sent', '');
        grabStartPoll();
      } else {
        grabSetStatus('✗ ' + (d.status || 'error'), 'err');
        grabSetBtnsDisabled(false);
      }
    })
    .catch(() => {
      grabSetStatus('✗ request failed', 'err');
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
