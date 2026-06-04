"""
Tests for the pure module-level label helpers in web_control_node.

Import strategy: the helpers are module-level functions with no ROS or aiohttp
dependency, so we can import the module without a running ROS context as long as
rclpy.init() is never called at import time.  The module imports rclpy and
aiohttp at the top level (for the class/handler definitions), but neither
requires init()/running instances to be *importable*.  We stub both if absent.
"""

import importlib
import os
import sys
import types

import pytest


# ---------------------------------------------------------------------------
# Stub infrastructure
# ---------------------------------------------------------------------------

def _make_stub(name: str) -> types.ModuleType:
    """Create and register a minimal stub module in sys.modules."""
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


def _ensure_attr(mod_name: str, *attrs):
    mod = sys.modules.get(mod_name)
    if mod is None:
        return
    for a in attrs:
        if not hasattr(mod, a):
            # Create a trivial sentinel class
            setattr(mod, a, type(a, (), {})())


def _install_stubs():
    """Install minimal stubs for rclpy and aiohttp if they are not present."""
    # ---- rclpy ----
    if 'rclpy' not in sys.modules:
        try:
            import rclpy  # noqa: F401 — use real package when available
        except ImportError:
            rclpy_stub = _make_stub('rclpy')
            node_stub = _make_stub('rclpy.node')
            node_stub.Node = object
            rclpy_stub.node = node_stub
            action_stub = _make_stub('rclpy.action')
            action_stub.ActionClient = object
            rclpy_stub.action = action_stub

    # ---- message types ----
    for pkg in [
        'geometry_msgs', 'geometry_msgs.msg',
        'nav_msgs', 'nav_msgs.msg',
        'sensor_msgs', 'sensor_msgs.msg',
        'nav2_msgs', 'nav2_msgs.action',
    ]:
        if pkg not in sys.modules:
            _make_stub(pkg)

    _ensure_attr('geometry_msgs.msg', 'Twist', 'PoseStamped', 'PoseWithCovarianceStamped')
    _ensure_attr('nav_msgs.msg', 'OccupancyGrid')
    _ensure_attr('sensor_msgs.msg', 'CompressedImage', 'Image')
    _ensure_attr('nav2_msgs.action', 'NavigateToPose')

    # ---- aiohttp ----
    # The module does `from aiohttp import web; import aiohttp` inside a
    # try/except ImportError block.  We need 'aiohttp' in sys.modules with a
    # `web` attribute so the from-import succeeds.
    if 'aiohttp' not in sys.modules:
        try:
            import aiohttp  # noqa: F401 — use real package when available
        except ImportError:
            web_stub = _make_stub('aiohttp.web')
            web_stub.Application = object
            web_stub.Request = object
            web_stub.Response = object
            web_stub.StreamResponse = object
            web_stub.WebSocketResponse = object
            web_stub.json_response = None

            aio_stub = _make_stub('aiohttp')
            # `from aiohttp import web` needs web as an attribute
            aio_stub.web = web_stub
            # `aiohttp.WSMsgType` is referenced at runtime only (not import time)
            aio_stub.WSMsgType = type('WSMsgType', (), {'TEXT': 1, 'ERROR': 2, 'CLOSE': 3})


# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------

_install_stubs()

try:
    pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)

    _mod = importlib.import_module('jetank_web_control.web_control_node')
    _safe_capture_name = _mod._safe_capture_name
    _yolo_parse = _mod._yolo_parse
    _yolo_serialize = _mod._yolo_serialize
except Exception as exc:
    pytest.skip(f'Could not import web_control_node: {exc}', allow_module_level=True)


# ---------------------------------------------------------------------------
# _safe_capture_name
# ---------------------------------------------------------------------------

VALID_NAME = '20260531T101010_0001.jpg'

VALID_BOX = {'cls': 0, 'cx': 0.5, 'cy': 0.5, 'w': 0.2, 'h': 0.3}
VALID_LINE = '0 0.500000 0.500000 0.200000 0.300000'


class TestSafeCaptureName:
    def test_valid_timestamped(self):
        assert _safe_capture_name(VALID_NAME) == VALID_NAME

    def test_valid_simple(self):
        assert _safe_capture_name('image.jpg') == 'image.jpg'

    def test_valid_uppercase_ext(self):
        assert _safe_capture_name('frame.JPG') == 'frame.JPG'

    def test_valid_mixed_ext(self):
        assert _safe_capture_name('frame.Jpg') == 'frame.Jpg'

    def test_valid_with_dots_dashes(self):
        assert _safe_capture_name('my-image.v2.jpg') == 'my-image.v2.jpg'

    def test_rejects_path_traversal(self):
        assert _safe_capture_name('../etc/passwd') is None

    def test_rejects_forward_slash(self):
        assert _safe_capture_name('a/b.jpg') is None

    def test_rejects_backslash(self):
        assert _safe_capture_name('a\\b.jpg') is None

    def test_rejects_wrong_extension(self):
        assert _safe_capture_name('evil.png') is None

    def test_rejects_dotdot_alone(self):
        assert _safe_capture_name('..') is None

    def test_rejects_empty_string(self):
        assert _safe_capture_name('') is None

    def test_accepts_double_dot_in_name(self):
        # 'foo..bar.jpg' is safe: the ".." is not a dot-separated path component
        # (split('.') gives ['foo','','bar','jpg'] — no '..' element)
        assert _safe_capture_name('foo..bar.jpg') == 'foo..bar.jpg'

    def test_rejects_dotdot_as_component(self):
        # The literal '..' directory-traversal component is always rejected
        # (the == '..' guard catches it before the regex)
        assert _safe_capture_name('..') is None
        # '..jpg' is NOT a path traversal - just an unusual name, is accepted
        # (the regex ^[A-Za-z0-9._-]+\.jpg$ matches it since '.' is allowed)
        # We test the meaningful security boundary instead: an explicit path
        assert _safe_capture_name('../secret.jpg') is None

    def test_rejects_no_extension(self):
        assert _safe_capture_name('imagejpg') is None

    def test_rejects_spaces(self):
        assert _safe_capture_name('my file.jpg') is None


# ---------------------------------------------------------------------------
# _yolo_parse
# ---------------------------------------------------------------------------


class TestYoloParse:
    def test_valid_single_line(self):
        boxes = _yolo_parse(VALID_LINE, n_classes=1)
        assert len(boxes) == 1
        b = boxes[0]
        assert b['cls'] == 0
        assert abs(b['cx'] - 0.5) < 1e-9
        assert abs(b['w'] - 0.2) < 1e-9

    def test_valid_multiple_lines(self):
        text = '0 0.1 0.2 0.3 0.4\n1 0.5 0.6 0.1 0.1\n'
        boxes = _yolo_parse(text, n_classes=2)
        assert len(boxes) == 2
        assert boxes[0]['cls'] == 0
        assert boxes[1]['cls'] == 1

    def test_skips_blank_lines(self):
        text = '\n  \n' + VALID_LINE + '\n\n'
        boxes = _yolo_parse(text, n_classes=1)
        assert len(boxes) == 1

    def test_skips_wrong_field_count(self):
        text = '0 0.5 0.5\n' + VALID_LINE
        boxes = _yolo_parse(text, n_classes=1)
        assert len(boxes) == 1

    def test_skips_non_numeric(self):
        text = 'a b c d e\n' + VALID_LINE
        boxes = _yolo_parse(text, n_classes=1)
        assert len(boxes) == 1

    def test_drops_out_of_range_cls_too_high(self):
        text = '5 0.5 0.5 0.1 0.1\n' + VALID_LINE
        boxes = _yolo_parse(text, n_classes=1)
        assert len(boxes) == 1
        assert boxes[0]['cls'] == 0

    def test_drops_negative_cls(self):
        text = '-1 0.5 0.5 0.1 0.1\n' + VALID_LINE
        boxes = _yolo_parse(text, n_classes=1)
        assert len(boxes) == 1

    def test_drops_cx_out_of_range(self):
        text = '0 1.5 0.5 0.1 0.1\n' + VALID_LINE
        boxes = _yolo_parse(text, n_classes=1)
        assert len(boxes) == 1

    def test_drops_negative_coords(self):
        text = '0 -0.1 0.5 0.1 0.1\n' + VALID_LINE
        boxes = _yolo_parse(text, n_classes=1)
        assert len(boxes) == 1

    def test_boundary_coords_accepted(self):
        text = '0 0.0 0.0 0.0 0.0\n0 1.0 1.0 1.0 1.0'
        boxes = _yolo_parse(text, n_classes=1)
        assert len(boxes) == 2

    def test_empty_text_returns_empty(self):
        assert _yolo_parse('', n_classes=2) == []

    def test_zero_classes_drops_all(self):
        boxes = _yolo_parse(VALID_LINE, n_classes=0)
        assert boxes == []

    def test_never_raises_on_garbage(self):
        result = _yolo_parse('this is completely garbage\x00\xff', n_classes=10)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# _yolo_serialize
# ---------------------------------------------------------------------------


class TestYoloSerialize:
    def test_empty_list(self):
        assert _yolo_serialize([]) == ''

    def test_single_box_format(self):
        text = _yolo_serialize([VALID_BOX])
        assert text == '0 0.500000 0.500000 0.200000 0.300000\n'

    def test_trailing_newline_per_line(self):
        text = _yolo_serialize([VALID_BOX, VALID_BOX])
        lines = text.splitlines()
        assert len(lines) == 2
        assert text.endswith('\n')

    def test_six_decimal_places(self):
        box = {'cls': 1, 'cx': 1 / 3, 'cy': 2 / 3, 'w': 0.1, 'h': 0.2}
        text = _yolo_serialize([box])
        parts = text.strip().split()
        assert parts[1] == f'{1/3:.6f}'
        assert parts[2] == f'{2/3:.6f}'

    def test_round_trip(self):
        boxes_in = [
            {'cls': 0, 'cx': 0.1, 'cy': 0.2, 'w': 0.3, 'h': 0.4},
            {'cls': 2, 'cx': 0.9, 'cy': 0.8, 'w': 0.05, 'h': 0.07},
        ]
        text = _yolo_serialize(boxes_in)
        boxes_out = _yolo_parse(text, n_classes=3)
        assert len(boxes_out) == 2
        for a, b in zip(boxes_in, boxes_out):
            assert a['cls'] == b['cls']
            assert abs(a['cx'] - b['cx']) < 1e-5
            assert abs(a['cy'] - b['cy']) < 1e-5
            assert abs(a['w'] - b['w']) < 1e-5
            assert abs(a['h'] - b['h']) < 1e-5
