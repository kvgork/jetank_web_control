"""
Shared test scaffolding for the jetank_web_control test suite.

The modules under test import ROS (rclpy + message packages) and aiohttp at
the top level, but the tested helpers are pure Python: no ``rclpy.init()``,
no running event loop. On a bare interpreter (no ROS / aiohttp installed)
this conftest stubs just enough of those packages for the imports to
succeed; when the real packages are available they are used untouched.

pytest imports this file before collecting any test module in this
directory, so the test files can import ``jetank_web_control.*`` directly
without repeating the stub/sys.path boilerplate.
"""

import importlib
import os
import sys
import types


def _make_stub(name: str) -> types.ModuleType:
    """Create and register a minimal stub module in sys.modules."""
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


def _ensure_attr(mod_name: str, *attrs):
    """Add trivial sentinel classes for missing attributes on a stub module."""
    mod = sys.modules.get(mod_name)
    if mod is None:
        return
    for a in attrs:
        if not hasattr(mod, a):
            setattr(mod, a, type(a, (), {}))


class _Vec3:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0


class StubTwist:
    """Minimal stand-in for geometry_msgs/Twist with .linear/.angular vectors."""

    def __init__(self):
        self.linear = _Vec3()
        self.angular = _Vec3()


class _StubHeader:
    def __init__(self):
        self.stamp = None
        self.frame_id = ''


class StubTwistStamped:
    """Minimal stand-in for geometry_msgs/TwistStamped (header + twist)."""

    def __init__(self):
        self.header = _StubHeader()
        self.twist = StubTwist()


def _install_stubs():
    """Stub ROS / aiohttp deps ONLY when the real packages are unavailable."""
    # ---- rclpy ----
    if 'rclpy' not in sys.modules:
        try:
            import rclpy  # noqa: F401 — prefer the real package when present
        except ImportError:
            rclpy_stub = _make_stub('rclpy')
            node_stub = _make_stub('rclpy.node')
            node_stub.Node = object
            rclpy_stub.node = node_stub
            action_stub = _make_stub('rclpy.action')
            action_stub.ActionClient = object
            rclpy_stub.action = action_stub
            qos_stub = _make_stub('rclpy.qos')
            for _q in ('DurabilityPolicy', 'HistoryPolicy', 'QoSProfile',
                       'ReliabilityPolicy'):
                setattr(qos_stub, _q, object)
            rclpy_stub.qos = qos_stub

    # ---- geometry_msgs ----
    # Functional Twist/TwistStamped stubs: the cmd_vel_bridge tests construct
    # and inspect these, so sentinel classes are not enough.
    if 'geometry_msgs.msg' not in sys.modules:
        try:
            import geometry_msgs.msg  # noqa: F401 — prefer real messages
        except ImportError:
            _make_stub('geometry_msgs')
            gm = _make_stub('geometry_msgs.msg')
            gm.Twist = StubTwist
            gm.TwistStamped = StubTwistStamped
            for attr in ('PoseStamped', 'PoseWithCovarianceStamped'):
                setattr(gm, attr, type(attr, (), {}))

    # ---- remaining message / optional packages ----
    # Prefer the real package in every case: real geometry_msgs message
    # constructors lazily import std_msgs.msg.Header, so an eager stub of an
    # otherwise-available package would break them at runtime.
    for pkg in [
        'nav_msgs', 'nav_msgs.msg',
        'sensor_msgs', 'sensor_msgs.msg',
        'nav2_msgs', 'nav2_msgs.action',
        'vision_msgs', 'vision_msgs.msg',
        'std_msgs', 'std_msgs.msg',
        # jetank_manipulation / jetank_mission are imported with try/except
        # in web_control_node, so no stub is strictly needed; add them anyway
        # for offline test runs.
        'jetank_manipulation', 'jetank_manipulation.action',
        'jetank_mission', 'jetank_mission.action',
    ]:
        if pkg in sys.modules:
            continue
        try:
            importlib.import_module(pkg)
        except ImportError:
            _make_stub(pkg)
    _ensure_attr('nav_msgs.msg', 'OccupancyGrid')
    _ensure_attr('sensor_msgs.msg', 'CompressedImage', 'Image')
    _ensure_attr('nav2_msgs.action', 'NavigateToPose')
    _ensure_attr('vision_msgs.msg', 'Detection2DArray')
    _ensure_attr('std_msgs.msg', 'String')

    # ---- aiohttp ----
    # web_control_node does `from aiohttp import web; import aiohttp` in a
    # try/except that raises SystemExit when missing, so 'aiohttp' must be in
    # sys.modules with a `web` attribute for the from-import to succeed.
    if 'aiohttp' not in sys.modules:
        try:
            import aiohttp  # noqa: F401 — prefer the real package when present
        except ImportError:
            web_stub = _make_stub('aiohttp.web')
            web_stub.Application = object
            web_stub.Request = object
            web_stub.Response = object
            web_stub.StreamResponse = object
            web_stub.WebSocketResponse = object
            web_stub.json_response = None
            aio_stub = _make_stub('aiohttp')
            aio_stub.web = web_stub
            # `aiohttp.WSMsgType` is referenced at runtime only (not import time)
            aio_stub.WSMsgType = type(
                'WSMsgType', (), {'TEXT': 1, 'ERROR': 2, 'CLOSE': 3})


# Run at import time: pytest loads conftest.py before the test modules, so
# the stubs and the package path are in place for their top-level imports.
_install_stubs()

_pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)
