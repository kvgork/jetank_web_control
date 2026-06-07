"""
Pure-logic tests for the cmd_vel mux/bridge and the web node's twist math.

These import the package's *real* modules and exercise their actual logic:

  * ``cmd_vel_bridge._is_nonzero`` — the 1e-3 dead-band threshold.
  * ``cmd_vel_bridge.CmdVelBridge._tick`` — the teleop-priority mux decision
    (active teleop wins; else fresh nav; else zero), run against a fake ``self``
    so no ROS context / timers are needed.
  * ``web_control_node.WebControlNode.apply_cmd`` — the [-1, 1] clamp then
    per-axis max-speed scaling, run against a fake ``self``.

Import strategy mirrors test_labels.py: stub rclpy / geometry_msgs only if the
real packages are unavailable (a bare env), so the module is importable either
way without ever calling rclpy.init().
"""

import importlib
import os
import sys
import types

import pytest


# ---------------------------------------------------------------------------
# Stub infrastructure (only used when the real ROS packages are absent)
# ---------------------------------------------------------------------------

def _make_stub(name):
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


class _Vec3:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0


class _StubTwist:
    """Minimal stand-in for geometry_msgs/Twist with .linear/.angular vectors."""

    def __init__(self):
        self.linear = _Vec3()
        self.angular = _Vec3()


class _StubHeader:
    def __init__(self):
        self.stamp = None
        self.frame_id = ''


class _StubTwistStamped:
    """Minimal stand-in for geometry_msgs/TwistStamped (header + twist)."""

    def __init__(self):
        self.header = _StubHeader()
        self.twist = _StubTwist()


def _ensure_msg_attr(mod_name, *attrs):
    mod = sys.modules.get(mod_name)
    if mod is None:
        return
    for a in attrs:
        if not hasattr(mod, a):
            setattr(mod, a, type(a, (), {}))


def _install_stubs():
    """Stub ROS deps ONLY when the real packages are unavailable (bare env)."""
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

    if 'geometry_msgs.msg' not in sys.modules:
        try:
            import geometry_msgs.msg  # noqa: F401 — prefer real messages
        except ImportError:
            _make_stub('geometry_msgs')
            gm = _make_stub('geometry_msgs.msg')
            gm.Twist = _StubTwist
            gm.TwistStamped = _StubTwistStamped
            for attr in ('PoseStamped', 'PoseWithCovarianceStamped'):
                setattr(gm, attr, type(attr, (), {}))

    for pkg in [
        'nav_msgs', 'nav_msgs.msg',
        'sensor_msgs', 'sensor_msgs.msg',
        'nav2_msgs', 'nav2_msgs.action',
        'vision_msgs', 'vision_msgs.msg',
        'jetank_manipulation', 'jetank_manipulation.action',
    ]:
        if pkg not in sys.modules:
            _make_stub(pkg)
    _ensure_msg_attr('nav_msgs.msg', 'OccupancyGrid')
    _ensure_msg_attr('sensor_msgs.msg', 'CompressedImage', 'Image')
    _ensure_msg_attr('nav2_msgs.action', 'NavigateToPose')
    _ensure_msg_attr('vision_msgs.msg', 'Detection2DArray')


_install_stubs()

_pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

try:
    _bridge = importlib.import_module('jetank_web_control.cmd_vel_bridge')
    _is_nonzero = _bridge._is_nonzero
    CmdVelBridge = _bridge.CmdVelBridge
    Twist = sys.modules['geometry_msgs.msg'].Twist
except Exception as exc:  # pragma: no cover
    pytest.skip(f'Could not import cmd_vel_bridge: {exc}', allow_module_level=True)

try:
    _wcn = importlib.import_module('jetank_web_control.web_control_node')
    WebControlNode = _wcn.WebControlNode
except Exception:  # pragma: no cover
    WebControlNode = None


def _twist(lx=0.0, ly=0.0, az=0.0):
    t = Twist()
    t.linear.x = lx
    t.linear.y = ly
    t.angular.z = az
    return t


# ---------------------------------------------------------------------------
# _is_nonzero: 1e-3 dead-band on linear.x, linear.y, angular.z
# ---------------------------------------------------------------------------

class TestIsNonzero:
    def test_all_zero_is_zero(self):
        assert _is_nonzero(_twist()) is False

    def test_linear_x_above_threshold(self):
        assert _is_nonzero(_twist(lx=0.5)) is True

    def test_linear_y_above_threshold(self):
        assert _is_nonzero(_twist(ly=0.5)) is True

    def test_angular_z_above_threshold(self):
        assert _is_nonzero(_twist(az=0.5)) is True

    def test_negative_value_counts(self):
        assert _is_nonzero(_twist(lx=-0.5)) is True

    def test_just_below_threshold_is_zero(self):
        # 1e-3 is the strict boundary: abs > 1e-3 required.
        assert _is_nonzero(_twist(lx=1e-3, ly=1e-3, az=1e-3)) is False

    def test_just_above_threshold_is_nonzero(self):
        assert _is_nonzero(_twist(lx=1.1e-3)) is True

    def test_tiny_noise_treated_as_zero(self):
        assert _is_nonzero(_twist(lx=1e-4, az=-5e-4)) is False


# ---------------------------------------------------------------------------
# CmdVelBridge._tick: teleop-priority mux decision (run on a fake self)
# ---------------------------------------------------------------------------

def _make_stamp():
    """Return a value assignable to a TwistStamped header.stamp.

    Uses a real builtin_interfaces Time when ROS is installed (the real message
    type validates the field); falls back to a plain object for the bare-env
    stub, whose header.stamp accepts anything.
    """
    try:
        from builtin_interfaces.msg import Time
        return Time()
    except ImportError:
        return object()


class _FakeClock:
    class _Now:
        def to_msg(self):
            return _make_stamp()

    def now(self):
        return self._Now()


class _FakeBridge:
    """Carries exactly the attributes CmdVelBridge._tick reads/writes."""

    def __init__(self, now, teleop=None, teleop_t=-1e9, nav=None, nav_t=-1e9,
                 teleop_timeout=0.4, nav_timeout=0.5):
        self._now_val = now
        self._teleop = teleop
        self._teleop_t = teleop_t
        self._nav = nav
        self._nav_t = nav_t
        self._teleop_timeout = teleop_timeout
        self._nav_timeout = nav_timeout
        self._frame_id = 'base_link'
        self.published = None

    def _now(self):
        return self._now_val

    def get_clock(self):
        return _FakeClock()

    class _Pub:
        def __init__(self, outer):
            self._outer = outer

        def publish(self, stamped):
            self._outer.published = stamped

    @property
    def _pub(self):
        return self._Pub(self)


def _run_tick(fake):
    # Invoke the real, unbound _tick against our fake self.
    CmdVelBridge._tick(fake)
    return fake.published.twist


class TestTickMux:
    def test_fresh_nonzero_teleop_wins(self):
        teleop = _twist(lx=0.4)
        nav = _twist(lx=0.1)
        fake = _FakeBridge(now=1.0, teleop=teleop, teleop_t=0.9, nav=nav, nav_t=0.9)
        out = _run_tick(fake)
        assert out is teleop

    def test_zero_teleop_falls_through_to_nav(self):
        # Teleop fresh but zero -> nav drives.
        nav = _twist(lx=0.1)
        fake = _FakeBridge(now=1.0, teleop=_twist(), teleop_t=0.9, nav=nav, nav_t=0.9)
        out = _run_tick(fake)
        assert out is nav

    def test_stale_teleop_falls_through_to_nav(self):
        teleop = _twist(lx=0.4)
        nav = _twist(lx=0.1)
        # teleop is 0.6s old (> 0.4 timeout) so it is ignored; nav is fresh.
        fake = _FakeBridge(now=1.0, teleop=teleop, teleop_t=0.4, nav=nav, nav_t=0.9)
        out = _run_tick(fake)
        assert out is nav

    def test_everything_stale_outputs_zero(self):
        teleop = _twist(lx=0.4)
        nav = _twist(lx=0.1)
        fake = _FakeBridge(now=10.0, teleop=teleop, teleop_t=0.0, nav=nav, nav_t=0.0)
        out = _run_tick(fake)
        assert out.linear.x == 0.0 and out.angular.z == 0.0
        assert out is not teleop and out is not nav

    def test_no_inputs_outputs_zero(self):
        fake = _FakeBridge(now=1.0)
        out = _run_tick(fake)
        assert out.linear.x == 0.0 and out.angular.z == 0.0

    def test_stamped_carries_frame_id(self):
        fake = _FakeBridge(now=1.0, teleop=_twist(lx=0.4), teleop_t=0.9)
        _run_tick(fake)
        assert fake.published.header.frame_id == 'base_link'


# ---------------------------------------------------------------------------
# WebControlNode.apply_cmd: clamp to [-1, 1] then scale by per-axis max speed
# ---------------------------------------------------------------------------

class _FakeLock:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeNode:
    def __init__(self, max_linear, max_angular):
        self._max_linear = max_linear
        self._max_angular = max_angular
        self._cmd_lock = _FakeLock()
        self._last_cmd_time = 0.0
        self.published = None

    def _publish_twist(self, lx, az):
        self.published = (lx, az)


@pytest.mark.skipif(WebControlNode is None, reason='web_control_node import failed')
class TestApplyCmdMath:
    def _apply(self, lin, ang, max_lin=0.5, max_ang=1.0):
        node = _FakeNode(max_lin, max_ang)
        WebControlNode.apply_cmd(node, lin, ang)
        return node.published

    def test_full_forward_scales_to_max_linear(self):
        lx, az = self._apply(1.0, 0.0)
        assert lx == pytest.approx(0.5)
        assert az == pytest.approx(0.0)

    def test_full_turn_scales_to_max_angular(self):
        lx, az = self._apply(0.0, 1.0)
        assert az == pytest.approx(1.0)

    def test_half_input_scales_linearly(self):
        lx, az = self._apply(0.5, 0.5)
        assert lx == pytest.approx(0.25)
        assert az == pytest.approx(0.5)

    def test_clamps_above_one(self):
        lx, az = self._apply(3.0, 7.0)
        assert lx == pytest.approx(0.5)   # 1.0 * 0.5
        assert az == pytest.approx(1.0)   # 1.0 * 1.0

    def test_clamps_below_minus_one(self):
        lx, az = self._apply(-3.0, -7.0)
        assert lx == pytest.approx(-0.5)
        assert az == pytest.approx(-1.0)

    def test_respects_custom_max_speeds(self):
        lx, az = self._apply(1.0, -1.0, max_lin=0.2, max_ang=2.5)
        assert lx == pytest.approx(0.2)
        assert az == pytest.approx(-2.5)

    def test_updates_last_cmd_time(self):
        node = _FakeNode(0.5, 1.0)
        WebControlNode.apply_cmd(node, 0.1, 0.1)
        assert node._last_cmd_time > 0.0
