"""
Pure-logic tests for the extended cmd_vel mux/bridge.

Tests cover:
  (a) _is_nonzero dead-band helper
  (b) _select_command priority: manip > teleop > nav
  (c) staleness / timeout fall-through to next source
  (d) silence when all sources idle (None returned)
  (e) output type: TwistStamped when output_stamped=True, Twist when False
  (f) empty manip_topic / nav_topic => no subscription created
  (g) WebControlNode.apply_cmd math (preserved from original suite)

No ROS spin, no hardware.  ROS / geometry_msgs packages are stubbed when
unavailable (bare env) using the same pattern as the other test files in this
package.
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
    TwistStamped = sys.modules['geometry_msgs.msg'].TwistStamped
except Exception as exc:  # pragma: no cover
    pytest.skip(f'Could not import cmd_vel_bridge: {exc}', allow_module_level=True)

try:
    _wcn = importlib.import_module('jetank_web_control.web_control_node')
    WebControlNode = _wcn.WebControlNode
except Exception:  # pragma: no cover
    WebControlNode = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _twist(lx=0.0, ly=0.0, az=0.0):
    t = Twist()
    t.linear.x = lx
    t.linear.y = ly
    t.angular.z = az
    return t


def _make_stamp():
    """Return a value assignable to a TwistStamped header.stamp."""
    try:
        from builtin_interfaces.msg import Time
        return Time()
    except ImportError:
        return object()


class _FakeClock:
    class _Now:
        def to_msg(self):
            return _make_stamp()

        @property
        def nanoseconds(self):
            return 0

    def now(self):
        return self._Now()


# ---------------------------------------------------------------------------
# Fake node for _select_command + _tick tests
# ---------------------------------------------------------------------------

class _FakeBridge:
    """Carries exactly the attributes CmdVelBridge._select_command and _tick
    read/write.  Mirrors the instance variables set in CmdVelBridge.__init__.

    Includes a _select_command delegation so that _tick (called as an unbound
    method on this fake) can call self._select_command(now) and reach the real
    implementation.
    """

    def __init__(self, now,
                 teleop=None, teleop_t=-1e9,
                 nav=None, nav_t=-1e9,
                 manip=None, manip_t=-1e9,
                 teleop_timeout=0.4, nav_timeout=0.5, manip_timeout=0.5,
                 nav_subscribed=True, output_stamped=True,
                 frame_id='base_link'):
        self._now_val = now
        self._teleop = teleop
        self._teleop_t = teleop_t
        self._nav = nav
        self._nav_t = nav_t
        self._manip = manip
        self._manip_t = manip_t
        self._teleop_timeout = teleop_timeout
        self._nav_timeout = nav_timeout
        self._manip_timeout = manip_timeout
        self._nav_subscribed = nav_subscribed
        self._output_stamped = output_stamped
        self._frame_id = frame_id
        self._stop_burst_ticks = 6   # 0.3 s * 20 Hz
        self._stop_burst = 0
        self.published = None        # last message given to _pub.publish()

    def _now(self):
        return self._now_val

    def _select_command(self, now):
        # Delegate to the real implementation so _tick can call self._select_command.
        return CmdVelBridge._select_command(self, now)

    def get_clock(self):
        return _FakeClock()

    class _Pub:
        def __init__(self, outer):
            self._outer = outer

        def publish(self, msg):
            self._outer.published = msg

    @property
    def _pub(self):
        return self._Pub(self)


def _select(fake):
    """Call the real unbound _select_command against our fake self."""
    return CmdVelBridge._select_command(fake, fake._now_val)


def _run_tick(fake):
    """Call the real unbound _tick against our fake self."""
    CmdVelBridge._tick(fake)
    return fake.published


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
        assert _is_nonzero(_twist(lx=1e-3, ly=1e-3, az=1e-3)) is False

    def test_just_above_threshold_is_nonzero(self):
        assert _is_nonzero(_twist(lx=1.1e-3)) is True

    def test_tiny_noise_treated_as_zero(self):
        assert _is_nonzero(_twist(lx=1e-4, az=-5e-4)) is False


# ---------------------------------------------------------------------------
# _select_command: priority manip > teleop > nav, staleness, silence
# ---------------------------------------------------------------------------

class TestSelectCommandPriority:
    """(a) Priority ordering."""

    def test_manip_beats_teleop_and_nav(self):
        manip = _twist(lx=1.0)
        teleop = _twist(lx=0.4)
        nav = _twist(lx=0.2)
        fake = _FakeBridge(
            now=1.0,
            manip=manip, manip_t=0.9,
            teleop=teleop, teleop_t=0.9,
            nav=nav, nav_t=0.9,
        )
        assert _select(fake) is manip

    def test_teleop_beats_nav_when_manip_absent(self):
        teleop = _twist(lx=0.4)
        nav = _twist(lx=0.2)
        fake = _FakeBridge(
            now=1.0,
            teleop=teleop, teleop_t=0.9,
            nav=nav, nav_t=0.9,
        )
        assert _select(fake) is teleop

    def test_nav_wins_when_teleop_zero_manip_absent(self):
        nav = _twist(lx=0.2)
        fake = _FakeBridge(
            now=1.0,
            teleop=_twist(), teleop_t=0.9,   # fresh but zero
            nav=nav, nav_t=0.9,
        )
        assert _select(fake) is nav

    def test_manip_zero_still_wins_over_teleop(self):
        """Manip zero means 'stop base' (e.g. end of approach); it still wins."""
        manip_zero = _twist()
        teleop = _twist(lx=0.5)
        fake = _FakeBridge(
            now=1.0,
            manip=manip_zero, manip_t=0.9,
            teleop=teleop, teleop_t=0.9,
        )
        assert _select(fake) is manip_zero


class TestSelectCommandStaleness:
    """(b) Timeout fall-through to next source."""

    def test_stale_manip_falls_through_to_teleop(self):
        manip = _twist(lx=1.0)
        teleop = _twist(lx=0.4)
        fake = _FakeBridge(
            now=2.0,
            manip=manip, manip_t=1.0,    # 1.0 s ago > 0.5 timeout -> stale
            teleop=teleop, teleop_t=1.9,  # fresh
        )
        assert _select(fake) is teleop

    def test_stale_teleop_falls_through_to_nav(self):
        teleop = _twist(lx=0.4)
        nav = _twist(lx=0.1)
        fake = _FakeBridge(
            now=1.0,
            teleop=teleop, teleop_t=0.4,  # 0.6 s ago > 0.4 timeout -> stale
            nav=nav, nav_t=0.9,
        )
        assert _select(fake) is nav

    def test_stale_nav_returns_none(self):
        nav = _twist(lx=0.1)
        fake = _FakeBridge(
            now=10.0,
            nav=nav, nav_t=0.0,          # 10 s ago > 0.5 timeout -> stale
        )
        assert _select(fake) is None

    def test_all_sources_stale_returns_none(self):
        fake = _FakeBridge(
            now=100.0,
            manip=_twist(lx=1.0), manip_t=0.0,
            teleop=_twist(lx=0.4), teleop_t=0.0,
            nav=_twist(lx=0.1), nav_t=0.0,
        )
        assert _select(fake) is None


class TestSelectCommandSilence:
    """(c) Silence when all sources idle."""

    def test_no_inputs_returns_none(self):
        fake = _FakeBridge(now=1.0)
        assert _select(fake) is None

    def test_nav_not_subscribed_blocks_nav(self):
        nav = _twist(lx=0.2)
        fake = _FakeBridge(
            now=1.0,
            nav=nav, nav_t=0.9,
            nav_subscribed=False,         # empty nav_topic -> guard active
        )
        assert _select(fake) is None

    def test_all_zero_returns_none(self):
        fake = _FakeBridge(
            now=1.0,
            teleop=_twist(), teleop_t=0.9,
            nav=_twist(), nav_t=0.9,
        )
        assert _select(fake) is None


# ---------------------------------------------------------------------------
# _tick: output type controlled by output_stamped  (d)
# ---------------------------------------------------------------------------

class TestTickOutputType:
    """(d) output_stamped=True -> TwistStamped; False -> plain Twist."""

    def test_output_stamped_true_publishes_twiststamped(self):
        teleop = _twist(lx=0.4)
        fake = _FakeBridge(
            now=1.0,
            teleop=teleop, teleop_t=0.9,
            output_stamped=True,
        )
        result = _run_tick(fake)
        assert result is not None
        # TwistStamped has a .header attribute; plain Twist does not
        assert hasattr(result, 'header'), \
            f'Expected TwistStamped (has .header), got {type(result)}'
        assert hasattr(result, 'twist')
        assert result.twist is teleop

    def test_output_stamped_false_publishes_plain_twist(self):
        teleop = _twist(lx=0.4)
        fake = _FakeBridge(
            now=1.0,
            teleop=teleop, teleop_t=0.9,
            output_stamped=False,
        )
        result = _run_tick(fake)
        assert result is not None
        # Plain Twist has .linear / .angular but NO .header
        assert not hasattr(result, 'header'), \
            f'Expected plain Twist (no .header), got {type(result)}'
        assert result is teleop

    def test_stamped_output_carries_frame_id(self):
        fake = _FakeBridge(
            now=1.0,
            teleop=_twist(lx=0.4), teleop_t=0.9,
            output_stamped=True,
            frame_id='base_link',
        )
        result = _run_tick(fake)
        assert result.header.frame_id == 'base_link'

    def test_manip_output_stamped_true(self):
        manip = _twist(lx=0.8)
        fake = _FakeBridge(
            now=1.0,
            manip=manip, manip_t=0.9,
            output_stamped=True,
        )
        result = _run_tick(fake)
        assert hasattr(result, 'header')
        assert result.twist is manip

    def test_manip_output_stamped_false(self):
        manip = _twist(lx=0.8)
        fake = _FakeBridge(
            now=1.0,
            manip=manip, manip_t=0.9,
            output_stamped=False,
        )
        result = _run_tick(fake)
        assert not hasattr(result, 'header')
        assert result is manip


# ---------------------------------------------------------------------------
# Empty-topic guard: no subscription created  (e)
# ---------------------------------------------------------------------------

class TestEmptyTopicGuard:
    """(e) Empty manip_topic / nav_topic => nav_subscribed=False =>
    those sources are excluded from selection.

    The guard itself is in __init__ (which we cannot call without ROS), but
    _select_command respects self._nav_subscribed for nav, and the manip path
    simply won't have a fresh timestamp when no subscription exists.
    We test the selection-logic side of the guard here.
    """

    def test_nav_subscribed_false_blocks_nav_even_when_fresh(self):
        nav = _twist(lx=0.3)
        fake = _FakeBridge(
            now=1.0,
            nav=nav, nav_t=0.9,
            nav_subscribed=False,   # simulates empty nav_topic
        )
        assert _select(fake) is None

    def test_nav_subscribed_true_allows_nav(self):
        nav = _twist(lx=0.3)
        fake = _FakeBridge(
            now=1.0,
            nav=nav, nav_t=0.9,
            nav_subscribed=True,
        )
        assert _select(fake) is nav

    def test_manip_never_received_gives_none(self):
        """No manip subscription => _manip stays None => manip path inactive."""
        fake = _FakeBridge(now=1.0)   # _manip=None, _manip_t=-1e9 by default
        assert _select(fake) is None  # no active source

    def test_only_teleop_active_when_nav_disabled(self):
        teleop = _twist(lx=0.4)
        fake = _FakeBridge(
            now=1.0,
            teleop=teleop, teleop_t=0.9,
            nav_subscribed=False,
        )
        assert _select(fake) is teleop


# ---------------------------------------------------------------------------
# Idle-silence / stop-burst behavior (preserved from original)
# ---------------------------------------------------------------------------

class TestIdleSilence:
    def test_idle_after_burst_returns_none_published(self):
        """When all sources idle and stop_burst exhausted, _tick publishes nothing."""
        fake = _FakeBridge(now=100.0)
        fake._stop_burst = 0   # burst already spent
        result = _run_tick(fake)
        assert result is None

    def test_stop_burst_decrements_and_publishes_zero(self):
        """With stop_burst > 0 and all idle, _tick publishes a zero then decrements."""
        fake = _FakeBridge(now=100.0, output_stamped=True)
        fake._stop_burst = 3
        result = _run_tick(fake)
        # Must publish something (the zero-stop burst)
        assert result is not None
        assert fake._stop_burst == 2  # decremented by 1

    def test_active_source_arms_stop_burst(self):
        """When an active source is present, stop_burst is reset to stop_burst_ticks."""
        teleop = _twist(lx=0.4)
        fake = _FakeBridge(now=1.0, teleop=teleop, teleop_t=0.9, output_stamped=True)
        fake._stop_burst = 0
        _run_tick(fake)
        assert fake._stop_burst == fake._stop_burst_ticks


# ---------------------------------------------------------------------------
# WebControlNode.apply_cmd: clamp to [-1, 1] then scale (preserved)
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
        assert lx == pytest.approx(0.5)
        assert az == pytest.approx(1.0)

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
