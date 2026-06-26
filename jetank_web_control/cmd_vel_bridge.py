#!/usr/bin/env python3
"""cmd_vel mux + Twist/TwistStamped bridge for simulation and hardware.

Three sources want to drive the robot (in priority order, highest first):
  - manip (base_approach / mission_coordinator) -> TwistStamped on `manip_topic`
  - web UI teleop (joystick / WASD)             -> Twist on `teleop_topic`
  - Nav2 (click-to-navigate)                    -> Twist on `nav_topic`

All three used to share the same topic, causing priority conflicts and idle
zero-velocity floods.  This node muxes them and republishes the winner.

Priority: manip (if fresh) > teleop (if fresh & nonzero) > nav (if fresh &
nonzero & subscribed) > silent.  When no source is active the node goes silent
after a short zero-burst to guarantee a clean stop (see idle-silence comment).

Simulation (output_stamped=True, default):
  Output is TwistStamped on /diff_drive_controller/cmd_vel — unchanged from
  the original behaviour.

Hardware (output_stamped=False):
  Output is plain Twist on output_topic (e.g. /cmd_vel for the motor driver).
  nav_topic should be set to '' on hardware (Nav2 already owns /cmd_vel via
  its velocity_smoother; the arbiter must NOT double-publish there).
  manip_topic set to /cmd_vel_manip (highest priority, base_approach and
  mission_coordinator write here during SEARCH/APPROACH).

Empty-topic guard: topics set to '' are NOT subscribed.  This prevents the
arbiter from subscribing to its own output topic and creating a feedback loop,
and disables nav on hardware where Nav2 is the sole nav publisher.

Parameters:
  teleop_topic   (str)   default '/cmd_vel_teleop'              (Twist in, web UI)
  nav_topic      (str)   default '/cmd_vel'                     (Twist in, Nav2)
                          set to '' to disable (no subscription created)
  output_topic   (str)   default '/diff_drive_controller/cmd_vel'
  frame_id       (str)   default 'base_link'
  rate_hz        (float) default 20.0
  teleop_timeout (float) default 0.4   (s; teleop considered stale after this)
  nav_timeout    (float) default 0.5   (s; nav considered stale after this)
  output_stamped (bool)  default True  True -> TwistStamped out (sim); False -> Twist out (HW)
  manip_topic    (str)   default ''    TwistStamped in from base_approach/mission_coordinator;
                          '' => disabled (no subscription created).  Highest priority.
  manip_timeout  (float) default 0.5   (s; manip considered stale after this)
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TwistStamped


def _is_nonzero(t: Twist) -> bool:
    return (abs(t.linear.x) > 1e-3 or abs(t.linear.y) > 1e-3
            or abs(t.angular.z) > 1e-3)


class CmdVelBridge(Node):
    def __init__(self):
        super().__init__('cmd_vel_bridge')
        self.declare_parameter('teleop_topic', '/cmd_vel_teleop')
        self.declare_parameter('nav_topic', '/cmd_vel')
        self.declare_parameter('output_topic', '/diff_drive_controller/cmd_vel')
        self.declare_parameter('frame_id', 'base_link')
        self.declare_parameter('rate_hz', 20.0)
        self.declare_parameter('teleop_timeout', 0.4)
        self.declare_parameter('nav_timeout', 0.5)
        # New params (Phase 1 — FROZEN contract with Phase 2)
        self.declare_parameter('output_stamped', True)
        self.declare_parameter('manip_topic', '')
        self.declare_parameter('manip_timeout', 0.5)

        teleop_topic = self.get_parameter('teleop_topic').value
        nav_topic = self.get_parameter('nav_topic').value
        out_topic = self.get_parameter('output_topic').value
        self._frame_id = self.get_parameter('frame_id').value
        rate = float(self.get_parameter('rate_hz').value)
        self._teleop_timeout = float(self.get_parameter('teleop_timeout').value)
        self._nav_timeout = float(self.get_parameter('nav_timeout').value)
        self._output_stamped = bool(self.get_parameter('output_stamped').value)
        manip_topic = self.get_parameter('manip_topic').value
        self._manip_timeout = float(self.get_parameter('manip_timeout').value)

        # Source state: message + timestamp of last received message
        self._teleop = None
        self._teleop_t = -1e9
        self._nav = None
        self._nav_t = -1e9
        self._manip = None           # TwistStamped (highest priority)
        self._manip_t = -1e9
        self._nav_subscribed = False  # tracks whether the nav sub was created

        # When no fresh source is driving, emit a short zero burst to guarantee a
        # clean stop, then go SILENT instead of flooding zeros forever. A third
        # node (base_approach, during a grasp APPROACH) publishes drive commands
        # directly to output_topic; a continuous idle-zero stream from this bridge
        # interleaves with those at the controller and cancels the motion (the base
        # stutters in place -> APPROACH times out -> every fetch mission fails).
        # Going silent lets base_approach own the topic; the diff_drive_controller's
        # own cmd_vel_timeout keeps the base stopped once every publisher is quiet.
        self._stop_burst_ticks = max(1, int(round(0.3 * rate)))
        self._stop_burst = 0

        # Publisher: TwistStamped (sim) or Twist (hardware) depending on output_stamped
        if self._output_stamped:
            self._pub = self.create_publisher(TwistStamped, out_topic, 10)
        else:
            self._pub = self.create_publisher(Twist, out_topic, 10)

        # Subscriptions — empty-topic guard: only subscribe when topic is non-empty
        self.create_subscription(Twist, teleop_topic, self._on_teleop, 10)

        if nav_topic:
            self.create_subscription(Twist, nav_topic, self._on_nav, 10)
            self._nav_subscribed = True

        if manip_topic:
            self.create_subscription(
                TwistStamped, manip_topic, self._on_manip, 10)

        self.create_timer(1.0 / rate, self._tick)

        out_type = 'TwistStamped' if self._output_stamped else 'Twist'
        self.get_logger().info(
            f'cmd_vel mux: manip[{manip_topic or "disabled"}] + '
            f'teleop[{teleop_topic}] + nav[{nav_topic or "disabled"}] '
            f'-> {out_topic} ({out_type}, manip>teleop>nav priority)')

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_teleop(self, msg: Twist):
        self._teleop = msg
        self._teleop_t = self._now()

    def _on_nav(self, msg: Twist):
        self._nav = msg
        self._nav_t = self._now()

    def _on_manip(self, msg: TwistStamped):
        # Store only the twist part for uniform handling; freshness tracked separately
        self._manip = msg.twist
        self._manip_t = self._now()

    def _select_command(self, now: float):
        """Pure selection logic — returns (chosen_twist_or_None, any_active).

        Returns the winning Twist (or None if all sources are idle/stale).
        This is a pure function of state, extracted for testability.

        Priority: manip > teleop (nonzero) > nav (nonzero, if subscribed) > None.
        Manip wins whenever it is fresh, regardless of whether the twist is zero
        (it owns the base during approach; a zero manip means "stop the base").
        """
        manip_fresh = (self._manip is not None
                       and (now - self._manip_t) < self._manip_timeout)
        teleop_fresh = (self._teleop is not None
                        and (now - self._teleop_t) < self._teleop_timeout)
        nav_fresh = (self._nav_subscribed
                     and self._nav is not None
                     and (now - self._nav_t) < self._nav_timeout)

        if manip_fresh:
            return self._manip         # highest priority — owns base during APPROACH
        if teleop_fresh and _is_nonzero(self._teleop):
            return self._teleop        # active manual driving
        if nav_fresh and _is_nonzero(self._nav):
            return self._nav           # let Nav2 drive
        return None                    # all sources idle / stale

    def _tick(self):
        now = self._now()
        out = self._select_command(now)

        if out is not None:
            self._stop_burst = self._stop_burst_ticks  # arm a stop burst for idle
        else:
            # No fresh nonzero source. Emit a brief zero burst to brake, then stay
            # silent so base_approach can drive the base directly (see __init__).
            if self._stop_burst <= 0:
                return
            self._stop_burst -= 1
            out = Twist()

        if self._output_stamped:
            stamped = TwistStamped()
            stamped.header.stamp = self.get_clock().now().to_msg()
            stamped.header.frame_id = self._frame_id
            stamped.twist = out
            self._pub.publish(stamped)
        else:
            self._pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = CmdVelBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
