#!/usr/bin/env python3
"""cmd_vel mux + Twist->TwistStamped bridge for simulation.

Two sources want to drive the robot:
  - the web UI teleop (joystick / WASD)  -> a Twist on `teleop_topic`
  - Nav2 (click-to-navigate)             -> a Twist on `nav_topic`

Both used to publish the same `/cmd_vel`, so Nav2's idle zero-velocity stream
(~20 Hz) drowned the web teleop and the robot would not move manually. This node
muxes them with teleop priority and republishes the result as a TwistStamped on
`output_topic`, which is what Gazebo's diff_drive_controller subscribes to.

Priority: while the web teleop is sending a fresh *non-zero* command it wins;
otherwise Nav2's command passes through; otherwise zero (stop).

Parameters:
  teleop_topic   (str)   default '/cmd_vel_teleop'   (Twist in, web UI)
  nav_topic      (str)   default '/cmd_vel'          (Twist in, Nav2)
  output_topic   (str)   default '/diff_drive_controller/cmd_vel' (TwistStamped)
  frame_id       (str)   default 'base_link'
  rate_hz        (float) default 20.0
  teleop_timeout (float) default 0.4   (s; teleop considered stale after this)
  nav_timeout    (float) default 0.5   (s; nav considered stale after this)
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

        teleop_topic = self.get_parameter('teleop_topic').value
        nav_topic = self.get_parameter('nav_topic').value
        out_topic = self.get_parameter('output_topic').value
        self._frame_id = self.get_parameter('frame_id').value
        rate = float(self.get_parameter('rate_hz').value)
        self._teleop_timeout = float(self.get_parameter('teleop_timeout').value)
        self._nav_timeout = float(self.get_parameter('nav_timeout').value)

        self._teleop = None
        self._teleop_t = -1e9
        self._nav = None
        self._nav_t = -1e9

        self._pub = self.create_publisher(TwistStamped, out_topic, 10)
        self.create_subscription(Twist, teleop_topic, self._on_teleop, 10)
        self.create_subscription(Twist, nav_topic, self._on_nav, 10)
        self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info(
            f'cmd_vel mux: teleop[{teleop_topic}] + nav[{nav_topic}] '
            f'-> {out_topic} (TwistStamped, teleop priority)')

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_teleop(self, msg: Twist):
        self._teleop = msg
        self._teleop_t = self._now()

    def _on_nav(self, msg: Twist):
        self._nav = msg
        self._nav_t = self._now()

    def _tick(self):
        now = self._now()
        out = Twist()  # default zero (stop)
        teleop_fresh = self._teleop is not None and (now - self._teleop_t) < self._teleop_timeout
        nav_fresh = self._nav is not None and (now - self._nav_t) < self._nav_timeout
        if teleop_fresh and _is_nonzero(self._teleop):
            out = self._teleop          # active manual driving wins
        elif nav_fresh:
            out = self._nav             # otherwise let Nav2 drive
        stamped = TwistStamped()
        stamped.header.stamp = self.get_clock().now().to_msg()
        stamped.header.frame_id = self._frame_id
        stamped.twist = out
        self._pub.publish(stamped)


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
