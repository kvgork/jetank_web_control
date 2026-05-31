#!/usr/bin/env python3
"""Twist -> TwistStamped bridge.

The web control (and most teleop tools) publish ``geometry_msgs/Twist`` on
``/cmd_vel``. Gazebo's ``diff_drive_controller`` subscribes to
``geometry_msgs/TwistStamped`` on ``/diff_drive_controller/cmd_vel``. This node
relays the former to the latter, stamping each message, so the web control can
drive the simulation unchanged.

Parameters:
  input_topic   (str)  default '/cmd_vel'                       (Twist in)
  output_topic  (str)  default '/diff_drive_controller/cmd_vel' (TwistStamped out)
  frame_id      (str)  default 'base_link'
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TwistStamped


class CmdVelBridge(Node):
    def __init__(self):
        super().__init__('cmd_vel_bridge')
        self.declare_parameter('input_topic', '/cmd_vel')
        self.declare_parameter('output_topic', '/diff_drive_controller/cmd_vel')
        self.declare_parameter('frame_id', 'base_link')

        in_topic = self.get_parameter('input_topic').value
        out_topic = self.get_parameter('output_topic').value
        self._frame_id = self.get_parameter('frame_id').value

        self._pub = self.create_publisher(TwistStamped, out_topic, 10)
        self.create_subscription(Twist, in_topic, self._on_twist, 10)
        self.get_logger().info(f'cmd_vel bridge: {in_topic} (Twist) -> {out_topic} (TwistStamped)')

    def _on_twist(self, msg: Twist):
        stamped = TwistStamped()
        stamped.header.stamp = self.get_clock().now().to_msg()
        stamped.header.frame_id = self._frame_id
        stamped.twist = msg
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
