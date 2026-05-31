#!/usr/bin/env python3
"""Web remote-control bring-up.

Real robot (default)::

    ros2 launch jetank_web_control web_control.launch.py
    # web publishes Twist on /cmd_vel; subscribes the compressed camera topic.

Simulation::

    ros2 launch jetank_web_control web_control.launch.py sim:=true
    # use_sim_time, raw camera Image (Gazebo has no compressed transport), and a
    # cmd_vel bridge Twist(/cmd_vel) -> TwistStamped(/diff_drive_controller/cmd_vel).
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    sim = LaunchConfiguration('sim').perform(context).lower() in ('true', '1')
    image_topic = LaunchConfiguration('image_topic').perform(context)
    if not image_topic:
        image_topic = ('/stereo_camera/left/image_raw' if sim
                       else '/stereo_camera/left/image_raw/compressed')

    web_node = Node(
        package='jetank_web_control',
        executable='web_control_node',
        name='web_control_node',
        output='screen',
        parameters=[{
            'web_port':          LaunchConfiguration('web_port'),
            'image_topic':       image_topic,
            'image_compressed':  not sim,
            'cmd_vel_topic':     LaunchConfiguration('cmd_vel_topic'),
            'max_linear_speed':  LaunchConfiguration('max_linear'),
            'max_angular_speed': LaunchConfiguration('max_angular'),
            'use_sim_time':      sim,
        }],
    )

    # In sim, bridge the web's Twist to the controller's TwistStamped topic.
    bridge_node = Node(
        package='jetank_web_control',
        executable='cmd_vel_bridge',
        name='cmd_vel_bridge',
        output='screen',
        condition=IfCondition(LaunchConfiguration('sim')),
        parameters=[{
            'input_topic':  LaunchConfiguration('cmd_vel_topic'),
            'output_topic': LaunchConfiguration('output_cmd_vel'),
            'use_sim_time': True,
        }],
    )

    return [web_node, bridge_node]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('web_port',       default_value='8080'),
        DeclareLaunchArgument('image_topic',    default_value='',
                              description='Camera topic; empty = auto (raw in sim, compressed on robot)'),
        DeclareLaunchArgument('cmd_vel_topic',  default_value='/cmd_vel'),
        DeclareLaunchArgument('max_linear',     default_value='0.5'),
        DeclareLaunchArgument('max_angular',    default_value='1.0'),
        DeclareLaunchArgument('sim',            default_value='false',
                              description='Simulation mode: raw camera + cmd_vel bridge + use_sim_time'),
        DeclareLaunchArgument('output_cmd_vel', default_value='/diff_drive_controller/cmd_vel',
                              description='Bridge output (TwistStamped) topic for the sim controller'),
        OpaqueFunction(function=launch_setup),
    ])
