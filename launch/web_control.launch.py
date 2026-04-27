#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('web_port',       default_value='8080'),
        DeclareLaunchArgument('image_topic',    default_value='/stereo_camera/left/image_raw/compressed'),
        DeclareLaunchArgument('cmd_vel_topic',  default_value='/cmd_vel'),
        DeclareLaunchArgument('max_linear',     default_value='0.5'),
        DeclareLaunchArgument('max_angular',    default_value='1.0'),

        Node(
            package='jetank_web_control',
            executable='web_control_node',
            name='web_control_node',
            output='screen',
            parameters=[{
                'web_port':          LaunchConfiguration('web_port'),
                'image_topic':       LaunchConfiguration('image_topic'),
                'cmd_vel_topic':     LaunchConfiguration('cmd_vel_topic'),
                'max_linear_speed':  LaunchConfiguration('max_linear'),
                'max_angular_speed': LaunchConfiguration('max_angular'),
            }],
        ),
    ])
