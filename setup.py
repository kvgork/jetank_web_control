from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'jetank_web_control'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='koen',
    maintainer_email='gorkom.projects@gmail.com',
    description='Web-based remote control interface for the JeTank robot',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'web_control_node = jetank_web_control.web_control_node:main',
            'cmd_vel_bridge = jetank_web_control.cmd_vel_bridge:main',
        ],
    },
)
