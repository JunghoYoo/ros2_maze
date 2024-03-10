#!/usr/bin/env python3

import os
from launch.substitutions import LaunchConfiguration
from launch.substitutions import ThisLaunchFileDir
from launch_ros.actions import Node
from launch import LaunchDescription
#from launch import In
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    included_launch = IncludeLaunchDescription(PythonLaunchDescriptionSource(\
        os.path.join(get_package_share_directory('turtlebot3_bringup'), 'launch/camera_robot.launch.py')))                

    return LaunchDescription([
        #included_launch, 
        Node(
            package='ros2_maze',
            executable='getObjectRange',
            name='getObjectRange',
            output='screen'),
        Node(
            package='ros2_maze',
            executable='imageClassifier',
            name='imageClassifier',
            output='screen'),
        Node(
            package='ros2_maze',
            executable='goToGoal',
            name='goToGoal',
            output='screen'),                    
    ])
