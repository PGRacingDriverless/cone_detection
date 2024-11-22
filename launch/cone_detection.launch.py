import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition

def generate_launch_description():
    params_file_path = os.path.join(
        get_package_share_directory("cone_detection"),
        "config",
        "params.yaml"
    )

    cone_detection_node = Node(
        executable="cone_detection_node",
        package="cone_detection",
        name="cone_detection_node",
        parameters = [params_file_path],
        output="screen"
    )

    launch_rviz = LaunchConfiguration("rviz")

    launch_rviz_arg = DeclareLaunchArgument(
        "rviz",
        default_value="False",
        description="Launch RViz2"
    )
   
    rviz_config_path = os.path.join(
        get_package_share_directory("cone_detection"),
        "config",
        "config.rviz"
    )

    rviz_node = Node(
        executable="rviz2",
        package="rviz2",
        name="RViz2",
        arguments=["-d", rviz_config_path],
        output="screen",
        condition=IfCondition(launch_rviz)
    )

    launch_description = LaunchDescription()

    launch_description.add_action(cone_detection_node)
    launch_description.add_action(launch_rviz_arg)
    launch_description.add_action(rviz_node)

    return launch_description
