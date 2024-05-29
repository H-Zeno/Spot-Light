from __future__ import annotations

import time

import cv2
import numpy as np
import os
import copy

from bosdyn.client import Sdk
from robot_utils.advanced_movement import move_body_distanced, push
from robot_utils.base import ControlFunction, take_control_with_function
from robot_utils.basic_movements import carry_arm, stow_arm, move_body, gaze, carry, move_arm_distanced, move_arm
from robot_utils.advanced_movement import push_light_switch, turn_light_switch
from robot_utils.frame_transformer import FrameTransformerSingleton
from robot_utils.video import localize_from_images, get_camera_rgbd, set_gripper_camera_params, set_gripper
from scipy.spatial.transform import Rotation
from utils.coordinates import Pose3D, Pose2D, pose_distanced, average_pose3Ds
from utils.recursive_config import Config
from utils.singletons import (
    GraphNavClientSingleton,
    ImageClientSingleton,
    RobotCommandClientSingleton,
    RobotSingleton,
    RobotStateClientSingleton,
    WorldObjectClientSingleton,
)
from utils.light_switch_detection import predict_light_switches
from utils.affordance_detection_light_switch import compute_affordance_VLM_GPT4, compute_advanced_affordance_VLM_GPT4
from bosdyn.api.image_pb2 import ImageResponse
from utils.object_detetion import BBox, Detection, Match
from robot_utils.video import frame_coordinate_from_depth_image, select_points_from_bounding_box

frame_transformer = FrameTransformerSingleton()
graph_nav_client = GraphNavClientSingleton()
image_client = ImageClientSingleton()
robot_command_client = RobotCommandClientSingleton()
robot = RobotSingleton()
robot_state_client = RobotStateClientSingleton()
world_object_client = WorldObjectClientSingleton()

from utils.pose_utils import calculate_light_switch_poses

from utils.pose_utils import (
    determine_handle_center,
    find_plane_normal_pose,
    calculate_handle_poses,
    cluster_handle_poses,
    filter_handle_poses,
    refine_handle_position,

)

AFFORDANCE_CLASSES = {0: "SINGLE PUSH",
                          1: "DOUBLE PUSH",
                          2: "ROTATING",
                          3: "something else"}

AFFORDANCE_DICT = {"switch type": ["push button switch", "rotating switch", "none"],
                       "button count": ["single", "double", "multi", "none"],
                       "button stacking": ["vertical", "horizontal", "none"]}

API_KEY = "..."
STAND_DISTANCE = 1.0
GRIPPER_WIDTH = 0.03
ADVANCED_AFFORDANCE = True

class _Push_Light_Switch(ControlFunction):
    def __call__(
        self,
        config: Config,
        sdk: Sdk,
        *args,
        **kwargs,
    ) -> str:

        frame_name = localize_from_images(config, vis_block=False)


        # position in front of shelf
        # x, y, angle = 1.3, 1.2, 180 #high cabinet, upper switch batch, close up
        x, y, angle = 1.5, 1.2, 180  #high cabinet, upper switch batch, far away

        pose = Pose2D(np.array([x, y]))
        pose.set_rot_from_angle(angle, degrees=True)
        move_body(
            pose=pose,
            frame_name=frame_name,
        )

        cabinet_pose = Pose3D((0.35, 1.1, 0.70)) #high cabinet
        # cabinet_pose = Pose3D((0.35, 1.1, 0.35))

        cabinet_pose.set_rot_from_rpy((0,0,angle), degrees=True)

        carry()

        set_gripper_camera_params('4096x2160')
        time.sleep(1)
        gaze(cabinet_pose, frame_name, gripper_open=True)
        depth_image_response, color_response = get_camera_rgbd(
            in_frame="image",
            vis_block=False,
            cut_to_size=False,
        )
        set_gripper_camera_params('640x480')
        stow_arm()

        boxes = predict_light_switches(color_response[0], vis_block=True)

        poses = calculate_light_switch_poses(boxes, depth_image_response, frame_name, frame_transformer)

        for pose in poses:
            move_body_distanced(pose.to_dimension(2), STAND_DISTANCE, frame_name)
            carry_arm()
            #################################
            # refine handle position
            #################################
            # camera_add_pose_refinement_right = Pose3D((-0.2, -0.05, -0.05))
            # camera_add_pose_refinement_right.set_rot_from_rpy((0, 10, 15), degrees=True)
            # camera_add_pose_refinement_left = Pose3D((-0.2, 0.05, -0.05))
            # camera_add_pose_refinement_left.set_rot_from_rpy((0, 10, -15), degrees=True)
            # camera_add_pose_refinement_bot = Pose3D((-0.2, -0.0, -0.1))
            # camera_add_pose_refinement_bot.set_rot_from_rpy((0, -10, 0), degrees=True)

            camera_add_pose_refinement_right = Pose3D((-0.25, -0.05, -0.04))
            camera_add_pose_refinement_right.set_rot_from_rpy((0, 10, 15), degrees=True)
            camera_add_pose_refinement_left = Pose3D((-0.25, 0.05, -0.04))
            camera_add_pose_refinement_left.set_rot_from_rpy((0, 10, -15), degrees=True)
            camera_add_pose_refinement_bot = Pose3D((-0.25, -0.0, -0.1))
            camera_add_pose_refinement_bot.set_rot_from_rpy((0, -10, 0), degrees=True)
            camera_add_pose_refinement_top = Pose3D((-0.25, -0.0, -0.01))
            camera_add_pose_refinement_top.set_rot_from_rpy((0, 10, 0), degrees=True)


            ref_add_poses = (camera_add_pose_refinement_right, camera_add_pose_refinement_left,
                             camera_add_pose_refinement_bot, camera_add_pose_refinement_top)

            refined_poses = []
            for ref_pose in ref_add_poses:

                # handle not finding plane
                try:
                    move_arm(pose @ ref_pose, frame_name)
                    depth_image_response, color_response = get_camera_rgbd(
                    in_frame="image", vis_block=False, cut_to_size=False
                    )
                    ref_boxes = predict_light_switches(color_response[0], vis_block=True)
                    refined_posess = calculate_light_switch_poses(ref_boxes, depth_image_response, frame_name, frame_transformer)
                    # filter refined poses
                    idx = np.argmin(np.linalg.norm(np.array([refined_pose.coordinates for refined_pose in refined_posess]) - pose.coordinates, axis=1))
                    refined_pose = refined_posess[idx]
                    refined_poses.append(refined_pose)
                    bbox = ref_boxes[idx]
                except:
                    continue

            refined_pose = average_pose3Ds(refined_poses)

            #################################
            # affordance detection
            #################################

            pose_affordance = copy.deepcopy(refined_pose)
            z_offset = -0.04
            pose_affordance.coordinates[2] += z_offset
            move_arm_distanced(pose_affordance, 0.08, frame_name)
            set_gripper(True)
            depth_image_response, color_response = get_camera_rgbd(
                in_frame="image",
                vis_block=False,
                cut_to_size=False,
            )
            # handle not finding a bounding box
            try:
                boxes = predict_light_switches(color_response[0], vis_block=True)
                bbox = boxes[0]
            except:
                continue

            cropped_image = color_response[0][int(bbox.ymin):int(bbox.ymax), int(bbox.xmin):int(bbox.xmax)]

            if ADVANCED_AFFORDANCE:
                # calculate advanced affordance (push, turn, double push)
                affordance_dict = compute_advanced_affordance_VLM_GPT4(cropped_image, AFFORDANCE_DICT, API_KEY)
                if affordance_dict["switch type"] == "rotating switch":
                    turn_light_switch(refined_pose, frame_name)
                elif affordance_dict["switch type"] == "push button switch" and affordance_dict["button count"] == "single":
                    push_light_switch(refined_pose, frame_name, z_offset=True)
                elif affordance_dict["switch type"] == "push button switch" and affordance_dict["button count"] == "double":
                    if affordance_dict["button stacking"] == "horizontal":
                        offsets = [GRIPPER_WIDTH//2, -GRIPPER_WIDTH//2]
                        for offset in offsets:
                            pose_offset = copy.deepcopy(refined_pose)
                            pose_offset.coordinates[1] += offset
                            push_light_switch(pose_offset, frame_name, z_offset=True)
                else:
                    print("THATS NOT A LIGHT SWITCH!")
            else:
                # calculate only simple affordance (push, turn)
                affordance_key = compute_affordance_VLM_GPT4(cropped_image, AFFORDANCE_CLASSES, API_KEY)
                if affordance_key == 0 or affordance_key == 1:
                    # z offset IFF push button
                    z_offset = 0.04
                    pose_offset = copy.deepcopy(refined_pose)
                    pose_offset.coordinates[2] += z_offset
                    push_light_switch(pose_offset, frame_name)
                elif affordance_key == 2:
                    turn_light_switch(refined_pose, frame_name)
                else:
                    print("Something else")

            stow_arm()


            a = 2

        stow_arm()

        return frame_name

        #TODO pose refineme


def main():
    config = Config()
    take_control_with_function(config, function=_Push_Light_Switch(), body_assist=True)


if __name__ == "__main__":
    main()
