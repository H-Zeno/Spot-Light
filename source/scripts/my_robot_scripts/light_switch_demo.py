from __future__ import annotations

import time

import cv2
import numpy as np
import os

from bosdyn.client import Sdk
from robot_utils.advanced_movement import move_body_distanced, push
from robot_utils.base import ControlFunction, take_control_with_function
from robot_utils.basic_movements import carry_arm, stow_arm, move_body, gaze, carry, move_arm_distanced
from robot_utils.advanced_movement import push_light_switch, turn_light_switch
from robot_utils.frame_transformer import FrameTransformerSingleton
from robot_utils.video import localize_from_images, get_camera_rgbd
from scipy.spatial.transform import Rotation
from utils.coordinates import Pose3D, Pose2D, pose_distanced
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
from utils.affordance_detection_light_switch import compute_affordance_VLM_GPT4, compute_affordance_VLM_llava
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

API_KEY = "..."
STAND_DISTANCE = 1.0

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
        x, y, angle = 1.5, 0.8, 180 #high cabinet, -z
        # x, y, angle = 1.1, -1.2, 270  # large cabinet, +z

        pose = Pose2D(np.array([x, y]))
        pose.set_rot_from_angle(angle, degrees=True)
        move_body(
            pose=pose,
            frame_name=frame_name,
        )

        cabinet_pose = Pose3D((0.35, 0.8, 0.70)) #high cabinet
        # cabinet_pose = Pose3D((0.8, 0.8, 0.75))
        cabinet_pose.set_rot_from_rpy((0,0,angle), degrees=True)

        carry()

        gaze(cabinet_pose, frame_name, gripper_open=True)
        depth_response, color_response = get_camera_rgbd(
            in_frame="image",
            vis_block=False,
            cut_to_size=False,
        )
        stow_arm()

        predictions = predict_light_switches(color_response[0], vis_block=True)

        # predictions to Bbox
        boxes = []
        for pred in predictions.xyxy:
            x1, y1, x2, y2 = pred
            boxes.append(BBox(xmax=x2, xmin=x1, ymax=y2, ymin=y1))

        a = 2

        #################################
        #
        #DBUGGING
        #
        #################################
        depth_image_response = depth_response

        centers = []

        depth_image, depth_response = depth_image_response

        # # determine center coordinates for all handles
        for bbox in boxes:
            center = determine_handle_center(depth_image, bbox)
            centers.append(center)
        # if len(centers) == 0:
        #     return []
        centers = np.stack(centers, axis=0)

        # use centers to get depth and position of handle in frame coordinates
        center_coordss = frame_coordinate_from_depth_image(
            depth_image=depth_image,
            depth_response=depth_response,
            pixel_coordinatess=centers,
            frame_name=frame_name,
            vis_block=False,
        ).reshape((-1, 3))

        # select all points within the point cloud that belong to a drawer (not a handle) and determine the planes
        # the axis of motion is simply the normal of that plane
        drawer_bbox_pointss = select_points_from_bounding_box(
            depth_image_response, boxes, frame_name, vis_block=False
        )

        points_frame = drawer_bbox_pointss[0]
        drawer_masks = drawer_bbox_pointss[1]


        # we use the current body position to get the normal that points towards the robot, not away
        current_body = frame_transformer.get_current_body_position_in_frame(
            frame_name, in_common_pose=True
        )
        poses = []
        for center_coords, bbox_mask in zip(center_coordss, drawer_masks):
            pose = find_plane_normal_pose(
                points_frame[bbox_mask],
                center_coords,
                current_body,
                threshold=0.03,
                min_samples=10,
                vis_block=False,
            )
            poses.append(pose)


        # predict affordance from bounding boxes
        affordance_keys = []
        for bbox in boxes:
            cropped_image = color_response[0][int(bbox.ymin):int(bbox.ymax), int(bbox.xmin):int(bbox.xmax)]
            affordance_keys.append(compute_affordance_VLM_GPT4(cropped_image, AFFORDANCE_CLASSES, API_KEY))
            # affordance_keys.append(compute_affordance_VLM_llava(cropped_image, AFFORDANCE_CLASSES))

            carry()

        for idx, affordance_key in enumerate(affordance_keys):

            body_pose = pose_distanced(poses[idx], STAND_DISTANCE).to_dimension(2)
            move_body(body_pose, frame_name)

            a = 2
            if affordance_key == 0:
                push_light_switch(poses[idx], frame_name)
            elif affordance_key == 1:
                push_light_switch(poses[idx], frame_name)
            elif affordance_key == 2:
                turn_light_switch(poses[idx], frame_name)
            else:
                print("Something else")


        stow_arm()
        #################################
        #
        #DBUGGING
        #
        #################################

        return frame_name

        #TODO pose refineme


def main():
    config = Config()
    take_control_with_function(config, function=_Push_Light_Switch(), body_assist=True)


if __name__ == "__main__":
    main()
