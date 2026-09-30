"""nuScenes protocol sampling and coordinate conversion; model axes are x forward, y left."""

from __future__ import annotations

from typing import cast

import numpy as np
import numpy.typing as npt
from py123d.api import SceneAPI
from py123d.datatypes import LidarID

from py123d_garage.datatypes.trajectory import TrajectorySE2
from py123d_garage.py123d_help.scene_readers.ego_state import sample_ego_se2

_LIDAR_TO_MODEL = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def trajectory_timestamps(
    scene: SceneAPI, num_steps: int, interval_us: int, nuscenes_protocol: str
) -> npt.NDArray[np.int64]:
    if nuscenes_protocol == "sparse_drive":
        return np.array([scene.get_timestamp_at_iteration(i).time_us for i in range(1, num_steps + 1)], dtype=np.int64)
    ego = scene.get_ego_state_se3_at_iteration(0)
    assert ego is not None
    return ego.timestamp.time_us + np.arange(1, num_steps + 1, dtype=np.int64) * interval_us


def world_lidar_pose(scene: SceneAPI, iteration: int) -> npt.NDArray[np.float64]:
    ego = scene.get_ego_state_se3_at_iteration(iteration)
    assert ego is not None
    extrinsic = scene.get_lidar_metadatas()[LidarID.LIDAR_TOP].lidar_to_imu_se3.transformation_matrix
    return ego.imu_se3.transformation_matrix @ extrinsic


def sample_nuscenes_trajectory(
    scene: SceneAPI, num_steps: int, interval_us: int, nuscenes_protocol: str
) -> TrajectorySE2:
    if nuscenes_protocol == "garage":
        return sample_ego_se2(scene, num_steps, interval_us, relative_to_anchor=True)
    origin_inverse = np.linalg.inv(world_lidar_pose(scene, 0))
    relative = np.stack([origin_inverse @ world_lidar_pose(scene, i) for i in range(1, num_steps + 1)])
    positions = relative[:, :3, 3] @ _LIDAR_TO_MODEL.T
    rotations = _LIDAR_TO_MODEL @ relative[:, :3, :3] @ _LIDAR_TO_MODEL.T
    yaw = np.arctan2(rotations[:, 1, 0], rotations[:, 0, 0])
    return TrajectorySE2(
        pose_se2_array=np.column_stack([positions[:, :2], yaw]),
        timestamps_us=trajectory_timestamps(scene, num_steps, interval_us, nuscenes_protocol),
    )


def future_lidar_boxes(scene: SceneAPI, num_steps: int) -> list[npt.NDArray[np.float64]]:
    """SparseDrive boxes: current raw LiDAR axes, dimensions (length, width, height)."""
    origin_inverse = np.linalg.inv(world_lidar_pose(scene, 0))
    future_boxes = []
    for iteration in range(1, num_steps + 1):
        future_pose = world_lidar_pose(scene, iteration)
        future_inverse = np.linalg.inv(future_pose)
        future_to_current = origin_inverse @ future_pose
        detections = scene.get_box_detections_se3_at_iteration(iteration)
        assert detections is not None
        boxes = []
        for detection in detections.box_detections:
            if cast(int, detection.attributes.num_lidar_points) <= 0:
                continue
            box = detection.bounding_box_se3
            local = future_inverse @ box.center_se3.transformation_matrix
            center = (future_to_current @ local)[:3, 3]
            # Match nuScenes Box.orientation.yaw_pitch_roll before the planar future-to-current rotation.
            yaw = np.arctan2(-local[0, 1], local[0, 0])
            direction = future_to_current[:2, :2] @ np.array([np.cos(yaw), np.sin(yaw)])
            boxes.append([*center, box.length, box.width, box.height, np.arctan2(direction[1], direction[0])])
        future_boxes.append(np.array(boxes, dtype=np.float64).reshape(-1, 7))
    return future_boxes


def model_to_lidar_xy(positions: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return positions @ _LIDAR_TO_MODEL[:2, :2]
