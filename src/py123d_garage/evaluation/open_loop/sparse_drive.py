"""SparseDrive planning metrics, adapted from official commit ffebeb40 (MIT; SparseDrive.LICENSE)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
from py123d.api import SceneAPI
from shapely.geometry import Polygon

from py123d_garage.py123d_help.scene_readers.nuscenes_protocol import (
    future_lidar_boxes,
    model_to_lidar_xy,
    sample_nuscenes_trajectory,
)


def _yaw(trajectory: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    if np.linalg.norm(trajectory[-1] - trajectory[0]) < 0.5:
        return np.full(len(trajectory), np.pi / 2)
    anchored = np.vstack([np.zeros((1, 2)), trajectory])
    delta = np.vstack([anchored[2:] - anchored[:-2], anchored[-1] - anchored[-2]])
    return np.arctan2(delta[:, 1], delta[:, 0])


def _box_corners(boxes: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    corners = boxes[:, None, 3:5] * np.array([[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]])
    c, s = np.cos(boxes[:, 6, None]), np.sin(boxes[:, 6, None])
    x, y = corners[..., 0], corners[..., 1]
    return np.stack([c * x - s * y, s * x + c * y], axis=-1) + boxes[:, None, :2]


def _collisions(
    trajectory: npt.NDArray[np.float64], future_boxes: list[npt.NDArray[np.float64]]
) -> npt.NDArray[np.bool_]:
    yaw = _yaw(trajectory)
    boxes = np.zeros((len(trajectory), 7))
    boxes[:, :2] = trajectory + 0.5 * np.column_stack([np.cos(yaw), np.sin(yaw)])
    boxes[:, 3:6] = [4.084, 1.85, 1.56]
    boxes[:, 6] = yaw
    return np.array(
        [
            any(Polygon(ego).intersects(Polygon(obstacle)) for obstacle in _box_corners(obstacles))
            for ego, obstacles in zip(_box_corners(boxes), future_boxes, strict=True)
        ],
        dtype=np.bool_,
    )


def sparse_drive_metrics(
    prediction: npt.NDArray[np.float64],
    ground_truth: npt.NDArray[np.float64],
    future_boxes: list[npt.NDArray[np.float64]],
) -> dict[str, npt.NDArray[np.float64] | npt.NDArray[np.bool_]]:
    """Model-axis trajectories, raw LiDAR boxes; L2 in meters, collisions as fractions."""
    gt_collision = _collisions(model_to_lidar_xy(ground_truth), future_boxes)
    pred_collision = _collisions(model_to_lidar_xy(prediction), future_boxes) & ~gt_collision
    return {
        "L2": np.linalg.norm(prediction - ground_truth, axis=-1),
        "obj_box_col": pred_collision,
    }


def score_sparse_drive(
    scene: SceneAPI, prediction: npt.NDArray[np.float64] | None, num_steps: int, interval_us: int
) -> dict[str, object]:
    truth = sample_nuscenes_trajectory(scene, num_steps, interval_us, "sparse_drive").pose_se2_array[:, :2]
    result: dict[str, object] = {
        "gt_traj": truth.tolist(),
        "pred_traj": None,
        "L2_per_step": None,
        "obj_box_col": None,
    }
    if prediction is not None:
        metrics = sparse_drive_metrics(prediction, truth, future_lidar_boxes(scene, num_steps))
        result.update(
            pred_traj=prediction.tolist(),
            L2_per_step=metrics["L2"].tolist(),
            obj_box_col=metrics["obj_box_col"].tolist(),
        )
    return result


def save_sparse_drive_results(
    results: list[tuple[str, dict[str, object]]], output_dir: Path, checkpoint: str, total_samples: int
) -> None:
    parsed_samples = sum(bool(score["parse_success"]) for _, score in results)
    scored = [score for _, score in results if score.get("L2_per_step") is not None]
    curves: dict[str, npt.NDArray[np.float64] | None] = {
        metric: np.array([score[key] for score in scored], dtype=np.float64).mean(axis=0) if scored else None
        for metric, key in (("obj_box_col", "obj_box_col"), ("L2", "L2_per_step"))
    }
    metrics: dict[str, dict[str, object]] = {}
    for strategy in ("uniad", "stp3"):
        metrics[strategy] = {
            metric: float(
                np.mean([curve[:step].mean() if strategy == "stp3" else curve[step - 1] for step in (2, 4, 6)])
            )
            if curve is not None
            else None
            for metric, curve in curves.items()
        }
    metrics["per_step"] = {metric: curve.tolist() if curve is not None else None for metric, curve in curves.items()}
    payload: dict[str, object] = {
        "metadata": {
            "checkpoint": checkpoint,
            "total_samples": total_samples,
            "evaluated_samples": len(results),
            "parsed_samples": parsed_samples,
            "parsed_rate": parsed_samples / len(results) if results else 0.0,
        },
        "metrics": metrics,
        "samples": {
            token: {key: score.get(key) for key in ("gt_traj", "pred_traj", "L2_per_step")} for token, score in results
        },
    }
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
