from __future__ import annotations

import faulthandler
import json
import logging
import os
from dataclasses import replace
from pathlib import Path
from typing import cast

import hydra
import lightning as L
import numpy as np
import pandas as pd
import torch.distributed as dist
from omegaconf import DictConfig
from py123d.api import SceneAPI
from py123d.datatypes import CameraID
from py123d.geometry.geometry_index import PoseSE2Index
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from typing_extensions import override

from py123d_garage.api.abstract_policy import AbstractPolicy, AnyPolicy
from py123d_garage.api.abstract_policy_config import AbstractPolicyConfig
from py123d_garage.cache import CacheStoreReader
from py123d_garage.common.config_help import (
    CONFIG_PATH,
    build_from_string,
    finalize_evaluation,
    hydra_overrides,
    register_schema,
    run_dir,
    save_config,
)
from py123d_garage.common.logging_setup import setup_logging
from py123d_garage.config.presets.python.offline_data_sources.navsim import navtest
from py123d_garage.config.schema.evaluation.open_loop_config import OpenLoopBenchmarkConfig
from py123d_garage.datatypes.trajectory import TrajectorySE2
from py123d_garage.evaluation.help.scene_inference import (
    OfflineEvaluationDataset,
    SceneSample,
    _collate_scene_samples,
    _save_scene_views,
    build_source_scenes,
)
from py123d_garage.evaluation.help.sharding_help import save_results
from py123d_garage.evaluation.open_loop.sparse_drive import save_sparse_drive_results, score_sparse_drive
from py123d_garage.py123d_help.scene_builders import VerboseProcessPoolExecutor
from py123d_garage.py123d_help.scene_readers.ego_state import sample_ego_se2
from py123d_garage.py123d_help.scene_readers.sensors import camera_at_anchor

LOG = logging.getLogger(__name__)


register_schema("evaluate_open_loop", OpenLoopBenchmarkConfig)


@hydra.main(config_path=str(CONFIG_PATH), config_name="evaluate_open_loop", version_base=None)
def main(cfg: DictConfig) -> None:
    setup_logging()
    faulthandler.enable()
    benchmark_config: OpenLoopBenchmarkConfig = finalize_evaluation(cfg, OpenLoopBenchmarkConfig, hydra_overrides())
    parallelization = benchmark_config.parallelization_config
    model = OpenLoopEvaluationModule(benchmark_config, run_dir())
    trainer = L.Trainer(
        accelerator=parallelization.accelerator,
        devices=parallelization.devices,
        default_root_dir=str(model.output_dir),
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
    )
    trainer.predict(model, return_predictions=False)


class _IndexedDataset(Dataset[tuple[int, SceneSample]]):
    def __init__(self, samples: ConcatDataset[SceneSample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    @override
    def __getitem__(self, index: int) -> tuple[int, SceneSample]:
        return index, self.samples[index]


def _collate_indexed_samples(samples: list[tuple[int, SceneSample]]) -> tuple[list[int], SceneSample]:
    indices, scene_samples = zip(*samples, strict=True)
    return list(indices), _collate_scene_samples(list(scene_samples))


class OpenLoopEvaluationModule(L.LightningModule):
    def __init__(self, benchmark_config: OpenLoopBenchmarkConfig, output_dir: Path) -> None:
        super().__init__()
        self.policy = cast(AnyPolicy, build_from_string(benchmark_config.policy_config, AbstractPolicy))
        if not benchmark_config.benchmark_offline_data_sources:
            benchmark_config = replace(
                benchmark_config,
                benchmark_offline_data_sources={"navtest": navtest(self.policy.policy_config)},
            )
        self.benchmark_config = benchmark_config
        self.output_dir = output_dir
        self.scenes: list[SceneAPI] = []
        self.sample_tokens: list[str] = []
        self.results: list[tuple[int, dict[str, object]]] = []
        self.save_visualizations = (
            benchmark_config.save_visualizations and self.policy.policy_config.nuscenes_protocol != "sparse_drive"
        )
        workers = benchmark_config.parallelization_config.max_workers
        self.max_workers = workers if workers is not None else len(os.sched_getaffinity(0))

    @override
    def setup(self, stage: str) -> None:
        self.output_dir = Path(self.trainer.strategy.broadcast(str(self.output_dir)))
        if self.trainer.is_global_zero:
            save_config(self.benchmark_config)
            LOG.info(f"Path where all results are stored: {self.output_dir}")
        self.policy.verify_contract(benchmark_config=self.benchmark_config)
        feature_policy = cast(AnyPolicy, build_from_string(self.benchmark_config.policy_config, AbstractPolicy))
        datasets: list[OfflineEvaluationDataset] = []
        executor = VerboseProcessPoolExecutor(max_workers=self.max_workers)
        for source in self.benchmark_config.benchmark_offline_data_sources.values():
            scenes = build_source_scenes(source, executor, self.policy)
            self.scenes.extend(scenes)
            if self.policy.policy_config.nuscenes_protocol == "sparse_drive":
                rows = [
                    json.loads(line) for line in Path(cast(str, source.sample_manifest_path)).read_text().splitlines()
                ]
                tokens = {(row["scene_name"], row["timestamp_us"]): row["id"] for row in rows}
                self.sample_tokens.extend(
                    tokens[(scene.log_name, camera_at_anchor(scene, CameraID.PCAM_F0).timestamp.time_us)]
                    for scene in scenes
                )
            cache_reader = (
                CacheStoreReader(source.cache_root, self.policy.cache_signature(scenes[0].scene_metadata.dataset))
                if source.cache_root is not None
                else None
            )
            datasets.append(
                OfflineEvaluationDataset(scenes, feature_policy, cache_reader, build_labels=self.save_visualizations)
            )
        executor._terminate()
        self.dataset = _IndexedDataset(ConcatDataset(datasets))
        self.policy.initialize(self.benchmark_config.policy_config.evaluation_checkpoint_file)

    @override
    def predict_dataloader(self) -> DataLoader[tuple[int, SceneSample]]:
        return DataLoader(
            self.dataset,
            batch_size=self.benchmark_config.parallelization_config.inference_batch_size,
            num_workers=self.max_workers,
            collate_fn=_collate_indexed_samples,
            pin_memory=self.device.type != "cpu",
        )

    @override
    def predict_step(self, batch: tuple[list[int], SceneSample], batch_idx: int) -> list[dict[str, object]]:
        indices, (features, labels, navigation) = batch
        scenes = [self.scenes[index] for index in indices]
        predictions = self.policy(features, navigation)
        trajectories = self.policy.trajectories_from_predictions(predictions, scenes)
        parse_success = getattr(predictions, "parse_success", None)
        parsed = parse_success.tolist() if parse_success is not None else [True] * len(scenes)
        scores = _score_scenes(scenes, trajectories, self.policy.policy_config, parsed)
        self.results.extend(zip(indices, scores, strict=True))
        if self.save_visualizations:
            _save_scene_views(
                self.policy,
                features.to("cpu"),
                labels.to("cpu") if labels is not None else None,
                navigation.to("cpu"),
                predictions.to("cpu"),
                scenes,
                self.output_dir / "visualizations",
            )
        return scores

    @override
    def on_predict_epoch_end(self) -> None:
        rank_results: list[list[tuple[int, dict[str, object]]]] = [self.results]
        if self.trainer.world_size > 1:
            rank_results = [[] for _ in range(self.trainer.world_size)]
            dist.all_gather_object(rank_results, self.results)  # pyright: ignore[reportUnknownMemberType]
        if self.trainer.is_global_zero:
            ordered = sorted((item for results in rank_results for item in results), key=lambda item: item[0])
            if self.policy.policy_config.nuscenes_protocol == "sparse_drive":
                save_sparse_drive_results(
                    [(self.sample_tokens[index], score) for index, score in ordered],
                    self.output_dir,
                    self.benchmark_config.policy_config.evaluation_checkpoint_file,
                    len(self.dataset),
                )
            else:
                save_results(pd.DataFrame([score for _, score in ordered]), str(self.output_dir))


def _score_scenes(
    scenes: list[SceneAPI],
    trajectories: list[TrajectorySE2],
    policy_config: AbstractPolicyConfig,
    parse_success: list[bool],
) -> list[dict[str, object]]:
    """
    Computes each scene's metrics under the configured trajectory protocol.

    Args:
        scenes: the scenes, in inference order.
        trajectories: the predicted trajectory of each scene, ego-relative.
        policy_config: provides the trajectory grid the errors are scored on.
        parse_success: whether each prediction contains a complete trajectory.

    Returns:
        one metric dict per scene.
    """
    results: list[dict[str, object]] = []
    for scene, trajectory, parsed in zip(scenes, trajectories, parse_success, strict=True):
        result: dict[str, object] = {"scene_uuid": scene.scene_uuid}
        try:
            if policy_config.nuscenes_protocol == "sparse_drive":
                result["parse_success"] = parsed
                metrics = score_sparse_drive(
                    scene,
                    trajectory.pose_se2_array[:, :2] if parsed else None,
                    policy_config.trajectory_num_steps,
                    policy_config.trajectory_interval_us,
                )
            else:
                metrics = _compute_displacement_errors(
                    scene,
                    trajectory,
                    policy_config.trajectory_num_steps,
                    policy_config.trajectory_interval_us,
                )
            result.update(metrics)
        except Exception as error:
            result["scoring_error"] = f"{type(error).__name__}: {error}"
            LOG.warning(
                f"scene {scene.scene_uuid} of log {scene.log_name} is unscorable: {error}",
            )
        results.append(result)
    return results


def _compute_displacement_errors(
    scene: SceneAPI,
    trajectory: TrajectorySE2,
    num_steps: int,
    interval_us: int,
) -> dict[str, float]:
    """
    Compares one predicted trajectory against the logged ego poses on the scoring grid.

    Args:
        scene: the scene the trajectory was predicted for, anchored at its current frame.
        trajectory: the predicted trajectory, relative to the anchor ego pose.
        num_steps: how many poses the scoring grid holds.
        interval_us: spacing of the scoring grid.

    Returns:
        the average and the final displacement error, in meters.
    """
    ground_truth = sample_ego_se2(
        scene_api=scene,
        num_steps=num_steps,
        interval_us=interval_us,
        relative_to_anchor=True,
    )
    ego_state_se3 = scene.get_ego_state_se3_at_iteration(0)
    assert ego_state_se3 is not None, "Ego state should be available for displacement scoring!"
    # The prediction's first pose sits one policy step past the anchor, so a finer
    # scoring grid would query before it; in its own frame the ego is at the origin
    # at the anchor, and prepending that pose makes the whole grid interpolable.
    anchored_prediction = TrajectorySE2(
        pose_se2_array=np.vstack(
            [np.zeros((1, len(PoseSE2Index)), dtype=np.float64), trajectory.pose_se2_array],
        ),
        timestamps_us=np.concatenate(
            [np.array([ego_state_se3.timestamp.time_us], dtype=np.int64), trajectory.timestamps_us],
        ),
    )
    predicted_se2_array = anchored_prediction.interpolate(ground_truth.timestamps_us)
    position_columns = [PoseSE2Index.X, PoseSE2Index.Y]
    displacements_m = np.linalg.norm(
        predicted_se2_array[:, position_columns] - ground_truth.pose_se2_array[:, position_columns],
        axis=1,
    )
    return {
        "average_displacement_error_m": float(displacements_m.mean()),
        "final_displacement_error_m": float(displacements_m[-1]),
    }


if __name__ == "__main__":
    main()
