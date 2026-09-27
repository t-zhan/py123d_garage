from __future__ import annotations

from dataclasses import dataclass, field

from py123d.datatypes import CameraID

from py123d_garage.api.abstract_policy_config import AbstractPolicyConfig
from py123d_garage.datatypes.numerics import NonNegativeInt, PositiveFloat, PositiveInt


@dataclass
class ImpromptuVLAConfig(AbstractPolicyConfig):
    model_path: str = ""
    sample_manifest_path: str = ""
    trajectory_horizon_us: PositiveInt = 3_000_000
    trajectory_interval_us: PositiveInt = 500_000
    required_target_point_distances_m: list[PositiveFloat] = field(default_factory=list)
    image_min_pixels: int = 1024
    image_max_pixels: int = 262144
    cutoff_len: int = 4096
    max_new_tokens: int = 512
    gradient_checkpointing: bool = True

    @property
    def required_history_duration_us(self) -> NonNegativeInt:
        return 0

    @property
    def required_cameras(self) -> dict[str, list[CameraID]]:
        return {"nuscenes": [CameraID.PCAM_F0]}
