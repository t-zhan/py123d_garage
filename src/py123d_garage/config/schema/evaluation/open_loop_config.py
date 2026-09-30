from __future__ import annotations

from dataclasses import dataclass, field

from py123d_garage.api.abstract_benchmark_config import AbstractBenchmarkConfig
from py123d_garage.config.schema.policy.policy_config import EvaluationPolicyConfig


@dataclass
class OpenLoopParallelizationConfig:
    accelerator: str = "auto"
    devices: int | str = "auto"
    inference_batch_size: int = 32
    max_workers: int | None = None


@dataclass
class OpenLoopBenchmarkConfig(AbstractBenchmarkConfig):
    # -- Config objects --

    # The policy under evaluation; evaluation_checkpoint_file selects its weights.
    policy_config: EvaluationPolicyConfig = field(default_factory=EvaluationPolicyConfig)
    # Lightning devices and per-process loader resources.
    parallelization_config: OpenLoopParallelizationConfig = field(
        default_factory=OpenLoopParallelizationConfig,
    )

    # -- Atomic settings --

    # Writes the policy's rendered view of every scored scene into <run dir>/visualizations.
    save_visualizations: bool = False
