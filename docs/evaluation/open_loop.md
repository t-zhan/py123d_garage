<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../assets/logo_dark.png">
  <img src="../assets/logo_light.png" alt="Py123d Garage" width="575">
</picture>

</div>

<h1 align="center">Open-loop evaluation</h1>

This evaluation protocol only requires an offline dataset.

## Default usage on nuPlan navtest

By default we use the navtest split:

```bash
python -m py123d_garage.evaluation.open_loop.evaluate \
    policy_config.evaluation_checkpoint_file=<checkpoint.pth> \
    parallelization_config.accelerator=gpu \
    parallelization_config.devices=auto
```

Lightning `Trainer.predict` manages devices and distributes samples without repetition. `devices=auto` uses all visible GPUs when `accelerator=gpu`. `inference_batch_size` and `max_workers` are per process. The main process gathers per-scene metrics and writes `results.csv` with an `average` row; no manual shards or intermediate shard CSVs are needed. Install the `train` extra to use this entry point.

For `nuscenes_protocol=sparse_drive`, the main process writes `results.json` instead of CSV. The source JSONL manifest supplies native nuScenes sample IDs. `metadata` records the checkpoint, total dataset size, scoring attempts (`evaluated_samples`), model parsing successes (`parsed_samples`), and their global parsed rate. Attempts include failed parses and scoring errors; parsing success is independent of scoring success. Each sample stores only `gt_traj`, `pred_traj`, and `L2_per_step`; failed parses keep the truth and store `null` for prediction and L2. The original `try/except` catches scoring errors, logs a warning and continues; all three result fields are `null` for those samples. Metrics average only successfully parsed and scored samples, with `null` metrics when none qualify. Predictions without a `parse_success` field are treated as successfully parsed.

`metrics.uniad` and `metrics.stp3` contain `obj_box_col` and `L2`: respectively the mean of steps 2/4/6 and the mean of the three prefix means at those steps. `metrics.per_step` stores their shared six-step curves. Collision geometry uses SparseDrive polygons and masks ground-truth collisions internally; these names describe temporal aggregation, not the full native UniAD/ST-P3 benchmarks. SparseDrive does not write visualizations. The project evaluation YAMLs disable Hydra job logging and set `hydra.output_subdir: null`, leaving only `results.json` and the effective `config.yaml` in `results/<timestamp>/`.

## Other datasets

For example, we evaluate a checkpoint on nuPlan test split with the config `src/py123d_garage/config/presets/yaml/offline_data_sources/tf_nuplan/nuplan_test.yaml` and the command:

```bash
python -m py123d_garage.evaluation.open_loop.evaluate \
    policy_config.evaluation_checkpoint_file=<checkpoint.pth> \
    +offline_data_sources/ltf_nuplan@benchmark_offline_data_sources.nuplan_test=nuplan_test \
    benchmark_offline_data_sources.nuplan_test.cache_root=null \
    parallelization_config.accelerator=gpu \
    parallelization_config.devices=auto
```
