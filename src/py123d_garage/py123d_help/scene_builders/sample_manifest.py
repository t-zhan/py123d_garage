"""Select scenes in a published sample manifest's order."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from py123d.api import SceneAPI
from py123d.api.scene.arrow.arrow_scene_api import ArrowSceneAPI
from py123d.datatypes import CameraID, ModalityType


def select_manifest_scenes(
    scenes: Sequence[SceneAPI],
    manifest_path: str,
    camera_id: CameraID,
    log_names: set[str] | None = None,
) -> list[SceneAPI]:
    rows = [json.loads(line) for line in Path(manifest_path).read_text().splitlines()]
    keys = [
        (row["scene_name"], row["timestamp_us"]) for row in rows if log_names is None or row["scene_name"] in log_names
    ]
    wanted = set(keys)
    wanted_logs = {key[0] for key in wanted}
    indexed: dict[tuple[str, int], SceneAPI] = {}
    for scene in scenes:
        if scene.log_name not in wanted_logs:
            continue
        timestamp_us = cast(ArrowSceneAPI, scene).get_modality_column_at_iteration(
            0,
            "timestamp_us",
            ModalityType.CAMERA,
            camera_id,
        )
        if timestamp_us is None:
            raise ValueError(f"No {camera_id.name} at {scene.scene_uuid}")
        key = (scene.log_name, int(timestamp_us))
        if key in wanted:
            if key in indexed:
                raise ValueError(f"Multiple 123D scenes match {key}")
            indexed[key] = scene
    missing = wanted - indexed.keys()
    if missing:
        raise KeyError(f"Manifest scenes missing from 123D: {sorted(missing)[:5]} ({len(missing)} total)")
    return [indexed[key] for key in keys]
