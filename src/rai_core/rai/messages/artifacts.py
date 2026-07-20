# Copyright (C) 2025 Robotec.AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
import json
import re
from pathlib import Path
from typing import Any, List, TypedDict


class MultimodalArtifact(TypedDict, total=False):
    images: List[str]  # base64 encoded images
    raw_images: List[str]  # base64 encoded images stored outside checkpoints
    audios: List[str]
    summary: str


class StoredArtifactReference(TypedDict):
    artifact_id: str
    summary: str
    images: int
    raw_images: int
    audios: int


class ToolArtifactRecord(TypedDict, total=False):
    summary: str
    images: List[str]
    raw_images: List[str]


def _default_artifact_root(db_path: str | Path = "data/artifacts") -> Path:
    return Path(db_path)


def _safe_tool_call_id(tool_call_id: str) -> str:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", tool_call_id)
    return safe_id or "unknown_tool_call"


def _tool_artifact_dir(tool_call_id: str, root: str | Path = "data/artifacts") -> Path:
    return _default_artifact_root(root) / _safe_tool_call_id(tool_call_id)


def _write_image(image_b64: str, output_path: Path) -> None:
    output_path.write_bytes(base64.b64decode(image_b64))


def _read_image(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def store_artifacts(
    tool_call_id: str,
    artifacts: List[Any],
    db_path: str | Path = "data/artifacts",
) -> None:
    artifact_dir = _tool_artifact_dir(tool_call_id, db_path)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, Any] = {
        "tool_call_id": tool_call_id,
        "artifacts": [],
    }
    image_index = 0
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            metadata["artifacts"].append({"value": artifact})
            continue

        stored_artifact: dict[str, Any] = {
            "summary": artifact.get("summary", ""),
            "audios": [],
            "images": [],
            "raw_images": [],
        }
        for key, extension in (
            ("images", "png"),
            ("raw_images", "png"),
            ("audios", "bin"),
        ):
            values = artifact.get(key, [])
            if not isinstance(values, list):
                continue
            for value_b64 in values:
                if not isinstance(value_b64, str):
                    continue
                filename = f"{key}_{image_index:04d}.{extension}"
                _write_image(value_b64, artifact_dir / filename)
                stored_artifact[key].append(filename)
                image_index += 1
        metadata["artifacts"].append(stored_artifact)

    (artifact_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def get_stored_artifacts(
    tool_call_id: str,
    db_path: str | Path = "data/artifacts",
) -> List[Any]:
    artifact_dir = _tool_artifact_dir(tool_call_id, db_path)
    metadata_path = artifact_dir / "metadata.json"
    if not metadata_path.is_file():
        return []

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    restored: list[Any] = []
    for artifact in metadata.get("artifacts", []):
        if not isinstance(artifact, dict):
            restored.append(artifact)
            continue
        restored_artifact: dict[str, Any] = {
            "summary": artifact.get("summary", ""),
            "audios": [],
            "images": [],
            "raw_images": [],
        }
        for key in ("images", "raw_images", "audios"):
            for filename in artifact.get(key, []):
                image_path = artifact_dir / filename
                if image_path.is_file():
                    restored_artifact[key].append(_read_image(image_path))
                elif key == "audios" and isinstance(filename, str):
                    # Older records stored audio base64 directly in metadata.
                    restored_artifact[key].append(filename)
        restored.append(restored_artifact)
    return restored


def store_tool_artifact_record(
    tool_call_id: str,
    record: ToolArtifactRecord,
    db_path: str | Path = "data/artifacts",
) -> None:
    store_artifacts(tool_call_id, [record], db_path=db_path)


def get_tool_artifact_record(
    tool_call_id: str,
    db_path: str | Path = "data/artifacts",
) -> ToolArtifactRecord | None:
    records = get_stored_artifacts(tool_call_id, db_path=db_path)
    if records and isinstance(records[0], dict):
        return records[0]
    return None


def stored_artifact_reference(
    tool_call_id: str,
    artifact: MultimodalArtifact,
) -> StoredArtifactReference:
    """Return checkpoint-safe metadata for an externally stored artifact."""

    def count(key: str) -> int:
        values = artifact.get(key, [])
        return len(values) if isinstance(values, list) else 0

    return StoredArtifactReference(
        artifact_id=tool_call_id,
        summary=str(artifact.get("summary", "")),
        images=count("images"),
        raw_images=count("raw_images"),
        audios=count("audios"),
    )
