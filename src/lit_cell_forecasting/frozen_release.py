"""Load and validate the frozen MDCK Bulk LIT-Cell fold states."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .model import CausalInnovationStateSpaceForecaster


DEFAULT_RELEASE_DIR = (
    Path(__file__).resolve().parents[2] / "models" / "lit_cell_mdck_bulk_primary"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def state_dict_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Hash tensor names, shapes, dtypes, and bytes independently of torch.save."""

    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        if tensor.dtype == torch.bfloat16:
            raw = tensor.view(torch.uint16).numpy().tobytes(order="C")
        else:
            raw = tensor.numpy().tobytes(order="C")
        digest.update(raw)
    return digest.hexdigest()


def load_release_manifest(release_dir: Path | str | None = None) -> dict[str, Any]:
    root = Path(release_dir) if release_dir is not None else DEFAULT_RELEASE_DIR
    manifest_path = root / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema") != "lit_cell_frozen_fold_release_v1":
        raise ValueError(f"Unsupported frozen release schema: {payload.get('schema')!r}")
    return payload


def checkpoint_entry(
    test_movie: int,
    seed: int,
    release_dir: Path | str | None = None,
) -> dict[str, Any]:
    manifest = load_release_manifest(release_dir)
    matches = [
        entry
        for entry in manifest["checkpoints"]
        if int(entry["test_movie"]) == int(test_movie)
        and int(entry["seed"]) == int(seed)
    ]
    if len(matches) != 1:
        raise KeyError(
            f"Expected one state for test_movie={test_movie}, seed={seed}; "
            f"found {len(matches)}"
        )
    return matches[0]


def build_model_from_payload(
    payload: Mapping[str, Any],
    *,
    device: str | torch.device = "cpu",
) -> CausalInnovationStateSpaceForecaster:
    state_dict = payload["state_dict"]
    args = payload["args"]
    variant = payload["variant"]
    first_weight = state_dict["static_encoder.0.weight"]
    model = CausalInnovationStateSpaceForecaster(
        static_dim=int(first_weight.shape[1]),
        hidden=int(first_weight.shape[0]),
        history_lags=int(args["history_lags"]),
        correction_bound=float(args["correction_bound"]),
        dropout=float(args["dropout"]),
        use_update=bool(variant["use_update"]),
        use_graph=bool(variant["use_graph"]),
        graph_heads=int(args["graph_heads"]),
        output_mode=str(variant["output_mode"]),
        target_mean=state_dict["target_mean"].detach().cpu().numpy(),
        target_scale=state_dict["target_scale"].detach().cpu().numpy(),
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


@dataclass(frozen=True)
class FrozenFoldState:
    entry: dict[str, Any]
    payload: dict[str, Any]
    model: CausalInnovationStateSpaceForecaster


def load_frozen_fold_state(
    test_movie: int,
    seed: int,
    *,
    release_dir: Path | str | None = None,
    device: str | torch.device = "cpu",
    verify: bool = True,
) -> FrozenFoldState:
    """Load one trusted checkpoint distributed with the LIT-Cell repository."""

    root = Path(release_dir) if release_dir is not None else DEFAULT_RELEASE_DIR
    entry = checkpoint_entry(test_movie, seed, root)
    path = root / entry["path"]
    if verify and file_sha256(path) != entry["sha256"]:
        raise ValueError(f"Checkpoint file hash mismatch: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if verify and state_dict_sha256(payload["state_dict"]) != entry["state_dict_sha256"]:
        raise ValueError(f"Checkpoint tensor hash mismatch: {path}")
    model = build_model_from_payload(payload, device=device)
    return FrozenFoldState(entry=entry, payload=payload, model=model)

