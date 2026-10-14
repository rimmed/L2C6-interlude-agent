"""
Export a trained :class:`CombatEventTransformer` checkpoint to ONNX.

Two files are written next to each other:

* ``<name>.onnx`` - the network: five positional inputs, three logit outputs, dynamic batch axis.
* ``<name>.json`` - everything a runtime needs to build the inputs **without** importing torch,
  pandas or the dataloader: window length, Top-K / panel sizes, the action label space and the
  feature-normalisation constants of :mod:`l2_NN.dataloader.dataloder` and
  ``l2_client/skill_features.py``.  The skill panel itself is built by the client from the
  character's own skills, so the model is not tied to the training characters.  The headless
  client (``l2_client/policy.py``) reads only this pair of files.

After the export the ONNX graph is run with ``onnxruntime`` on a left-padded probe batch and
compared to the PyTorch model, so a broken export fails here instead of inside the bot.

Usage::

    uv run l2_NN/export_onnx.py                                   # checkpoints/best_model.pth -> checkpoints/combat_policy.onnx
    uv run l2_NN/export_onnx.py -c checkpoints/best_model.pth -o models/combat_policy.onnx
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import numpy as np
import torch
from torch import nn

# ``uv run l2_NN/export_onnx.py`` puts *this file's* directory on ``sys.path``; the project root (the
# folder holding ``l2_NN/``) has to be added before the package can be imported.
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from l2_client import skill_features as sf  # noqa: E402
from l2_NN.dataloader import dataloder as dl  # noqa: E402 - PROJECT_ROOT is added just above
from l2_NN.model.CombatEventTransformer import CombatEventTransformer  # noqa: E402

INPUT_NAMES: Final[tuple[str, ...]] = ("player_state", "delta_t", "mobs_matrix", "skills_matrix", "seq_mask")
OUTPUT_NAMES: Final[tuple[str, ...]] = ("logits_action_type", "logits_pointer_idx", "logits_skill_slot")

#: Bumped whenever the meaning of a metadata field changes, so the client can refuse a stale file.
METADATA_VERSION: Final[int] = 2  # 2: skill slots described by datapack features, pointer heads


class OnnxPolicy(nn.Module):
    """Positional-argument wrapper: ONNX graphs take a flat tensor list, not the model's batch dict."""

    def __init__(self, model: CombatEventTransformer) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        player_state: torch.Tensor,
        delta_t: torch.Tensor,
        mobs_matrix: torch.Tensor,
        skills_matrix: torch.Tensor,
        seq_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.model(
            {
                "player_state": player_state,
                "delta_t": delta_t,
                "mobs_matrix": mobs_matrix,
                "skills_matrix": skills_matrix,
                "seq_mask": seq_mask,
            }
        )
        return tuple(out[name] for name in OUTPUT_NAMES)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a CombatEventTransformer checkpoint to ONNX + JSON metadata.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-c", "--checkpoint", type=Path, default=Path("checkpoints/best_model.pth"))
    parser.add_argument(
        "-o", "--output", type=Path, default=None, help="Target .onnx path (default: <checkpoint dir>/combat_policy.onnx)"
    )
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--no-verify", action="store_true", help="Skip the onnxruntime vs PyTorch comparison")
    parser.add_argument("--atol", type=float, default=1e-4, help="Max abs logit difference accepted by the check")
    return parser.parse_args(argv)


def load_checkpoint(path: Path) -> tuple[CombatEventTransformer, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = CombatEventTransformer(**checkpoint["model_config"])
    load_policy_weights(model, checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def load_policy_weights(model: CombatEventTransformer, state_dict: dict) -> None:
    """
    Load a checkpoint; behaviour-cloning checkpoints predate the RL value head, so only
    ``head_value.*`` may be missing (it keeps its fresh initialisation).
    """
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    wrong = unexpected + [key for key in missing if not key.startswith("head_value.")]
    if wrong:
        raise RuntimeError(f"checkpoint does not match the model: {wrong}")


def probe_inputs(config: dict, seq_len: int, batch_size: int = 4, seed: int = 0) -> tuple[torch.Tensor, ...]:
    """A random left-padded batch shaped like the dataloader output (rows with 1..L real events)."""
    generator = torch.Generator().manual_seed(seed)
    max_mobs, max_skills = config["max_mobs"], config["max_skills"]
    player_state = torch.rand(batch_size, seq_len, config["player_dim"], generator=generator)
    delta_t = torch.rand(batch_size, seq_len, generator=generator)
    mobs = torch.rand(batch_size, seq_len, max_mobs, config["mob_features"], generator=generator)
    mobs[..., dl.MOB_PADDED_INDEX] = (torch.rand(batch_size, seq_len, max_mobs, generator=generator) > 0.7).float()
    skills = (torch.rand(batch_size, seq_len, max_skills, config["skill_features"], generator=generator) > 0.5).float()

    seq_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    for row in range(batch_size):
        real = 1 + (row * (seq_len - 1)) // max(1, batch_size - 1)  # 1 real event ... a full window
        seq_mask[row, seq_len - real :] = True
    padded = ~seq_mask
    player_state[padded] = 0.0
    delta_t[padded] = 0.0
    mobs[padded] = 0.0
    mobs[..., dl.MOB_PADDED_INDEX][padded] = 1.0
    skills[padded] = 0.0
    return player_state, delta_t, mobs, skills, seq_mask


def build_metadata(checkpoint: dict, onnx_path: Path) -> dict:
    args = checkpoint.get("args", {})
    config = checkpoint["model_config"]
    return {
        "metadata_version": METADATA_VERSION,
        "onnx_file": onnx_path.name,
        "inputs": list(INPUT_NAMES),
        "outputs": list(OUTPUT_NAMES),
        "seq_len": int(args.get("seq_len", 16)),
        "max_mobs": config["max_mobs"],
        "max_skills": config["max_skills"],
        # train.py zeroes is_current_target at the decision step unless --keep-last-target-flag was set;
        # inference has to feed the network the same way.
        "hide_last_target": not bool(args.get("keep_last_target_flag", False)),
        "action_to_id": checkpoint["action_to_id"],
        "features": {
            "player": list(dl.PLAYER_FEATURE_NAMES),
            "mob": list(dl.MOB_FEATURE_NAMES),
            "skill": list(dl.SKILL_FEATURE_NAMES),
        },
        "constants": {
            "max_level": dl.MAX_LEVEL,
            "max_distance": dl.MAX_DISTANCE,
            "max_cast_speed": dl.MAX_CAST_SPEED,
            "max_atk_speed": dl.MAX_ATK_SPEED,
            "max_move_speed": dl.MAX_MOVE_SPEED,
            "default_cast_speed": dl.DEFAULT_CAST_SPEED,
            "default_atk_speed": dl.DEFAULT_ATK_SPEED,
            # The logs never carried a move speed, so every training row used this default.
            "default_move_speed": dl.DEFAULT_MOVE_SPEED,
            "skill_cooldown_norm_seconds": dl.SKILL_COOLDOWN_NORM_SECONDS,
        },
        # The skill panel is the character's own combat skills (skill_data.combat_panel), encoded by
        # l2_client/skill_features.py; the client checks that its encoder still matches these scales.
        "skill_encoding": {
            "power_log_scale": sf.POWER_LOG_SCALE,
            "cast_range_scale": sf.CAST_RANGE_SCALE,
            "hit_time_scale_ms": sf.HIT_TIME_SCALE_MS,
            "reuse_log_scale": sf.REUSE_LOG_SCALE,
            "cooldown_scale_s": sf.COOLDOWN_SCALE_S,
        },
        "training": {
            "epoch": checkpoint.get("epoch"),
            "val_metrics": checkpoint.get("val_metrics"),
            "split": checkpoint.get("split"),
        },
    }


def verify(onnx_path: Path, wrapper: OnnxPolicy, inputs: tuple[torch.Tensor, ...], atol: float) -> float:
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    feeds = {name: tensor.numpy() for name, tensor in zip(INPUT_NAMES, inputs)}
    onnx_out = session.run(list(OUTPUT_NAMES), feeds)
    with torch.no_grad():
        torch_out = wrapper(*inputs)
    worst = 0.0
    for name, expected, got in zip(OUTPUT_NAMES, torch_out, onnx_out):
        if not np.isfinite(got).all():
            raise RuntimeError(f"onnxruntime produced non-finite values in {name}")
        worst = max(worst, float(np.max(np.abs(expected.numpy() - got))))
    if worst > atol:
        raise RuntimeError(f"ONNX output differs from PyTorch by {worst:.3e} (> atol {atol:.1e})")
    return worst


def export_policy(model: CombatEventTransformer, checkpoint: dict, onnx_path: Path, *, opset: int = 18,
                  atol: float | None = 1e-4) -> dict:
    """
    Write ``onnx_path`` and its ``.json`` metadata for ``model`` (on the CPU, any mode).

    ``checkpoint`` supplies ``model_config``, ``action_to_id`` and the training ``args`` (see
    :func:`build_metadata`). With ``atol`` the graph is checked against PyTorch with onnxruntime.
    Used by this script and by the RL trainer, which publishes every new policy this way.
    """
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = build_metadata(checkpoint, onnx_path)
    wrapper = OnnxPolicy(model).eval()
    # The probe batch must not satisfy batch * nhead == seq_len: PyTorch's attention compares the
    # (batch * nhead, L, L) mask against an (L, L) one, and torch.export turns that coincidental
    # equality into a guard that freezes the batch axis.
    probe_batch = next(b for b in range(3, 64) if b * checkpoint["model_config"]["nhead"] != metadata["seq_len"])
    inputs = probe_inputs(checkpoint["model_config"], metadata["seq_len"], batch_size=probe_batch)
    batch = torch.export.Dim("batch", min=1)
    dynamic_shapes = {name: {0: batch} for name in INPUT_NAMES}
    torch.onnx.export(
        wrapper,
        inputs,
        str(onnx_path),
        input_names=list(INPUT_NAMES),
        output_names=list(OUTPUT_NAMES),
        dynamic_shapes=dynamic_shapes,
        opset_version=opset,
        dynamo=True,
        external_data=False,  # one self-contained file; the model is ~1 MB
        verbose=False,
    )
    onnx_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2))

    if atol is not None:
        worst = verify(onnx_path, wrapper, inputs, atol)
        # Other batch sizes (1 = the client's case) prove the batch axis really is dynamic.
        for batch_size in (1, 4, 32):
            probe = probe_inputs(checkpoint["model_config"], metadata["seq_len"], batch_size=batch_size, seed=batch_size)
            worst = max(worst, verify(onnx_path, wrapper, probe, atol))
        metadata["verified_max_abs_diff"] = worst
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    onnx_path = args.output or args.checkpoint.with_name("combat_policy.onnx")
    model, checkpoint = load_checkpoint(args.checkpoint)
    print(f"Loaded {args.checkpoint} (epoch {checkpoint.get('epoch')})")
    metadata = export_policy(model, checkpoint, onnx_path, opset=args.opset, atol=None if args.no_verify else args.atol)
    print(f"Wrote {onnx_path} ({onnx_path.stat().st_size / 1024:.0f} KiB) and {onnx_path.with_suffix('.json')}")
    if "verified_max_abs_diff" in metadata:
        print(f"Verified with onnxruntime: max |torch - onnx| = {metadata['verified_max_abs_diff']:.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
