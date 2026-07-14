"""Export a compact EMA-only RGB-Joint deployment checkpoint."""

from __future__ import annotations

import argparse
import math
import os
from collections.abc import Mapping
from pathlib import Path

import torch

try:
    from .contracts import base_policy_contract
    from .scaler_M import Scaler
except ImportError:  # direct script execution
    from contracts import base_policy_contract
    from scaler_M import Scaler


def _load_checkpoint(path: Path) -> Mapping[str, object]:
    try:
        checkpoint = torch.load(
            path, map_location="cpu", weights_only=False, mmap=True
        )
    except TypeError:  # older torch
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Training checkpoint must be a mapping")
    return checkpoint


def _validation_metadata(
    checkpoint: Mapping[str, object], source: Path
) -> dict[str, object]:
    """Extract a score only when ``source`` is an exact monitored top-k model.

    Lightning's ``current_score`` can be stale in a rotating resume checkpoint
    when that epoch did not enter top-k.  ``best_k_models`` binds a score to a
    specific file, so it is the only safe source of selection metadata here.
    """

    callbacks = checkpoint.get("callbacks")
    matched_scores: list[float] = []
    if isinstance(callbacks, Mapping):
        for callback_key, state in callbacks.items():
            if (
                not isinstance(state, Mapping)
                or state.get("monitor") != "val_loss"
                # PL 2.4 omits mode from ModelCheckpoint.state_dict(), but it
                # remains part of the callback state key.
                or not isinstance(callback_key, str)
                or "'mode': 'min'" not in callback_key
            ):
                continue
            best_k_models = state.get("best_k_models")
            if not isinstance(best_k_models, Mapping):
                continue
            for model_path, candidate in best_k_models.items():
                if not isinstance(model_path, str):
                    continue
                if Path(model_path).expanduser().resolve() != source:
                    continue
                if torch.is_tensor(candidate):
                    if candidate.numel() != 1:
                        raise ValueError("A top-k validation score must be scalar")
                    candidate = candidate.detach().cpu().item()
                matched_scores.append(float(candidate))
    if not matched_scores:
        return {}
    score = matched_scores[0]
    if any(candidate != score for candidate in matched_scores[1:]):
        raise ValueError(
            f"Monitored callbacks disagree about the validation score for {source}"
        )
    epoch = int(checkpoint.get("epoch", -1))
    global_step = int(checkpoint.get("global_step", -1))
    ema_step = int(checkpoint.get("ema_optimization_step", -1))
    if not math.isfinite(score) or epoch < 0 or global_step < 0 or ema_step < 0:
        raise ValueError("Training checkpoint has invalid best-validation metadata")
    return {
        "val_loss": score,
        "epoch": epoch,
        "global_step": global_step,
        "ema_optimization_step": ema_step,
        "monitor": "val_loss",
        "mode": "min",
    }


def _validate_complete_ema_policy(
    ema_state: Mapping[str, object], lightning_state: Mapping[str, object]
) -> None:
    """Require the deploy EMA namespace to match the full raw policy exactly."""

    raw_policy = {
        key[len("policy."):]: value
        for key, value in lightning_state.items()
        if isinstance(key, str)
        and key.startswith("policy.")
        and torch.is_tensor(value)
    }
    if not raw_policy:
        raise ValueError("Training checkpoint has no raw policy namespace")
    raw_keys = set(raw_policy)
    ema_keys = set(ema_state)
    if raw_keys != ema_keys:
        raise ValueError(
            "EMA policy is incomplete: "
            f"missing={sorted(raw_keys - ema_keys)}, "
            f"unexpected={sorted(ema_keys - raw_keys)}"
        )
    for key, raw_tensor in raw_policy.items():
        ema_tensor = ema_state[key]
        if not torch.is_tensor(ema_tensor):
            raise TypeError(f"EMA policy entry {key!r} is not a tensor")
        if ema_tensor.shape != raw_tensor.shape or ema_tensor.dtype != raw_tensor.dtype:
            raise ValueError(
                f"EMA policy entry {key!r} differs from raw policy metadata: "
                f"shape {tuple(ema_tensor.shape)} vs {tuple(raw_tensor.shape)}, "
                f"dtype {ema_tensor.dtype} vs {raw_tensor.dtype}"
            )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Strip optimizer/raw weights and retain EMA policy + scaler contract"
    )
    parser.add_argument("--input", required=True, help="Full Lightning .ckpt")
    parser.add_argument("--output", required=True, help="Compact deployment .pth")
    args = parser.parse_args(argv)

    source = Path(args.input).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Training checkpoint does not exist: {source}")
    if source == output:
        raise ValueError("Deployment output must differ from the source checkpoint")
    checkpoint = _load_checkpoint(source)

    contract = checkpoint.get("policy_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("Training checkpoint is missing policy_contract")
    mismatches = {
        key: (contract.get(key), expected)
        for key, expected in base_policy_contract().items()
        if contract.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Incompatible RGB-Joint policy contract: {mismatches}")

    policy_state = checkpoint.get("ema_policy_state_dict")
    if not isinstance(policy_state, Mapping) or not policy_state:
        raise ValueError("Training checkpoint has no complete EMA policy state")
    if not all(isinstance(key, str) and torch.is_tensor(value) for key, value in policy_state.items()):
        raise TypeError("EMA policy state must map string keys to tensors")

    lightning_state = checkpoint.get("state_dict")
    if not isinstance(lightning_state, Mapping):
        raise ValueError("Training checkpoint has no Lightning state_dict")
    _validate_complete_ema_policy(policy_state, lightning_state)
    scaler_state = {
        key[len("scaler."):]: value
        for key, value in lightning_state.items()
        if isinstance(key, str) and key.startswith("scaler.") and torch.is_tensor(value)
    }
    scaler_fingerprint = Scaler.state_fingerprint(scaler_state)
    if scaler_fingerprint != contract.get("scaler_fingerprint"):
        raise ValueError("Embedded scaler does not match policy_contract")

    payload = {
        "format": "chronos_rgb_joint_deploy_v1",
        "policy_contract": dict(contract),
        "ema_policy_state_dict": dict(policy_state),
        "scaler_state_dict": scaler_state,
        **_validation_metadata(checkpoint, source),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.parent / f".{output.name}.{os.getpid()}.partial"
    partial.unlink(missing_ok=True)
    try:
        torch.save(payload, partial)
        file_fd = os.open(partial, os.O_RDONLY)
        try:
            os.fsync(file_fd)
        finally:
            os.close(file_fd)
        os.replace(partial, output)
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    print(
        f"Exported {len(policy_state)} EMA tensors to {output} "
        f"({output.stat().st_size / 2**30:.2f} GiB)"
    )


if __name__ == "__main__":
    main()
