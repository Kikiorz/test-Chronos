"""Export a compact EMA-only RGB-Joint deployment checkpoint."""

from __future__ import annotations

import argparse
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
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    partial.unlink(missing_ok=True)
    torch.save(payload, partial)
    partial.replace(output)
    print(
        f"Exported {len(policy_state)} EMA tensors to {output} "
        f"({output.stat().st_size / 2**30:.2f} GiB)"
    )


if __name__ == "__main__":
    main()
