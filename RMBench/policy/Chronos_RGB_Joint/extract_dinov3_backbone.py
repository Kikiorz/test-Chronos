"""Extract a strict standalone DINOv3-B/16 state without publishing weights."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from safetensors.torch import save_file

try:
    from .dinov3_backbone import (
        DINOV3_MODEL_NAME,
        _load_tensor_mapping,
        _select_strict_backbone_state,
        build_dinov3_vitb16,
    )
except ImportError:  # direct script execution
    from dinov3_backbone import (  # type: ignore
        DINOV3_MODEL_NAME,
        _load_tensor_mapping,
        _select_strict_backbone_state,
        build_dinov3_vitb16,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    loaded = _load_tensor_mapping(source)
    reference = build_dinov3_vitb16(pretrained=False, weights_path=None)
    state = _select_strict_backbone_state(loaded, set(reference.state_dict()))
    reference.load_state_dict(state, strict=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {key: value.detach().cpu().contiguous() for key, value in state.items()},
        str(output),
        metadata={
            "model_name": DINOV3_MODEL_NAME,
            "source_sha256": _sha256(source),
            "tensor_count": str(len(state)),
        },
    )
    # Re-open through the production loader to prove namespace and shapes.
    verified = build_dinov3_vitb16(pretrained=False, weights_path=output)
    if len(verified.state_dict()) != len(state):
        raise RuntimeError("Standalone DINOv3 verification failed")
    print(
        f"Extracted {len(state)} DINOv3 tensors to {output} "
        f"(sha256={_sha256(output)})"
    )


if __name__ == "__main__":
    main()
