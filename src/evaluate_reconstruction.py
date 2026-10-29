"""Evaluate saved, paired reconstruction and input NPZ files offline."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from eval import compute_reconstruction_metrics


def load_images(path: Path) -> np.ndarray:
    with np.load(path) as archive:
        if "arr_0" not in archive:
            raise KeyError(f"{path} must contain arr_0")
        return archive["arr_0"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="Input crops saved by stage1_sample_ddp.py")
    parser.add_argument("--reconstruction", type=Path, required=True, help="Reconstructed images saved by stage1_sample_ddp.py")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--output", type=Path, help="Optional JSON results path")
    args = parser.parse_args()

    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    reference = load_images(args.reference)
    reconstruction = load_images(args.reconstruction)
    results = compute_reconstruction_metrics(reference, reconstruction, torch.device(device_name), args.batch_size)
    output = {
        "reference": str(args.reference),
        "reconstruction": str(args.reconstruction),
        "num_images": int(reference.shape[0]),
        "metrics": results,
    }
    rendered = json.dumps(output, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
