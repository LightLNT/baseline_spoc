#!/usr/bin/env python3
"""Benchmark inference FPS for offline transformer models."""

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace
from typing import Any, Dict

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
import yaml

from training.offline.train_pl import (
    LitModel,
    arg_parser_for_offline_training,
    get_dataloader,
)


def _load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        if config_path.endswith(".json"):
            return json.load(f)
        return yaml.safe_load(f)


def _build_args_from_config(config: Dict[str, Any], overrides: argparse.Namespace) -> argparse.Namespace:
    base_parser = arg_parser_for_offline_training()
    # parse empty list to get defaults
    args = base_parser.parse_args([])
    for key, value in config.items():
        setattr(args, key, value)
    for key, value in vars(overrides).items():
        if value is not None:
            setattr(args, key, value)
    return args


def benchmark(args: argparse.Namespace):
    device = torch.device(args.device)
    args.ckpt_pth = args.ckpt_path
    args._prepared_input_sensors = False

    dataloader = get_dataloader(args.data_subset, args)

    lit_model = LitModel(args)
    lit_model.eval()
    lit_model.to(device)
    lit_model.preproc.device = device

    is_cuda = device.type == "cuda"
    total_frames = 0
    measured_time = 0.0
    processed_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= args.num_batches:
                break
            if batch is None or len(batch) == 0:
                continue
            start = time.perf_counter()
            outputs, proc_batch = lit_model.forward_batch(batch)
            if is_cuda:
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            frames = float(proc_batch["lengths"].sum().item())
            if batch_idx >= args.warmup_batches:
                total_frames += frames
                measured_time += elapsed
                processed_batches += 1

    if processed_batches == 0 or measured_time == 0:
        fps = 0.0
    else:
        fps = total_frames / measured_time

    print(
        json.dumps(
            {
                "ckpt": args.ckpt_path,
                "model_version": args.model_version,
                "num_batches": processed_batches,
                "frames": total_frames,
                "elapsed_sec": measured_time,
                "fps": fps,
            },
            indent=2,
        )
    )


def parse_cli():
    parser = argparse.ArgumentParser(description="Benchmark offline model FPS")
    parser.add_argument("--config", required=True, help="Path to training config.yaml (or json)")
    parser.add_argument("--ckpt_path", required=True, help="Path to checkpoint file (model.ckpt)")
    parser.add_argument("--model_version", required=False, help="Override model_version from config")
    parser.add_argument("--input_sensors", nargs="+", help="Optional override for input sensors")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_batches", type=int, default=50)
    parser.add_argument("--warmup_batches", type=int, default=5)
    parser.add_argument("--data_subset", default="val", help="train/val split for dataloader")
    parser.add_argument("--keep_manip_camera", action="store_true")
    parser.add_argument("--precision", help="Override precision (e.g., 16-mixed)")
    parser.add_argument("--per_gpu_batch", type=int, help="Override batch size as needed")
    parser.add_argument("--detector_usage")
    parser.add_argument("--data_dir")
    parser.add_argument("--dataset_version")
    args = parser.parse_args()

    config = _load_config(args.config)
    overrides = SimpleNamespace(
        model_version=args.model_version,
        input_sensors=args.input_sensors,
        device=args.device,
        num_batches=args.num_batches,
        warmup_batches=args.warmup_batches,
        data_subset=args.data_subset,
        keep_manip_camera=args.keep_manip_camera,
        precision=args.precision,
        per_gpu_batch=args.per_gpu_batch,
        detector_usage=args.detector_usage,
        data_dir=args.data_dir,
        dataset_version=args.dataset_version,
    )
    combined_args = _build_args_from_config(config, overrides)
    combined_args.device = args.device
    combined_args.num_batches = args.num_batches
    combined_args.warmup_batches = args.warmup_batches
    combined_args.data_subset = args.data_subset
    combined_args.keep_manip_camera = args.keep_manip_camera
    return combined_args, args


if __name__ == "__main__":
    combined_args, raw_args = parse_cli()
    benchmark(SimpleNamespace(**{**vars(combined_args), "ckpt_path": raw_args.ckpt_path}))
