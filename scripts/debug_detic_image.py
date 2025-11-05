#!/usr/bin/env python
"""Run Detic on a single RGB image and dump detection metadata for debugging.

This small utility mirrors the vocabulary/allowed-phrase logic used in the
training pipeline so you can quickly validate whether Detic can locate the
objects implied by a goal description.

Example:
    python scripts/debug_detic_image.py --image path/to/frame.png \
        --goal "pick up the red mug" --output detic_log.json

Ensure DETIC_REPO_PATH is set so `utils.detic_utils.DeticPredictor` can locate
its dependencies.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List, Sequence

import numpy as np
import torch
from PIL import Image

from utils.detic_utils import DeticPredictor


def _normalize_phrases(phrases: Sequence[str] | None) -> List[str]:
    if not phrases:
        return []
    normalized: List[str] = []
    for phrase in phrases:
        if not isinstance(phrase, str):
            continue
        cleaned = phrase.strip()
        if cleaned:
            normalized.append(cleaned)
    return normalized


def _build_detector_vocabulary(
    base_phrases: Sequence[str] | None,
    goal_text: str | None,
    max_vocab: int,
) -> tuple[List[str], List[str]]:
    base = _normalize_phrases(base_phrases)
    if not base:
        base = ["object"]

    candidates: List[str] = list(base)
    if goal_text:
        cleaned = goal_text.strip()
        if cleaned:
            candidates.append(cleaned)

    seen = set()
    vocabulary: List[str] = []
    for phrase in candidates:
        if phrase in seen:
            continue
        vocabulary.append(phrase)
        seen.add(phrase)
        if max_vocab and len(vocabulary) >= max_vocab:
            break

    if not vocabulary:
        vocabulary = ["object"]

    allowed = list({phrase for phrase in base if phrase in vocabulary})
    if goal_text:
        cleaned = goal_text.strip()
        if cleaned and cleaned in vocabulary and cleaned not in allowed:
            allowed.append(cleaned)
    if not allowed:
        allowed = list(vocabulary)

    return vocabulary, allowed


def _load_image_tensor(image_path: str) -> tuple[torch.Tensor, tuple[int, int]]:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    array = np.array(image, dtype=np.uint8)
    tensor = torch.from_numpy(array).permute(2, 0, 1).float()
    return tensor, (height, width)


def _filter_detections(
    predictor: DeticPredictor,
    image_tensor: torch.Tensor,
    vocabulary: Sequence[str],
    allowed: Sequence[str],
    max_detections: int,
) -> tuple[list[dict], dict]:
    detections = predictor(image_tensor.unsqueeze(0))
    result: list[dict] = []
    extra: dict = {
        "vocabulary": list(vocabulary),
        "allowed_phrases": list(allowed),
    }

    if not detections:
        extra["status"] = "empty_predictions"
        return result, extra

    det = detections[0]
    instances = det.get("instances") if isinstance(det, dict) else None
    if instances is None or len(instances) == 0:
        extra["status"] = "no_instances"
        return result, extra

    instances = instances.to("cpu")
    boxes = instances.pred_boxes.tensor.detach().clone()
    scores = instances.scores.detach().clone()
    classes = instances.pred_classes.detach().clone().to(torch.int64)

    label_lookup = {idx: vocabulary[idx] for idx in range(len(vocabulary))}
    allowed_set = set(allowed)

    filtered_indices: List[int] = []
    for idx, cls_idx in enumerate(classes.tolist()):
        label = label_lookup.get(cls_idx)
        if label is None:
            continue
        if label in allowed_set:
            filtered_indices.append(idx)

    if filtered_indices:
        boxes = boxes[filtered_indices]
        scores = scores[filtered_indices]
        classes = classes[filtered_indices]
    else:
        extra["status"] = "filtered_out"
        return result, extra

    if max_detections and boxes.shape[0] > max_detections:
        topk_scores, topk_idx = torch.topk(scores, max_detections)
        boxes = boxes[topk_idx]
        scores = topk_scores
        classes = classes[topk_idx]

    image_height, image_width = instances.image_size
    width_scale = float(image_width) if image_width > 0 else 1.0
    height_scale = float(image_height) if image_height > 0 else 1.0

    for idx in range(boxes.shape[0]):
        label = label_lookup.get(int(classes[idx]), "?")
        box_pixels = boxes[idx].tolist()
        box_norm = [
            float(box_pixels[0] / width_scale),
            float(box_pixels[1] / height_scale),
            float(box_pixels[2] / width_scale),
            float(box_pixels[3] / height_scale),
        ]
        result.append(
            {
                "label": label,
                "score": float(scores[idx].item()),
                "box_xyxy_pixels": [float(x) for x in box_pixels],
                "box_xyxy_normalized": box_norm,
            }
        )

    if not result:
        extra["status"] = "no_active_detections"
    else:
        extra["status"] = "ok"

    return result, extra


def run(args: argparse.Namespace) -> dict:
    vocabulary, allowed = _build_detector_vocabulary(
        base_phrases=args.base_phrase,
        goal_text=args.goal,
        max_vocab=args.max_vocab,
    )

    predictor = DeticPredictor(
        vocabulary=vocabulary,
        prompt=args.prompt,
        config_file=args.config_file,
        model_weights_file=args.weights_file,
        min_size_test=args.min_size_test,
        max_size_test=args.max_size_test,
        confidence_threshold=args.confidence_threshold,
        device=args.device,
    )
    predictor.to(torch.device(args.device))

    image_tensor, (height, width) = _load_image_tensor(args.image)
    detections, extra = _filter_detections(
        predictor=predictor,
        image_tensor=image_tensor,
        vocabulary=vocabulary,
        allowed=allowed,
        max_detections=args.max_detections,
    )

    report = {
        "image_path": os.path.abspath(args.image),
        "image_size": {
            "height": int(height),
            "width": int(width),
        },
        "goal": args.goal,
        "detic": {
            "vocabulary": extra.get("vocabulary", []),
            "allowed_phrases": extra.get("allowed_phrases", []),
            "status": extra.get("status", "unknown"),
        },
        "detections": detections,
    }

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Detic on a single image for debugging.")
    parser.add_argument("--image", required=True, help="Path to the RGB image (PNG/JPEG).")
    parser.add_argument("--goal", default="", help="Goal or task description to include in the vocabulary.")
    parser.add_argument(
        "--base-phrase",
        action="append",
        help="Base vocabulary phrase(s). Can be specified multiple times. Defaults to 'object'.",
    )
    parser.add_argument("--max-vocab", type=int, default=128, help="Maximum vocabulary size.")
    parser.add_argument("--max-detections", type=int, default=10, help="Maximum detections to keep.")
    parser.add_argument("--prompt", default="a ", help="Text prompt prefix used by Detic.")
    parser.add_argument(
        "--config-file",
        default="Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml",
        help="Detic config file name or path.",
    )
    parser.add_argument(
        "--weights-file",
        default="Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth",
        help="Detic model weights file name or path.",
    )
    parser.add_argument("--min-size-test", type=int, default=None, help="Optional min size for resizing.")
    parser.add_argument("--max-size-test", type=int, default=None, help="Optional max size for resizing.")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.30,
        help="Score threshold for filtering Detic predictions.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for Detic (e.g., 'cuda', 'cuda:0', or 'cpu').",
    )
    parser.add_argument("--output", help="Optional path to write the JSON report.")

    args = parser.parse_args()
    report = run(args)

    json_str = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(json_str + "\n")
        print(f"Saved Detic report to {os.path.abspath(args.output)}")
    else:
        print(json_str)


if __name__ == "__main__":
    main()
