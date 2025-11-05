import os
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from utils.visualization_utils import (
    draw_grounding_dino_detections,
    overlay_object_token_attention,
    shade_object_region,
)


def _to_uint8_image(
    frame: torch.Tensor,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Convert a CHW torch tensor in [0,1] (or normalized) to an uint8 HWC image."""

    img = frame.detach().cpu().clone()
    if mean is not None and std is not None:
        mean_t = torch.tensor(mean, dtype=img.dtype, device=img.device).view(-1, 1, 1)
        std_t = torch.tensor(std, dtype=img.dtype, device=img.device).view(-1, 1, 1)
        img = img * std_t + mean_t
    img = img.clamp(0.0, 1.0)
    img = (img * 255.0).permute(1, 2, 0).to(torch.uint8).numpy()
    return img


def render_object_token_batch(
    frames: Dict[str, torch.Tensor],
    latest_object_data: Dict[str, Dict[str, object]],
    output_dir: str,
    patch_grid: Tuple[int, int],
    *,
    prefix: str = "batch",
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
    max_batches: int = 2,
    max_timesteps: int = 4,
    overlay_attention: bool = True,
) -> None:
    """Render debug visualizations for object tokens and save them as PNGs.

    Args:
        frames: mapping of sensor name -> tensor shaped [B, T, C, H, W].
        latest_object_data: cached metadata from ObjectTokenVisualEncoder.latest_object_data.
        output_dir: directory to store rendered images.
        patch_grid: (height, width) of the visual backbone's patch layout.
        prefix: filename prefix for saved images.
        mean/std: optional normalization stats used to undo preprocessing.
        max_batches: limit on batch samples to visualize.
        max_timesteps: limit on temporal steps per sample.
        overlay_attention: whether to overlay the strongest object's attention map.
    """

    os.makedirs(output_dir, exist_ok=True)
    grid_h, grid_w = patch_grid

    for sensor, sensor_frames in frames.items():
        if sensor not in latest_object_data:
            continue

        sensor_data = latest_object_data[sensor]
        sensor_dir = os.path.join(output_dir, sensor)
        os.makedirs(sensor_dir, exist_ok=True)

        frames_cpu = sensor_frames.detach().cpu()
        B, T = frames_cpu.shape[:2]

        for b in range(min(B, max_batches)):
            for t in range(min(T, max_timesteps)):
                frame_rgb = _to_uint8_image(frames_cpu[b, t], mean=mean, std=std)
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                boxes = sensor_data.get("boxes")
                scores = sensor_data.get("scores")
                object_mask = sensor_data.get("object_mask")
                attention = sensor_data.get("attention")
                labels_nested = sensor_data.get("labels")

                if boxes is None or scores is None or object_mask is None:
                    annotated = frame_bgr
                else:
                    boxes_np = boxes[b, t].detach().cpu().numpy()
                    scores_np = scores[b, t].detach().cpu().numpy() if scores is not None else None
                    mask_np = object_mask[b, t].detach().cpu().numpy().astype(bool)

                    active_indices = [idx for idx, flag in enumerate(mask_np.tolist()) if flag]
                    if labels_nested is not None:
                        labels_sample = labels_nested[b][t]
                        object_labels = [labels_sample[idx + 1] for idx in active_indices]
                    else:
                        object_labels = []

                    object_boxes = boxes_np[mask_np]
                    object_scores = scores_np[mask_np] if scores_np is not None else None

                    annotated = draw_grounding_dino_detections(
                        frame_bgr.copy(),
                        object_boxes,
                        labels=object_labels,
                        scores=object_scores,
                        inplace=True,
                    )

                    if len(active_indices) > 0:
                        top_idx_rel = 0
                        if object_scores is not None and object_scores.size > 0:
                            top_idx_rel = int(np.argmax(object_scores))
                        top_idx = active_indices[top_idx_rel]
                        annotated = shade_object_region(
                            annotated,
                            boxes_np[top_idx],
                            color=(255, 255, 0),
                            alpha=0.25,
                            inplace=True,
                        )
                        if overlay_attention and attention is not None:
                            attn_flat = attention[b, t, top_idx].detach().cpu().numpy()
                            annotated = overlay_object_token_attention(
                                annotated,
                                attn_flat,
                                patch_grid=(grid_h, grid_w),
                                alpha=0.4,
                                inplace=True,
                            )

                filename = f"{prefix}_{sensor}_b{b}_t{t}.png"
                output_path = os.path.join(sensor_dir, filename)
                cv2.imwrite(output_path, annotated)