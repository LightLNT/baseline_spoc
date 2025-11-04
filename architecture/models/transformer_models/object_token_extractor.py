from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn

from architecture.models.transformer_models.image_encoders import Dinov2


@dataclass
class ObjectTokenExtractorConfig:
    max_object_tokens: int = 10
    pooling: str = "attention"
    min_patch_overlap: int = 1
    fallback_to_cls: bool = True
    return_attention: bool = True


@dataclass
class ObjectTokenExtractionOutput:
    cls_tokens: torch.Tensor
    object_tokens: torch.Tensor
    object_mask: torch.Tensor
    boxes: torch.Tensor
    scores: torch.Tensor
    attention_maps: torch.Tensor


class ObjectTokenExtractor(nn.Module):
    def __init__(self, dinov2: Dinov2, cfg: ObjectTokenExtractorConfig):
        super().__init__()
        self.dinov2 = dinov2
        self.cfg = cfg
        self.token_dim = dinov2.cfg.output_size[0]
        self.patch_grid = dinov2.cfg.patch_grid
        self.input_height, self.input_width = dinov2.cfg.input_size
        self.width_crop = dinov2.cfg.width_crop
        self.effective_width = self.input_width - 2 * self.width_crop
        self.patch_height = self.input_height / self.patch_grid[0]
        self.patch_width = self.effective_width / self.patch_grid[1]
        if cfg.pooling == "attention":
            self.attention_fc = nn.Linear(self.token_dim, 1)
        else:
            self.attention_fc = None

    def forward(
        self,
        images: torch.Tensor,
        boxes: Sequence[torch.Tensor],
        scores: Optional[Sequence[torch.Tensor]] = None,
    ) -> ObjectTokenExtractionOutput:
        cls_tokens, patch_tokens = self.dinov2.forward_patch_tokens(images, detach=True)
        if cls_tokens is None:
            raise ValueError("DINOv2 forward_features did not return cls tokens; cannot fallback.")
        batch = patch_tokens.shape[0]
        grid_h, grid_w = self.patch_grid
        device = patch_tokens.device
        cls_tokens = cls_tokens.to(device)
        patch_tokens = patch_tokens.to(device)
        patch_tokens = patch_tokens.reshape(batch, grid_h, grid_w, self.token_dim)

        max_tokens = self.cfg.max_object_tokens
        object_tokens = torch.zeros(batch, max_tokens, self.token_dim, device=device)
        object_mask = torch.zeros(batch, max_tokens, dtype=torch.bool, device=device)
        attention_maps = torch.zeros(batch, max_tokens, grid_h * grid_w, device=device)
        padded_boxes = torch.zeros(batch, max_tokens, 4, device=device)
        padded_scores = torch.zeros(batch, max_tokens, device=device)

        if scores is None:
            scores = [None for _ in range(batch)]

        assert len(boxes) == batch, "Mismatch between images and bounding boxes."  # type: ignore[arg-type]

        for idx in range(batch):
            curr_boxes = boxes[idx]
            curr_scores = scores[idx]
            if curr_boxes is None or curr_boxes.numel() == 0:
                continue
            curr_boxes = curr_boxes.to(device)
            if curr_scores is not None:
                curr_scores = curr_scores.to(device)
            num_boxes = min(curr_boxes.shape[0], max_tokens)
            for box_idx in range(num_boxes):
                box = curr_boxes[box_idx]
                score = curr_scores[box_idx] if curr_scores is not None else torch.tensor(1.0, device=device)
                x0 = box[0] * self.input_width - self.width_crop
                y0 = box[1] * self.input_height
                x1 = box[2] * self.input_width - self.width_crop
                y1 = box[3] * self.input_height
                x0 = torch.clamp(x0, 0, self.effective_width)
                x1 = torch.clamp(x1, 0, self.effective_width)
                y0 = torch.clamp(y0, 0, self.input_height)
                y1 = torch.clamp(y1, 0, self.input_height)

                x0_idx = torch.floor(x0 / self.patch_width).to(torch.int64)
                y0_idx = torch.floor(y0 / self.patch_height).to(torch.int64)
                x1_idx = torch.ceil(x1 / self.patch_width).to(torch.int64)
                y1_idx = torch.ceil(y1 / self.patch_height).to(torch.int64)

                x0_idx = torch.clamp(x0_idx, 0, grid_w - 1)
                y0_idx = torch.clamp(y0_idx, 0, grid_h - 1)
                x1_idx = torch.clamp(x1_idx, x0_idx + 1, grid_w)
                y1_idx = torch.clamp(y1_idx, y0_idx + 1, grid_h)

                selected = patch_tokens[idx, y0_idx:y1_idx, x0_idx:x1_idx, :]
                if selected.numel() == 0:
                    pooled = cls_tokens[idx]
                    weights = torch.zeros(1, device=device)
                else:
                    flat = selected.reshape(-1, self.token_dim)
                    if flat.shape[0] < self.cfg.min_patch_overlap:
                        continue
                    if self.cfg.pooling == "attention" and self.attention_fc is not None:
                        logits = self.attention_fc(flat)
                        weights = torch.softmax(logits.squeeze(-1), dim=0)
                        pooled = torch.sum(weights.unsqueeze(-1) * flat, dim=0)
                    else:
                        weights = torch.full((flat.shape[0],), 1.0 / flat.shape[0], device=device)
                        pooled = flat.mean(dim=0)

                padded_boxes[idx, box_idx] = box
                padded_scores[idx, box_idx] = score
                object_tokens[idx, box_idx] = pooled
                object_mask[idx, box_idx] = True

                if selected.numel() != 0 and self.cfg.return_attention:
                    yy = torch.arange(y0_idx, y1_idx, device=device)
                    xx = torch.arange(x0_idx, x1_idx, device=device)
                    mesh_y, mesh_x = torch.meshgrid(yy, xx, indexing="ij")
                    flat_indices = (mesh_y * grid_w + mesh_x).reshape(-1)
                    attention_maps[idx, box_idx, flat_indices] = weights

            if not object_mask[idx].any() and self.cfg.fallback_to_cls:
                object_tokens[idx, 0] = cls_tokens[idx]
                object_mask[idx, 0] = True
                padded_scores[idx, 0] = 1.0

        return ObjectTokenExtractionOutput(
            cls_tokens=cls_tokens,
            object_tokens=object_tokens,
            object_mask=object_mask,
            boxes=padded_boxes,
            scores=padded_scores,
            attention_maps=attention_maps,
        )
