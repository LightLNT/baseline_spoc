from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


@dataclass
class GroundingDINOResult:
    boxes: torch.Tensor
    scores: torch.Tensor
    labels: List[str]
    boxes_absolute: torch.Tensor
    image_size: Tuple[int, int]

    def to(self, device: torch.device) -> "GroundingDINOResult":
        return GroundingDINOResult(
            boxes=self.boxes.to(device),
            scores=self.scores.to(device),
            labels=list(self.labels),
            boxes_absolute=self.boxes_absolute.to(device),
            image_size=self.image_size,
        )


class GroundingDINOPredictor:
    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-tiny",
        device: Union[str, torch.device] = "cuda",
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        max_detections: Optional[int] = None,
    ):
        if isinstance(device, str):
            if "cuda" in device and not torch.cuda.is_available():
                print("GroundingDINOPredictor: CUDA requested but unavailable, falling back to CPU.")
                device = "cpu"
            self.device = torch.device(device)
        else:
            self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
        self.model.eval()
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.max_detections = max_detections

    def to(self, device: Union[str, torch.device]) -> "GroundingDINOPredictor":
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.model.to(self.device)
        return self

    def __call__(
        self,
        images: Sequence[Union[Image.Image, np.ndarray, torch.Tensor]],
        text_queries: Union[str, Sequence[str]],
        box_threshold: Optional[float] = None,
        text_threshold: Optional[float] = None,
        max_detections: Optional[int] = None,
    ) -> List[GroundingDINOResult]:
        # Support either a single prompt (str), a sequence of phrases (old behavior),
        # or a per-image list of prompts (len == len(images)).
        per_image_prompts = None
        global_prompt = None
        global_phrases = None
        if isinstance(text_queries, str):
            global_phrases = [phrase.strip() for phrase in text_queries.split(".") if phrase.strip()]
            if len(global_phrases) == 0:
                raise ValueError("text_queries must provide at least one phrase for detection.")
            global_prompt = " . ".join(global_phrases) + "."
        else:
            # try sequence
            text_list = list(text_queries)
            if len(text_list) == len(images):
                # per-image prompts
                per_image_prompts = [str(tq) for tq in text_list]
            else:
                # combine sequence into global phrase list (old behavior)
                global_phrases = [str(phrase).strip() for phrase in text_list if str(phrase).strip()]
                if len(global_phrases) == 0:
                    raise ValueError("text_queries must provide at least one phrase for detection.")
                global_prompt = " . ".join(global_phrases) + "."

        box_threshold = self.box_threshold if box_threshold is None else box_threshold
        text_threshold = self.text_threshold if text_threshold is None else text_threshold
        max_detections = self.max_detections if max_detections is None else max_detections

        results: List[GroundingDINOResult] = []
        # Precompute per-image phrase lists if needed
        per_image_phrases = None
        if per_image_prompts is not None:
            per_image_phrases = [
                [phrase.strip() for phrase in p.split(".") if phrase.strip()] if isinstance(p, str) else [str(p)]
                for p in per_image_prompts
            ]

        for idx, image in enumerate(images):
            pil_image = self._to_pil(image)
            if per_image_prompts is not None:
                prompt = per_image_prompts[idx]
                phrases = per_image_phrases[idx]
            else:
                prompt = global_prompt
                phrases = global_phrases

            inputs = self.processor(images=pil_image, text=prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            target_size = torch.tensor([pil_image.size[::-1]], device=self.device)
            processed = self.processor.post_process_grounded_object_detection(
                outputs,
                threshold=box_threshold,
                target_sizes=target_size,
            )[0]
            boxes_absolute = processed["boxes"].detach().cpu()
            scores = processed["scores"].detach().cpu()
            # 兼容新版 transformers
            labels_field = processed.get("text_labels", None)
            if labels_field is not None:
                # 新版，直接是字符串列表
                labels = labels_field
                labels_idx = None
            else:
                # 旧版，labels 是 tensor 或 list
                labels_idx = processed["labels"]
                if isinstance(labels_idx, torch.Tensor):
                    labels_idx = labels_idx.detach().cpu().to(torch.int64)
                else:
                    labels_idx = torch.tensor(labels_idx, dtype=torch.int64)
                # ensure phrases is a list to avoid None indexing
                if phrases is None:
                    phrases = []
                labels = [phrases[idx] if 0 <= idx < len(phrases) else "" for idx in labels_idx.tolist()]

            if scores.numel() == 0:
                results.append(
                    GroundingDINOResult(
                        boxes=torch.zeros((0, 4), dtype=torch.float32),
                        scores=torch.zeros(0, dtype=torch.float32),
                        labels=[],
                        boxes_absolute=torch.zeros((0, 4), dtype=torch.float32),
                        image_size=tuple(pil_image.size[::-1]),
                    )
                )
                continue

            keep = scores >= text_threshold
            boxes_absolute = boxes_absolute[keep]
            scores = scores[keep]
            # labels_idx = labels_idx[keep]
            if labels_idx is not None:
                labels_idx = labels_idx[keep]

            if scores.numel() == 0:
                results.append(
                    GroundingDINOResult(
                        boxes=torch.zeros((0, 4), dtype=torch.float32),
                        scores=torch.zeros(0, dtype=torch.float32),
                        labels=[],
                        boxes_absolute=torch.zeros((0, 4), dtype=torch.float32),
                        image_size=tuple(pil_image.size[::-1]),
                    )
                )
                continue

            if max_detections is not None and scores.numel() > max_detections:
                topk_scores, topk_idx = torch.topk(scores, max_detections)
                boxes_absolute = boxes_absolute[topk_idx]
                if labels_idx is not None:
                    labels_idx = labels_idx[topk_idx]
                scores = topk_scores

            if labels_idx is not None:
                if phrases is None:
                    phrases = []
                labels = [phrases[idx] if 0 <= idx < len(phrases) else "" for idx in labels_idx.tolist()]
            boxes_normalized = self._normalize_boxes(boxes_absolute, tuple(pil_image.size[::-1]))
            results.append(
                GroundingDINOResult(
                    boxes=boxes_normalized,
                    scores=scores,
                    labels=labels,
                    boxes_absolute=boxes_absolute,
                    image_size=tuple(pil_image.size[::-1]),
                )
            )
        return results

    @staticmethod
    def _normalize_boxes(boxes: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
        height, width = hw
        scale = torch.tensor([1.0 / width, 1.0 / height, 1.0 / width, 1.0 / height], dtype=boxes.dtype)
        return boxes * scale

    @staticmethod
    def _to_pil(image: Union[Image.Image, np.ndarray, torch.Tensor]) -> Image.Image:
        if isinstance(image, Image.Image):
            return image
        if torch.is_tensor(image):
            img = image.detach().cpu()
            if img.ndim == 3 and img.shape[0] in (1, 3):
                img = img.permute(1, 2, 0)
            img = img.numpy()
        else:
            img = np.asarray(image)
        if img.dtype != np.uint8:
            img = np.clip(img, 0.0, 1.0)
            img = (img * 255).astype(np.uint8)
        return Image.fromarray(img)
