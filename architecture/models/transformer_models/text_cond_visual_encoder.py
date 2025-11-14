from __future__ import annotations

import math
import warnings

# from utils.transformation_util import get_full_transformation_list, sample_a_specific_transform
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from open_clip import create_model_from_pretrained
from open_clip.transformer import TextTransformer
from transformers import T5EncoderModel

from architecture.models.transformer_models.image_encoders import IMAGE_ENCODERS
from architecture.models.transformer_models.object_token_extractor import (
    ObjectTokenExtractionOutput,
    ObjectTokenExtractor,
    ObjectTokenExtractorConfig,
)
from utils.bbox_utils import get_best_of_two_bboxes
from utils.detic_utils import DeticPredictor
from utils.sensor_constant_utils import is_a_visual_sensor
from transformers import AutoTokenizer


@dataclass
class TransformerConfig:
    num_layers: int = 3
    d_model: int = 512
    nhead: int = 8


TEXT_ENCODER_DIMS = {
    "t5-small": 512,
    "t5-base": 768,
    "t5-large": 1024,
    "SigLIPBase": 768,
    "SigLIPLarge": 1024,
}


def create_text_encoder(encoder_name):
    if "siglip" in encoder_name.lower():
        _, cfg = IMAGE_ENCODERS[encoder_name]
        encoder = create_model_from_pretrained(f"hf-hub:timm/{cfg.model}")[0].text
        encoder.output_tokens = True
        return encoder
    elif "t5" in encoder_name.lower():
        return T5EncoderModel.from_pretrained(encoder_name)
    else:
        raise NotImplementedError("Only SigLIP and T5 text encoders are supported.")


@dataclass
class TextCondVisualEncoderConfig:
    image_encoder: str = "Dinov2Small"
    text_encoder: str = "t5-small"
    fusion_xformer: TransformerConfig = field(default_factory=lambda: TransformerConfig(3,512,8))
    input_sensors: List[str] = None
    bbox_encoding_type: Literal["positional"] = "positional"


@dataclass
class ObjectTokenVisualEncoderConfig(TextCondVisualEncoderConfig):
    max_object_tokens: int = 10
    detector_config_file: str = "~/detic/Detic/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml"
    detector_weights_file: str = "~/detic/Detic/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"
    detector_prompt: str = "a "
    detector_device: Union[str, Sequence[str]] = "cuda"
    detector_confidence_threshold: Optional[float] = None
    detector_min_size_test: Optional[int] = 512#None
    detector_max_size_test: Optional[int] = 640#None
    detector_phrases: Sequence[str] = field(default_factory=lambda: ["object"])
    max_detector_vocabulary: int = 32 #128
    pooling: Literal["attention", "mean"] = "mean"
    min_patch_overlap: int = 1
    use_detector: bool = True
    detector_usage: Literal["always", "eval_only", "train_only", "never"] = "always"
    ground_truth_sensor_map: Dict[str, Sequence[str]] = field(default_factory=dict)
    bbox_rotary_dim: int = 0
    # legacy fields kept for backward compatibility and gracefully ignored when Detic is used
    detector_model_id: Optional[str] = None
    detector_box_threshold: float = 0.25
    detector_text_threshold: float = 0.25


class TextCondMultiCameraVisualEncoder(nn.Module):
    def __init__(self, cfg: TextCondVisualEncoderConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.image_encoder == "dinov2" and cfg.image_encoder not in IMAGE_ENCODERS:
            cfg.image_encoder = "Dinov2Small"
            print("REAPLACING DINOV2 WITH DINOV2SMALL")

        if cfg.image_encoder in IMAGE_ENCODERS:
            image_encoder_model_cls, image_encoder_cfg = IMAGE_ENCODERS[cfg.image_encoder]
            self.image_encoder = image_encoder_model_cls(image_encoder_cfg)
        else:
            raise NotImplementedError()

        self.visual_compressor = self.create_compressor()
        self.visual_adapter = nn.Sequential(
            nn.Linear(self.cfg.fusion_xformer.d_model, self.cfg.fusion_xformer.d_model),
            nn.LayerNorm(self.cfg.fusion_xformer.d_model),
            nn.ReLU(),
        )

        self.text_encoder = create_text_encoder(cfg.text_encoder)

        self.text_encoder.eval()

        self.text_adapter = nn.Sequential(
            nn.Linear(TEXT_ENCODER_DIMS[cfg.text_encoder], self.cfg.fusion_xformer.d_model),
            nn.LayerNorm(self.cfg.fusion_xformer.d_model),
            nn.ReLU(),
        )
        self.fusion_xformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=cfg.fusion_xformer.d_model, nhead=cfg.fusion_xformer.nhead, batch_first=True
            ),
            num_layers=cfg.fusion_xformer.num_layers,
        )
        self.fusion_token = nn.Parameter(0.1 * torch.rand(cfg.fusion_xformer.d_model))
        self.visual_sensors = [sensor for sensor in cfg.input_sensors if is_a_visual_sensor(sensor)]
        # KE: This is absolutely important! # KE2: Actually not so much anymore lol
        self.visual_sensors = sorted(self.visual_sensors)
        for sensor in self.visual_sensors:
            setattr(
                self,
                f"visual_sensor_token_{sensor}",
                nn.Parameter(0.1 * torch.rand(cfg.fusion_xformer.d_model)),
            )

    def encode_text(self, preproc_text_input):
        with torch.no_grad():
            if isinstance(self.text_encoder, TextTransformer):
                cls_feats, text_feats = self.text_encoder(preproc_text_input)
                text_feats = torch.cat([text_feats, cls_feats.unsqueeze(1)], dim=1)
            else:
                text_feats = self.text_encoder(**preproc_text_input).last_hidden_state

        return self.text_adapter(text_feats)

    def encode_imgs(self, imgs):
        B, T, C, H, W = imgs.shape
        feats = self.visual_compressor(
            self.image_encoder(imgs.reshape(B * T, C, H, W))
        )  # BTxC_xH_xW_
        _, C_, H_, W_ = feats.shape
        feats = feats.reshape(B * T, C_, H_ * W_).permute(0, 2, 1)  # BTxH_W_xC_
        return self.visual_adapter(feats)

    def create_compressor(self):
        return nn.Sequential(
            nn.Conv2d(self.image_encoder.cfg.output_size[0], self.cfg.fusion_xformer.d_model, 1),
            nn.ReLU(),
            nn.Conv2d(self.cfg.fusion_xformer.d_model, self.cfg.fusion_xformer.d_model, 1),
            nn.ReLU(),
        )

    def get_image_text_feats(self, frames, goals, text_feats):
        all_img_features = {}
        images_chw = None
        for sensor in frames.keys():
            assert is_a_visual_sensor(sensor)
            imgs = frames[sensor]
            B, T, C, H, W = imgs.shape

            if images_chw is None:
                images_chw = (C, H, W)

            assert images_chw == (C, H, W)

            image_feats = self.encode_imgs(imgs)  # BTxHWxD
            all_img_features[sensor] = image_feats

        concatenated_feats = []

        for k in self.visual_sensors:
            corresponding_camera_token = getattr(self, f"visual_sensor_token_{k}")
            concatenated_feats.append(all_img_features[k] + corresponding_camera_token)

        concatenated_feats = torch.cat(concatenated_feats, dim=1)

        if text_feats is None:
            text_feats = self.encode_text(goals)  # BxLxD
        B, L, D = text_feats.shape
        text_feats_ = text_feats.unsqueeze(1).tile(1, T, 1, 1).reshape(B * T, L, D)
        fusion_token = self.fusion_token.reshape(1, 1, D).tile(B * T, 1, 1)
        return fusion_token, concatenated_feats, text_feats, text_feats_, B, T, D

    def forward(
        self,
        frames,
        goals,
        text_feats=None,
        non_visual_sensors=None,
    ):
        # KE: I checked the gradients on all the values.
        (
            fusion_token,
            concatenated_feats,
            text_feats,
            text_feats_,
            B,
            T,
            D,
        ) = self.get_image_text_feats(frames, goals, text_feats)
        input_features = [fusion_token, concatenated_feats, text_feats_]

        fused_feats = self.fusion_xformer(torch.cat(input_features, 1))

        fused_feats = fused_feats[:, 0, :]  # BTxD

        return fused_feats.reshape(B, T, D), text_feats


class TextCondMultiCameraVisualEncoderWDoubleDet(TextCondMultiCameraVisualEncoder):
    def __init__(self, cfg: TextCondVisualEncoderConfig):
        super().__init__(cfg)
        assert "manip_task_relevant_object_bbox" in cfg.input_sensors
        assert "nav_task_relevant_object_bbox" in cfg.input_sensors
        assert "nav_accurate_object_bbox" in cfg.input_sensors
        assert "manip_accurate_object_bbox" in cfg.input_sensors
        num_boxes = 2
        num_cameras = 2
        self.len_bounding_boxes = num_boxes * 5 * num_cameras
        self.bbox_pos_encoder = nn.Sequential(
            PositionalEncoder(32),
            nn.Linear(32, self.cfg.fusion_xformer.d_model),
            nn.LayerNorm(self.cfg.fusion_xformer.d_model),
            nn.ReLU(),
        )
        self.coord_pos_enc = nn.Embedding(self.len_bounding_boxes, self.cfg.fusion_xformer.d_model)

        # self.manip_coord_pos_enc = nn.Embedding(
        #     self.len_bounding_boxes, self.cfg.fusion_xformer.d_model
        # )

    def forward(
        self,
        frames,
        goals,
        text_feats=None,
        non_visual_sensors=None,
    ):
        # KE: I checked the gradients on all the values.
        (
            fusion_token,
            concatenated_feats,
            text_feats,
            text_feats_,
            B,
            T,
            D,
        ) = self.get_image_text_feats(frames, goals, text_feats)
        input_features = [fusion_token, concatenated_feats, text_feats_]

        task_relevant_object_bbox = non_visual_sensors["nav_task_relevant_object_bbox"]
        manip_task_relevant_object_bbox = non_visual_sensors["manip_task_relevant_object_bbox"]
        nav_accurate_object_bbox = non_visual_sensors["nav_accurate_object_bbox"]
        manip_accurate_object_bbox = non_visual_sensors["manip_accurate_object_bbox"]

        best_nav_boxes = get_best_of_two_bboxes(task_relevant_object_bbox, nav_accurate_object_bbox)
        best_manip_boxes = get_best_of_two_bboxes(
            manip_task_relevant_object_bbox, manip_accurate_object_bbox
        )

        combined_boxes = torch.concat([best_nav_boxes, best_manip_boxes], dim=2)
        B, T, N = combined_boxes.shape
        combined_boxes = combined_boxes.reshape(B * T, N)
        pos_encoded_boxes = self.bbox_pos_encoder(combined_boxes)
        pos_encoded_boxes = pos_encoded_boxes + self.coord_pos_enc(
            torch.tensor(
                [[i for i in range((self.len_bounding_boxes))]],
                device=pos_encoded_boxes.device,
            ).tile(B * T, 1)
        )
        input_features.append(pos_encoded_boxes)

        fused_feats = self.fusion_xformer(torch.cat(input_features, 1))

        fused_feats = fused_feats[:, 0, :]  # BTxD

        return fused_feats.reshape(B, T, D), text_feats


class ObjectTokenVisualEncoder(TextCondMultiCameraVisualEncoder):
    def __init__(self, cfg: ObjectTokenVisualEncoderConfig):
        super().__init__(cfg)
        if not hasattr(self.image_encoder, "forward_patch_tokens"):
            raise ValueError(
                "ObjectTokenVisualEncoder requires image encoders exposing forward_patch_tokens."
            )
        self.cfg: ObjectTokenVisualEncoderConfig = cfg
        extractor_cfg = ObjectTokenExtractorConfig(
            max_object_tokens=cfg.max_object_tokens,
            pooling=cfg.pooling,
            min_patch_overlap=cfg.min_patch_overlap,
        )
        self.object_token_extractor = ObjectTokenExtractor(self.image_encoder, extractor_cfg)
        self.visual_adapter = nn.Sequential(
            nn.LayerNorm(self.image_encoder.cfg.output_size[0]),
            nn.Linear(self.image_encoder.cfg.output_size[0], self.cfg.fusion_xformer.d_model),
            nn.ReLU(),
        )
        rotary_dim = getattr(cfg, "bbox_rotary_dim", 0) or 0
        self.bbox_rotary_embed = (
            RotaryBBoxEncoding(self.cfg.fusion_xformer.d_model, rotary_dim)
            if rotary_dim > 0
            else None
        )
        self.detector: Optional[DeticPredictor] = None
        self.detector_device = torch.device("cpu")
        self.detector_pool: List[Tuple[DeticPredictor, torch.device]] = []
        self.detector_devices: List[torch.device] = []
        self._detector_usage = getattr(cfg, "detector_usage", "always")
        self._ground_truth_sensor_map = dict(getattr(cfg, "ground_truth_sensor_map", {}) or {})
        if not self._ground_truth_sensor_map:
            self._ground_truth_sensor_map = {
                "raw_navigation_camera": (
                    "nav_task_relevant_object_bbox",
                    "nav_accurate_object_bbox",
                ),
                "raw_manipulation_camera": (
                    "manip_task_relevant_object_bbox",
                    "manip_accurate_object_bbox",
                ),
            }
        self._detector_debug_logged: Dict[str, bool] = {}
        self._missing_gt_warned: Set[str] = set()
        self._goal_tokenizer = None
        if cfg.use_detector:
            if cfg.detector_model_id:
                warnings.warn(
                    "detector_model_id is deprecated; Detic uses detector_config_file/detector_weights_file.",
                    RuntimeWarning,
                )
            resolved_devices = self._resolve_detector_devices(cfg.detector_device)
            if not resolved_devices:
                resolved_devices = [torch.device("cpu")]
            self.detector_devices = resolved_devices
            detector_device = resolved_devices[0]
            confidence_threshold = (
                cfg.detector_confidence_threshold
                if cfg.detector_confidence_threshold is not None
                else cfg.detector_box_threshold
            )
            initial_vocabulary = [
                str(phrase) for phrase in self._normalize_detector_phrases(cfg.detector_phrases)
            ]
            if not initial_vocabulary:
                initial_vocabulary = ["object"]
            for device in resolved_devices:
                try:
                    predictor = DeticPredictor(
                        vocabulary=initial_vocabulary,
                        prompt=cfg.detector_prompt,
                        config_file=cfg.detector_config_file,
                        model_weights_file=cfg.detector_weights_file,
                        min_size_test=cfg.detector_min_size_test,
                        max_size_test=cfg.detector_max_size_test,
                        confidence_threshold=confidence_threshold,
                        device=str(device),
                    )
                    predictor.to(device)
                    self.detector_pool.append((predictor, device))
                except Exception as exc:
                    vocab_sample = initial_vocabulary[:5]
                    vocab_types = [type(item).__name__ for item in vocab_sample]
                    vocab_values = [repr(item) for item in vocab_sample]
                    import traceback

                    tb = traceback.format_exc()
                    warnings.warn(
                        (
                            f"Failed to initialize DeticPredictor on {device} ({type(exc).__name__}: {exc}); "
                            f"initial_vocabulary_types={vocab_types}, initial_vocabulary_values={vocab_values}. "
                            f"traceback={tb.strip()} "
                            "Continuing without detector on this device."
                        ),
                        RuntimeWarning,
                    )
            if self.detector_pool:
                self.detector_devices = [dev for _, dev in self.detector_pool]
                self.detector = self.detector_pool[0][0]
                self.detector_device = self.detector_pool[0][1]
            else:
                self.detector_devices = []
                self.detector = None
                self.detector_device = torch.device("cpu")

            if self.detector is not None:
                try:
                    self._goal_tokenizer = AutoTokenizer.from_pretrained(cfg.text_encoder)
                except Exception:
                    self._goal_tokenizer = None
        self.latest_object_data: Dict[str, Dict[str, object]] = {}

    def _detect_boxes(
        self,
        images: torch.Tensor,
        B: Optional[int] = None,
        T: Optional[int] = None,
        goals: Optional[dict] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[str]], Dict[str, Any]]:
        batch = images.shape[0]
        device = images.device
        if not self.detector_pool:
            zeros_boxes = [torch.zeros((0, 4), dtype=torch.float32, device=device) for _ in range(batch)]
            zeros_scores = [torch.zeros(0, dtype=torch.float32, device=device) for _ in range(batch)]
            empty_labels = [[] for _ in range(batch)]
            return zeros_boxes, zeros_scores, empty_labels, {
                "vocabulary": [],
                "per_image_allowed": [],
            }

        goal_texts = self._decode_goal_strings(goals, B, T)
        vocabulary, per_image_allowed = self._build_detector_vocabulary(goal_texts, batch)
        self._update_detector_vocabulary(vocabulary)

        detections: List[Any] = []
        total_devices = len(self.detector_pool)
        base_chunk = batch // total_devices
        remainder = batch % total_devices
        start = 0
        chunk_plan: List[Dict[str, Any]] = []
        for idx, (predictor, det_device) in enumerate(self.detector_pool):
            chunk = base_chunk + (1 if idx < remainder else 0)
            if chunk <= 0 or start >= batch:
                continue
            end = min(batch, start + chunk)
            current_inputs = images[start:end].detach().to(det_device, dtype=torch.float32)
            try:
                current_detections = predictor(current_inputs)
            except Exception as exc:
                warnings.warn(
                    f"Detic inference failed on device {det_device} ({exc}); returning empty detections for assigned batch.",
                    RuntimeWarning,
                )
                current_detections = [None] * (end - start)
            if not isinstance(current_detections, list):
                current_detections = [current_detections]
            detections.extend(current_detections)
            chunk_plan.append(
                {
                    "device": str(det_device),
                    "start": int(start),
                    "end": int(end),
                    "batch": int(end - start),
                }
            )
            start = end

        if len(detections) < batch:
            missing = batch - len(detections)
            detections.extend([None] * missing)

        boxes_list: List[torch.Tensor] = []
        scores_list: List[torch.Tensor] = []
        labels_list: List[List[str]] = []
        vocabulary_reference = self.detector.vocabulary if self.detector is not None else []
        vocabulary_set = set(vocabulary_reference)
        for idx, det in enumerate(detections):
            instances = det.get("instances", None) if isinstance(det, dict) else None
            if instances is None or len(instances) == 0:
                boxes_list.append(torch.zeros((0, 4), dtype=torch.float32, device=device))
                scores_list.append(torch.zeros(0, dtype=torch.float32, device=device))
                labels_list.append(["" for _ in range(self.cfg.max_object_tokens)])
                continue

            instances = instances.to("cpu")
            boxes = instances.pred_boxes.tensor.detach().clone()
            scores = instances.scores.detach().clone()
            classes = instances.pred_classes.detach().clone().to(torch.int64)
            labels = [self.detector.vocabulary[int(cls)] for cls in classes.tolist() if 0 <= int(cls) < len(self.detector.vocabulary)]

            allowed_labels = per_image_allowed[idx] if idx < len(per_image_allowed) else vocabulary_set
            filtered_indices: List[int] = [i for i, label in enumerate(labels) if label in allowed_labels]
            if filtered_indices:
                boxes = boxes[filtered_indices]
                scores = scores[filtered_indices]
                labels = [labels[i] for i in filtered_indices]
            else:
                boxes = boxes[:0]
                scores = scores[:0]
                labels = []

            if boxes.numel() > 0:
                image_height, image_width = instances.image_size
                scale = torch.tensor(
                    [1.0 / float(image_width), 1.0 / float(image_height), 1.0 / float(image_width), 1.0 / float(image_height)],
                    dtype=torch.float32,
                )
                boxes = boxes * scale
                boxes = boxes.clamp(0.0, 1.0)

            if boxes.shape[0] > self.cfg.max_object_tokens:
                topk_scores, topk_idx = torch.topk(scores, self.cfg.max_object_tokens)
                boxes = boxes[topk_idx]
                scores = topk_scores
                labels = [labels[i] for i in topk_idx.tolist()]

            boxes_list.append(boxes.to(device=device, dtype=torch.float32))
            scores_list.append(scores.to(device=device, dtype=torch.float32))
            padded_labels = labels + ["" for _ in range(self.cfg.max_object_tokens - len(labels))]
            labels_list.append(padded_labels)

        return boxes_list, scores_list, labels_list, {
            "vocabulary": list(vocabulary_reference),
            "per_image_allowed": per_image_allowed,
            "mode": "detector",
            "stats": {
                "flattened_batch": int(batch),
                "sequence_batch": int(B) if B is not None else None,
                "sequence_length": int(T) if T is not None else None,
                "image_shape": tuple(images.shape[1:]),
                "device_plan": chunk_plan,
            },
        }


    def _should_use_detector(self) -> bool:
        if not self.cfg.use_detector:
            return False
        if not self.detector_pool:
            return False
        mode = getattr(self, "_detector_usage", "always")
        if mode == "always":
            return True
        if mode == "never":
            return False
        if mode == "eval_only":
            return not self.training
        if mode == "train_only":
            return self.training
        return True

    def _ground_truth_sources_for_sensor(self, sensor: str) -> Sequence[str]:
        direct = self._ground_truth_sensor_map.get(sensor)
        if direct:
            return direct
        for key, sources in self._ground_truth_sensor_map.items():
            if key in sensor:
                return sources
        sensor_lower = sensor.lower()
        if "nav" in sensor_lower:
            return (
                "nav_task_relevant_object_bbox",
                "nav_accurate_object_bbox",
            )
        if "manip" in sensor_lower:
            return (
                "manip_task_relevant_object_bbox",
                "manip_accurate_object_bbox",
            )
        return ()

    def _build_ground_truth_detections(
        self,
        sensor: str,
        non_visual_sensors: Optional[Dict[str, torch.Tensor]],
        B: int,
        T: int,
        device: torch.device,
        image_shape: Tuple[int, int],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[str]], Dict[str, Any]]:
        raw_height, raw_width = image_shape
        height = max(float(raw_height), 1.0)
        width = max(float(raw_width), 1.0)
        if non_visual_sensors is None:
            if sensor not in self._missing_gt_warned:
                warnings.warn(
                    f"Ground-truth boxes requested for sensor '{sensor}' but non-visual sensors are missing.",
                    RuntimeWarning,
                )
                self._missing_gt_warned.add(sensor)
            zeros_boxes = [torch.zeros((0, 4), dtype=torch.float32, device=device) for _ in range(B * T)]
            zeros_scores = [torch.zeros(0, dtype=torch.float32, device=device) for _ in range(B * T)]
            empty_labels = [[] for _ in range(B * T)]
            return zeros_boxes, zeros_scores, empty_labels, {
                "vocabulary": [],
                "per_image_allowed": [set() for _ in range(B * T)],
                "mode": "ground_truth",
                "stats": {
                    "flattened_batch": int(B * T),
                    "sequence_batch": int(B),
                    "sequence_length": int(T),
                    "image_shape": (3, int(raw_height), int(raw_width)),
                    "device_plan": [],
                    "source_sensors": [],
                },
            }

        source_names = [s for s in self._ground_truth_sources_for_sensor(sensor) if s in non_visual_sensors]
        if not source_names:
            if sensor not in self._missing_gt_warned:
                warnings.warn(
                    f"No configured ground-truth sensors found for visual sensor '{sensor}'.",
                    RuntimeWarning,
                )
                self._missing_gt_warned.add(sensor)
            zeros_boxes = [torch.zeros((0, 4), dtype=torch.float32, device=device) for _ in range(B * T)]
            zeros_scores = [torch.zeros(0, dtype=torch.float32, device=device) for _ in range(B * T)]
            empty_labels = [[] for _ in range(B * T)]
            return zeros_boxes, zeros_scores, empty_labels, {
                "vocabulary": [],
                "per_image_allowed": [set() for _ in range(B * T)],
                "mode": "ground_truth",
                "stats": {
                    "flattened_batch": int(B * T),
                    "sequence_batch": int(B),
                    "sequence_length": int(T),
                    "image_shape": (3, int(raw_height), int(raw_width)),
                    "device_plan": [],
                    "source_sensors": [],
                },
            }

        tensors: Dict[str, torch.Tensor] = {}
        for name in source_names:
            tensor = non_visual_sensors[name]
            if not torch.is_tensor(tensor):
                tensor = torch.as_tensor(tensor)
            tensors[name] = tensor.to(device=device, dtype=torch.float32, non_blocking=True)

        flattened_batch = B * T
        boxes_list: List[torch.Tensor] = []
        scores_list: List[torch.Tensor] = []
        labels_list: List[List[str]] = []
        per_image_allowed: List[Set[str]] = []
        vocabulary: List[str] = []
        vocabulary_set: Set[str] = set()

        for b in range(B):
            for t in range(T):
                frame_boxes: List[torch.Tensor] = []
                frame_scores: List[torch.Tensor] = []
                frame_labels: List[str] = []
                for source_name, tensor in tensors.items():
                    entry = tensor[b, t]
                    if entry.numel() == 0:
                        continue
                    flat = entry.view(-1)
                    if flat.numel() == 0:
                        continue
                    if torch.all(flat < 0):
                        continue
                    num_boxes = flat.numel() // 5
                    if num_boxes == 0:
                        continue
                    reshaped = flat.reshape(num_boxes, 5)
                    for idx_box in range(num_boxes):
                        coords = reshaped[idx_box, :4]
                        if torch.any(torch.isnan(coords)):
                            continue
                        x1, y1, x2, y2 = [float(val) for val in coords.tolist()]
                        if max(x1, y1, x2, y2) >= 999 or min(x1, y1, x2, y2) < 0:
                            continue
                        if x2 <= x1 or y2 <= y1:
                            continue
                        norm_box = torch.tensor(
                            [x1 / float(width), y1 / float(height), x2 / float(width), y2 / float(height)],
                            dtype=torch.float32,
                            device=device,
                        ).clamp(0.0, 1.0)
                        frame_boxes.append(norm_box)
                        extra = reshaped[idx_box, 4].item()
                        area_score = float(extra)
                        if not math.isfinite(area_score) or area_score <= 0:
                            area_score = float((x2 - x1) * (y2 - y1))
                        frame_scores.append(torch.tensor(area_score, dtype=torch.float32, device=device))
                        label = f"{source_name}_{idx_box}"
                        frame_labels.append(label)
                        if label not in vocabulary_set:
                            vocabulary_set.add(label)
                            vocabulary.append(label)
                if frame_boxes:
                    boxes_tensor = torch.stack(frame_boxes, dim=0)
                    scores_tensor = torch.stack(frame_scores, dim=0)
                else:
                    boxes_tensor = torch.zeros((0, 4), dtype=torch.float32, device=device)
                    scores_tensor = torch.zeros(0, dtype=torch.float32, device=device)
                boxes_list.append(boxes_tensor)
                scores_list.append(scores_tensor)
                labels_list.append(frame_labels)
                per_image_allowed.append(set(frame_labels))

        return boxes_list, scores_list, labels_list, {
            "vocabulary": vocabulary,
            "per_image_allowed": per_image_allowed,
            "mode": "ground_truth",
            "stats": {
                "flattened_batch": int(flattened_batch),
                "sequence_batch": int(B),
                "sequence_length": int(T),
                "image_shape": (3, int(raw_height), int(raw_width)),
                "device_plan": [],
                "source_sensors": list(tensors.keys()),
            },
        }

    def _gather_detections(
        self,
        sensor: str,
        images: torch.Tensor,
        B: int,
        T: int,
        goals: Optional[dict],
        non_visual_sensors: Optional[Dict[str, torch.Tensor]],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[str]], Dict[str, Any]]:
        if self._should_use_detector():
            boxes_list, scores_list, labels_list, detector_meta = self._detect_boxes(images, B=B, T=T, goals=goals)
            if detector_meta is None:
                detector_meta = {}
            else:
                detector_meta = dict(detector_meta)
            detector_meta.setdefault("mode", "detector")
            return boxes_list, scores_list, labels_list, detector_meta

        height = int(images.shape[2]) if images.ndim >= 3 else 0
        width = int(images.shape[3]) if images.ndim >= 4 else 0
        return self._build_ground_truth_detections(
            sensor=sensor,
            non_visual_sensors=non_visual_sensors,
            B=B,
            T=T,
            device=images.device,
            image_shape=(height, width),
        )
    def _decode_goal_strings(
        self,
        goals: Optional[dict],
        B: Optional[int],
        T: Optional[int],
    ) -> Optional[List[str]]:
        if (
            goals is None
            or self._goal_tokenizer is None
            or B is None
            or T is None
            or "input_ids" not in goals
        ):
            return None

        input_ids = goals["input_ids"]
        try:
            decoded = self._goal_tokenizer.batch_decode(
                input_ids.detach().cpu().tolist(), skip_special_tokens=True
            )
        except Exception:
            return None

        per_image_texts: List[str] = []
        for text in decoded:
            cleaned = text.strip() if isinstance(text, str) else ""
            for _ in range(T):
                per_image_texts.append(cleaned)
        return per_image_texts if per_image_texts else None

    def _build_detector_vocabulary(
        self,
        goal_texts: Optional[List[str]],
        batch: int,
    ) -> Tuple[List[str], List[Set[str]]]:
        base_phrases = [str(phrase) for phrase in self._normalize_detector_phrases(self.cfg.detector_phrases)]
        if not base_phrases:
            base_phrases = ["object"]

        cleaned_goal_texts: List[str] = []
        if goal_texts is not None:
            for text in goal_texts:
                cleaned_goal_texts.append(text.strip() if isinstance(text, str) else str(text))
        candidate_phrases = base_phrases + [t for t in cleaned_goal_texts if t]

        unique_vocab: List[str] = []
        seen: Set[str] = set()
        max_vocab = getattr(self.cfg, "max_detector_vocabulary", 128) or 128
        for phrase in candidate_phrases:
            if phrase in seen:
                continue
            phrase_str = str(phrase)
            unique_vocab.append(phrase_str)
            seen.add(phrase_str)
            if len(unique_vocab) >= max_vocab:
                break

        if not unique_vocab:
            unique_vocab = ["object"]
            seen = {"object"}

        per_image_allowed: List[Set[str]] = []
        for idx in range(batch):
            allowed = set(base_phrases)
            if cleaned_goal_texts and idx < len(cleaned_goal_texts):
                goal_phrase = cleaned_goal_texts[idx]
                if goal_phrase:
                    allowed.add(str(goal_phrase))
            allowed = {phrase for phrase in allowed if phrase in seen}
            if not allowed:
                allowed = set(unique_vocab)
            per_image_allowed.append(allowed)

        return unique_vocab, per_image_allowed

    @staticmethod
    def _normalize_detector_phrases(phrases: Optional[Sequence[str]]) -> List[str]:
        if phrases is None:
            return []
        normalized: List[str] = []
        for phrase in phrases:
            cleaned = str(phrase).strip()
            if cleaned:
                normalized.append(cleaned)
        return normalized

    @staticmethod
    def _reshape_bt(tensor: torch.Tensor, B: int, T: int):
        return tensor.reshape(B, T, *tensor.shape[1:])

    def _prepare_sensor_tokens(
        self, extracted: ObjectTokenExtractionOutput
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cls_tokens = extracted.cls_tokens
        object_tokens = extracted.object_tokens
        combined = torch.cat([cls_tokens.unsqueeze(1), object_tokens], dim=1)
        cls_mask = torch.ones(
            (cls_tokens.shape[0], 1), dtype=torch.bool, device=combined.device
        )
        token_mask = torch.cat([cls_mask, extracted.object_mask], dim=1)
        flat = combined.reshape(-1, combined.shape[-1])
        adapted = self.visual_adapter(flat)
        adapted = adapted.reshape(combined.shape[0], combined.shape[1], -1)
        adapted = adapted * token_mask.unsqueeze(-1).float()
        zeros = torch.zeros(
            extracted.boxes.shape[0], 1, extracted.boxes.shape[-1], device=extracted.boxes.device
        )
        token_boxes = torch.cat([zeros, extracted.boxes], dim=1)
        return adapted, token_mask, token_boxes

    def _update_detector_vocabulary(self, vocabulary: Sequence[str]) -> None:
        if not self.detector_pool:
            return
        string_vocab = [str(item) for item in vocabulary]
        for predictor, _ in self.detector_pool:
            try:
                predictor.vocabulary = string_vocab
            except Exception as exc:
                warnings.warn(
                    f"Failed to update Detic vocabulary ({exc}); retaining previous vocabulary on this device.",
                    RuntimeWarning,
                )

    def _resolve_detector_devices(self, requested: Union[str, Sequence[str], torch.device, None]) -> List[torch.device]:
        if isinstance(requested, torch.device):
            specs: List[Union[str, torch.device, None]] = [requested]
        elif isinstance(requested, (list, tuple, set)):
            specs = list(requested)
        elif isinstance(requested, str) and "," in requested:
            specs = [item.strip() for item in requested.split(",") if item.strip()]
        else:
            specs = [requested]

        devices: List[torch.device] = []
        for spec in specs:
            devices.append(self._resolve_detector_device(spec))
        unique_devices: List[torch.device] = []
        for device in devices:
            if device not in unique_devices:
                unique_devices.append(device)
        return unique_devices

    @staticmethod
    def _resolve_detector_device(requested: Optional[Union[str, torch.device]]) -> torch.device:
        if requested is None or requested == "":
            requested = "auto"
        if isinstance(requested, torch.device):
            device = requested
            if device.type == "cuda" and not torch.cuda.is_available():
                warnings.warn(
                    "CUDA requested for Detic but no GPU is available; using CPU instead.",
                    RuntimeWarning,
                )
                return torch.device("cpu")
            return device

        if isinstance(requested, str):
            requested_str = requested.strip()
            if requested_str.isdigit():
                requested_str = f"cuda:{requested_str}"
            elif requested_str.lower().startswith("gpu:") and requested_str[4:].isdigit():
                requested_str = f"cuda:{requested_str[4:]}"
            elif requested_str.lower().startswith("cuda") and requested_str[4:].isdigit():
                requested_str = f"cuda:{requested_str[4:]}"
        else:
            requested_str = str(requested)

        requested_lower = requested_str.lower()
        # If user asked for auto/cuda:auto or simply 'cuda', map to the current CUDA device
        if requested_lower in {"auto", "cuda:auto", "cuda"}:
            if torch.cuda.is_available():
                # Map to the current process's CUDA device index to avoid cross-process device mixups
                try:
                    idx = torch.cuda.current_device()
                    return torch.device(f"cuda:{idx}")
                except Exception:
                    # Fallback to default CUDA device if current_device() fails
                    return torch.device("cuda")
            else:
                return torch.device("cpu")

        # Otherwise try to construct the device directly (e.g. 'cuda:0' or 'cpu')
        try:
            device = torch.device(requested_str)
        except (TypeError, RuntimeError):
            warnings.warn(
                f"Invalid detector device specification '{requested}', falling back to CPU.",
                RuntimeWarning,
            )
            device = torch.device("cpu")

        if device.type == "cuda" and not torch.cuda.is_available():
            warnings.warn(
                "CUDA requested for Detic but no GPU is available; using CPU instead.",
                RuntimeWarning,
            )
            device = torch.device("cpu")

        return device

    def _store_latest(
        self,
        sensor: str,
        extracted: ObjectTokenExtractionOutput,
        token_mask: torch.Tensor,
        labels_list: List[List[str]],
        B: int,
        T: int,
        detector_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        max_tokens = self.cfg.max_object_tokens
        boxes = self._reshape_bt(extracted.boxes, B, T)
        scores = self._reshape_bt(extracted.scores, B, T)
        object_mask = self._reshape_bt(extracted.object_mask, B, T)
        attention = self._reshape_bt(extracted.attention_maps, B, T)
        token_mask_bt = self._reshape_bt(token_mask, B, T)
        cls_tokens = self._reshape_bt(extracted.cls_tokens, B, T)

        labels_nested: List[List[List[str]]] = []
        idx = 0
        for _b in range(B):
            per_batch: List[List[str]] = []
            for _t in range(T):
                per_sample = ["[cls]"]
                per_sample.extend(labels_list[idx])
                valid_mask = token_mask_bt[_b, _t]
                for label_idx, label in enumerate(per_sample):
                    if label == "" and bool(valid_mask[label_idx]):
                        if label_idx == 0:
                            per_sample[label_idx] = "[cls]"
                        else:
                            per_sample[label_idx] = "[token]"
                per_batch.append(per_sample)
                idx += 1
            labels_nested.append(per_batch)

        detector_info: Optional[Dict[str, Any]] = None
        if detector_meta is not None:
            vocabulary = detector_meta.get("vocabulary") or []
            per_image_allowed = detector_meta.get("per_image_allowed") or []
            formatted_allowed: List[List[List[str]]] = []
            flat_idx = 0
            for _b in range(B):
                per_batch: List[List[str]] = []
                for _t in range(T):
                    if flat_idx < len(per_image_allowed):
                        allowed_items = per_image_allowed[flat_idx]
                        if isinstance(allowed_items, (set, list, tuple)):
                            per_batch.append(sorted(list(allowed_items)))
                        else:
                            per_batch.append([str(allowed_items)])
                    else:
                        per_batch.append([])
                    flat_idx += 1
                formatted_allowed.append(per_batch)

            detector_info = {
                "vocabulary": list(vocabulary),
                "per_image_allowed": formatted_allowed,
            }

            mode = detector_meta.get("mode") if isinstance(detector_meta, dict) else None
            if mode:
                detector_info["mode"] = mode

            stats = detector_meta.get("stats") if isinstance(detector_meta, dict) else None
            if stats:
                detector_info["stats"] = stats
                already_logged = self._detector_debug_logged.get(sensor, False)
                if not already_logged:
                    flattened_batch = stats.get("flattened_batch")
                    sequence_batch = stats.get("sequence_batch")
                    sequence_length = stats.get("sequence_length")
                    plan = stats.get("device_plan") or []
                    plan_str = ", ".join(
                        f"{item.get('device')}:batch={item.get('batch')}"
                        for item in plan
                        if item.get("batch")
                    )
                    if not plan_str:
                        plan_str = "none"
                    warnings.warn(
                        (
                            f"Detic input stats for sensor '{sensor}' (mode={mode or 'unknown'}): flattened_batch={flattened_batch}, "
                            f"sequence_batch={sequence_batch}, sequence_length={sequence_length}, "
                            f"image_shape={stats.get('image_shape')}, per_device={plan_str}."
                        ),
                        RuntimeWarning,
                    )
                    self._detector_debug_logged[sensor] = True

        self.latest_object_data[sensor] = {
            "boxes": boxes,
            "scores": scores,
            "object_mask": object_mask,
            "token_mask": token_mask_bt,
            "attention": attention,
            "labels": labels_nested,
            "cls_tokens": cls_tokens,
            "detector_meta": detector_info,
        }

    def get_image_text_feats(self, frames, goals, text_feats, non_visual_sensors=None):
        all_img_features = {}
        images_chw = None
        self.latest_object_data = {}
        for sensor in frames.keys():
            assert is_a_visual_sensor(sensor)
            imgs = frames[sensor]
            B, T, C, H, W = imgs.shape
            if images_chw is None:
                images_chw = (C, H, W)
            assert images_chw == (C, H, W)
            flattened = imgs.reshape(B * T, C, H, W)
            boxes_list, scores_list, labels_list, detector_meta = self._gather_detections(
                sensor=sensor,
                images=flattened,
                B=B,
                T=T,
                goals=goals,
                non_visual_sensors=non_visual_sensors,
            )
            extracted = self.object_token_extractor(flattened, boxes_list, scores_list)
            sensor_tokens, token_mask, token_boxes = self._prepare_sensor_tokens(extracted)
            if self.bbox_rotary_embed is not None:
                sensor_tokens = self.bbox_rotary_embed(sensor_tokens, token_boxes, token_mask)
            all_img_features[sensor] = sensor_tokens
            self._store_latest(sensor, extracted, token_mask, labels_list, B, T, detector_meta)

        concatenated_feats = []
        for sensor in self.visual_sensors:
            camera_token = getattr(self, f"visual_sensor_token_{sensor}")
            concatenated_feats.append(all_img_features[sensor] + camera_token)

        concatenated_feats = torch.cat(concatenated_feats, dim=1)

        if text_feats is None:
            text_feats = self.encode_text(goals)
        B, L, D = text_feats.shape
        text_feats_ = text_feats.unsqueeze(1).tile(1, T, 1, 1).reshape(B * T, L, D)
        fusion_token = self.fusion_token.reshape(1, 1, D).tile(B * T, 1, 1)
        return fusion_token, concatenated_feats, text_feats, text_feats_, B, T, D

    def forward(
        self,
        frames,
        goals,
        text_feats=None,
        non_visual_sensors=None,
    ):
        (
            fusion_token,
            concatenated_feats,
            text_feats,
            text_feats_,
            B,
            T,
            D,
        ) = self.get_image_text_feats(frames, goals, text_feats, non_visual_sensors)
        input_features = [fusion_token, concatenated_feats, text_feats_]

        fused_feats = self.fusion_xformer(torch.cat(input_features, 1))

        fused_feats = fused_feats[:, 0, :]

        return fused_feats.reshape(B, T, D), text_feats


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).reshape_as(x)


class RotaryBBoxEncoding(nn.Module):
    def __init__(self, model_dim: int, rotary_dim: int = 128, base: float = 10000.0):
        super().__init__()
        rotary_dim = min(model_dim, rotary_dim)
        rotary_dim = max(0, rotary_dim - (rotary_dim % 8))
        self.rotary_dim = rotary_dim
        if rotary_dim == 0:
            self.register_buffer("inv_freq", torch.tensor([]), persistent=False)
            return
        self.coords = 4
        self.per_coord_dim = rotary_dim // self.coords
        if self.per_coord_dim % 2 != 0:
            self.per_coord_dim -= 1
        if self.per_coord_dim <= 0:
            raise ValueError("Rotary dimension too small for bbox encoding.")
        dim_half = self.per_coord_dim // 2
        inv_freq = torch.exp(
            torch.arange(0, dim_half, dtype=torch.float32) * (-math.log(base) / max(dim_half, 1))
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, token_feats: torch.Tensor, boxes: torch.Tensor, token_mask: torch.Tensor):
        if self.rotary_dim == 0 or self.inv_freq.numel() == 0:
            return token_feats

        x_rot = token_feats[..., : self.rotary_dim]
        x_pass = token_feats[..., self.rotary_dim :]

        cos, sin = self._build_sin_cos(boxes, token_mask, x_rot.dtype, x_rot.device)
        x_rotated = x_rot * cos + _rotate_half(x_rot) * sin
        return torch.cat([x_rotated, x_pass], dim=-1)

    def _build_sin_cos(self, boxes, token_mask, dtype, device):
        # boxes: (B, N, 4)
        x0, y0, x1, y1 = boxes.unbind(-1)
        centers = torch.stack([(x0 + x1) * 0.5, (y0 + y1) * 0.5], dim=-1)
        sizes = torch.stack([(x1 - x0).abs(), (y1 - y0).abs()], dim=-1)
        coords = torch.cat([centers, sizes], dim=-1).clamp(min=0.0, max=1.0)
        sin_parts = []
        cos_parts = []
        freq = self.inv_freq.to(device=device, dtype=dtype)
        for idx in range(coords.shape[-1]):
            coord = coords[..., idx].unsqueeze(-1)
            angles = coord * freq
            sin = torch.repeat_interleave(torch.sin(angles), 2, dim=-1)
            cos = torch.repeat_interleave(torch.cos(angles), 2, dim=-1)
            target_dim = self.per_coord_dim
            if sin.shape[-1] < target_dim:
                pad = target_dim - sin.shape[-1]
                sin = F.pad(sin, (0, pad))
                cos = F.pad(cos, (0, pad))
            sin_parts.append(sin[..., :target_dim])
            cos_parts.append(cos[..., :target_dim])

        sin = torch.cat(sin_parts, dim=-1)
        cos = torch.cat(cos_parts, dim=-1)
        mask = token_mask.unsqueeze(-1).to(dtype=dtype, device=device)
        sin = sin * mask
        cos = cos * mask + (1.0 - mask)
        return cos, sin


class PositionalEncoder(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.register_buffer("div_term", div_term)

    def forward(self, position):
        """
        Args:
            position: Tensor, shape [batch_size, seq_len]
        """
        B, L = position.shape
        position = position.unsqueeze(-1)  # BxLx1
        pe = torch.zeros([B, L, self.d_model], device=position.device)
        pe[:, :, 0::2] = torch.sin(position * self.div_term)
        pe[:, :, 1::2] = torch.cos(position * self.div_term)
        return pe


@dataclass
class NonTxVisualEncoderConfig:
    image_encoder: str = "Dinov2Small"
    text_encoder: str = "t5-small"
    input_sensors: List[str] = None
    compressor_hidden_dims: List[int] = (128, 32)
    text_adapter_output_dim: int = 32
    image_text_combiner_hidden_dims: List[int] = (64, 32)
    per_cam_feat_dim: int = 2688
    final_out_dim: int = 512


class NonTxMultiCameraVisualEncoder(nn.Module):
    def __init__(self, cfg: NonTxVisualEncoderConfig):
        super().__init__()
        self.cfg = cfg

        if cfg.image_encoder == "dinov2" and cfg.image_encoder not in IMAGE_ENCODERS:
            cfg.image_encoder = "Dinov2Small"
            print("REAPLACING DINOV2 WITH DINOV2SMALL")

        if cfg.image_encoder in IMAGE_ENCODERS:
            image_encoder_model_cls, image_encoder_cfg = IMAGE_ENCODERS[cfg.image_encoder]
            self.image_encoder = image_encoder_model_cls(image_encoder_cfg)
        else:
            raise NotImplementedError()
        self.visual_compressor = self.create_compressor()

        self.text_encoder = create_text_encoder(cfg.text_encoder)

        # text_adapter maps T5/SigLIP text embeddings to the action decoders dimension for use as memory
        self.text_adapter = nn.Sequential(
            nn.Linear(TEXT_ENCODER_DIMS[cfg.text_encoder], self.cfg.final_out_dim),
            nn.LayerNorm(self.cfg.final_out_dim),
            nn.ReLU(),
        )
        # text_adapter_for_combiner maps the text embedding (after text_adapter) to the dimension required for image text combination
        self.text_adapter_for_combiner = nn.Sequential(
            nn.Linear(self.cfg.final_out_dim, self.cfg.text_adapter_output_dim),
            nn.LayerNorm(self.cfg.text_adapter_output_dim),
            nn.ReLU(),
        )

        self.image_text_combiner = self.create_image_text_combiner()
        self.visual_sensors = [sensor for sensor in cfg.input_sensors if is_a_visual_sensor(sensor)]
        self.final_adapter = nn.Sequential(
            nn.Linear(len(self.visual_sensors) * 32 * 7 * 12, self.cfg.final_out_dim),
            nn.LayerNorm(self.cfg.final_out_dim),
            nn.ReLU(),
        )

    def encode_text(self, preproc_text_input):
        with torch.no_grad():
            if isinstance(self.text_encoder, TextTransformer):
                cls_feats, text_feats = self.text_encoder(preproc_text_input)
                text_feats = torch.cat([text_feats, cls_feats.unsqueeze(1)], dim=1)
            else:
                text_feats = self.text_encoder(**preproc_text_input).last_hidden_state

        return self.text_adapter(text_feats)

    def encode_imgs(self, imgs):
        B, T, C, H, W = imgs.shape
        feats = self.visual_compressor(
            self.image_encoder(imgs.reshape(B * T, C, H, W))
        )  # BTxC_xH_xW_
        return feats

    def create_compressor(self):
        assert len(self.cfg.compressor_hidden_dims) == 2
        return nn.Sequential(
            nn.Conv2d(self.image_encoder.cfg.output_size[0], self.cfg.compressor_hidden_dims[0], 1),
            nn.ReLU(),
            nn.Conv2d(self.cfg.compressor_hidden_dims[0], self.cfg.compressor_hidden_dims[1], 1),
            nn.ReLU(),
        )

    def create_image_text_combiner(self):
        assert len(self.cfg.image_text_combiner_hidden_dims) == 2
        return nn.Sequential(
            nn.Conv2d(
                self.cfg.compressor_hidden_dims[-1] + self.cfg.text_adapter_output_dim,
                self.cfg.image_text_combiner_hidden_dims[0],
                1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                self.cfg.image_text_combiner_hidden_dims[0],
                self.cfg.image_text_combiner_hidden_dims[1],
                1,
            ),
            nn.ReLU(),
        )

    def forward(
        self,
        frames,
        goals,
        text_feats=None,
        task_relevant_object_bbox=None,
        manip_task_relevant_object_bbox=None,
    ):
        assert task_relevant_object_bbox is None and manip_task_relevant_object_bbox is None

        all_img_features = {}
        images_chw = None
        for sensor in frames.keys():
            assert is_a_visual_sensor(sensor)
            imgs = frames[sensor]
            B, T, C, H, W = imgs.shape

            if images_chw is None:
                images_chw = (C, H, W)

            assert images_chw == (C, H, W)

            image_feats = self.encode_imgs(imgs)  # BTxCxHxW
            all_img_features[sensor] = image_feats

            _, fC, fH, fW = image_feats.shape
        if text_feats is None:
            text_feats = self.encode_text(goals)  # BxLxD

        text_feats_ = self.text_adapter_for_combiner(text_feats)
        text_feats_ = text_feats_.mean(dim=1, keepdim=True).tile(1, T, 1).reshape(B * T, -1)  # BTxD
        text_feats_ = text_feats_.unsqueeze(-1).unsqueeze(-1).tile(1, 1, fH, fW)  # BTxDxHxW

        all_cam_feats = []
        for sensor in frames.keys():
            all_cam_feats.append(
                self.image_text_combiner(
                    torch.cat([all_img_features[sensor], text_feats_], dim=1)
                ).reshape(B, T, -1)
            )

        fused_feats = self.final_adapter(torch.cat(all_cam_feats, dim=-1))
        return fused_feats, text_feats
