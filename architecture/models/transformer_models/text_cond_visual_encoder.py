import math

# from utils.transformation_util import get_full_transformation_list, sample_a_specific_transform
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from open_clip import create_model_from_pretrained
from open_clip.transformer import TextTransformer
from transformers import T5EncoderModel

from architecture.models.transformer_models.image_encoders import IMAGE_ENCODERS, Dinov2
from architecture.models.transformer_models.object_token_extractor import (
    ObjectTokenExtractionOutput,
    ObjectTokenExtractor,
    ObjectTokenExtractorConfig,
)
from utils.bbox_utils import get_best_of_two_bboxes
from utils.grounding_dino_utils import GroundingDINOPredictor
from utils.sensor_constant_utils import is_a_visual_sensor


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
    fusion_xformer: TransformerConfig = TransformerConfig(3, 512, 8)
    input_sensors: List[str] = None
    bbox_encoding_type: Literal["positional"] = "positional"


@dataclass
class ObjectTokenVisualEncoderConfig(TextCondVisualEncoderConfig):
    max_object_tokens: int = 10
    detector_model_id: str = "IDEA-Research/grounding-dino-tiny"
    detector_device: str = "cuda"
    detector_box_threshold: float = 0.25
    detector_text_threshold: float = 0.25
    detector_phrases: Sequence[str] = field(default_factory=lambda: ["object"])
    pooling: Literal["attention", "mean"] = "attention"
    min_patch_overlap: int = 1
    use_detector: bool = True


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
        if not isinstance(self.image_encoder, Dinov2):
            raise ValueError("ObjectTokenVisualEncoder currently supports Dinov2 image encoders only.")
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
        if cfg.use_detector:
            self.detector: Optional[GroundingDINOPredictor] = GroundingDINOPredictor(
                model_id=cfg.detector_model_id,
                device=cfg.detector_device,
                box_threshold=cfg.detector_box_threshold,
                text_threshold=cfg.detector_text_threshold,
                max_detections=cfg.max_object_tokens,
            )
        else:
            self.detector = None
        self.latest_object_data: Dict[str, Dict[str, object]] = {}

    def _detect_boxes(
        self, images: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[str]]]:
        batch = images.shape[0]
        device = images.device
        if self.detector is None:
            zeros_boxes = [torch.zeros((0, 4), dtype=torch.float32, device=device) for _ in range(batch)]
            zeros_scores = [torch.zeros(0, dtype=torch.float32, device=device) for _ in range(batch)]
            empty_labels = [[] for _ in range(batch)]
            return zeros_boxes, zeros_scores, empty_labels

        pil_inputs = [GroundingDINOPredictor._to_pil(img.detach().cpu()) for img in images]
        detections = self.detector(
            pil_inputs,
            text_queries=self.cfg.detector_phrases,
            max_detections=self.cfg.max_object_tokens,
        )

        boxes_list: List[torch.Tensor] = []
        scores_list: List[torch.Tensor] = []
        labels_list: List[List[str]] = []
        for det in detections:
            boxes = det.boxes.to(device)
            scores = det.scores.to(device)
            labels = list(det.labels)
            if boxes.shape[0] > self.cfg.max_object_tokens:
                topk_scores, topk_idx = torch.topk(scores, self.cfg.max_object_tokens)
                boxes = boxes[topk_idx]
                labels = [labels[i] for i in topk_idx.tolist()]
                scores = topk_scores
            boxes_list.append(boxes)
            scores_list.append(scores)
            padded_labels = labels + ["" for _ in range(self.cfg.max_object_tokens - len(labels))]
            labels_list.append(padded_labels)

        return boxes_list, scores_list, labels_list

    @staticmethod
    def _reshape_bt(tensor: torch.Tensor, B: int, T: int):
        return tensor.reshape(B, T, *tensor.shape[1:])

    def _prepare_sensor_tokens(
        self, extracted: ObjectTokenExtractionOutput
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        return adapted, token_mask

    def _store_latest(
        self,
        sensor: str,
        extracted: ObjectTokenExtractionOutput,
        token_mask: torch.Tensor,
        labels_list: List[List[str]],
        B: int,
        T: int,
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

        self.latest_object_data[sensor] = {
            "boxes": boxes,
            "scores": scores,
            "object_mask": object_mask,
            "token_mask": token_mask_bt,
            "attention": attention,
            "labels": labels_nested,
            "cls_tokens": cls_tokens,
        }

    def get_image_text_feats(self, frames, goals, text_feats):
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
            boxes_list, scores_list, labels_list = self._detect_boxes(flattened)
            extracted = self.object_token_extractor(flattened, boxes_list, scores_list)
            sensor_tokens, token_mask = self._prepare_sensor_tokens(extracted)
            all_img_features[sensor] = sensor_tokens
            self._store_latest(sensor, extracted, token_mask, labels_list, B, T)

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
