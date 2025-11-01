import math

# from utils.transformation_util import get_full_transformation_list, sample_a_specific_transform
from dataclasses import dataclass
from typing import List, Literal,Dict, Optional, Tuple, Protocol
from einops import rearrange
import torch
import torch.nn as nn
from torch import Tensor
from open_clip import create_model_from_pretrained
from open_clip.transformer import TextTransformer
from transformers import T5EncoderModel

from architecture.models.transformer_models.image_encoders import IMAGE_ENCODERS
from utils.bbox_utils import get_best_of_two_bboxes
from utils.sensor_constant_utils import is_a_visual_sensor
from perceiver_io.encoder import PerceiverEncoder
from perceiver_io.decoders import PerceiverDecoder
from perceiver_io import PerceiverIO

class Attention(Protocol):
    def forward(
            self,
            x: torch.Tensor,
            context: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Implementation of attention. Should return tuple of (feature, attention_map).
        """


def plain_attention(
        query: torch.Tensor,
        key_value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        attention_dropout: float = 0.0,
        training: bool = True,
) -> Tuple[Tensor, Tensor]:
    """
    Args:
        query: (batch, out_seq_len, dim)
        key_value: (batch, in_seq_len, 2, dim)
        mask: (batch, out_seq_len, in_seq_len)
        attention_dropout: dropout probability
        training: whether in training mode

    Returns:
        Tuple[Tensor, Tensor]: (feature, attention_map)
        where:
            feature: (batch, out_seq_len, dim)
            attention_map: (batch, heads, out_seq_len, in_seq_len)
    """
    key, value = key_value.unbind(-3)

    # keyT = key.permute(0, 2, 3, 1)  # transpose to (batch, heads, dim, in_seq_len)
    keyT = key.permute(0, 1, 3, 4, 2)
    value = value.transpose(2, 3)  # transpose to (batch, heads, in_seq_len, dim)
    query = query.transpose(2, 3)  # transpose to (batch, heads, out_seq_len, dim)

    softmax_scale = query.shape[-1] ** (-0.5)
    dots = torch.matmul(query * softmax_scale, keyT)
    if mask is not None:
        assert (
                mask.shape[-2:] == dots.shape[-2:]
        ), f"Mask shape {mask.shape} does not match attention shape {dots.shape}"
        inv_mask = (
            (~mask).unsqueeze(-3).expand_as(dots)
        )  # pylint: disable=invalid-unary-operand-type
        dots.masked_fill_(inv_mask, float("-inf"))

    attn = dots.softmax(dim=-1, dtype=torch.float).to(
        value.dtype
    )  # (batch, heads, out_seq_len, in_seq_len)
    if attention_dropout > 0:
        attn = F.dropout(attn, p=attention_dropout, training=training)

    y = torch.matmul(attn, value).transpose(
        2, 3
    )  # transpose to (batch, seq_len, heads, dim)
    return y, attn


def make_ffn(dim, mult=4):
    return nn.Sequential(
        nn.Linear(dim, dim * mult, bias=False),
        nn.GELU(),
        nn.Linear(dim * mult, dim, bias=False),
    )
class PlainAttention(nn.Module):
    """
    Attention module from original Transformer paper.
    """

    def __init__(
            self,
            model_dim: int,
            context_dim: Optional[int] = None,
            num_heads: int = 8,
            attention_dropout: float = 0.0,
            head_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        context_dim = model_dim if context_dim is None else context_dim
        if head_dim is None:
            assert (
                    model_dim % num_heads == 0
            ), f"model_dim ({model_dim}) must be divisible by num_heads ({num_heads})"
            head_dim = model_dim // num_heads
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.to_q = nn.Linear(model_dim, head_dim * num_heads, bias=False)
        self.to_kv = nn.Linear(context_dim, head_dim * num_heads * 2, bias=False)
        self.to_out = nn.Linear(head_dim * num_heads, model_dim)

    def forward(
            self, x, context=None, mask: Optional[torch.Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        :param x: [batch, seq_len, model_dim]
        :param context: [batch, context_len, context_dim]
        :param mask: [batch, seq_len, context_len]
        """

        context = x if context is None else context
        query = rearrange(
            self.to_q(x),
            "batch seq (head feature) -> batch seq head feature",
            head=self.num_heads,
        )
        key_value = rearrange(
            self.to_kv(context),
            "batch seq (n head feature) -> batch seq n head feature",
            head=self.num_heads,
            n=2,
        )
        y, attn = plain_attention(
            query=query,
            key_value=key_value,
            mask=mask,
            attention_dropout=self.attention_dropout,
            training=self.training,
        )
        y = self.to_out(y.flatten(-2))
        return y, attn


class TransformerBlock(nn.Module):
    """
    A transformer block with pre-normalization.
    """

    def __init__(
            self,
            model_dim: int,
            attention: Attention,
            context_dim: Optional[int] = None,
            extra_context_norm: bool = False,
    ):
        super().__init__()
        context_dim = model_dim if context_dim is None else context_dim
        self.attention = attention
        self.ff = make_ffn(model_dim)
        self.pre_norm1 = nn.LayerNorm(model_dim)
        self.pre_norm2 = nn.LayerNorm(context_dim) if extra_context_norm else None
        self.pre_norm3 = nn.LayerNorm(model_dim)

    def forward(self, x, context=None, mask=None) -> Tuple[Tensor, Tensor]:
        context = x if context is None else context
        if self.pre_norm2 is not None:
            y, attn = self.attention.forward(
                self.pre_norm1(x), context=self.pre_norm2(context), mask=mask
            )
        elif x is not context:
            y, attn = self.attention.forward(
                self.pre_norm1(x), context=self.pre_norm1(context), mask=mask
            )
        else:
            y, attn = self.attention.forward(self.pre_norm1(x), mask=mask)
        x = x + y
        x = x + self.ff(self.pre_norm3(x))
        return x, attn

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
        # 使用官方PerceiverIO
        num_latents = 8
        latent_dim = self.cfg.fusion_xformer.d_model
        input_dim = self.cfg.fusion_xformer.d_model
        decoder_query_dim = self.cfg.fusion_xformer.d_model
        encoder = PerceiverEncoder(
            num_latents=num_latents,
            latent_dim=latent_dim,
            input_dim=input_dim,
            num_self_attn_per_block=cfg.fusion_xformer.nhead,
            num_blocks=cfg.fusion_xformer.num_layers
        )
        decoder = PerceiverDecoder(
            latent_dim=latent_dim,
            query_dim=decoder_query_dim
        )
        self.fusion_xformer = PerceiverIO(encoder, decoder)
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
    # image-text feature
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
        context = torch.cat(input_features, 1)  # [B*T, seq, D]
        # 以第一个token作为query
        query = context[:, :1, :]  # [B*T, 1, D]
        fused_feats = self.fusion_xformer(context, query)  # [B*T, 1, D]
        fused_feats = fused_feats[:, 0, :]  # [B*T, D]
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
