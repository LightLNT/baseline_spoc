from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional, Tuple

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from open_clip import create_model_from_pretrained


@dataclass
class ClipResNetConfig:
    model: str = "RN50"
    pool: bool = False
    device: str = "cpu"
    output_size: Tuple[int, int, int] = (2048, 7, 12)


class ClipResNet(nn.Module):
    def __init__(self, cfg: ClipResNetConfig):  # pool=True, device='cpu'):
        super().__init__()
        self.cfg = cfg
        self.model = clip.load(cfg.model, device=torch.device(cfg.device), jit=False)[0]
        self.eval()

    def forward(self, x):
        m = self.model.visual
        with torch.no_grad():

            def stem(x):
                x = m.relu1(m.bn1(m.conv1(x)))
                x = m.relu2(m.bn2(m.conv2(x)))
                x = m.relu3(m.bn3(m.conv3(x)))
                x = m.avgpool(x)
                return x

            x = x.type(m.conv1.weight.dtype)
            x = stem(x)
            x = m.layer1(x)
            x = m.layer2(x)
            x = m.layer3(x)
            x = m.layer4(x)
            if self.cfg.pool:
                x = F.adaptive_avg_pool2d(x, (1, 1))
                x = torch.flatten(x, 1)

            return x


@dataclass
class Dinov2Config:
    model: str = "dinov2_vits14"
    output_size: Tuple[int, int, int] = (384, 7, 12)
    input_size: Tuple[int, int] = (224, 384)
    patch_grid: Tuple[int, int] = (16, 27)
    width_crop: int = 3
    detach_features: bool = True


class Dinov2(nn.Module):
    def __init__(self, cfg: Dinov2Config):
        super().__init__()
        self.cfg = cfg
        self.model = torch.hub.load("facebookresearch/dinov2", cfg.model)
        self.pool = nn.AdaptiveAvgPool2d(cfg.output_size[1:])
        self.eval()

    def _prepare_inputs(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-2:] == tuple(self.cfg.input_size), (
            f"Expected shape is {self.cfg.input_size}; got {x.shape}"
        )
        if self.cfg.width_crop > 0:
            return x[:, :, :, self.cfg.width_crop : -self.cfg.width_crop]
        return x

    def _forward_features(self, x: torch.Tensor, detach: Optional[bool] = None):
        should_detach = self.cfg.detach_features if detach is None else detach
        ctx = torch.no_grad() if should_detach else nullcontext()
        with ctx:
            inputs = self._prepare_inputs(x)
            features = self.model.forward_features(inputs)
        return features

    def forward_patch_tokens(self, x: torch.Tensor, detach: Optional[bool] = None):
        features = self._forward_features(x, detach=detach)
        patch_tokens = features["x_norm_patchtokens"]
        cls_token = features.get("x_norm_clstoken")
        return cls_token, patch_tokens

    def forward(self, x):
        cls_token, patch_tokens = self.forward_patch_tokens(x, detach=True)
        del cls_token  # cls token is unused for the pooled output
        B, _, D = patch_tokens.shape
        grid_h, grid_w = self.cfg.patch_grid
        assert grid_h * grid_w == patch_tokens.shape[1], (
            f"Expected {grid_h * grid_w} patch tokens, got {patch_tokens.shape[1]}"
        )
        patch_tokens = patch_tokens.permute(0, 2, 1)
        patch_tokens = patch_tokens.reshape(B, D, grid_h, grid_w)
        patch_tokens = self.pool(patch_tokens)
        return patch_tokens


@dataclass
class SigLIPConfig:
    model: str = "ViT-B-16-SigLIP-256"
    output_size: Tuple[int, int, int] = (768, 7, 12)
    input_size: Tuple[int, int] = (256, 256)
    patch_grid: Tuple[int, int] = (16, 16)
    width_crop: int = 0
    detach_features: bool = True


class SigLIP(nn.Module):
    def __init__(self, cfg: SigLIPConfig):
        super().__init__()
        self.cfg = cfg
        siglip_full_model = create_model_from_pretrained("hf-hub:timm/{}".format(cfg.model))
        self.model = siglip_full_model[0].visual.trunk
        self.context_length = siglip_full_model[0].context_length
        self.pool = nn.AdaptiveAvgPool2d(cfg.output_size[1:])
        self.eval()

    def _forward_features(self, x: torch.Tensor, detach: Optional[bool] = None):
        assert x.shape[-2:] == tuple(self.cfg.input_size), (
            f"Expected shape is {self.cfg.input_size}; got {x.shape[-2:]}"
        )
        should_detach = self.cfg.detach_features if detach is None else detach
        ctx = torch.no_grad() if should_detach else nullcontext()
        with ctx:
            features = self.model.forward_features(x)
        return features

    def forward_patch_tokens(self, x: torch.Tensor, detach: Optional[bool] = None):
        features = self._forward_features(x, detach=detach)
        if isinstance(features, dict):
            patch_tokens = features.get("x_prenorm") or features.get("x_norm_patchtokens")
        else:
            patch_tokens = features
        if patch_tokens is None:
            raise ValueError("SigLIP forward_features did not return patch tokens.")
        if patch_tokens.shape[1] == self.cfg.patch_grid[0] * self.cfg.patch_grid[1] + 1:
            cls_tokens = patch_tokens[:, 0]
            patch_tokens = patch_tokens[:, 1:]
        else:
            cls_tokens = patch_tokens.mean(dim=1)
        return cls_tokens, patch_tokens

    def forward(self, x):
        _, patch_tokens = self.forward_patch_tokens(x, detach=True)
        B, tokens, D = patch_tokens.shape
        grid_h, grid_w = self.cfg.patch_grid
        assert tokens == grid_h * grid_w, (
            f"Expected {grid_h * grid_w} patch tokens, got {tokens}"
        )
        patch_tokens = patch_tokens.permute(0, 2, 1)
        patch_tokens = patch_tokens.reshape(B, D, grid_h, grid_w)
        patch_tokens = self.pool(patch_tokens)
        return patch_tokens


IMAGE_ENCODERS = dict(
    Dinov2Small=(Dinov2, Dinov2Config()),
    Dinov2Base=(Dinov2, Dinov2Config(model="dinov2_vitb14", output_size=(768, 7, 12))),
    ClipResNet50=(ClipResNet, ClipResNetConfig()),
    SigLIPBase=(SigLIP, SigLIPConfig()),
    SigLIPLarge=(SigLIP, SigLIPConfig(model="ViT-L-16-SigLIP-256", output_size=(1024, 7, 12))),
)
