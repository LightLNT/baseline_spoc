from __future__ import annotations

import os
import sys
from typing import List, Optional, Sequence

import torch
from torch.distributions.utils import lazy_property
from torch.nn import functional as F
from torchvision.transforms import Resize

__all__ = ["DeticPredictor", "resize_boxes"]


def _discover_detic_repo() -> Optional[str]:
    env_path = os.environ.get("DETIC_REPO_PATH")
    if env_path:
        env_path = os.path.abspath(env_path)
        if os.path.isdir(env_path):
            if env_path not in sys.path:
                sys.path.insert(0, env_path)
            return env_path
        return None

    for candidate in sys.path:
        if candidate.endswith("Detic") or candidate.endswith("Detic/"):
            if os.path.isdir(candidate):
                normalized = candidate.rstrip("/")
                if normalized not in sys.path:
                    sys.path.insert(0, normalized)
                return normalized
    return None


DETIC_REPO_PATH = _discover_detic_repo()
_DETIC_AVAILABLE = DETIC_REPO_PATH is not None
_missing_reason = "Detic repository path not found" if not _DETIC_AVAILABLE else ""

if _DETIC_AVAILABLE:
    centernet_path = os.path.join(DETIC_REPO_PATH, "third_party", "CenterNet2")
    if os.path.isdir(centernet_path) and centernet_path not in sys.path:
        sys.path.insert(0, centernet_path)
    try:
        from centernet.config import add_centernet_config  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        _DETIC_AVAILABLE = False
        _missing_reason = f"centernet import failed ({exc})"

if _DETIC_AVAILABLE:
    try:
        from detectron2.checkpoint import DetectionCheckpointer  # type: ignore
        from detectron2.config import get_cfg  # type: ignore
        from detectron2.data.transforms import ResizeShortestEdge  # type: ignore
        from detectron2.modeling import build_model  # type: ignore
        from detic.config import add_detic_config  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        _DETIC_AVAILABLE = False
        _missing_reason = f"detectron2/detic import failed ({exc})"


def resize_boxes(boxes, original_size, new_size, cutoff_amount=6):
    """
    Resize bounding boxes from original image size to new image size.

    Args:
    - boxes (list of lists): List of bounding boxes in the format [x1, y1, x2, y2].
    - original_size (tuple): Original image size in the format (original_height, original_width).
    - new_size (tuple): New image size in the format (new_height, new_width).

    Returns:
    - list of lists: Resized bounding boxes.
    """

    original_height, original_width = original_size
    new_height, new_width = new_size

    scale_x = new_width / original_width
    scale_y = new_height / original_height

    resized_boxes = []

    for box in boxes:
        x1, y1, x2, y2 = box
        resized_x1 = int(x1 * scale_x) - cutoff_amount
        resized_y1 = int(y1 * scale_y)
        resized_x2 = int(x2 * scale_x) - cutoff_amount
        resized_y2 = int(y2 * scale_y)
        resized_boxes.append([resized_x1, resized_y1, resized_x2, resized_y2])

    return resized_boxes


if not _DETIC_AVAILABLE:

    class DeticPredictor:  # type: ignore[misc]
        """Fallback predictor that raises an informative error when Detic is unavailable."""

        def __init__(self, *args, **kwargs):
            message = (
                "DeticPredictor dependencies are not available. "
                f"{_missing_reason or 'Set DETIC_REPO_PATH to the Detic repository and install Detic/Detectron2.'}"
            )
            raise ImportError(message)

else:

    def create_detic_cfg(
        config_file: str,
        opts: Optional[List[str]],
        confidence_threshold: float,
        pred_all_class: bool,
        device: str | torch.device,
    ):
        cfg = get_cfg()
        device_arg = device
        if isinstance(device_arg, torch.device):
            device_arg = str(device_arg)
        cfg.MODEL.DEVICE = device_arg

        add_centernet_config(cfg)
        add_detic_config(cfg)
        cfg.merge_from_file(config_file)
        if opts:
            cfg.merge_from_list([str(item) for item in opts])

        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
        cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = "rand"
        if not pred_all_class:
            cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True

        cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH = os.path.join(
            DETIC_REPO_PATH, cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH
        )

        cfg.freeze()
        return cfg

    class DeticPredictor:
        """
        Simple end-to-end Detic predictor that runs on a single device for a batch of input images.

        Note:
        1. Always assume you are given a batch of RGB images as input of shape B x C x H x W.
        2. Will apply resizing defined by min_size_test/max_size_test.
        """

        def __init__(
            self,
            vocabulary: Sequence[str] = ("apple", "potato"),
            prompt: str = "a ",
            config_file: str = "Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml",
            model_weights_file: str = "Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth",
            min_size_test: Optional[int] = None,
            max_size_test: Optional[int] = None,
            confidence_threshold: float = 0.3,
            pred_all_class: bool = False,
            device: str | torch.device = "cpu",
        ):
            if DETIC_REPO_PATH is None:
                raise FileNotFoundError(
                    "DETIC_REPO_PATH could not be resolved; ensure Detic is installed."
                )

            if not os.path.exists(config_file):
                candidate = os.path.join(DETIC_REPO_PATH, "configs", config_file)
                if not os.path.exists(candidate):
                    raise FileNotFoundError(f"Could not locate Detic config file: {config_file}")
                config_file = candidate

            if not os.path.exists(model_weights_file):
                candidate = os.path.join(DETIC_REPO_PATH, "models", model_weights_file)
                if not os.path.exists(candidate):
                    raise FileNotFoundError(f"Could not locate Detic model weights: {model_weights_file}")
                model_weights_file = candidate

            opts: List[str] = [
                "MODEL.WEIGHTS",
                model_weights_file,
            ]

            if min_size_test is not None:
                opts.extend(["INPUT.MIN_SIZE_TEST", str(min_size_test)])

            if max_size_test is not None:
                opts.extend(["INPUT.MAX_SIZE_TEST", str(max_size_test)])

            cfg = create_detic_cfg(
                config_file=config_file,
                opts=opts,
                confidence_threshold=confidence_threshold,
                pred_all_class=pred_all_class,
                device=device,
            )

            self.cfg = cfg.clone()
            self.prompt = prompt

            self.model = build_model(self.cfg)
            checkpointer = DetectionCheckpointer(self.model)
            checkpointer.load(cfg.MODEL.WEIGHTS)
            self.model.eval()

            self._vocabulary: Optional[List[str]] = None
            self.vocabulary = list(vocabulary)

            assert cfg.INPUT.FORMAT == "RGB"

        def to(self, device: torch.device):
            self.model.to(device)
            self.text_encoder.to(device)
            return self

        @property
        def vocabulary(self) -> Sequence[str]:
            return self._vocabulary or []

        @lazy_property
        def text_encoder(self):
            from detic.modeling.text.text_encoder import build_text_encoder

            text_encoder = build_text_encoder(pretrain=True)
            text_encoder.eval()
            text_encoder.to(self.model.device)
            return text_encoder

        def get_clip_embeddings(self, vocabulary, prompt="a "):
            texts = [prompt + x for x in vocabulary]
            with torch.no_grad():
                return self.text_encoder(texts).detach().permute(1, 0).contiguous()

        @vocabulary.setter
        def vocabulary(self, vocabulary: Sequence[str]):
            vocab_list = list(vocabulary)
            if self._vocabulary is not None and list(self._vocabulary) == vocab_list:
                return
            self._vocabulary = vocab_list

            num_classes = len(self._vocabulary)
            self.model.roi_heads.num_classes = num_classes

            zs_weight = self.get_clip_embeddings(self._vocabulary, prompt=self.prompt)
            zs_weight = torch.cat(
                [zs_weight, zs_weight.new_zeros((zs_weight.shape[0], 1), device=self.model.device)],
                dim=1,
            )

            if self.model.roi_heads.box_predictor[0].cls_score.norm_weight:
                zs_weight = F.normalize(zs_weight, p=2, dim=0)

            for predictor in self.model.roi_heads.box_predictor:
                del predictor.cls_score.zs_weight
                predictor.cls_score.zs_weight = zs_weight

        def resize_images(self, images: torch.Tensor):
            b, c, h, w = images.shape
            new_h, new_w = ResizeShortestEdge.get_output_shape(
                oldh=h,
                oldw=w,
                short_edge_length=self.cfg.INPUT.MIN_SIZE_TEST,
                max_size=self.cfg.INPUT.MAX_SIZE_TEST,
            )

            return Resize((new_h, new_w), antialias=True)(images)

        def __call__(self, images: torch.Tensor):
            with torch.no_grad():
                nbatch, _, height, width = images.shape
                images = self.resize_images(images)
                images = images.float()

                inputs = []
                for i in range(nbatch):
                    inputs.append({"image": images[i], "height": height, "width": width})

                predictions = self.model(inputs)
                return predictions
