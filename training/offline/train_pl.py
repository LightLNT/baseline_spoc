import argparse
import json
import os
import random
import warnings
from typing import Any, Dict, List, Mapping, Optional, Sequence

import lightning.pytorch as pl
import numpy as np
import torch
import wandb
from allenact.utils.misc_utils import str2bool
from torch import optim
from torch.utils.data import DataLoader
from torchmetrics import F1Score
from torchmetrics.aggregation import SumMetric

from architecture.models.transformer_models import REGISTERED_MODELS
from online_evaluation.local_logging_utils import LocalWandbLogger
from training.offline.chores_dataset import ChoresMultitaskDataset
from training.offline.dataset_mixtures import get_mixture_by_name
from training.offline.train_utils import get_latest_local_ckpt_pth
from utils.object_token_debug import render_object_token_batch
from utils.sensor_constant_utils import is_a_non_visual_sensor, is_a_visual_sensor


def arg_parser_for_offline_training():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="EarlyFusionCnnTransformer")
    parser.add_argument("--model_version", type=str, default="small_3")
    parser.add_argument("--loss", type=str, default="action")
    parser.add_argument("--dataset_version", type=str, default="object_nav_v0.3")
    parser.add_argument("--data_dir", type=str, default="/data/datasets")
    parser.add_argument("--output_dir", type=str, default="/data/results")
    parser.add_argument("--max_samples", type=int, default=100000)
    parser.add_argument("--eval_max_samples", type=int, default=1600)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--wandb_logging", default=True, type=str2bool)

    parser.add_argument("--wandb_project_name", default="", type=str)
    parser.add_argument("--wandb_entity_name", default="", type=str)
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--log_video_every", type=int, default=2000)
    parser.add_argument("--max_epochs", type=int, default=250)
    parser.add_argument("--per_gpu_batch", type=int, default=16)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--sliding_window", type=int, default=100)
    parser.add_argument("--init_prob_sample_last_steps", type=float, default=0.0)
    parser.add_argument("--final_prob_sample_last_steps", type=float, default=0.0)
    parser.add_argument("--reduce_action_redundancy", type=str2bool, default=False)
    parser.add_argument("--precision", type=str, default="32-true", choices=["32-true", "16-mixed"])
    # resume training from last local checkpoint
    parser.add_argument("--resume_local", action=argparse.BooleanOptionalAction)
    # resume from specified run id and step
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction)
    parser.add_argument("--use_non_strict_ckpt_loading", action=argparse.BooleanOptionalAction)
    parser.add_argument("--restart_optimizer", action=argparse.BooleanOptionalAction)
    # initialize model from a specified run_id and step
    parser.add_argument("--init_model", action=argparse.BooleanOptionalAction)
    # specify run id for --resume or --init_model
    parser.add_argument("--run_id", type=str)
    parser.add_argument("--step", type=int, default=-1)
    parser.add_argument(
        "--input_sensors",
        nargs="+",
    default=["raw_navigation_camera"],
    )
    parser.add_argument(
        "--detector_usage",
        type=str,
        default="always",
        choices=["always", "eval_only", "train_only", "never"],
        help="Control when object detectors run (only relevant for object-token models).",
    )
    parser.add_argument(
        "--visualize_object_tokens",
        type=str2bool,
        default=False,
        help="If true, render object-token visualizations once per training epoch (rank 0 only).",
    )
    parser.add_argument(
        "--visual_debug_max_batch",
        type=int,
        default=2,
        help="Max number of batch elements to render when visualizing object tokens.",
    )
    parser.add_argument(
        "--visual_debug_max_timestep",
        type=int,
        default=4,
        help="Max number of timesteps per sample to render during visualization.",
    )
    parser.add_argument(
        "--keep_manip_camera",
        action=argparse.BooleanOptionalAction,
        help="If set, keep raw_manipulation_camera in the input sensors despite default filtering.",
    )
    return parser

def _prepare_input_sensors(args) -> None:
    """Ensure required bbox sensors are included when detector runs are skipped."""
    if getattr(args, "_prepared_input_sensors", False):
        return

    if "object_token" in str(args.model_version).lower() and getattr(
        args, "detector_usage", "always"
    ) in {"eval_only", "train_only", "never"}:
        required_bbox_sensors = [
            "nav_task_relevant_object_bbox",
            "nav_accurate_object_bbox",
        ]
        if "raw_manipulation_camera" in args.input_sensors:
            required_bbox_sensors.extend(
                [
                    "manip_task_relevant_object_bbox",
                    "manip_accurate_object_bbox",
                ]
            )
        for sensor in required_bbox_sensors:
            if sensor not in args.input_sensors:
                args.input_sensors.append(sensor)

    if not getattr(args, "keep_manip_camera", False):
        filtered_input_sensors = [
            sensor for sensor in args.input_sensors if sensor != "raw_manipulation_camera"
        ]
        if len(filtered_input_sensors) != len(args.input_sensors):
            warnings.warn(
                "Removing 'raw_manipulation_camera' from input sensors to reduce inference load.",
                RuntimeWarning,
            )
            args.input_sensors = filtered_input_sensors

    args._prepared_input_sensors = True


class AdamWSkipLoadStateDict(optim.AdamW):
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        warnings.warn("AdamWSkipLoadStateDict IS IGNORING A REQUEST TO LOAD A STATE DICT")
        return


class LitModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.use_non_strict_ckpt_loading = args.use_non_strict_ckpt_loading
        self.restart_optimizer = args.restart_optimizer
        _prepare_input_sensors(args)
        model, preproc = REGISTERED_MODELS[args.model].build_model(
            model_version=args.model_version,
            input_sensors=args.input_sensors,
            loss=args.loss,
            ckpt_pth=args.ckpt_pth,
            detector_usage=getattr(args, "detector_usage", None),
        )
        self.model = model
        self.preproc = preproc
        self.args = args
        self.metrics = self.get_metrics()
        self.train_steps = 0
        self.num_frames = 0
        self.frames_metric = SumMetric()
        self.log_video_every = args.log_video_every
        self.visualize_object_tokens = bool(args.visualize_object_tokens)
        self.visual_debug_max_batch = args.visual_debug_max_batch
        self.visual_debug_max_timestep = args.visual_debug_max_timestep
        self.visual_debug_output_dir = os.path.join(args.output_dir, "object_token_debug")
        self._visual_mean_std = self._extract_image_stats()
        self._visual_debug_cache = None

    def on_fit_start(self):
        self.preproc.device = self.device
        self.frames_metric.reset()
        self._visual_debug_cache = None

    def log_videos(self, batch, outputs, train_or_val):
        items_to_log = random.choices(range(len(batch)), k=min(10, len(batch)))
        columns = ["Task", "Observation", "Actions_gt", "Actions_pred", "Sensor_path"]
        data = []
        for item_to_log in items_to_log:
            batch_item = batch[item_to_log]
            output_item = outputs["actions_logits"][item_to_log]
            pred = output_item.argmax(-1).cpu().tolist()
            actions_pred = [self.preproc.cfg.action_list[action_idx] for action_idx in pred]
            actions_gt = list(batch_item["observations"]["actions"])
            task = batch_item["observations"]["goal"]

            nav_frames = batch_item["observations"].get("raw_navigation_camera")
            if nav_frames is None:
                continue
            nav_frames = nav_frames.cpu().numpy()
            nav_frames = np.transpose(nav_frames, (0, 3, 1, 2))
            video = wandb.Video(nav_frames, fps=5)

            sensor_path = batch_item["raw_navigation_camera"]
            data.append([task, video, actions_gt, actions_pred, sensor_path])

        if hasattr(self.logger, "log_table"):
            self.logger.log_table(
                key=f"video_action_table/{train_or_val}/{self.train_steps}",
                columns=columns,
                data=data,
            )

    def _extract_image_stats(self):
        default_mean = (0.48145466, 0.4578275, 0.40821073)
        default_std = (0.26862954, 0.26130258, 0.27577711)
        try:
            transforms = getattr(self.preproc.image_preprocessor, "transforms", None)
        except Exception:
            transforms = None
        if transforms:
            for transform in reversed(transforms):
                if hasattr(transform, "mean") and hasattr(transform, "std"):
                    try:
                        mean = tuple(float(m) for m in transform.mean)
                        std = tuple(float(s) for s in transform.std)
                        return mean, std
                    except TypeError:
                        continue
        return default_mean, default_std

    def _cache_visual_debug_batch(
        self,
        proc_batch,
        raw_batch: Optional[Sequence[Mapping[str, Any]]] = None,
    ):
        if not self.visualize_object_tokens:
            return
        if self.trainer is not None:
            if not self.trainer.is_global_zero:
                return
            if getattr(self.trainer, "sanity_checking", False):
                return
        if not hasattr(self.model.visual_encoder, "latest_object_data"):
            return
        if self._visual_debug_cache is not None:
            return

        frames = {}
        for sensor, tensor in proc_batch.items():
            if is_a_visual_sensor(sensor) and torch.is_tensor(tensor):
                frames[sensor] = tensor.detach().cpu()

        if not frames:
            return

        max_batch = max(1, int(self.visual_debug_max_batch))
        # Limit cached tensors to the configured batch budget to control memory usage.
        for sensor in list(frames.keys()):
            frames[sensor] = frames[sensor][:max_batch].clone()

        non_visual = {}
        for sensor, tensor in proc_batch.items():
            if is_a_non_visual_sensor(sensor) and torch.is_tensor(tensor):
                non_visual[sensor] = tensor.detach().cpu()

        for sensor in list(non_visual.keys()):
            non_visual[sensor] = non_visual[sensor][:max_batch].clone()

        tasks = None
        if raw_batch is not None:
            tasks = []
            for sample in raw_batch[:max_batch]:
                goal_text = None
                if isinstance(sample, Mapping):
                    observations = sample.get("observations")
                    if isinstance(observations, Mapping):
                        goal_text = observations.get("goal")
                tasks.append(goal_text)

        goal_key = self.preproc.cfg.goal_sensor_uuid
        goals_val = proc_batch.get(goal_key)
        if isinstance(goals_val, dict):
            goals = {
                k: v.detach().cpu()[:max_batch].clone()
                for k, v in goals_val.items()
                if torch.is_tensor(v)
            }
        elif torch.is_tensor(goals_val):
            goals = goals_val.detach().cpu()[:max_batch].clone()
        else:
            goals = None

        self._visual_debug_cache = {
            "frames": frames,
            "non_visual": non_visual,
            "goals": goals,
            "train_step": int(self.train_steps),
            "tasks": tasks,
        }

    def _render_visual_debug(self, epoch_idx: int):
        if not self.visualize_object_tokens or self._visual_debug_cache is None:
            return
        if self.trainer is not None:
            if not self.trainer.is_global_zero:
                self._visual_debug_cache = None
                return
            if getattr(self.trainer, "sanity_checking", False):
                self._visual_debug_cache = None
                return
        if not hasattr(self.model.visual_encoder, "latest_object_data"):
            self._visual_debug_cache = None
            return

        cache = self._visual_debug_cache
        frames_cpu = cache["frames"]
        if not frames_cpu:
            self._visual_debug_cache = None
            return

        frames_device = {k: v.to(self.device) for k, v in frames_cpu.items()}
        non_visual_cpu = cache["non_visual"]
        non_visual_device = {k: v.to(self.device) for k, v in non_visual_cpu.items()}
        goals_cpu = cache["goals"]
        if isinstance(goals_cpu, dict):
            goals_device = {k: v.to(self.device) for k, v in goals_cpu.items()}
        elif torch.is_tensor(goals_cpu):
            goals_device = goals_cpu.to(self.device)
        else:
            goals_device = goals_cpu

        if goals_device is None:
            self._visual_debug_cache = None
            return

        with torch.no_grad():
            # Re-run the visual encoder to populate latest_object_data for visualization.
            self.model.visual_encoder(
                frames_device,
                goals_device,
                text_feats=None,
                non_visual_sensors=non_visual_device if non_visual_device else None,
            )

        image_cfg = getattr(self.model.visual_encoder.image_encoder, "cfg", None)
        if image_cfg is not None and hasattr(image_cfg, "patch_grid"):
            patch_grid = image_cfg.patch_grid
        elif image_cfg is not None and hasattr(image_cfg, "output_size"):
            patch_grid = image_cfg.output_size[1:]
        else:
            patch_grid = (16, 27)

        epoch_dir = os.path.join(
            self.visual_debug_output_dir,
            f"epoch_{epoch_idx:04d}",
        )
        os.makedirs(epoch_dir, exist_ok=True)
        mean, std = self._visual_mean_std
        train_step = cache.get("train_step", int(self.train_steps))
        tasks = cache.get("tasks")

        log_records = self._collect_visual_debug_logs(
            latest_object_data=self.model.visual_encoder.latest_object_data,
            frames_cpu=frames_cpu,
            tasks=tasks,
            epoch_idx=epoch_idx,
            train_step=train_step,
        )
        if log_records:
            log_path = os.path.join(epoch_dir, "detic_predictions.jsonl")
            with open(log_path, "w", encoding="utf-8") as log_file:
                for record in log_records:
                    log_file.write(json.dumps(record, ensure_ascii=False) + "\n")

        render_object_token_batch(
            frames=frames_cpu,
            latest_object_data=self.model.visual_encoder.latest_object_data,
            output_dir=epoch_dir,
            patch_grid=patch_grid,
            prefix=f"epoch{epoch_idx:04d}_step{train_step:06d}",
            mean=mean,
            std=std,
            max_batches=self.visual_debug_max_batch,
            max_timesteps=self.visual_debug_max_timestep,
            overlay_attention=True,
        )

        self._visual_debug_cache = None

    def _collect_visual_debug_logs(
        self,
        *,
        latest_object_data: Dict[str, Dict[str, Any]],
        frames_cpu: Dict[str, torch.Tensor],
        tasks: Optional[Sequence[Any]],
        epoch_idx: int,
        train_step: int,
    ) -> Sequence[Dict[str, Any]]:
        logs: List[Dict[str, Any]] = []
        max_batches = max(1, int(self.visual_debug_max_batch))
        max_timesteps = max(1, int(self.visual_debug_max_timestep))

        for sensor, _ in frames_cpu.items():
            sensor_data = latest_object_data.get(sensor)
            if sensor_data is None:
                logs.append(
                    {
                        "epoch": int(epoch_idx),
                        "train_step": int(train_step),
                        "sensor": sensor,
                        "status": "no_detector_data",
                    }
                )
                continue

            detector_meta = sensor_data.get("detector_meta") if isinstance(sensor_data, dict) else None
            detector_vocab = None
            detector_allowed = None
            if isinstance(detector_meta, dict):
                detector_vocab = detector_meta.get("vocabulary")
                detector_allowed = detector_meta.get("per_image_allowed")

            boxes = sensor_data.get("boxes")
            object_mask = sensor_data.get("object_mask")
            scores = sensor_data.get("scores")
            labels_nested = sensor_data.get("labels")

            if boxes is None or object_mask is None:
                logs.append(
                    {
                        "epoch": int(epoch_idx),
                        "train_step": int(train_step),
                        "sensor": sensor,
                        "status": "missing_boxes_or_mask",
                    }
                )
                continue

            boxes_np = boxes.detach().cpu().numpy()
            mask_np = object_mask.detach().cpu().numpy().astype(bool)
            scores_np = scores.detach().cpu().numpy() if scores is not None else None

            if isinstance(labels_nested, torch.Tensor):
                labels_data = labels_nested.detach().cpu().tolist()
            else:
                labels_data = labels_nested

            B, T = boxes_np.shape[0], boxes_np.shape[1]
            limit_b = min(B, max_batches)
            limit_t = min(T, max_timesteps)

            for b in range(limit_b):
                goal_text = None
                if tasks and b < len(tasks):
                    goal_text = tasks[b]

                for t in range(limit_t):
                    mask_flags = mask_np[b, t].tolist()
                    active_indices = [idx for idx, flag in enumerate(mask_flags) if flag]

                    labels_for_step: List[Any] = []
                    if labels_data is not None:
                        try:
                            labels_sample = labels_data[b][t]
                        except (TypeError, IndexError):
                            labels_sample = None
                        if labels_sample is not None:
                            for idx in active_indices:
                                label_idx = idx + 1 if len(labels_sample) > (idx + 1) else idx
                                labels_for_step.append(labels_sample[label_idx])

                    detections = []
                    for pos, idx in enumerate(active_indices):
                        det: Dict[str, Any] = {
                            "index": int(idx),
                            "box_xyxy": [float(coord) for coord in boxes_np[b, t, idx].tolist()],
                        }
                        if scores_np is not None and idx < scores_np.shape[-1]:
                            det["score"] = float(scores_np[b, t, idx])
                        if pos < len(labels_for_step):
                            det["label"] = labels_for_step[pos]
                        detections.append(det)

                    record: Dict[str, Any] = {
                        "epoch": int(epoch_idx),
                        "train_step": int(train_step),
                        "sensor": sensor,
                        "batch_index": int(b),
                        "timestep": int(t),
                        "goal": goal_text,
                        "num_candidates": int(boxes_np.shape[2]),
                        "num_active": len(active_indices),
                        "detections": detections,
                    }
                    if scores_np is None:
                        record["status"] = "scores_missing"
                    elif not detections:
                        record.setdefault("status", "no_active_detections")
                    if detector_vocab:
                        record["detector_vocabulary"] = list(detector_vocab)
                    if detector_allowed and b < len(detector_allowed):
                        allowed_seq = detector_allowed[b]
                        if isinstance(allowed_seq, (list, tuple)) and t < len(allowed_seq):
                            record["detector_allowed"] = list(allowed_seq[t])
                    logs.append(record)

        return logs

    def forward_batch(self, batch):
        if len(batch) == 0:
            from utils.debug_utils import ForkedPdb

            ForkedPdb().set_trace()

        proc_batch = self.preproc.process(batch)
        outputs = self.model(proc_batch)
        return outputs, proc_batch

    def training_step(self, batch, batch_idx):
        self.train_steps += 1
        outputs, proc_batch = self.forward_batch(batch)
        self._cache_visual_debug_batch(proc_batch, batch)
        self.frames_metric.update(proc_batch["lengths"])
        train_frames = 0
        if self.train_steps % 10 == 0:
            train_frames = self.frames_metric.compute()

        losses = dict()
        for k, v in outputs.items():
            if "loss" in k:
                losses[f"{k}/train"] = v

        self.log_dict(
            {
                **losses,
                "train_steps": float(self.train_steps),
                "train_frames": train_frames,
                "current_prob_to_sample_last_steps": float(
                    min([b["prob_sample_last_steps"] for b in batch])
                ),
            },
            on_step=True,
            on_epoch=False,
            logger=True,
            batch_size=len(batch),
            sync_dist=True,
        )

        if self.train_steps % self.log_video_every == 0:
            self.log_videos(batch, outputs, "train")
        return outputs

    def on_train_epoch_end(self) -> None:
        if not self.visualize_object_tokens:
            return
        if self.trainer is not None:
            if not self.trainer.is_global_zero:
                self._visual_debug_cache = None
                return
            if getattr(self.trainer, "sanity_checking", False):
                self._visual_debug_cache = None
                return
        os.makedirs(self.visual_debug_output_dir, exist_ok=True)
        epoch_idx = int(getattr(self, "current_epoch", 0))
        self._render_visual_debug(epoch_idx)

    def get_metrics(self):
        metrics = dict()
        metrics["f1score_weighted"] = F1Score(
            task="multiclass",
            num_classes=self.model.cfg.num_actions,
            ignore_index=-1,
            average="weighted",
        )
        metrics["f1score_macro"] = F1Score(
            task="multiclass",
            num_classes=self.model.cfg.num_actions,
            ignore_index=-1,
            average="macro",
        )
        metrics["f1score"] = F1Score(
            task="multiclass",
            num_classes=self.model.cfg.num_actions,
            ignore_index=-1,
            average=None,
        )
        return metrics

    def on_train_epoch_start(self) -> None:
        prob_decay_size = (
            self.args.init_prob_sample_last_steps - self.args.final_prob_sample_last_steps
        ) / args.max_epochs
        current_prob = (
            self.args.init_prob_sample_last_steps - prob_decay_size * self.trainer.current_epoch
        )
        next_prob = self.args.init_prob_sample_last_steps - prob_decay_size * (
            self.trainer.current_epoch + 1
        )
        # 4 is the current number of workers we use in the dataloader
        self.trainer.train_dataloader.dataset.init_prob_sample_last_steps(
            init_prob=current_prob,
            final_prob=next_prob,
            num_workers=4,
            num_gpu_per_node=max(torch.cuda.device_count(), 1),
            num_node=self.args.num_nodes,
        )

    def on_validation_epoch_start(self):
        for metric_name, metric in self.metrics.items():
            self.metrics[metric_name] = metric.to(self.device)

    def validation_step(self, batch, batch_idx):
        outputs, proc_batch = self.forward_batch(batch)
        losses = dict()
        for k, v in outputs.items():
            if "loss" in k:
                losses[f"{k}/val"] = v

        self.log_dict(
            {
                **losses,
                "train_steps": float(self.train_steps),
            },
            on_step=True,
            on_epoch=False,
            logger=True,
            batch_size=len(batch),
            sync_dist=True,
        )
        pred = outputs["actions_logits"].argmax(-1)
        gt = proc_batch["actions"]

        if batch_idx == 0:
            self.log_videos(batch, outputs, "val")

        for metric_name in self.metrics:
            self.metrics[metric_name](pred, gt)

    def on_validation_epoch_end(self):
        metrics_to_log = {}
        for metric_name, metric in self.metrics.items():
            if metric_name == "f1score":
                action_f1scores = metric.compute()
                for action_idx, action_name in enumerate(self.preproc.cfg.action_list):
                    metrics_to_log[f"{metric_name}/{action_name}/val"] = action_f1scores[action_idx]
            else:
                metrics_to_log[f"{metric_name}/val"] = metric.compute()

        self.log_dict(
            dict(**metrics_to_log, train_steps=self.train_steps),
            sync_dist=False,
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        for metric in self.metrics.values():
            metric.reset()

    def configure_optimizers(self):
        if self.restart_optimizer:
            return AdamWSkipLoadStateDict(self.model.parameters(), lr=self.args.lr)
        else:
            return optim.AdamW(self.model.parameters(), lr=self.args.lr)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["train_steps"] = self.train_steps
        if hasattr(self.logger, "_run_id"):
            self.logger._checkpoint_name = f"ckpt-{self.logger._run_id}-{self.train_steps}"
        else:
            self.logger._checkpoint_name = f"ckpt-{self.logger.experiment.id}-{self.train_steps}"

    def on_load_checkpoint(self, checkpoint):
        self.train_steps = checkpoint["train_steps"]
        self.trainer.fit_loop.epoch_progress.current.completed = checkpoint["epoch"]

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: Optional[bool] = None):
        state_dict = {
            k.replace(
                "model.visual_encoder.image_encoder.model.visual.trunk",
                "model.visual_encoder.image_encoder.model",
            ): v
            for k, v in state_dict.items()
        }
        state_dict = {
            k.replace(
                "model.visual_encoder.image_encoder.model.text.transformer",
                "model.visual_encoder.text_encoder.transformer",
            ): v
            for k, v in state_dict.items()
        }
        for k in [
            "logit_scale",
            "logit_bias",
            "text.positional_embedding",
            "text.token_embedding.weight",
            "text.ln_final.weight",
            "text.ln_final.bias",
            "text.text_projection.weight",
            "text.text_projection.bias",
        ]:
            k = f"model.visual_encoder.image_encoder.model.{k}"
            if k in state_dict:
                del state_dict[k]

        assert strict is None or strict == (not self.use_non_strict_ckpt_loading)
        strict = not self.use_non_strict_ckpt_loading

        return super().load_state_dict(state_dict, strict=strict)


def identity_collate(batch):
    return [sample for sample in batch if sample is not None]


def get_dataloader(subset: str, args):
    dataset = ChoresMultitaskDataset(
        base_data_dir=args.data_dir,
        dataset_names=get_mixture_by_name(args.dataset_version),
        subset=subset,  # temporary
        max_samples=args.max_samples if subset == "train" else args.eval_max_samples,
        proc_idx=0,  # can't use with DDP
        num_procs=1,  # can't use with DDP
        sliding_window=args.sliding_window,
        input_sensors=args.input_sensors,
        reduce_action_redundancy=args.reduce_action_redundancy if subset == "train" else False,
    )

    return DataLoader(
        dataset,
        batch_size=args.per_gpu_batch,
        num_workers=4 if torch.cuda.is_available() else 1,
        prefetch_factor=2,
        collate_fn=identity_collate,
        persistent_workers=False,
        pin_memory=True,
    )


def launch_training(args):
    _prepare_input_sensors(args)
    local_world_size = max(torch.cuda.device_count(), 1)

    # create data loaders
    data_loaders = dict(
        train=get_dataloader("train", args),
        val=get_dataloader("val", args),
    )

    # set args
    args.num_datasets = len(data_loaders["train"].dataset.dataset_names)
    # max_samples is per dataset, so we need to multiply by num_datasets
    args.max_samples = min(
        args.max_samples * args.num_datasets,
        len(data_loaders["train"].dataset),
    )
    args.exp_name = ",".join(
        [
            f"pl-model={args.model}/{args.model_version}",
            f"dataset={args.dataset_version}",
            f"batch_size={args.per_gpu_batch * local_world_size * args.num_nodes}",
            f"lr={args.lr}",
            f"scale={args.max_samples}",
        ]
    )
    args.exp_dir = os.path.join(args.output_dir, args.exp_name)

    # create logger
    assert (
        args.wandb_entity_name != "" and args.wandb_project_name != ""
    ), "wandb_entity_name and wandb_project_name must be provided"
    logger: Optional[pl.loggers.wandb.WandbLogger]
    if args.wandb_logging:
        logger = pl.loggers.wandb.WandbLogger(
            project=args.wandb_project_name,
            entity=args.wandb_entity_name,
            name=args.exp_name,
            save_dir=args.output_dir,
            config=vars(args),
            log_model="all",
        )
    else:
        logger = LocalWandbLogger(
            project=args.wandb_project_name,
            entity=args.wandb_entity_name,
            name=args.exp_name,
            save_dir=args.output_dir,
            config=vars(args),
            log_model="all",
        )

    if args.init_model:
        init_model_dir = os.path.join(args.exp_dir, args.run_id, str(args.step))
        logger.download_artifact(
            f"{args.wandb_entity_name}/{args.wandb_project_name}/ckpt-{args.run_id}-{args.step}:latest",
            save_dir=init_model_dir,
        )
        args.ckpt_pth = os.path.join(init_model_dir, "model.ckpt")
    else:
        args.ckpt_pth = None

    # create model
    lit_model = LitModel(args)

    # create checkpoint callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.exp_dir,
        filename="checkpoint_{train_steps:.0f}",
        save_top_k=-1,
        verbose=True,
        every_n_train_steps=args.save_every,
    )

    # create trainer and train
    if torch.cuda.is_available():
        devices = local_world_size
        accelerator = "gpu"
        strategy = pl.strategies.DDPStrategy(find_unused_parameters=True)
    else:
        devices = accelerator = strategy = "auto"
        args.precision = "32-true"  # mixed precision doesn't work on cpu

    trainer = pl.Trainer(
        devices=devices,
        num_nodes=args.num_nodes,
        accelerator=accelerator,
        strategy=strategy,
        callbacks=[checkpoint_callback],
        default_root_dir=args.output_dir,
        val_check_interval=args.eval_every,
        log_every_n_steps=10,
        max_epochs=args.max_epochs,
        logger=logger,
        precision=args.precision,
    )

    resume_ckpt_path = None
    if args.resume:
        ckpt_dir = os.path.join(args.exp_dir, args.run_id, str(args.step))
        resume_ckpt_path = os.path.join(ckpt_dir, "model.ckpt")
        if not os.path.exists(resume_ckpt_path):
            logger.download_artifact(
                f"{args.wandb_entity_name}/{args.wandb_project_name}/ckpt-{args.run_id}-{args.step}:latest",
                save_dir=ckpt_dir,
            )
        print("Resuming from:", resume_ckpt_path)
    elif args.resume_local:
        resume_ckpt_path = get_latest_local_ckpt_pth(args.exp_dir)
        if resume_ckpt_path is None:
            print("No local ckpt found. Training from scratch.")
        else:
            print("Resuming from local ckpt:", resume_ckpt_path)
    else:
        print(
            'Training from scratch. Set "--resume" (along with "--run_id" and "--step") to resume from a checkpoint.'
        )

    trainer.fit(
        lit_model,
        data_loaders["train"],
        data_loaders["val"],
        ckpt_path=resume_ckpt_path,
    )


if __name__ == "__main__":
    args = arg_parser_for_offline_training().parse_args()
    if args.wandb_logging:
        assert args.wandb_project_name != ""
        assert args.wandb_entity_name != ""
    os.environ["TOKENIZERS_PARALLELISM"] = "False"
    torch.hub._validate_not_a_forked_repo = (
        lambda a, b, c: True
    )  # This is for getting around the http limit rate error. From https://github.com/pytorch/vision/issues/4156#issuecomment-886005117
    try:
        torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    except Exception as exc:
        warnings.warn(
            f"Skipping Dinov2 torch.hub preload due to error: {exc}. Cached weights will be used if available.",
            RuntimeWarning,
        )

    # Reduced matmul precision for NVIDIA A6000 GPUs
    if torch.cuda.is_available():
        if args.precision == "16-mixed":
            torch.set_float32_matmul_precision("medium")
        elif args.precision == "32-true":
            pass
        else:
            raise NotImplementedError(f"Unknown precision {args.precision}")

    launch_training(args)
