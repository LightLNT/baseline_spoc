# Sparse 1+k Architecture Cheat Sheet

This note captures the reusable configuration we will keep for all sparse-model experiments (Table 2 & Table 3).

## Token layout

- **CLS tokens**: 2（均以导航视角为核心；原来的 manipulation CLS 不再接收独立画面，仅作为额外的导航全局 token）。
- **Object tokens**: `k` slots populated from detector outputs (adaptive count per frame, padded to max_k).
- **Fallback variants**:
  - *No CLS* (Table 2 Variant 1): set `num_cls_tokens=0`, keep only object tokens.
  - *k = 0* (Table 2 Variant 2): disable detector, keep exactly the 2 CLS tokens.

## Sensor bundle

ObjNav-only 消融实验现仅使用导航相机与其 GT bbox，统一传感器如下：

```
raw_navigation_camera
nav_task_relevant_object_bbox
nav_accurate_object_bbox
```

Command-line flags：

- 不再需要 `--keep_manip_camera`，保持导航相机即可。
- 使用 `--detector_usage never`，完全依赖仿真器生成的 GT bbox。

## Model versions

| Name | Backbone | Notes |
| --- | --- | --- |
| `object_token_siglip_small_manip` | SigLIP + T5 | Table 3 baseline（SigLIP 视觉 + T5 文本）。 |
| `object_token_dinov2_small_manip` | DINOv2 + T5 | Table 2 base / Table 3 final。 |
| `object_token_dinov2_small_no_cls` | DINOv2 + T5 | Table 2 Variant 1：禁用 CLS（仅对象 token）。 |
| `object_token_dinov2_small_k0` | DINOv2 + T5 | Table 2 Variant 2：`k=0`，仅保留导航 CLS。 |

> 所有上述消融模型默认**冻结图像与文本编码器**（仅训练稀疏融合/解码层），以减轻显存压力并隔离结构差异的影响。

(上述 `*_manip` / `no_cls` / `k0` 变体已在代码中实现；若需新增其它稀疏模型，可仿照这些配置设置 `max_object_tokens=k` 并保持 `--detector_usage never` 以复用仿真器 GT bbox。)

## Training recipe

Example template (fill in dataset paths and model version per row in the tables):

```
python -m training.offline.train_pl \
  --model EarlyFusionCnnTransformer \
  --model_version <one of the names above> \
  --input_sensors raw_navigation_camera \
      nav_task_relevant_object_bbox nav_accurate_object_bbox \
  --detector_usage never \  # 只使用仿真器提供的 GT bbox
  ... (shared hyper-parameters)
```

## Evaluation

- Use the same `input_sensors` list when calling `training.offline.online_eval`（不再需要 `--keep_manip_camera`）。
- For FPS benchmarking, load the ckpt (baseline vs final) with this sensor bundle and run the shared `benchmark_fps.py` script (to be added) to log SR + FPS together.

Keep this file updated whenever we add new sparse variants so Table 2/3 experiments stay consistent.
