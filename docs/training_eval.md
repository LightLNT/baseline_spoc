# 训练 & 评测流程指南

> 目的：统一 Sparse 1+k 架构（含操作臂视角）在训练、在线评测与 FPS 基准测试时的指令与注意事项，保证表 2/表 3 实验可复现。

## 1. 环境准备

1. 激活 `spoc` conda 环境，并将仓库根目录加入 `PYTHONPATH`：
   ```bash
   conda activate spoc
   export PYTHONPATH=$(pwd):$PYTHONPATH
   ```
2. 数据与结果目录约定：
   - `--data_dir` 指向 `/home/zsf/data/fifteen`（或你自己的数据根目录）。
   - `--output_dir` 指向 `/home/zsf/baseline_spoc/imitation_result`。
3. 统一的传感器组合（训练 & 评测保持一致，ObjNav-only）：
   ```
   raw_navigation_camera
   nav_task_relevant_object_bbox
   nav_accurate_object_bbox
   ```
  所有消融实验均不再使用操作臂相机，命令中无需 `--keep_manip_camera`。

## 2. 离线训练指令

当前仓库已内置以下稀疏 1+k `model_version`，可直接在 CLI 中切换：

| 名称 | Backbone | 说明 |
| --- | --- | --- |
| `object_token_siglip_small_manip` | SigLIP + T5 | 表 3 baseline，SigLIP 视觉 + T5 文本。 |
| `object_token_dinov2_small_manip` | DINOv2 + T5 | 表 2 base / 表 3 final。 |
| `object_token_dinov2_small_no_cls` | DINOv2 + T5 | Variant 1，禁用 CLS（仅对象 token）。 |
| `object_token_dinov2_small_k0` | DINOv2 + T5 | Variant 2，`k=0` 仅保留导航 CLS。 |

以 `object_token_dinov2_small_manip` 为例，下面模板涵盖双相机输入、仿真器 GT bbox 与常用超参：

```bash
python -m training.offline.train_pl \
  --model EarlyFusionCnnTransformer \
  --model_version object_token_dinov2_small_manip \
  --loss action \
  --dataset_version ObjectNavType \
  --data_dir /home/zsf/data/fifteen \
  --output_dir /home/zsf/baseline_spoc/imitation_result \
  --per_gpu_batch 96 \
  --precision 16-mixed \
  --input_sensors raw_navigation_camera \
    nav_task_relevant_object_bbox nav_accurate_object_bbox \
  --detector_usage never \
  --wandb_logging True \
  --wandb_project_name <project> \
  --wandb_entity_name <entity>
```

关键点：
- `_prepare_input_sensors` 会在内部根据 `--keep_manip_camera`、`--detector_usage` 自动补齐/过滤 bbox 传感器，无需手动修改源码（设置 `--detector_usage never` 即表示完全依赖仿真器 GT bbox）。
- 若在本地调试可设置 `--wandb_logging False` 以使用 `LocalWandbLogger`，其它参数保持一致。

## 3. 在线评测（`training.offline.online_eval`）

在线评测沿用训练时的导航传感器集合：

```bash
python -m training.offline.online_eval \
  --training_run_id <wandb_run_id> \
  --ckptStep 13000 \
  --num_workers 4 \
  --eval_subset val \
  --task_type ObjectNavType \
  --dataset_type ObjectNavType \
  --dataset_path /home/zsf/data/fifteen \
  --input_sensors raw_navigation_camera \
    nav_task_relevant_object_bbox nav_accurate_object_bbox \
  --gpu_devices 0 1 \
  --wandb_logging True \
  --wandb_project_name <project> \
  --wandb_entity_name <entity>
```

说明：
- `online_eval` 会从对应 WandB run 读取训练配置并保持导航相机传感器，与训练期完全一致。
- 若使用本地 checkpoint，可将 `--wandb_logging False` 并设置 `--local_checkpoint_dir` 指向保存目录。

## 4. FPS 基准脚本

`scripts/benchmark_fps.py` 用于快速对比稀疏架构（例如表 3 baseline vs final）的 SR/FPS：

```bash
python scripts/benchmark_fps.py \
  --config imitation_result/local_eval_ckpts/zgl2koao/config.yaml \
  --ckpt_path imitation_result/local_eval_ckpts/zgl2koao/checkpoint_train_steps=26000.ckpt \
  --device cuda:0 \
  --num_batches 50 \
  --warmup_batches 5 \
  --data_subset val \
  --keep_manip_camera  # （可选）保留兼容性，如不需要可省略
```

脚本行为：
- 读取训练期 `config.yaml`，复用所有数据/模型参数，并可通过 CLI 覆盖 `model_version`、`input_sensors`、`detector_usage` 等关键信息。
- 构建 `LitModel` 与 `val` dataloader，跳过前 `--warmup_batches` 后开始计时；每个 batch 根据 `proc_batch["lengths"]` 统计帧数并输出 JSON：
  ```json
  {
    "ckpt": "...",           # 输入 checkpoint
    "model_version": "...",
    "num_batches": 45,
    "frames": 5400.0,
    "elapsed_sec": 27.3,
    "fps": 197.8
  }
  ```
- 通过 `--keep_manip_camera` 保证与训练同样的输入负载；若需要只测 baseline，可不加此 flag 以观察两路相机对 FPS 的影响。

## 5. 建议的实验记录

| 项目 | 记录内容 |
| --- | --- |
| 训练 | wandb run id、模型版本、`input_sensors`、`detector_usage`、`keep_manip_camera` 状态 |
| 在线评测 | `benchmark_revision`、`task_type`、`det_type`、`keep_manip_camera`、SR/成功率 |
| FPS | `scripts/benchmark_fps.py` 输出 JSON（保存到同一 wandb/表格中），备注 `num_batches` 与 GPU 型号 |

按照以上流程执行，可确保训练、评测与 FPS 基准完全对齐，并为表 2/表 3 填表提供可追踪的日志。