# BudVLN Training Code

本目录包含 BudVLN 的完整训练代码，基于 [StreamVLN](https://github.com/InternRobotics/StreamVLN) 开发，可独立运行。

## 📁 目录结构

```
opensource_training/                   # 项目根目录
├── README.md                          # 本文件
├── QUICKSTART.md                      # 快速开始指南
├── requirements.txt                   # Python依赖
│
├── scripts/                           # 启动脚本
│   ├── train_hybrid.sh                # ⭐ 混合训练（GRPO + SFT）推荐
│   ├── train_hybrid_resume.sh         # 从断点恢复训练
│   ├── train_grpo.sh                  # 纯GRPO训练
│   ├── train_sft_twophase_merged.sh   # 两阶段SFT训练
│   ├── merge_r2r_rxr_envdrop_scalevln.py  # 数据集合并工具
│   ├── streamvln_eval_multi_gpu.sh    # 多GPU评估
│   ├── zero2.json                     # DeepSpeed ZeRO-2 配置
│   └── zero3.json                     # DeepSpeed ZeRO-3 配置
│
├── config/                            # Habitat 环境配置
│   ├── vln_r2r_rxr.yaml               # R2R + RxR（推荐）
│   ├── vln_r2r.yaml                   # R2R 单数据集
│   ├── vln_merged_standard.yaml       # 四数据集合并
│   ├── vln_merged_fast.yaml           # 快速训练
│   └── ...
│
├── streamvln/                         # 核心训练代码
│   ├── streamvln_grpo_train.py        # GRPO训练入口
│   ├── streamvln_eval.py              # 评估脚本
│   ├── streamvln_agent.py             # Agent推理
│   ├── args.py                        # 参数定义
│   ├── model/                         # BudVLN模型
│   │   └── stream_video_vln.py
│   ├── rewards/                       # 奖励函数
│   │   └── vln_reward.py
│   ├── dataset/                       # 数据加载
│   │   └── vln_action_dataset.py
│   ├── habitat_extensions/            # Habitat自定义组件
│   │   ├── measures.py
│   │   └── maps.py
│   └── utils/                         # 工具函数
│       ├── utils.py
│       └── dist.py
│
├── llava/                             # LLaVA 多模态基座模型
│   ├── model/                         # 模型架构
│   │   ├── language_model/            # 语言模型（Qwen等）
│   │   ├── multimodal_encoder/        # 视觉编码器
│   │   ├── multimodal_projector/      # 多模态投影器
│   │   └── multimodal_resampler/      # 多模态重采样器
│   └── train/                         # 训练器
│       └── llava_trainer.py
│
├── trl/                               # 自定义TRL库（强化学习）
│   ├── trainer/                       # DPO/PPO/SFT Trainer
│   ├── models/                        # Value Head等
│   ├── extras/                        # 采样工具
│   └── environment/                   # RL环境基类
│
└── docs/                              # 详细文档
    ├── TRAINING_GUIDE.md              # 完整训练指南
    └── PARAMETERS_EXPLAINED.md        # 参数详解
```

## 🚀 快速开始

### 1. 安装依赖

```bash
# 创建 conda 环境
conda create -n budvln python=3.9
conda activate budvln

# 安装 habitat-sim
conda install habitat-sim==0.2.4 withbullet headless -c conda-forge -c aihabitat

# 安装 habitat-lab & habitat-baselines
git clone --branch v0.2.4 https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
pip install -e habitat-lab
pip install -e habitat-baselines
cd ..

# 安装 Python 依赖
pip install -r requirements.txt
```

### 2. 准备数据

你需要准备三类数据：场景数据、VLN-CE Episodes、预训练模型。

#### 2.1 场景数据

- **Matterport3D (MP3D)**：用于 R2R / RxR / EnvDrop。从 [Matterport3D 官方页面](https://niessner.github.io/Matterport/) 下载，放到 `data/scene_datasets/mp3d/`
- **HM3D**：用于 ScaleVLN。从 [HM3D 官方页面](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#habitat-matterport-3d-research-dataset-hm3d) 下载 `train` split，放到 `data/scene_datasets/hm3d/`

#### 2.2 VLN-CE Episodes

下载 VLN-CE episodes 并放到 `data/datasets/` 目录：

- [R2R](https://github.com/jacobkrantz/VLN-CE) — 重命名 `R2R_VLNCE_v1/` → `r2r/`
- [RxR](https://github.com/jacobkrantz/VLN-CE) — 重命名 `RxR_VLNCE_v0/` → `rxr/`
- EnvDrop — 重命名 `R2R_VLNCE_v1-3_preprocessed/envdrop/` → `envdrop/`
- ScaleVLN — VLN-CE 格式的子集，参考 [ScaleVLN 官方仓库](https://github.com/wz0919/ScaleVLN)

#### 2.3 预训练模型

```bash
mkdir -p checkpoints
# 将模型放到 checkpoints/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3
```

模型下载地址请参考 [StreamVLN Model Zoo](https://github.com/InternRobotics/StreamVLN#-model-zoo)。

#### 2.4 合并多数据集（可选）

```bash
python scripts/merge_r2r_rxr_envdrop_scalevln.py
```

#### 数据目录结构

准备完成后，目录结构应如下：

```
data/
├── datasets/
│   ├── r2r/
│   │   ├── train/
│   │   ├── val_seen/
│   │   │   └── val_seen.json.gz
│   │   └── val_unseen/
│   │       └── val_unseen.json.gz
│   ├── rxr/
│   │   ├── train/
│   │   ├── val_seen/
│   │   │   ├── val_seen_guide.json.gz
│   │   │   └── ...
│   │   └── val_unseen/
│   │       ├── val_unseen_guide.json.gz
│   │       └── ...
│   ├── envdrop/
│   │   ├── envdrop.json.gz
│   │   └── ...
│   └── scalevln/
│       └── scalevln_subset_150k.json.gz
├── scene_datasets/
│   ├── mp3d/
│   │   ├── 17DRP5sb8fy/
│   │   ├── 1LXtFkjw3qL/
│   │   └── ...
│   └── hm3d/
│       ├── 00000-kfPV7w3FaU5/
│       ├── 00001-UVdNNRcVyV1/
│       └── ...
└── trajectory_data/          # 可选，用于 SFT 训练
    ├── R2R/
    │   ├── images/
    │   └── annotations.json
    └── RxR/
        ├── images/
        └── annotations.json
```

### 3. 开始训练

```bash
# ⭐ 推荐：混合训练（GRPO + SFT）
bash scripts/train_hybrid.sh

# 纯GRPO训练
bash scripts/train_grpo.sh

# 两阶段SFT训练
bash scripts/train_sft_twophase_merged.sh

# 从断点恢复
# 先编辑 scripts/train_hybrid_resume.sh 设置检查点路径
bash scripts/train_hybrid_resume.sh
```

### 4. 评估模型

```bash
# 编辑 scripts/streamvln_eval_multi_gpu.sh 设置检查点路径
bash scripts/streamvln_eval_multi_gpu.sh
```

## 🔥 训练方法

### 混合训练（Hybrid Training）⭐ 推荐

结合 GRPO（强化学习）和 SFT（监督学习），核心公式：

$$\mathcal{L} = \mathcal{L}_{GRPO} + \lambda(t) \cdot \mathcal{L}_{SFT}$$

其中 $\lambda(t)$ 从 1.0 → 0.9 cosine 衰减。

**关键特性：**
- 🎯 GRPO 通过奖励信号优化策略
- 📚 SFT 从专家演示中学习
- 🛡️ 专家干预机制自动纠偏
- 📉 动态权重平衡两种学习信号

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use_hybrid_training` | - | 启用混合训练 |
| `--sft_loss_start_weight` | 1.0 | SFT初始权重 |
| `--sft_loss_end_weight` | 0.9 | SFT最终权重 |
| `--enable_recovery` | True | 启用专家干预 |
| `--offtrack_dist_thresh` | 3.0m | 偏离距离阈值 |
| `--num_updates` | 500 | 训练更新次数 |
| `--learning_rate` | 5e-7 | 学习率 |

更多参数说明请查看 [docs/PARAMETERS_EXPLAINED.md](docs/PARAMETERS_EXPLAINED.md)。

## ⚠️ 环境要求

- **GPU**: NVIDIA H800（≥40GB显存）
- **Python**: 3.9+
- **CUDA**: 11.7+
- **存储**: ≥500GB（场景数据 + 模型）

## 📝 引用

```bibtex
@article{he2026nipping,
  title={Nipping the Drift in the Bud: Retrospective Rectification for Robust Vision-Language Navigation},
  author={He, Gang and Liu, Zhenyang and Xu, Kepeng and Xu, Li and Qiao, Tong and Yu, Wenxin and Wu, Chang and Xie, Weiying},
  journal={arXiv preprint arXiv:2602.06356},
  year={2026}
}
```

## 📚 详细文档

- [QUICKSTART.md](QUICKSTART.md) — 5分钟上手
- [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md) — 完整训练指南
- [docs/PARAMETERS_EXPLAINED.md](docs/PARAMETERS_EXPLAINED.md) — 参数详解

## 📄 License

This work is under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](LICENSE).
