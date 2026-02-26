# BudVLN Training Guide

本指南提供 BudVLN 训练的完整流程说明。

## 🎯 训练流程概览

```
1. 环境准备 → 2. 数据准备 → 3. 模型训练 → 4. 模型评估 → 5. 部署
```

---

## 1️⃣ 环境准备

### 系统要求

- **操作系统**: Linux（推荐Ubuntu 20.04+）
- **GPU**: NVIDIA V100/A100（40GB显存）
- **CUDA**: 11.7+
- **Python**: 3.9+
- **存储空间**: 至少500GB

### 安装依赖

```bash
# 克隆仓库
git clone https://github.com/lzy2208929036/BudVLN.git
cd BudVLN

# 安装Python依赖
pip install -r opensource_training/requirements.txt

# 安装Habitat-Lab
cd habitat-lab
pip install -e .
cd ..
```

### 配置WandB（可选但推荐）

```bash
# 登录WandB
wandb login

# 或设置环境变量
export WANDB_API_KEY="your_api_key"
```

---

## 2️⃣ 数据准备

### 下载数据集

```bash
# 下载Matterport3D场景数据
# 请访问: https://niessner.github.io/Matterport/

# 下载VLN数据集
mkdir -p data/datasets
cd data/datasets

# R2R数据集
wget https://www.dropbox.com/s/.../R2R_train.json
wget https://www.dropbox.com/s/.../R2R_val_seen.json
wget https://www.dropbox.com/s/.../R2R_val_unseen.json

# RxR数据集
wget https://storage.googleapis.com/rxr-datasets/rxr_marky_train_guide.jsonl.gz
# 下载其他RxR数据...
```

### 合并数据集

```bash
cd opensource_training/scripts
python merge_r2r_rxr_envdrop_scalevln.py \
    --output_dir ../../data/merged_datasets
```

### 下载预训练模型

```bash
# 从Hugging Face下载
mkdir -p checkpoints
cd checkpoints

# 下载预训练模型
# 具体下载方式请参考主README
```

---

## 3️⃣ 模型训练

### 方案A: 混合训练（推荐）

**最佳性能，收敛最快**

```bash
cd opensource_training/scripts
bash train_hybrid.sh
```

**训练特点：**
- 结合GRPO（强化学习）和SFT（监督学习）
- 动态权重衰减：SFT权重从1.0逐渐降到0.9
- 专家干预机制自动纠正偏离轨迹
- 训练时间：约3-4天（单GPU A100）

**关键参数调整：**

```bash
# 在 train_hybrid.sh 中修改：

# 加快训练速度（牺牲性能）
--num_updates 300
--group_size 1

# 提高性能（增加训练时间）
--num_updates 800
--group_size 3
--sft_loss_start_weight 1.5
```

### 方案B: 纯GRPO训练

**用于消融研究**

```bash
bash train_grpo.sh
```

### 方案C: 两阶段SFT训练

**更稳定但可能性能略低**

```bash
bash train_sft_twophase_merged.sh
```

### 断点恢复

如果训练中断，可以从检查点恢复：

```bash
# 1. 编辑 train_hybrid_resume.sh
# 修改: RESUME_CHECKPOINT="result/your_checkpoint/checkpoint_XXX"

# 2. 运行恢复脚本
bash train_hybrid_resume.sh
```

---

## 4️⃣ 训练监控

### 实时日志

```bash
# 查看实时训练日志
tail -f result/grpo_hybrid_trainingV10_multi_dataset/training.log

# 查看关键指标
grep "Update.*SR:" result/*/training.log
```

### WandB可视化

训练会自动上传到WandB：

1. 查看训练日志中的WandB链接：
```bash
grep "View run at" result/*/training.log
```

2. 关键指标：
   - `train/success_rate`: 成功率
   - `train/spl`: SPL指标
   - `train/oracle_rate`: 专家干预率
   - `train/grpo_loss`: GRPO损失
   - `train/sft_loss`: SFT损失
   - `train/sft_weight`: 当前SFT权重

### 检查点管理

```bash
# 列出所有检查点
ls -lh result/your_output_dir/checkpoint_*

# 查看最新检查点
ls -t result/your_output_dir/checkpoint_* | head -1
```

---

## 5️⃣ 模型评估

### 单GPU评估

```bash
python streamvln/streamvln_eval.py \
    --model_path result/your_checkpoint/checkpoint_XXX \
    --habitat_config_path config/vln_r2r_rxr.yaml \
    --split val_unseen
```

### 多GPU评估（推荐）

```bash
cd opensource_training/scripts

# 编辑 streamvln_eval_multi_gpu.sh 设置：
# - CHECKPOINT_PATH
# - NUM_GPUS

bash streamvln_eval_multi_gpu.sh
```

### 评估结果

评估完成后会生成：
- `eval_results.json`: 详细结果
- 控制台输出关键指标：
  - Success Rate (SR)
  - Success weighted by Path Length (SPL)
  - Oracle Success Rate (OSR)
  - Navigation Error (NE)

---

## 🎓 训练最佳实践

### 1. 训练策略

**第一阶段（前100-200次更新）：**
- 高SFT权重（1.0-1.5）
- 强专家干预（dist_thresh=3.0）
- 让模型快速学习基本导航

**第二阶段（200-500次更新）：**
- 降低SFT权重（0.5-0.9）
- 让GRPO主导学习
- 模型自主探索

**第三阶段（500+次更新）：**
- 稳定SFT权重（0.5-0.8）
- 性能平台期
- 选择最佳检查点

### 2. 超参数调优

**学习率：**
- 从 5e-7 开始
- 如果不收敛，降到 1e-7
- 如果收敛太慢，提到 1e-6

**SFT权重衰减：**
- 快速实验：100-200 updates
- 标准训练：400-600 updates
- 稳定训练：800+ updates

**专家干预强度：**
```bash
# 强干预（新手模型）
--offtrack_dist_thresh 2.0
--offtrack_patience 5

# 中等干预（标准）
--offtrack_dist_thresh 3.0
--offtrack_patience 8

# 弱干预（成熟模型）
--offtrack_dist_thresh 5.0
--offtrack_patience 12
```

### 3. 数据集选择

| 配置文件 | 数据集 | 训练速度 | 性能 | 推荐场景 |
|---------|--------|---------|------|---------|
| `vln_r2r.yaml` | R2R | 快 | 基准 | 快速验证 |
| `vln_r2r_rxr.yaml` | R2R+RxR | 中 | 更好 | 标准训练 |
| `vln_merged_standard.yaml` | 4个数据集 | 慢 | 最佳 | 完整训练 |
| `vln_merged_fast.yaml` | 采样版 | 中 | 好 | 平衡方案 |

---

## 🔧 故障排除

### 问题1: CUDA Out of Memory

**解决方案：**
```bash
# 方法1: 降低批量大小
--mini_batch_size 1
--group_size 1

# 方法2: 使用梯度累积
--gradient_accumulation_steps 2

# 方法3: 启用DeepSpeed ZeRO-2
--use_deepspeed
--deepspeed_config scripts/zero2.json
```

### 问题2: 训练不收敛

**检查清单：**
1. ✅ SFT权重是否足够高（start_weight >= 1.0）
2. ✅ 专家干预是否正常触发（oracle_rate 10-30%）
3. ✅ 学习率是否合适（尝试5e-7）
4. ✅ 数据集是否正确加载

**调试命令：**
```bash
# 检查专家干预率
grep "Oracle Rate" result/*/training.log

# 检查损失曲线
grep "GRPO Loss\|SFT Loss" result/*/training.log | tail -20
```

### 问题3: 专家干预率过高（>50%）

**可能原因：**
- 阈值设置过严
- 模型质量差

**解决方案：**
```bash
# 放宽阈值
--offtrack_dist_thresh 4.0
--offtrack_patience 12

# 或先进行SFT预训练
bash train_sft_twophase_merged.sh
```

### 问题4: 专家干预率过低（<5%）

**可能原因：**
- 阈值设置过松
- 模型已经很好（不是问题）

**解决方案：**
```bash
# 收紧阈值
--offtrack_dist_thresh 2.5
--offtrack_patience 5
```

---

## 📦 训练输出

每次训练会生成：

```
result/your_output_dir/
├── training.log              # 完整训练日志
├── checkpoint_10/            # 每10次更新保存
├── checkpoint_20/
├── ...
└── checkpoint_XXX/          # 最佳检查点
    ├── adapter_config.json
    ├── adapter_model.bin    # LoRA权重
    └── trainer_state.json
```

---

## 🚀 进阶主题

### 自定义数据集

1. 准备数据格式（参考R2R格式）
2. 创建Habitat配置文件
3. 修改训练脚本中的 `HABITAT_CONFIG`

### 多卡训练

```bash
# 使用DeepSpeed
deepspeed --num_gpus=4 streamvln/streamvln_grpo_train.py \
    --use_deepspeed \
    --deepspeed_config scripts/zero3.json \
    # 其他参数...
```

### 训练监控脚本

```bash
# 创建监控脚本
cat > watch_training.sh << 'EOF'
#!/bin/bash
while true; do
    clear
    echo "=== Latest Training Progress ==="
    tail -50 result/your_output_dir/training.log | grep -E "Update|SR:|SPL:"
    echo ""
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
    sleep 60
done
EOF

chmod +x watch_training.sh
./watch_training.sh
```

---

## 📚 相关文档

- [PARAMETERS_EXPLAINED.md](PARAMETERS_EXPLAINED.md) - 参数详解
- [../README.md](../README.md) - 主README
- [../../README.md](../../README.md) - 项目README

---

## 💬 获取帮助

- **GitHub Issues**: 提交bug报告或功能请求
- **讨论**: GitHub Discussions
- **邮件**: 联系论文作者

---

**祝训练顺利！🎉**
