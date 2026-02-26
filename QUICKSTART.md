# Training Code Quick Start

快速启动BudVLN训练的最简步骤。

## ⚡ 5分钟开始训练

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 准备数据（假设已下载Matterport3D和VLN数据集）
cd scripts
python merge_r2r_rxr_envdrop_scalevln.py

# 3. 开始训练
bash train_hybrid.sh
```

## 📋 训练前检查清单

- [ ] 已安装所有Python依赖
- [ ] 已下载Matterport3D场景数据
- [ ] 已下载VLN数据集（R2R/RxR等）
- [ ] 已下载预训练模型到 `checkpoints/`
- [ ] GPU显存足够（建议40GB+）
- [ ] 已配置WandB（可选）

## 🎯 主要训练脚本

| 脚本 | 说明 | 训练时间 | 推荐 |
|------|------|----------|------|
| `train_hybrid.sh` | GRPO+SFT混合训练 | 3-4天 | ⭐⭐⭐⭐⭐ |
| `train_hybrid_resume.sh` | 断点恢复 | 视情况 | ⭐⭐⭐⭐ |
| `train_grpo.sh` | 纯强化学习 | 4-5天 | ⭐⭐⭐ |
| `train_sft_twophase_merged.sh` | 监督学习 | 2-3天 | ⭐⭐⭐ |

## 📝 最小配置示例

如果显存有限，可以使用最小配置：

```bash
# 编辑 train_hybrid.sh，修改以下参数：
--num_episodes_per_update 1    # 减少episode数
--group_size 1                 # 减少group size
--mini_batch_size 1            # 减少批量大小
--num_updates 300              # 减少总更新次数
```

## 🔍 验证训练是否正常

训练开始后，检查以下内容：

```bash
# 1. 检查GPU使用
nvidia-smi

# 2. 查看日志（应该看到更新进度）
tail -f result/grpo_hybrid_trainingV10_multi_dataset/training.log

# 3. 查看WandB链接（复制到浏览器）
grep "View run at" result/*/training.log | tail -1
```

**正常训练的标志：**
- ✅ GPU利用率 > 80%
- ✅ 日志中出现 "Update X/500"
- ✅ 专家干预率在10-30%之间
- ✅ 成功率逐渐提高

## ⚠️ 常见错误

### 错误1: 找不到数据集
```
FileNotFoundError: data/datasets/R2R_VLNCE_v1-3/train/...
```
**解决**: 运行数据集合并脚本或检查数据路径

### 错误2: 显存溢出
```
RuntimeError: CUDA out of memory
```
**解决**: 降低 `mini_batch_size` 和 `group_size`

### 错误3: 找不到模型
```
FileNotFoundError: checkpoints/StreamVLN_Video_...
```
**解决**: 下载预训练模型到正确路径

## 📞 获取帮助

- 详细参数说明：[docs/PARAMETERS_EXPLAINED.md](docs/PARAMETERS_EXPLAINED.md)
- 完整训练指南：[docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md)
- GitHub Issues: 提交问题

---

**开始你的BudVLN训练之旅吧！** 🚀
