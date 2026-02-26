import gzip
import json
import os
import argparse

def load_dataset(path, dataset_name):
    """加载数据集并处理可能的格式问题"""
    print(f"Loading {dataset_name} from {path}...")
    if not os.path.exists(path):
        print(f"⚠️  Warning: {path} not found, skipping {dataset_name}")
        return None
    
    with gzip.open(path, 'rt') as f:
        data = json.load(f)
    
    if 'episodes' not in data:
        print(f"⚠️  Warning: {dataset_name} has no 'episodes' key, skipping")
        return None
    
    print(f"✓ Loaded {len(data['episodes'])} episodes from {dataset_name}")
    return data

def process_episodes(episodes, prefix, dataset_name):
    """统一处理 episodes 格式"""
    processed = []
    
    for ep in episodes:
        # 确保 episode_id 唯一
        ep['episode_id'] = f"{prefix}_{ep['episode_id']}"
        
        # 确保有 trajectory_id
        if 'trajectory_id' not in ep:
            ep['trajectory_id'] = ep['episode_id']
        
        # 统一 instruction 格式
        if 'instruction' in ep and isinstance(ep['instruction'], dict):
            instruction = ep['instruction']
            ep['instruction'] = {
                'instruction_text': instruction.get('instruction_text', ''),
                'instruction_tokens': instruction.get('instruction_tokens', [])
            }
        
        processed.append(ep)
    
    return processed

def merge_datasets(r2r_path=None, rxr_path=None, envdrop_path=None, scalevln_path=None, output_path=None):
    """
    合并多个 VLN 数据集
    
    Args:
        r2r_path: R2R 数据集路径
        rxr_path: RxR 数据集路径
        envdrop_path: EnvDrop 数据集路径
        scalevln_path: ScaleVLN 数据集路径
        output_path: 输出文件路径
    """
    merged_episodes = []
    instruction_vocab = {}
    
    # 加载并处理各个数据集
    datasets = [
        (r2r_path, "r2r", "R2R"),
        (rxr_path, "rxr", "RxR"),
        (envdrop_path, "envdrop", "EnvDrop"),
        (scalevln_path, "scalevln", "ScaleVLN")
    ]
    
    for path, prefix, name in datasets:
        if path is None:
            continue
        
        data = load_dataset(path, name)
        if data is None:
            continue
        
        # 处理 episodes
        processed = process_episodes(data['episodes'], prefix, name)
        merged_episodes.extend(processed)
        
        # 保留第一个数据集的 instruction_vocab（如果有）
        if not instruction_vocab and 'instruction_vocab' in data:
            instruction_vocab = data['instruction_vocab']
    
    if not merged_episodes:
        print("❌ Error: No episodes found in any dataset!")
        return
    
    # 创建合并数据
    merged_data = {
        'episodes': merged_episodes,
        'instruction_vocab': instruction_vocab
    }
    
    print(f"\n{'='*60}")
    print(f"📊 Total episodes: {len(merged_episodes)}")
    print(f"{'='*60}")
    
    # 保存
    print(f"\n💾 Saving to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with gzip.open(output_path, 'wt') as f:
        json.dump(merged_data, f)
    
    print("✅ Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="合并多个 VLN 数据集")
    
    # 方案一：标准四数据集整合（推荐）
    parser.add_argument("--r2r", type=str, 
                        default="data/datasets/r2r/train/train.json.gz",
                        help="R2R 训练数据路径")
    parser.add_argument("--rxr", type=str, 
                        default="data/datasets/rxr/train/train_follower_en.json.gz",
                        help="RxR 训练数据路径（建议用 train_follower_en.json.gz）")
    parser.add_argument("--envdrop", type=str, 
                        default="data/datasets/envdrop/train/train.json.gz",
                        help="EnvDrop 训练数据路径（train.json.gz 或 envdrop/envdrop.json.gz）")
    parser.add_argument("--scalevln", type=str, 
                        default="data/datasets/scalevln/scalevln_subset_150k.json.gz",
                        help="ScaleVLN 训练数据路径")
    parser.add_argument("--output", type=str, 
                        default="data/datasets/merged_train/train/train.json.gz",
                        help="输出文件路径")
    
    # 可选：不合并某个数据集（设为 None）
    parser.add_argument("--skip-rxr", action="store_true", help="跳过 RxR 数据集")
    parser.add_argument("--skip-envdrop", action="store_true", help="跳过 EnvDrop 数据集")
    parser.add_argument("--skip-scalevln", action="store_true", help="跳过 ScaleVLN 数据集")
    
    args = parser.parse_args()
    
    # 应用跳过选项
    rxr_path = None if args.skip_rxr else args.rxr
    envdrop_path = None if args.skip_envdrop else args.envdrop
    scalevln_path = None if args.skip_scalevln else args.scalevln
    
    print("🚀 开始合并数据集...")
    print(f"方案配置:")
    print(f"  - R2R: {args.r2r}")
    print(f"  - RxR: {rxr_path or '(跳过)'}")
    print(f"  - EnvDrop: {envdrop_path or '(跳过)'}")
    print(f"  - ScaleVLN: {scalevln_path or '(跳过)'}")
    print(f"  - Output: {args.output}\n")
    
    merge_datasets(
        r2r_path=args.r2r,
        rxr_path=rxr_path,
        envdrop_path=envdrop_path,
        scalevln_path=scalevln_path,
        output_path=args.output
    )
