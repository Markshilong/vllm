#!/usr/bin/env python3
import sys
import os
import argparse

# 设置 HuggingFace 缓存目录到有更多空间的位置
# 必须在导入 transformers 之前设置，因为 transformers 在导入时会读取并缓存路径
os.environ["HF_HOME"] = "/mnt/sdb/huggingface_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/mnt/sdb/huggingface_cache"

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

# Model names from test_deepseek_reason.py
TARGET_NAME = "Qwen/Qwen2.5-Math-7B"
SMALL_BASE_NAME = "Qwen/Qwen2.5-Math-1.5B"
DRAFT_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

"""
对齐三个模型的 vocab_size，用于 Reward-Shifted Speculative Sampling：
- target (π_target): Qwen/Qwen2.5-Math-7B
- small_base (π_sft/π_small_base): Qwen/Qwen2.5-Math-1.5B
- draft (π_r): deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

将所有模型对齐到最大的 vocab_size，并保存到当前目录下。
"""

def parse_args():
    parser = argparse.ArgumentParser(
        description="Pad three models' vocabularies to match the largest vocab size"
    )
    parser.add_argument(
        "--target", type=str,
        default=TARGET_NAME,
        help=f"Target model name or path (default: {TARGET_NAME})"
    )
    parser.add_argument(
        "--small_base", type=str,
        default=SMALL_BASE_NAME,
        help=f"Small base model name or path (default: {SMALL_BASE_NAME})"
    )
    parser.add_argument(
        "--draft", type=str,
        default=DRAFT_NAME,
        help=f"Draft model name or path (default: {DRAFT_NAME})"
    )
    parser.add_argument(
        "--suffix", type=str, default="-padded",
        help="Suffix to append to the output directory (default: -padded)"
    )
    return parser.parse_args()


def load_tokenizer(name):
    print(f"Loading tokenizer for '{name}'...")
    return AutoTokenizer.from_pretrained(name, trust_remote_code=True)


def safety_check(tokenizers, tolerance=500):
    """检查所有 tokenizer 的前 N-tolerance 个 token 是否对齐"""
    vocab_dicts = [tok.get_vocab() for tok in tokenizers]
    N = min(len(vocab) for vocab in vocab_dicts) - tolerance
    print(f"Running safety check: comparing first {N} tokens across all models (tolerance={tolerance})...")
    
    id_to_token_lists = [{idx: tok for tok, idx in vocab.items()} for vocab in vocab_dicts]
    
    mismatch_count = 0
    for i in range(N):
        tokens = [id_to_token[i] for id_to_token in id_to_token_lists]
        if not all(t == tokens[0] for t in tokens):
            mismatch_count += 1
            print(f"⚠️ Token mismatch at ID {i}: {tokens}")
            if mismatch_count > 10:
                print("❌ Too many mismatches, exiting.")
                sys.exit(1)
    print("✅ Safety check passed for main vocab region.")


def pad_model(model_name, target_vocab_size, suffix):
    """将模型 padding 到目标 vocab_size"""
    print(f"\n{'='*80}")
    print(f"Processing model: {model_name}")
    print(f"{'='*80}")
    
    # Load config to get current vocab size
    print(f"Loading config for '{model_name}'...")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    current_size = config.vocab_size
    print(f"Current vocab size: {current_size}")
    
    if current_size == target_vocab_size:
        print(f"✅ Model '{model_name}' already has target vocab size. No padding needed.")
        return model_name
    
    # Load tokenizer
    tokenizer = load_tokenizer(model_name)
    
    # Load model
    print(f"Loading model for '{model_name}'...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, config=config, trust_remote_code=True
    )
    
    pad_size = target_vocab_size - current_size
    print(f"Padding from {current_size} to {target_vocab_size} tokens (+{pad_size})...")
    
    # Resize token embeddings
    print("Resizing token embeddings...")
    model.config.vocab_size = target_vocab_size
    model.resize_token_embeddings(target_vocab_size)
    
    # Add padding tokens to tokenizer
    print("Adding padding tokens to tokenizer...")
    pad_tokens = [f"<extra_pad_{i}>" for i in range(pad_size)]
    tokenizer.add_special_tokens({
        "additional_special_tokens": pad_tokens,
        "pad_token": pad_tokens[0] if pad_tokens else None
    })
    print(f"New tokenizer size: {tokenizer.vocab_size}")
    
    # Save padded model
    base_name = model_name.rstrip('/').split('/')[-1]
    output_dir = f"./{base_name}{suffix}"
    print(f"Saving padded model and tokenizer to '{output_dir}'...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"✅ Padding complete! Padded model available at: {output_dir}")
    return output_dir


def main():
    args = parse_args()
    
    print("=" * 80)
    print("对齐三个模型的 vocab_size")
    print("=" * 80)
    print(f"Target model: {args.target}")
    print(f"Small base model: {args.small_base}")
    print(f"Draft model: {args.draft}")
    print("=" * 80)
    
    # Load configs to get vocab sizes
    print("\n检查各模型的 vocab_size...")
    configs = []
    model_names = [args.target, args.small_base, args.draft]
    for name in model_names:
        print(f"Loading config for '{name}'...")
        config = AutoConfig.from_pretrained(name, trust_remote_code=True)
        configs.append(config)
        print(f"  Vocab size: {config.vocab_size}")
    
    vocab_sizes = [config.vocab_size for config in configs]
    max_vocab_size = max(vocab_sizes)
    print(f"\n最大 vocab_size: {max_vocab_size}")
    
    # Load tokenizers and verify alignment
    print("\n加载 tokenizers 并验证对齐...")
    tokenizers = [load_tokenizer(name) for name in model_names]
    # 使用较大的 tolerance，允许末尾几百个特殊 token 不匹配
    safety_check(tokenizers, tolerance=500)
    
    # Pad all models to max vocab size
    print(f"\n将所有模型对齐到 vocab_size: {max_vocab_size}")
    padded_paths = {}
    
    for i, (name, current_size) in enumerate(zip(model_names, vocab_sizes)):
        if current_size < max_vocab_size:
            padded_path = pad_model(name, max_vocab_size, args.suffix)
            padded_paths[name] = padded_path
        else:
            print(f"\n{'='*80}")
            print(f"处理模型: {name}")
            print(f"{'='*80}")
            print(f"模型 '{name}' 已经是最大 vocab_size ({current_size})，无需 padding。")
            # 即使不需要 padding，也保存到当前目录以便统一使用
            base_name = name.rstrip('/').split('/')[-1]
            output_dir = f"./{base_name}{args.suffix}"
            print(f"复制模型到 '{output_dir}'...")
            # 加载并保存模型和 tokenizer
            config = AutoConfig.from_pretrained(name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                name, config=config, trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            padded_paths[name] = output_dir
            print(f"✅ 模型已保存到: {output_dir}")
    
    print("\n" + "=" * 80)
    print("✅ 所有模型对齐完成！")
    print("=" * 80)
    print("\n对齐后的模型路径：")
    for original, padded in padded_paths.items():
        print(f"  {original} -> {padded}")
    print("=" * 80)

if __name__ == "__main__":
    main()

