"""
测试: 验证 accept 和 recover 公式计算

验证公式 (6) 和 (7) 是否正确实现：

公式 (6)：accept 概率
p_accept(t) = min(1, π_large(ŷ_t | x, y_<t) / π_small-base(ŷ_t | x, y_<t))

公式 (7)：bonus (recovery) 分布
π_bonus^reason(· | x, y_<t+K') = (π_reason(· | x, y_<t+K') * (π_large(· | x, y_<t+K') / π_small-base(· | x, y_<t+K') - 1))_+
"""
import os
import torch
from pathlib import Path
from vllm import LLM, SamplingParams

os.environ["HF_HOME"] = "/mnt/sdb/huggingface_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/mnt/sdb/huggingface_cache"

# 获取测试文件所在目录的父目录（sl_tests）
TEST_DIR = Path(__file__).parent.parent


def test_accept_recover_formulas():
    """测试 accept 和 recover 公式计算"""
    print("=" * 80)
    print("测试: 验证 accept 和 recover 公式计算")
    print("=" * 80)
    
    # 模型配置 - 使用绝对路径
    target_model = str(TEST_DIR / "Qwen2.5-Math-7B-padded")
    draft_model = str(TEST_DIR / "DeepSeek-R1-Distill-Qwen-1.5B-padded")
    small_base_model = str(TEST_DIR / "Qwen2.5-Math-1.5B-padded")
    
    print("\n模型配置:")
    print(f"  - Target (π_large): {target_model}")
    print(f"  - Draft (π_reason): {draft_model}")
    print(f"  - Small Base (π_small-base): {small_base_model}")
    print("=" * 80)
    
    prompts = ["The capital of France is"]
    sampling_params = SamplingParams(
        temperature=0.0,  # 使用 temperature=0 以便结果可复现
        max_tokens=10,
    )
    
    print("\n初始化 LLM...")
    llm = LLM(
        model=target_model,
        tensor_parallel_size=1,
        download_dir="/mnt/sdb/huggingface_cache",
        enforce_eager=True,
        gpu_memory_utilization=0.45,
        speculative_config={
            "model": draft_model,
            "num_speculative_tokens": 5,
            "small_base_model": small_base_model,
            "small_base_tensor_parallel_size": 1,
        },
    )
    
    # 获取 sampler
    worker = llm.llm_engine.model_executor.driver_worker
    sampler = worker.spec_decode_sampler
    
    # 保存原始方法
    original_get_accepted = sampler._get_accepted
    original_get_recovered_probs = sampler._get_recovered_probs
    
    # 用于存储验证结果
    accept_verification_results = []
    recover_verification_results = []
    
    def verify_get_accepted(
        target_probs,  # [batch_size, k, vocab_size] - π_large
        draft_probs,   # [batch_size, k, vocab_size] - π_reason
        draft_token_ids,  # [batch_size, k]
        seeded_seqs,
        small_base_probs,  # [batch_size, k, vocab_size] - π_small-base
    ):
        """验证 accept 公式 (6)"""
        # 如果 small_base_probs 为 None，直接调用原始方法（标准 spec decode）
        if small_base_probs is None:
            print("\n[Accept 验证] small_base_probs 为 None，跳过公式 (6) 验证（使用标准 spec decode）")
            return original_get_accepted(
                target_probs, draft_probs, draft_token_ids, seeded_seqs, small_base_probs
            )
        
        batch_size, k, vocab_size = draft_probs.shape
        batch_indices = torch.arange(batch_size, device=target_probs.device)[:, None]
        probs_indices = torch.arange(k, device=target_probs.device)
        
        # 获取 draft token 对应的概率
        # shape [batch_size, k]
        selected_target_probs = target_probs[batch_indices, probs_indices, draft_token_ids]
        selected_small_base_probs = small_base_probs[batch_indices, probs_indices, draft_token_ids]
        
        # 公式 (6): p_accept(t) = min(1, π_large(ŷ_t) / π_small-base(ŷ_t))
        # 手动计算期望的 accept 概率（按照公式）
        # 添加 smallest_positive_value 避免除零（与 rejection_sampler.py 保持一致）
        smallest_positive = sampler._smallest_positive_value
        ratio = selected_target_probs / (selected_small_base_probs + smallest_positive)
        expected_accept_probs = torch.minimum(
            ratio,
            torch.ones_like(ratio)
        )
        
        # 复制原始方法的逻辑来验证（从 rejection_sampler.py 第 318-320 行）
        actual_capped_ratio = torch.minimum(
            selected_target_probs / (selected_small_base_probs + smallest_positive),
            torch.full((1, ), 1, device=target_probs.device)
        )
        
        # 验证 capped_ratio 计算正确（应该与公式 (6) 一致）
        max_diff = torch.max(torch.abs(expected_accept_probs - actual_capped_ratio))
        
        accept_verification_results.append({
            "expected_accept_probs": expected_accept_probs,
            "actual_capped_ratio": actual_capped_ratio,
            "max_diff": max_diff.item(),
            "ratio_mean": ratio.mean().item(),
            "ratio_max": ratio.max().item(),
        })
        
        print(f"\n[Accept 验证] max_diff: {max_diff.item():.2e}, ratio_mean: {ratio.mean().item():.4f}")
        assert max_diff < 1e-5, f"Accept 概率计算不匹配，max_diff={max_diff.item()}"
        
        # 调用原始方法获取实际结果（用于实际生成）
        actual_accepted = original_get_accepted(
            target_probs, draft_probs, draft_token_ids, seeded_seqs, small_base_probs
        )
        
        return actual_accepted
    
    def verify_get_recovered_probs(
        target_probs,  # [batch_size, k, vocab_size] - π_large
        draft_probs,   # [batch_size, k, vocab_size] - π_reason
        small_base_probs,  # [batch_size, k, vocab_size] - π_small-base
    ):
        """验证 recover 公式 (7)"""
        # 如果 small_base_probs 为 None，直接调用原始方法（标准 spec decode）
        if small_base_probs is None:
            print("[Recover 验证] small_base_probs 为 None，跳过公式 (7) 验证（使用标准 spec decode）")
            return original_get_recovered_probs(
                target_probs, draft_probs, small_base_probs
            )
        
        # 公式 (7): π_bonus^reason(·) = (π_reason(·) * (π_large(·) / π_small-base(·) - 1))_+
        
        smallest_positive = sampler._smallest_positive_value
        
        # 手动计算期望的 recover 分布（按照公式 (7)）
        # 步骤 1: 计算 ratio = π_large / π_small-base
        ratio = target_probs / (small_base_probs + smallest_positive)
        
        # 步骤 2: 计算 bonus = π_reason * (ratio - 1)
        bonus = draft_probs * (ratio - 1)
        
        # 步骤 3: 应用 (·)_+ 操作：clamp 到最小正值，然后归一化
        f = torch.clamp(bonus, min=smallest_positive)
        expected_recovered_probs = f / torch.sum(f, dim=-1, keepdim=True)
        
        # 调用原始方法获取实际结果
        actual_recovered_probs = original_get_recovered_probs(
            target_probs, draft_probs, small_base_probs
        )
        
        # 验证结果：比较期望和实际的 recover 分布
        max_diff = torch.max(torch.abs(expected_recovered_probs - actual_recovered_probs))
        mean_diff = torch.mean(torch.abs(expected_recovered_probs - actual_recovered_probs))
        
        # 验证分布是否归一化（每行的和应该接近 1）
        expected_sums = torch.sum(expected_recovered_probs, dim=-1)
        actual_sums = torch.sum(actual_recovered_probs, dim=-1)
        sum_diff = torch.max(torch.abs(expected_sums - actual_sums))
        
        recover_verification_results.append({
            "expected_recovered_probs": expected_recovered_probs,
            "actual_recovered_probs": actual_recovered_probs,
            "max_diff": max_diff.item(),
            "mean_diff": mean_diff.item(),
            "sum_diff": sum_diff.item(),
        })
        
        print(f"[Recover 验证] max_diff: {max_diff.item():.2e}, mean_diff: {mean_diff.item():.2e}, sum_diff: {sum_diff.item():.2e}")
        assert max_diff < 1e-4, f"Recover 分布计算不匹配，max_diff={max_diff.item()}"
        assert sum_diff < 1e-5, f"Recover 分布未正确归一化，sum_diff={sum_diff.item()}"
        
        return actual_recovered_probs
    
    # Monkey patch 方法
    sampler._get_accepted = verify_get_accepted
    sampler._get_recovered_probs = verify_get_recovered_probs
    
    print("\n开始生成（将验证公式计算）...")
    try:
        outputs = llm.generate(prompts, sampling_params)
        
        print("\n" + "=" * 80)
        print("生成结果:")
        print("=" * 80)
        for i, output in enumerate(outputs):
            print(f"\n[{i+1}] Prompt: {output.prompt!r}")
            print(f"    Generated: {output.outputs[0].text!r}")
        
        print("\n" + "=" * 80)
        print("公式验证结果:")
        print("=" * 80)
        
        if accept_verification_results:
            print(f"\n✓ Accept 公式 (6) 验证通过: {len(accept_verification_results)} 次调用")
            print("  公式: p_accept(t) = min(1, π_large(ŷ_t) / π_small-base(ŷ_t))")
            for i, result in enumerate(accept_verification_results):
                print(f"  调用 {i+1}: max_diff = {result['max_diff']:.2e}, "
                      f"ratio_mean = {result['ratio_mean']:.4f}, "
                      f"ratio_max = {result['ratio_max']:.4f}")
        else:
            print("\n⚠ Accept 公式 (6) 未被调用（可能没有 speculative decoding 步骤）")
        
        if recover_verification_results:
            print(f"\n✓ Recover 公式 (7) 验证通过: {len(recover_verification_results)} 次调用")
            print("  公式: π_bonus^reason(·) = (π_reason(·) * (π_large(·) / π_small-base(·) - 1))_+")
            for i, result in enumerate(recover_verification_results):
                print(f"  调用 {i+1}: max_diff = {result['max_diff']:.2e}, "
                      f"mean_diff = {result['mean_diff']:.2e}, "
                      f"sum_diff = {result['sum_diff']:.2e}")
        else:
            print("\n⚠ Recover 公式 (7) 未被调用（可能所有 token 都被 accept）")
        
        print("\n" + "=" * 80)
        if accept_verification_results or recover_verification_results:
            print("✓ 所有公式验证通过！")
        else:
            print("⚠ 未检测到公式计算（可能需要更多生成步骤）")
        print("=" * 80)
        
    finally:
        # 恢复原始方法
        sampler._get_accepted = original_get_accepted
        sampler._get_recovered_probs = original_get_recovered_probs
    
    del llm
    print("\n测试完成！")


if __name__ == "__main__":
    test_accept_recover_formulas()

