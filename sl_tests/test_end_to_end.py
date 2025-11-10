"""
测试 4: 端到端测试 - 验证完整的 Reward-Shifted Speculative Sampling 流程

这个测试验证：
1. 完整的生成流程能够正常运行
2. 输出是合理的（不崩溃、有输出）
3. 与标准 spec decode 的对比（输出可能不同，但都应该有效）
"""
import os
from vllm import LLM, SamplingParams

os.environ["HF_HOME"] = "/mnt/sdb/huggingface_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/mnt/sdb/huggingface_cache"

def test_end_to_end():
    """端到端测试"""
    print("=" * 60)
    print("测试 4: 端到端测试 - Reward-Shifted Speculative Sampling")
    print("=" * 60)
    
    prompts = [
        "The president of the United States is",
        "The future of artificial intelligence",
        "Once upon a time",
    ]
    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=20)
    
    # 测试 Reward-Shifted Speculative Sampling
    print("\n测试 A: Reward-Shifted Speculative Sampling (带 small_base)")
    llm_rsss = LLM(
        model="facebook/opt-6.7b",
        tensor_parallel_size=1,
        download_dir="/mnt/sdb/huggingface_cache",
        enforce_eager=True,
        speculative_config={
            "model": "facebook/opt-125m",  # draft/reasoning model
            "num_speculative_tokens": 5,
            "small_base_model": "facebook/opt-125m",  # small_base model
            "small_base_tensor_parallel_size": 1,
        },
    )
    
    print("生成中...")
    outputs_rsss = llm_rsss.generate(prompts, sampling_params)
    
    print("\n输出结果 (Reward-Shifted SS):")
    for i, output in enumerate(outputs_rsss):
        print(f"  [{i+1}] Prompt: {output.prompt!r}")
        print(f"      Generated: {output.outputs[0].text!r}")
        assert len(output.outputs[0].text) > 0, "输出不应为空"
    
    del llm_rsss
    
    # 测试标准 Speculative Decoding (对比)
    print("\n测试 B: 标准 Speculative Decoding (不带 small_base)")
    llm_standard = LLM(
        model="facebook/opt-6.7b",
        tensor_parallel_size=1,
        download_dir="/mnt/sdb/huggingface_cache",
        enforce_eager=True,
        speculative_config={
            "model": "facebook/opt-125m",
            "num_speculative_tokens": 5,
            # 不提供 small_base_model
        },
    )
    
    print("生成中...")
    outputs_standard = llm_standard.generate(prompts, sampling_params)
    
    print("\n输出结果 (标准 Spec Decode):")
    for i, output in enumerate(outputs_standard):
        print(f"  [{i+1}] Prompt: {output.prompt!r}")
        print(f"      Generated: {output.outputs[0].text!r}")
        assert len(output.outputs[0].text) > 0, "输出不应为空"
    
    del llm_standard
    
    # # 测试无 Spec Decode (基准)
    # print("\n测试 C: 无 Spec Decode (基准)")
    # llm_baseline = LLM(
    #     model="facebook/opt-6.7b",
    #     tensor_parallel_size=1,
    #     download_dir="/mnt/sdb/huggingface_cache",
    #     enforce_eager=True,
    #     # 不使用 speculative_config
    # )
    
    # print("生成中...")
    # outputs_baseline = llm_baseline.generate(prompts, sampling_params)
    
    # print("\n输出结果 (无 Spec Decode):")
    # for i, output in enumerate(outputs_baseline):
    #     print(f"  [{i+1}] Prompt: {output.prompt!r}")
    #     print(f"      Generated: {output.outputs[0].text!r}")
    #     assert len(output.outputs[0].text) > 0, "输出不应为空"
    
    # del llm_baseline
    
    # print("\n✓ 测试 4 通过: 端到端流程正常运行")
    # print("=" * 60)
    # print("\n注意: Reward-Shifted SS 的输出可能与标准 Spec Decode 不同，")
    # print("      这是预期的，因为 accept 和 recover 逻辑不同。")

if __name__ == "__main__":
    test_end_to_end()

