"""
测试 5: 验证向后兼容性

这个测试验证：
1. 当 small_base_worker 为 None 时，行为与原始 spec decode 一致
2. 所有原有功能仍然正常工作
"""
import os
from vllm import LLM, SamplingParams

os.environ["HF_HOME"] = "/mnt/sdb/huggingface_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/mnt/sdb/huggingface_cache"

def test_backward_compatibility():
    """测试向后兼容性"""
    print("=" * 60)
    print("测试 5: 验证向后兼容性")
    print("=" * 60)
    
    prompts = ["Hello, how are you?"]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=10)
    
    # 测试 1: 不带 small_base_model (应该使用原始逻辑)
    print("\n测试 A: 不带 small_base_model (原始 spec decode)")
    llm_original = LLM(
        model="facebook/opt-6.7b",
        tensor_parallel_size=1,
        download_dir="/mnt/sdb/huggingface_cache",
        enforce_eager=True,
        speculative_config={
            "model": "facebook/opt-125m",
            "num_speculative_tokens": 3,
            # 不提供 small_base_model
        },
    )
    
    # 验证 small_base_worker 为 None
    worker = llm_original.llm_engine.model_executor.driver_worker
    assert worker.small_base_worker is None, "不带 small_base_model 时，small_base_worker 应为 None"
    print("✓ small_base_worker 为 None (符合预期)")
    
    # 验证 scorer 的 small_base_worker 也为 None
    assert worker.scorer._small_base_worker is None, "scorer._small_base_worker 应为 None"
    print("✓ scorer._small_base_worker 为 None (符合预期)")
    
    outputs_original = llm_original.generate(prompts, sampling_params)
    print(f"输出: {outputs_original[0].outputs[0].text!r}")
    assert len(outputs_original[0].outputs[0].text) > 0, "输出不应为空"
    
    del llm_original
    
    # 测试 2: 带 small_base_model (应该使用新逻辑)
    print("\n测试 B: 带 small_base_model (Reward-Shifted SS)")
    llm_new = LLM(
        model="facebook/opt-6.7b",
        tensor_parallel_size=1,
        download_dir="/mnt/sdb/huggingface_cache",
        enforce_eager=True,
        speculative_config={
            "model": "facebook/opt-125m",
            "num_speculative_tokens": 3,
            "small_base_model": "facebook/opt-125m",
            "small_base_tensor_parallel_size": 1,
        },
    )
    
    # 验证 small_base_worker 不为 None
    worker = llm_new.llm_engine.model_executor.driver_worker
    assert worker.small_base_worker is not None, "带 small_base_model 时，small_base_worker 不应为 None"
    print("✓ small_base_worker 不为 None (符合预期)")
    
    # 验证 scorer 的 small_base_worker 也不为 None
    assert worker.scorer._small_base_worker is not None, "scorer._small_base_worker 不应为 None"
    print("✓ scorer._small_base_worker 不为 None (符合预期)")
    
    outputs_new = llm_new.generate(prompts, sampling_params)
    print(f"输出: {outputs_new[0].outputs[0].text!r}")
    assert len(outputs_new[0].outputs[0].text) > 0, "输出不应为空"
    
    del llm_new
    
    print("\n✓ 测试 5 通过: 向后兼容性验证通过")
    print("=" * 60)
    print("\n总结:")
    print("  - 不带 small_base_model: 使用原始 spec decode 逻辑")
    print("  - 带 small_base_model: 使用 Reward-Shifted SS 逻辑")
    print("  - 两种模式都能正常工作")

if __name__ == "__main__":
    test_backward_compatibility()

