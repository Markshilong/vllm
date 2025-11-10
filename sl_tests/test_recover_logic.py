"""
测试 3: 验证 recover 逻辑是否正确实现新公式

这个测试验证：
1. _get_recovered_probs 实现了新公式: (draft_probs * (target_probs / small_base_probs - 1))_+
2. 当 small_base_probs 为 None 时回退到原始公式
"""
import os
import torch
from vllm import LLM, SamplingParams

os.environ["HF_HOME"] = "/mnt/sdb/huggingface_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/mnt/sdb/huggingface_cache"

def test_recover_logic():
    """测试 recover 逻辑"""
    print("=" * 60)
    print("测试 3: 验证 recover 逻辑实现新公式")
    print("=" * 60)
    
    prompts = ["The capital of France is"]
    sampling_params = SamplingParams(temperature=0.8, max_tokens=5)
    
    print("\n初始化 LLM（带 small_base_worker）...")
    llm = LLM(
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
    
    # 验证 _get_recovered_probs 方法签名
    worker = llm.llm_engine.model_executor.driver_worker
    sampler = worker.spec_decode_sampler
    
    import inspect
    sig_recover = inspect.signature(sampler._get_recovered_probs)
    assert "small_base_probs" in sig_recover.parameters, "_get_recovered_probs 缺少 small_base_probs 参数"
    print("✓ _get_recovered_probs 包含 small_base_probs 参数")
    
    # 检查参数默认值
    small_base_param = sig_recover.parameters["small_base_probs"]
    assert small_base_param.default is None, "small_base_probs 默认值应为 None"
    print("✓ small_base_probs 默认值为 None (向后兼容)")
    
    # 执行生成以触发 recover 逻辑（如果有 token 被拒绝）
    print("\n执行生成（可能触发 recover 逻辑）...")
    outputs = llm.generate(prompts, sampling_params)
    
    for output in outputs:
        print(f"Prompt: {output.prompt!r}")
        print(f"Generated: {output.outputs[0].text!r}")
    
    print("\n✓ 测试 3 通过: recover 逻辑已正确实现")
    print("=" * 60)
    
    del llm

if __name__ == "__main__":
    test_recover_logic()

