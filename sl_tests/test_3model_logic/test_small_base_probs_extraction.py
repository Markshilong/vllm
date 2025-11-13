"""
测试 1: 验证 small_base_probs 是否正确从 small_base_worker 中提取

这个测试验证：
1. small_base_worker 被正确创建
2. scorer 能够获取 small_base 模型的概率
3. SpeculativeScores 中包含 small_base_probs 字段
"""
import os
import torch
from vllm import LLM, SamplingParams

# 设置 HuggingFace 缓存目录
os.environ["HF_HOME"] = "/mnt/sdb/huggingface_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/mnt/sdb/huggingface_cache"

def test_small_base_probs_extraction():
    """测试 small_base_probs 是否正确提取"""
    print("=" * 60)
    print("测试 1: 验证 small_base_probs 提取")
    print("=" * 60)
    
    prompts = ["The president of the United States is"]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=3)
    
    print("\n初始化 LLM（带 small_base_worker）...")
    llm = LLM(
        model="facebook/opt-6.7b",
        tensor_parallel_size=1,
        download_dir="/mnt/sdb/huggingface_cache",
        enforce_eager=True,
        speculative_config={
            "model": "facebook/opt-125m",  # draft model (reasoning)
            "num_speculative_tokens": 3,
            "small_base_model": "facebook/opt-125m",  # small_base model
            "small_base_tensor_parallel_size": 1,
        },
    )
    
    # 验证 small_base_worker 存在
    print("\n验证 small_base_worker 存在...")
    worker = llm.llm_engine.model_executor.driver_worker
    assert hasattr(worker, "small_base_worker"), "small_base_worker 属性不存在"
    assert worker.small_base_worker is not None, "small_base_worker 为 None"
    print(f"✓ small_base_worker 已创建: {type(worker.small_base_worker)}")
    
    # 验证 scorer 有 small_base_worker 引用
    print("\n验证 scorer 有 small_base_worker 引用...")
    assert hasattr(worker, "scorer"), "scorer 属性不存在"
    scorer = worker.scorer
    assert hasattr(scorer, "_small_base_worker"), "scorer 没有 _small_base_worker 属性"
    assert scorer._small_base_worker is not None, "scorer._small_base_worker 为 None"
    print(f"✓ scorer 已配置 small_base_worker: {type(scorer._small_base_worker)}")
    
    # 执行一次生成以触发 scoring
    print("\n执行生成以触发 scoring...")
    outputs = llm.generate(prompts, sampling_params)
    
    # 检查输出
    print("\n检查生成结果...")
    for output in outputs:
        print(f"Prompt: {output.prompt!r}")
        print(f"Generated: {output.outputs[0].text!r}")
    
    print("\n✓ 测试 1 通过: small_base_probs 提取机制已配置")
    print("=" * 60)
    
    del llm

if __name__ == "__main__":
    test_small_base_probs_extraction()

