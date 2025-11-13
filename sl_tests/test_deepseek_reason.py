"""
测试 DeepSeek-R1 和 Qwen2.5-Math 模型的 Speculative Decoding Accept Rate

配置：
- target (π_target): Qwen/Qwen2.5-Math-7B
- small_base (π_sft/π_small_base): Qwen/Qwen2.5-Math-1.5B
- draft (π_r): deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
"""
import os
from vllm import LLM, SamplingParams

os.environ["HF_HOME"] = "/mnt/sdb/huggingface_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/mnt/sdb/huggingface_cache"


def test_qwen_accept_rate():
    """测试 DeepSeek-R1 和 Qwen2.5-Math 模型的 accept rate"""
    print("=" * 80)
    print("测试 DeepSeek-R1 和 Qwen2.5-Math 模型的 Speculative Decoding Accept Rate")
    print("=" * 80)
    print("\n模型配置:")
    print("  - Target (π_target): ./Qwen2.5-Math-7B-padded")
    print("  - Small Base (π_sft/π_small_base): ./Qwen2.5-Math-1.5B-padded")
    print("  - Draft (π_r): ./DeepSeek-R1-Distill-Qwen-1.5B-padded")
    print("=" * 80)
    
    # 测试 prompts
    prompts = [
        "The capital of France is",
        "The future of artificial intelligence",
        "Once upon a time, in a distant galaxy",
        "The most important thing in life is",
        "In the field of machine learning,",
    ]
    
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=50,
    )
    
    print("\n初始化 LLM（带 Reward-Shifted Speculative Sampling）...")
    # 使用对齐后的模型路径（从当前目录读取）
    target_model = "./Qwen2.5-Math-7B-padded"
    draft_model = "./DeepSeek-R1-Distill-Qwen-1.5B-padded"
    small_base_model = "./Qwen2.5-Math-1.5B-padded"
    
    llm = LLM(
        model=target_model,  # target model (π_target)
        tensor_parallel_size=1,
        download_dir="/mnt/sdb/huggingface_cache",
        enforce_eager=True,
        gpu_memory_utilization=0.45,
        speculative_config={
            "model": draft_model,  # draft model (π_r)
            "num_speculative_tokens": 5,
            "small_base_model": small_base_model,  # small_base model (π_sft/π_small_base)
            "small_base_tensor_parallel_size": 1,
        },
    )
    
    print("\n开始生成...")
    outputs = llm.generate(prompts, sampling_params)
    
    # 获取 accept rate 统计数据
    worker = llm.llm_engine.model_executor.driver_worker
    sampler = worker.spec_decode_sampler
    
    # 从 sampler 获取统计数据
    num_accepted_tokens = sampler.num_accepted_tokens.item() if hasattr(sampler.num_accepted_tokens, 'item') else sampler.num_accepted_tokens
    num_draft_tokens = sampler.num_draft_tokens
    
    # 计算 accept rate
    if num_draft_tokens > 0:
        accept_rate = num_accepted_tokens / num_draft_tokens
    else:
        accept_rate = 0.0
    
    print("\n" + "=" * 80)
    print("生成结果:")
    print("=" * 80)
    for i, output in enumerate(outputs):
        print(f"\n[{i+1}] Prompt: {output.prompt!r}")
        print(f"    Generated: {output.outputs[0].text!r}")
    
    print("\n" + "=" * 80)
    print("Accept Rate 统计:")
    print("=" * 80)
    print(f"  Accepted tokens: {num_accepted_tokens}")
    print(f"  Draft tokens: {num_draft_tokens}")
    print(f"  Accept rate: {accept_rate:.4f} ({accept_rate * 100:.2f}%)")
    print("=" * 80)
    
    # 尝试从 metrics 获取更详细的信息（如果可用）
    try:
        if hasattr(llm.llm_engine, 'stat_loggers') and llm.llm_engine.stat_loggers:
            stat_logger = llm.llm_engine.stat_loggers.get('prometheus')
            if stat_logger:
                try:
                    metric_accept_rate = (
                        stat_logger.metrics.gauge_spec_decode_draft_acceptance_rate
                        .labels(**stat_logger.labels)._value.get()
                    )
                    print(f"\n从 Prometheus metrics 获取的 Accept rate: {metric_accept_rate:.4f} ({metric_accept_rate * 100:.2f}%)")
                except Exception as e:
                    print(f"\n无法从 Prometheus metrics 获取数据: {e}")
    except Exception as e:
        print(f"\n无法访问 stat_loggers: {e}")
    
    del llm
    print("\n测试完成！")


if __name__ == "__main__":
    test_qwen_accept_rate()

