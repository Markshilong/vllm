import os
from vllm import LLM, SamplingParams

# 设置 HuggingFace 缓存目录到有更多空间的位置
os.environ["HF_HOME"] = "/mnt/sdb/huggingface_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/mnt/sdb/huggingface_cache"

prompts = [
    "The president of the United States is",
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=50)

# 测试 small-base worker
# target model: opt-6.7b (大模型)
# draft model: opt-125m (用于生成候选token)
# small_base model: opt-125m (用于 Reward-Shifted Speculative Sampling 的分母)
print("正在初始化 LLM，包含 small-base worker...")
llm = LLM(
    model="facebook/opt-6.7b",
    tensor_parallel_size=1,
    download_dir="/mnt/sdb/huggingface_cache",  # 指定下载目录
    enforce_eager=True,
    speculative_config={
        "model": "facebook/opt-125m",  # draft model
        "num_speculative_tokens": 5,
        "small_base_model": "facebook/opt-125m",  # small-base model (用于 Reward-Shifted Speculative Sampling)
        "small_base_tensor_parallel_size": 1,  # 可选，默认为1
    },
)

# 验证 small_base_worker 是否被创建
# 通过检查 worker 的内部结构来验证
if hasattr(llm.llm_engine, "model_executor"):
    executor = llm.llm_engine.model_executor
    if hasattr(executor, "driver_worker"):
        worker = executor.driver_worker
        if hasattr(worker, "small_base_worker"):
            small_base_worker = worker.small_base_worker
            if small_base_worker is not None:
                print(f"✓ small_base_worker 已成功创建: {type(small_base_worker)}")
            else:
                print("✗ small_base_worker 为 None，可能未正确初始化")
        else:
            print("✗ worker 中没有 small_base_worker 属性")
    else:
        print("⚠ 无法访问 driver_worker，跳过验证")

print("\n开始生成...")
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}")
    print(f"Generated text: {generated_text!r}")
    print("-" * 50)

print("\n测试完成！small-base worker 已成功运行。")

