import os
from vllm import LLM, SamplingParams

# 设置 HuggingFace 缓存目录到有更多空间的位置
os.environ["HF_HOME"] = "/mnt/sdb/huggingface_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/mnt/sdb/huggingface_cache"

prompts = [
    "The president of the United States is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="facebook/opt-6.7b",
    tensor_parallel_size=1,
    download_dir="/mnt/sdb/huggingface_cache",  # 指定下载目录
    speculative_config={
        "model": "facebook/opt-125m",
        "num_speculative_tokens": 5,
    },
)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")