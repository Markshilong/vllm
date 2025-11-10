# CUBLAS_STATUS_INTERNAL_ERROR 调试指南

## 问题描述

在执行 `small_base_worker.execute_model` 时出现 `CUBLAS_STATUS_INTERNAL_ERROR`，错误发生在矩阵乘法操作中。

## 错误堆栈关键位置

```
File "vllm/spec_decode/mqa_scorer.py", line 95
    small_base_sampler_output = self._small_base_worker.execute_model(...)
    
File "vllm/model_executor/layers/linear.py", line 967
    RuntimeError: CUDA error: CUBLAS_STATUS_INTERNAL_ERROR when calling cublasLtMatmul
```

## 🔴 根本原因（已确认）

**`small_base_worker` 的 KV cache 未被初始化！**

在 `vllm/spec_decode/spec_decode_worker.py:520-527` 的 `initialize_cache` 方法中：

```python
def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
    """Initialize the cache engine of the scorer and proposer workers."""
    self.scorer_worker.initialize_cache(num_gpu_blocks=num_gpu_blocks,
                                       num_cpu_blocks=num_cpu_blocks)
    self.proposer_worker.initialize_cache(num_gpu_blocks=num_gpu_blocks,
                                          num_cpu_blocks=num_cpu_blocks)
    # ❌ 缺少：self.small_base_worker.initialize_cache(...)
```

**问题**：只初始化了 `scorer_worker` 和 `proposer_worker` 的 cache，但**没有初始化 `small_base_worker` 的 cache**。

当 `small_base_worker.execute_model` 尝试使用 `block_tables` 访问 KV cache 时，由于 cache 未初始化，导致 `CUBLAS_STATUS_INTERNAL_ERROR`。

## 可能的原因（其他潜在问题）

### 1. KV Cache 问题
- ✅ **已确认**：`small_base_worker` 的 KV cache 未初始化（根本原因）
- `block_tables` 中的 block 索引可能无效或超出范围
- 序列长度与 KV cache 不匹配

### 2. 序列元数据问题
- 传递给 `small_base_worker` 的 `target_seq_group_metadata_list` 可能包含无效数据
- `block_tables` 可能引用了错误的 block
- 序列 ID 映射可能有问题

### 3. 张量形状/设备问题
- 输入张量形状不匹配
- 张量在错误的设备上（CPU vs GPU）
- 数据类型不匹配

### 4. 内存问题
- GPU 内存不足
- 内存碎片化
- 并发访问冲突

## 调试步骤

### 步骤 1: 运行调试脚本

```bash
python sl_tests/debug_small_base_error.py
```

这个脚本会：
- 检查 `small_base_worker` 的初始化状态
- 检查 KV cache 状态
- 打印传递给 `execute_model` 的详细参数
- 捕获并显示详细的错误信息

### 步骤 2: 检查 KV Cache 初始化

在 `vllm/spec_decode/spec_decode_worker.py` 中查找 `small_base_worker` 的初始化代码，确认：
- KV cache 是否被正确创建
- `initialize_cache` 是否被调用
- cache 大小是否足够

### 步骤 3: 检查 block_tables

问题可能出在 `target_seq_group_metadata_list` 中的 `block_tables`。检查：
- `block_tables` 是否引用了 `small_base_worker` 的 KV cache blocks
- 还是引用了 `target_model` 或 `proposer` 的 blocks

**关键问题**：`small_base_worker` 可能有自己独立的 KV cache，但 `block_tables` 可能仍然指向主模型的 cache！

### 步骤 4: 添加断点和日志

在以下位置添加日志：

1. **mqa_scorer.py:75** - 调用 `small_base_worker.execute_model` 之前
   ```python
   print(f"DEBUG: small_base_req.block_tables = {small_base_req.seq_group_metadata_list[0].block_tables}")
   print(f"DEBUG: small_base_worker.kv_cache = {small_base_worker.kv_cache}")
   ```

2. **worker_base.py:420** - `execute_model` 内部
   ```python
   print(f"DEBUG: kv_caches = {kv_caches}")
   print(f"DEBUG: kv_caches[0][0].shape = {kv_caches[0][0].shape if kv_caches else 'None'}")
   ```

3. **draft_model_runner.py:282** - 模型执行前
   ```python
   print(f"DEBUG: input_ids.shape = {model_input.input_tokens.shape}")
   print(f"DEBUG: positions.shape = {model_input.input_positions.shape}")
   ```

### 步骤 5: 检查 block_tables 映射

**最可能的问题**：`small_base_worker` 需要自己的 `block_tables`，但当前代码可能传递了主模型的 `block_tables`。

检查 `mqa_scorer.py` 中构建 `target_seq_group_metadata_list` 的代码：

```python
block_tables={
    target_seq_id: seq_group_metadata.block_tables[seq_id],
},
```

这里直接复制了原始序列的 `block_tables`，但这些 blocks 可能属于主模型，而不是 `small_base_worker`！

### 步骤 6: 验证假设

如果 `small_base_worker` 有独立的 KV cache，那么：
1. 它需要自己的 block 分配
2. `block_tables` 需要指向 `small_base_worker` 的 cache blocks
3. 可能需要先为这些序列分配 blocks

## 修复建议

### ✅ 方案 1: 初始化 small_base_worker 的 KV cache（推荐）

在 `vllm/spec_decode/spec_decode_worker.py` 的 `initialize_cache` 方法中添加：

```python
def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
    """Initialize the cache engine of the scorer and proposer workers."""
    self.scorer_worker.initialize_cache(num_gpu_blocks=num_gpu_blocks,
                                        num_cpu_blocks=num_cpu_blocks)
    self.proposer_worker.initialize_cache(num_gpu_blocks=num_gpu_blocks,
                                          num_cpu_blocks=num_cpu_blocks)
    # 添加 small_base_worker 的 cache 初始化
    if self.small_base_worker is not None:
        self.small_base_worker.initialize_cache(num_gpu_blocks=num_gpu_blocks,
                                                num_cpu_blocks=num_cpu_blocks)
```

**注意**：还需要考虑 `block_tables` 的映射问题。`small_base_worker` 可能有自己独立的 KV cache，所以传递给它的 `block_tables` 需要指向 `small_base_worker` 的 cache blocks，而不是主模型的 blocks。

### 方案 2: 为 small_base_worker 创建新的 block_tables

如果 `small_base_worker` 有独立的 KV cache，需要：
1. 为 `target_seq_group_metadata_list` 中的序列分配新的 blocks
2. 使用 `small_base_worker` 的 cache 分配器
3. 更新 `block_tables` 指向新分配的 blocks

### 方案 2: 检查 small_base_worker 是否应该共享 KV cache

如果 `small_base_worker` 应该与主模型共享 KV cache，需要：
1. 确认 `small_base_worker` 使用的是同一个 cache
2. 验证 block 索引的有效性

### 方案 3: 临时禁用 small_base_worker 以验证

在 `mqa_scorer.py` 中临时注释掉 `small_base_worker` 的调用，看是否其他部分正常工作：

```python
# small_base_probs = None
# if self._small_base_worker is not None:
#     small_base_sampler_output = self._small_base_worker.execute_model(...)
```

## 快速检查清单

- [ ] `small_base_worker.kv_cache` 是否已初始化？
- [ ] `block_tables` 中的 block 索引是否有效？
- [ ] `block_tables` 是否指向正确的 cache（small_base vs target）？
- [ ] 输入张量的形状是否正确？
- [ ] 张量是否在正确的设备上？
- [ ] GPU 内存是否充足？
- [ ] 是否有其他 CUDA 错误（使用 `CUDA_LAUNCH_BLOCKING=1`）？

## 使用 CUDA 调试工具

设置环境变量以获取更详细的 CUDA 错误信息：

```bash
export CUDA_LAUNCH_BLOCKING=1
export CUDA_DEBUG=1
python sl_tests/test_small_base_probs_extraction.py
```

## 下一步

1. 运行 `debug_small_base_error.py` 收集详细信息
2. 检查 `small_base_worker` 的 KV cache 初始化代码
3. 验证 `block_tables` 的映射关系
4. 根据发现的问题实施修复

