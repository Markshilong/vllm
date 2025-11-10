# Reward-Shifted Speculative Sampling 测试套件

本目录包含用于验证 Reward-Shifted Speculative Sampling (SSS) 实现的测试用例。

## 测试文件说明

### 1. `test_small_base_probs_extraction.py`
**目的**: 验证 small_base_probs 是否正确从 small_base_worker 中提取

**验证内容**:
- small_base_worker 被正确创建
- scorer 能够获取 small_base 模型的概率
- SpeculativeScores 中包含 small_base_probs 字段

**运行方式**:
```bash
cd /home/shilong/shilong_projects/vllm
python sl_tests/test_small_base_probs_extraction.py
```

### 2. `test_accept_logic.py`
**目的**: 验证 accept 逻辑是否正确使用 small_base_probs 作为分母

**验证内容**:
- sampler.forward 方法包含 small_base_probs 参数
- _get_accepted 方法使用 small_base_probs 作为分母（如果提供）
- 当 small_base_probs 为 None 时回退到原始逻辑

**运行方式**:
```bash
python sl_tests/test_accept_logic.py
```

### 3. `test_recover_logic.py`
**目的**: 验证 recover 逻辑是否正确实现新公式

**验证内容**:
- _get_recovered_probs 实现了新公式: `(draft_probs * (target_probs / small_base_probs - 1))_+`
- 当 small_base_probs 为 None 时回退到原始公式: `(target_probs - draft_probs)_+`

**运行方式**:
```bash
python sl_tests/test_recover_logic.py
```

### 4. `test_end_to_end.py`
**目的**: 端到端测试 - 验证完整的 Reward-Shifted Speculative Sampling 流程

**验证内容**:
- 完整的生成流程能够正常运行
- 输出是合理的（不崩溃、有输出）
- 与标准 spec decode 的对比（输出可能不同，但都应该有效）

**运行方式**:
```bash
python sl_tests/test_end_to_end.py
```

### 5. `test_backward_compatibility.py`
**目的**: 验证向后兼容性

**验证内容**:
- 当 small_base_worker 为 None 时，行为与原始 spec decode 一致
- 所有原有功能仍然正常工作

**运行方式**:
```bash
python sl_tests/test_backward_compatibility.py
```

### 6. `run_all_tests.py`
**目的**: 运行所有测试用例

按顺序运行所有测试，逐步验证 Reward-Shifted Speculative Sampling 的实现。

**运行方式**:
```bash
python sl_tests/run_all_tests.py
```

## 运行所有测试

要运行所有测试，可以使用：

```bash
cd /home/shilong/shilong_projects/vllm
python sl_tests/run_all_tests.py
```

或者逐个运行：

```bash
# 测试 1
python sl_tests/test_small_base_probs_extraction.py

# 测试 2
python sl_tests/test_accept_logic.py

# 测试 3
python sl_tests/test_recover_logic.py

# 测试 4
python sl_tests/test_end_to_end.py

# 测试 5
python sl_tests/test_backward_compatibility.py
```

## 测试环境要求

1. **模型**: 测试使用 `facebook/opt-6.7b` 和 `facebook/opt-125m`
   - 确保这些模型已下载到 `/mnt/sdb/huggingface_cache`
   - 或修改测试文件中的模型路径和缓存目录

2. **环境变量**: 测试会自动设置 HuggingFace 缓存目录
   ```python
   os.environ["HF_HOME"] = "/mnt/sdb/huggingface_cache"
   os.environ["HUGGINGFACE_HUB_CACHE"] = "/mnt/sdb/huggingface_cache"
   ```

3. **GPU**: 需要至少 1 个 GPU（tensor_parallel_size=1）

## 测试验证点总结

### 实现验证
- ✅ small_base_worker 正确创建和初始化
- ✅ scorer 能够获取 small_base 模型概率
- ✅ SpeculativeScores 包含 small_base_probs 字段
- ✅ accept 逻辑使用 small_base_probs 作为分母
- ✅ recover 逻辑实现新公式
- ✅ 参数正确传递到 sampler

### 功能验证
- ✅ 端到端生成流程正常运行
- ✅ 输出合理且非空
- ✅ 向后兼容性（small_base_worker=None 时使用原始逻辑）

### 代码质量
- ✅ 方法签名正确
- ✅ 参数默认值正确
- ✅ 错误处理合理

## 注意事项

1. **输出差异**: Reward-Shifted SS 的输出可能与标准 Spec Decode 不同，这是预期的，因为 accept 和 recover 逻辑不同。

2. **随机性**: 由于 accept 过程涉及随机采样，相同输入可能产生不同输出（除非使用 temperature=0）。

3. **性能**: 这些测试主要验证正确性，不关注性能指标。性能测试需要单独的基准测试。

4. **模型选择**: 测试中使用 opt-125m 同时作为 draft 和 small_base 模型，实际使用中应该是不同的模型（draft 是 reasoning 模型，small_base 是 base 模型）。

## 故障排查

如果测试失败，请检查：

1. **模型是否下载**: 确保 opt-6.7b 和 opt-125m 已下载
2. **GPU 内存**: 确保有足够的 GPU 内存
3. **CUDA 环境**: 确保 CUDA 环境正确配置
4. **代码更新**: 确保所有实现代码已正确提交

查看详细错误信息，运行单个测试文件而不是 `run_all_tests.py`。

