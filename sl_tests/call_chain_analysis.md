# Speculative Decoding 调用链分析

## 概述
本文档梳理了从 `llm.generate()` 调用到生成 target 和 draft model 概率的完整调用链。

## 完整调用链

### 1. 入口点：LLM.generate()
**文件**: `vllm/entrypoints/llm.py`

```382:474:vllm/entrypoints/llm.py
def generate(...) -> list[RequestOutput]:
    # 验证和添加请求
    self._validate_and_add_requests(...)
    # 运行引擎
    outputs = self._run_engine(use_tqdm=use_tqdm)
    return self.engine_class.validate_outputs(outputs, RequestOutput)
```

### 2. LLM._run_engine()
**文件**: `vllm/entrypoints/llm.py`

```1404:1450:vllm/entrypoints/llm.py
def _run_engine(self, *, use_tqdm: bool):
    # 循环调用 engine.step() 直到所有请求完成
    while self.llm_engine.has_unfinished_requests():
        step_outputs = self.llm_engine.step()
        # 收集完成的输出
        for output in step_outputs:
            if output.finished:
                outputs.append(output)
```

### 3. LLMEngine.step()
**文件**: `vllm/engine/llm_engine.py`

```1273:1413:vllm/engine/llm_engine.py
def step(self) -> List[Union[RequestOutput, PoolingRequestOutput]]:
    # 1. 调度序列
    (seq_group_metadata_list, scheduler_outputs, ...) = \
        self.scheduler[virtual_engine].schedule()
    
    # 2. 创建 ExecuteModelRequest
    execute_model_req = ExecuteModelRequest(
        seq_group_metadata_list=seq_group_metadata_list,
        num_lookahead_slots=scheduler_outputs.num_lookahead_slots,
        ...
    )
    
    # 3. 执行模型
    outputs = self.model_executor.execute_model(
        execute_model_req=execute_model_req)
```

### 4. ModelExecutor.execute_model()
**文件**: `vllm/executor/ray_distributed_executor.py` 或类似

```444:462:vllm/executor/ray_distributed_executor.py
def execute_model(self, execute_model_req: ExecuteModelRequest) -> List[SamplerOutput]:
    # 调用 worker 的 execute_model 方法
    return self.driver_worker.execute_method("execute_model", execute_model_req)
```

### 5. SpecDecodeWorker.execute_model()
**文件**: `vllm/spec_decode/spec_decode_worker.py`

```533:800:vllm/spec_decode/spec_decode_worker.py
def execute_model(self, execute_model_req: Optional[ExecuteModelRequest] = None):
    # 检查是否需要执行 speculative decoding
    no_spec = (num_lookahead_slots == 0 or disable_all_speculation 
               or all_zero_spec_tokens)
    
    if no_spec:
        # 无 speculative decoding，直接运行 target model
        return self._run_no_spec(execute_model_req, skip_proposer=False)
    else:
        # 执行 speculative decoding
        return self._run_speculative_decoding_step(
            execute_model_req, num_lookahead_slots)
```

### 6. SpecDecodeWorker._run_speculative_decoding_step()
**文件**: `vllm/spec_decode/spec_decode_worker.py`

这是核心方法，负责协调 draft model 和 target model 的执行：

```836:913:vllm/spec_decode/spec_decode_worker.py
def _run_speculative_decoding_step(
        self, execute_model_req: ExecuteModelRequest,
        num_lookahead_slots: int) -> List[SamplerOutput]:
    
    # 步骤 1: 使用 draft worker 生成 proposals（包含 draft model 的概率）
    with Timer() as proposal_timer:
        proposals = self.proposer_worker.get_spec_proposals(
            execute_model_req, self._seq_with_bonus_token_in_last_step)
    
    # 步骤 2: 使用 target model 对 proposals 进行评分（生成 target model 的概率）
    with Timer() as scoring_timer:
        proposal_scores = self.scorer.score_proposals(
            execute_model_req, proposals)
    
    # 步骤 3: 验证 tokens（决定哪些 speculative tokens 被接受）
    accepted_token_ids, target_logprobs = self._verify_tokens(
        execute_model_req.seq_group_metadata_list, proposal_scores,
        proposals, execute_model_req.num_lookahead_slots)
    
    # 步骤 4: 创建输出
    return self._create_output_sampler_list(...)
```

## Draft Model 概率生成路径

### 7. ProposerWorker.get_spec_proposals()
**文件**: `vllm/spec_decode/top1_proposer.py` (Top1Proposer)

```44:100:vllm/spec_decode/top1_proposer.py
def get_spec_proposals(self, execute_model_req, ...) -> SpeculativeProposals:
    # 调用 worker 的 sampler_output 方法
    maybe_sampler_output, transposed = self._worker.sampler_output(
        execute_model_req=nonzero_execute_model_req,
        sample_len=proposal_len,
        ...)
    
    # 合并输出，提取 proposal_tokens 和 proposal_probs
    proposal_tokens, proposal_probs, proposal_lens = self._merge_outputs(...)
    
    return SpeculativeProposals(
        proposal_token_ids=proposal_tokens,
        proposal_probs=proposal_probs,  # Draft model 的概率
        proposal_lens=proposal_lens)
```

### 8. ProposerWorker.sampler_output()
**文件**: `vllm/spec_decode/proposer_worker_base.py` 或子类

该方法会：
1. 调用 `proposer_worker.execute_model()` 执行 draft model
2. Draft model 通过 forward pass 生成 logits
3. 通过 sampler 将 logits 转换为概率和采样 tokens

**关键路径**:
- `proposer_worker.execute_model()` 
  → `worker_base.execute_model()` 
  → `model_runner.execute_model()` 
  → `model.forward()` (draft model forward pass)
  → `model.compute_logits()` 
  → `sampler.forward()` (生成概率)

**Draft Model 概率生成**:
```python
# 在 ModelRunner.execute_model() 中
logits = self.model.compute_logits(hidden_states, ...)
output = self.sampler(logits=logits, ...)  # 生成概率和采样 tokens
# output.sampled_token_probs 包含 draft model 的概率
```

## Target Model 概率生成路径

### 9. Scorer.score_proposals()
**文件**: `vllm/spec_decode/mqa_scorer.py` (MQAScorer)

```14:100:vllm/spec_decode/mqa_scorer.py
def score_proposals(self, execute_model_req, proposals) -> SpeculativeScores:
    # 为每个 proposal token 创建新的序列元数据
    target_seq_group_metadata_list = []
    # ... 构建包含 proposal tokens 的序列 ...
    
    # 执行 target model
    target_sampler_output = self._scorer_worker.execute_model(
        execute_model_req=execute_model_req.clone(
            seq_group_metadata_list=target_seq_group_metadata_list))
    
    # 提取 target model 的概率
    target_token_ids = target_sampler_output.sampled_token_ids
    target_probs = target_sampler_output.sampled_token_probs  # Target model 的概率
    target_logprobs = target_sampler_output.logprobs
    
    return SpeculativeScores(
        probs=target_probs,  # Target model 对 proposal tokens 的概率
        logprobs=target_logprobs,
        token_ids=target_token_ids,
        ...)
```

**Target Model 概率生成路径**:
- `scorer_worker.execute_model()` 
  → `worker_base.execute_model()` 
  → `model_runner.execute_model()` 
  → `model.forward()` (target model forward pass)
  → `model.compute_logits()` 
  → `sampler.forward()` (生成概率)

**Target Model 概率生成**:
```python
# 在 ModelRunner.execute_model() 中
logits = self.model.compute_logits(hidden_states, ...)
output = self.sampler(logits=logits, ...)  # 生成概率和采样 tokens
# output.sampled_token_probs 包含 target model 的概率
```

## 概率生成的关键代码位置

### Sampler.forward() - 概率计算
**文件**: `vllm/model_executor/layers/sampler.py`

```216:296:vllm/model_executor/layers/sampler.py
def forward(self, logits: torch.Tensor, sampling_metadata: SamplingMetadata):
    # 应用各种惩罚和温度缩放
    logits = apply_penalties(...)
    logits.div_(temperatures.unsqueeze(dim=1))
    
    # 计算概率
    probs = torch.softmax(logits, dim=-1, dtype=torch.float)
    logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)
    
    # 采样 tokens
    sampled_tokens = _sample(probs, logprobs, ...)
    
    return SamplerOutput(
        sampled_token_ids=sampled_tokens,
        sampled_token_probs=probs,  # 这是最终的概率
        logprobs=logprobs,
        ...)
```

## 总结

### 调用链流程图

```
LLM.generate()
  ↓
LLM._run_engine()
  ↓
LLMEngine.step()
  ↓
ModelExecutor.execute_model()
  ↓
SpecDecodeWorker.execute_model()
  ↓
SpecDecodeWorker._run_speculative_decoding_step()
  ├─→ ProposerWorker.get_spec_proposals()
  │     ↓
  │   ProposerWorker.sampler_output()
  │     ↓
  │   WorkerBase.execute_model()
  │     ↓
  │   ModelRunner.execute_model()
  │     ↓
  │   Draft Model.forward() → compute_logits() → Sampler()
  │     ↓
  │   生成 Draft Model 概率 (proposal_probs)
  │
  └─→ Scorer.score_proposals()
        ↓
      ScorerWorker.execute_model()
        ↓
      WorkerBase.execute_model()
        ↓
      ModelRunner.execute_model()
        ↓
      Target Model.forward() → compute_logits() → Sampler()
        ↓
      生成 Target Model 概率 (target_probs)
```

### 关键数据结构

1. **SpeculativeProposals**: 包含 draft model 的 proposals
   - `proposal_token_ids`: Draft model 生成的 token IDs
   - `proposal_probs`: Draft model 的概率分布
   - `proposal_lens`: 每个序列的 proposal 长度

2. **SpeculativeScores**: 包含 target model 的评分
   - `probs`: Target model 对 proposal tokens 的概率分布
   - `logprobs`: Target model 的 log 概率
   - `token_ids`: Target model 采样的 token IDs

3. **SamplerOutput**: 包含模型输出的概率信息
   - `sampled_token_probs`: 采样 token 的概率分布 [batch_size, vocab_size]
   - `logprobs`: Log 概率
   - `sampled_token_ids`: 采样的 token IDs

### 概率生成的关键点

1. **Draft Model 概率**: 在 `proposer_worker.execute_model()` → `sampler.forward()` 中生成
2. **Target Model 概率**: 在 `scorer_worker.execute_model()` → `sampler.forward()` 中生成
3. **概率计算**: 都在 `Sampler.forward()` 中通过 `torch.softmax(logits, dim=-1)` 计算
4. **概率使用**: 在 `_verify_tokens()` 中使用两种概率来决定哪些 speculative tokens 被接受

