"""
调试脚本：定位 small_base_worker 执行时的 CUBLAS_STATUS_INTERNAL_ERROR 问题

这个脚本会：
1. 检查 small_base_worker 的初始化状态
2. 检查传递给 execute_model 的参数
3. 检查 KV cache 状态
4. 检查张量形状和设备
5. 添加详细的错误捕获和日志
"""
import os
import torch
import traceback
from vllm import LLM, SamplingParams
from vllm.sequence import ExecuteModelRequest

# 设置 HuggingFace 缓存目录
os.environ["HF_HOME"] = "/mnt/sdb/huggingface_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/mnt/sdb/huggingface_cache"

def debug_small_base_error():
    """调试 small_base_worker 执行错误"""
    print("=" * 60)
    print("调试 small_base_worker CUBLAS 错误")
    print("=" * 60)
    
    prompts = ["The president of the United States is"]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=3)
    
    print("\n1. 初始化 LLM...")
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
    
    worker = llm.llm_engine.model_executor.driver_worker
    scorer = worker.scorer
    small_base_worker = scorer._small_base_worker
    
    print("\n2. 检查 small_base_worker 状态...")
    print(f"   small_base_worker 类型: {type(small_base_worker)}")
    print(f"   small_base_worker 是否为 None: {small_base_worker is None}")
    
    if small_base_worker is not None:
        # 检查 KV cache
        print("\n3. 检查 KV cache...")
        print("   🔍 关键检查：small_base_worker 的 KV cache 是否已初始化？")
        
        # 检查是否有 cache_engine（Worker 类通常有）
        cache_engine = None
        if hasattr(small_base_worker, 'cache_engine'):
            cache_engine = small_base_worker.cache_engine
            print(f"   cache_engine 类型: {type(cache_engine)}")
            print(f"   cache_engine 是否为 None: {cache_engine is None}")
            if cache_engine is None:
                print("   ❌ 问题确认：cache_engine 为 None，KV cache 未初始化！")
                print("   💡 修复：需要在 spec_decode_worker.initialize_cache() 中初始化")
            else:
                print("   ✓ cache_engine 已初始化")
        else:
            print("   ⚠ small_base_worker 没有 cache_engine 属性")
        
        # 检查 gpu_cache
        if hasattr(small_base_worker, 'gpu_cache'):
            gpu_cache = small_base_worker.gpu_cache
            print(f"   gpu_cache 类型: {type(gpu_cache)}")
            print(f"   gpu_cache 是否为 None: {gpu_cache is None}")
            if gpu_cache is None:
                print("   ❌ gpu_cache 为 None")
            else:
                print(f"   ✓ gpu_cache 已初始化，长度: {len(gpu_cache)}")
        
        # 检查 kv_cache 属性（可能通过 property 访问）
        if hasattr(small_base_worker, 'kv_cache'):
            kv_cache = small_base_worker.kv_cache
            print(f"   kv_cache 类型: {type(kv_cache)}")
            print(f"   kv_cache 是否为 None: {kv_cache is None}")
            if kv_cache is not None:
                print(f"   kv_cache 长度: {len(kv_cache)}")
                if len(kv_cache) > 0 and len(kv_cache[0]) > 0:
                    print(f"   第一个 cache 的形状: {kv_cache[0][0].shape if hasattr(kv_cache[0][0], 'shape') else 'N/A'}")
        else:
            print("   ⚠ small_base_worker 没有 kv_cache 属性（可能通过其他方式访问）")
        
        # 检查 MultiStepWorker 的内部 worker（委托模式）
        print("\n   检查 MultiStepWorker 的内部 worker（委托模式）...")
        if hasattr(small_base_worker, 'worker'):
            inner_worker = small_base_worker.worker
            print(f"   inner_worker 类型: {type(inner_worker)}")
            if hasattr(inner_worker, 'cache_engine'):
                inner_cache_engine = inner_worker.cache_engine
                print(f"   inner_worker.cache_engine: {type(inner_cache_engine)}")
                print(f"   inner_worker.cache_engine 是否为 None: {inner_cache_engine is None}")
                if inner_cache_engine is None:
                    print("   ❌ 问题确认：inner_worker 的 cache_engine 也为 None！")
                    print("   💡 这证实了 small_base_worker.initialize_cache() 未被调用")
            if hasattr(inner_worker, 'gpu_cache'):
                inner_gpu_cache = inner_worker.gpu_cache
                print(f"   inner_worker.gpu_cache: {type(inner_gpu_cache)}")
                print(f"   inner_worker.gpu_cache 是否为 None: {inner_gpu_cache is None}")
        
        # 对比检查：scorer_worker 的 cache
        print("\n   对比：检查 scorer_worker 的 cache（应该已初始化）...")
        if hasattr(worker, 'scorer_worker'):
            scorer_worker = worker.scorer_worker
            if hasattr(scorer_worker, 'cache_engine'):
                scorer_cache_engine = scorer_worker.cache_engine
                print(f"   scorer_worker.cache_engine: {type(scorer_cache_engine)}")
                print(f"   scorer_worker.cache_engine 是否为 None: {scorer_cache_engine is None}")
                if scorer_cache_engine is not None:
                    print(f"   scorer_worker.cache_engine 长度: {len(scorer_cache_engine)}")
                if scorer_cache_engine is not None and cache_engine is None:
                    print("   ⚠ 对比结果：scorer_worker 有 cache，但 small_base_worker 没有！")
        
        # 检查 model_runner
        print("\n4. 检查 model_runner...")
        if hasattr(small_base_worker, 'model_runner'):
            model_runner = small_base_worker.model_runner
            print(f"   model_runner 类型: {type(model_runner)}")
            print(f"   model_runner 是否为 None: {model_runner is None}")
            if model_runner is not None:
                if hasattr(model_runner, 'device'):
                    print(f"   device: {model_runner.device}")
        else:
            print("   ⚠ small_base_worker 没有 model_runner 属性")
    
    # 修改 scorer 以添加调试信息
    print("\n5. 添加调试钩子到 scorer.score_proposals...")
    original_score_proposals = scorer.score_proposals
    
    def debug_score_proposals(execute_model_req, proposals):
        """带调试信息的 score_proposals 包装器"""
        print("\n" + "=" * 60)
        print("DEBUG: 进入 score_proposals")
        print("=" * 60)
        
        # 检查 execute_model_req
        print(f"\n执行模型请求信息:")
        print(f"  num_lookahead_slots: {execute_model_req.num_lookahead_slots}")
        print(f"  seq_group_metadata_list 长度: {len(execute_model_req.seq_group_metadata_list)}")
        
        # 构建 target_seq_group_metadata_list（复制原始逻辑）
        from vllm.sequence import SequenceData, SequenceGroupMetadata, get_all_seq_ids
        target_seq_group_metadata_list = []
        target_seq_id_start = max(
            get_all_seq_ids(execute_model_req.seq_group_metadata_list)) + 1
        all_proposal_tokens = proposals.proposal_token_ids.tolist()
        all_proposal_lengths = proposals.proposal_lens.tolist()
        
        print(f"\n提案信息:")
        print(f"  proposal_token_ids 形状: {proposals.proposal_token_ids.shape}")
        print(f"  proposal_lens: {all_proposal_lengths}")
        print(f"  target_seq_id_start: {target_seq_id_start}")
        
        for i, seq_group_metadata in enumerate(execute_model_req.seq_group_metadata_list):
            if all_proposal_lengths[i] == 0:
                target_seq_group_metadata_list.append(seq_group_metadata)
                continue
            
            seq_data_dict = seq_group_metadata.seq_data
            seq_id = next(iter(seq_data_dict.keys()))
            seq_data: SequenceData = seq_data_dict[seq_id]
            
            output_token_ids = seq_data.get_output_token_ids()
            proposal_token_ids = all_proposal_tokens[i][:all_proposal_lengths[i]]
            new_output_token_ids = [*output_token_ids, *proposal_token_ids]
            
            print(f"\n序列 {i}:")
            print(f"  原始 seq_id: {seq_id}")
            print(f"  原始 output_token_ids 长度: {len(output_token_ids)}")
            print(f"  提案 token_ids: {proposal_token_ids}")
            print(f"  新 output_token_ids 长度: {len(new_output_token_ids)}")
            
            target_seq_id = target_seq_id_start + i
            new_seq_data = SequenceData.from_seqs(
                prompt_token_ids=seq_data.get_prompt_token_ids(),
                output_token_ids=new_output_token_ids,
            )
            new_seq_data.update_num_computed_tokens(
                len(seq_data.get_prompt_token_ids()) + len(output_token_ids) - 1)
            
            new_seq_group_metadata = SequenceGroupMetadata(
                request_id=seq_group_metadata.request_id,
                is_prompt=seq_group_metadata.is_prompt,
                seq_data={target_seq_id: new_seq_data},
                sampling_params=seq_group_metadata.sampling_params,
                block_tables={
                    target_seq_id: seq_group_metadata.block_tables[seq_id],
                },
                lora_request=None,
            )
            target_seq_group_metadata_list.append(new_seq_group_metadata)
            print(f"  新 seq_id: {target_seq_id}")
            print(f"  block_tables: {new_seq_group_metadata.block_tables}")
        
        # 先执行 target_sampler_output（这个应该能成功）
        print("\n执行 target_sampler_output...")
        try:
            target_sampler_output = scorer._scorer_worker.execute_model(
                execute_model_req=execute_model_req.clone(
                    seq_group_metadata_list=target_seq_group_metadata_list))
            target_sampler_output = target_sampler_output[0]
            print(f"✓ target_sampler_output 成功")
            print(f"  sampled_token_ids 形状: {target_sampler_output.sampled_token_ids.shape}")
        except Exception as e:
            print(f"✗ target_sampler_output 失败: {e}")
            traceback.print_exc()
            raise
        
        # 现在尝试执行 small_base_worker
        print("\n执行 small_base_worker...")
        print(f"  small_base_worker 类型: {type(scorer._small_base_worker)}")
        
        # 检查要传递的请求
        small_base_req = execute_model_req.clone(
            seq_group_metadata_list=target_seq_group_metadata_list)
        
        print(f"\n传递给 small_base_worker 的请求:")
        print(f"  num_lookahead_slots: {small_base_req.num_lookahead_slots}")
        print(f"  seq_group_metadata_list 长度: {len(small_base_req.seq_group_metadata_list)}")
        
        # 检查每个序列的 block_tables
        for i, seq_meta in enumerate(small_base_req.seq_group_metadata_list):
            print(f"\n序列组 {i}:")
            print(f"  is_prompt: {seq_meta.is_prompt}")
            print(f"  block_tables: {seq_meta.block_tables}")
            for seq_id, block_table in seq_meta.block_tables.items():
                print(f"    seq_id {seq_id}: block_table 长度 = {len(block_table)}")
                if len(block_table) > 0:
                    print(f"      第一个 block: {block_table[0]}")
        
        # 检查 CUDA 状态
        print(f"\nCUDA 状态:")
        print(f"  CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  当前设备: {torch.cuda.current_device()}")
            print(f"  设备名称: {torch.cuda.get_device_name()}")
            print(f"  已分配内存: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            print(f"  缓存内存: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        
        try:
            # 尝试执行
            print("\n开始调用 small_base_worker.execute_model...")
            small_base_sampler_output = scorer._small_base_worker.execute_model(
                execute_model_req=small_base_req)
            print(f"✓ small_base_worker.execute_model 成功")
            if small_base_sampler_output and len(small_base_sampler_output) > 0:
                print(f"  sampled_token_ids 形状: {small_base_sampler_output[0].sampled_token_ids.shape}")
        except RuntimeError as e:
            error_msg = str(e)
            print(f"\n✗ RuntimeError: {error_msg}")
            if "CUBLAS" in error_msg or "CUDA" in error_msg:
                print("\n可能的错误原因:")
                print("  1. KV cache 未正确初始化")
                print("  2. block_tables 中的 block 索引无效")
                print("  3. 序列长度与 KV cache 不匹配")
                print("  4. 张量形状不匹配")
                print("  5. 设备不匹配（CPU vs GPU）")
            traceback.print_exc()
            raise
        except Exception as e:
            print(f"\n✗ 其他错误: {e}")
            traceback.print_exc()
            raise
        
        # 如果成功，继续执行原始逻辑
        print("\n继续执行原始 score_proposals 逻辑...")
        return original_score_proposals(execute_model_req, proposals)
    
    # 替换方法
    scorer.score_proposals = debug_score_proposals
    
    print("\n6. 执行生成以触发错误...")
    try:
        outputs = llm.generate(prompts, sampling_params)
        print("\n✓ 生成成功！")
        for output in outputs:
            print(f"Prompt: {output.prompt!r}")
            print(f"Generated: {output.outputs[0].text!r}")
    except Exception as e:
        print(f"\n✗ 生成失败: {e}")
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("调试完成")
    print("=" * 60)

if __name__ == "__main__":
    debug_small_base_error()

