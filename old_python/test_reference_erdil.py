import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import json
import numpy as np
from erdil import *

results = {}

# Test 1: Model properties
results["model_properties"] = {
    "llama_3_70b": {
        "total_params": Llama_3_70B.total_params,
        "total_active_params": Llama_3_70B.total_active_params,
        "total_ff_params": Llama_3_70B.total_ff_params,
        "total_attn_params": Llama_3_70B.total_attn_params,
        "kv_cache_size_per_input_bytes": Llama_3_70B.kv_cache_size_per_input_bytes,
        "d_head": Llama_3_70B.d_head,
        "num_kv_heads": Llama_3_70B.num_kv_heads,
        "sparsity_factor": Llama_3_70B.sparsity_factor,
        "embedding_params": Llama_3_70B.embedding_params,
    },
    "deepseek_v3": {
        "total_params": DeepSeek_V3.total_params,
        "total_active_params": DeepSeek_V3.total_active_params,
        "kv_cache_size_per_input_bytes": DeepSeek_V3.kv_cache_size_per_input_bytes,
        "sparsity_factor": DeepSeek_V3.sparsity_factor,
    },
    "gpt_4": {
        "total_params": GPT_4.total_params,
        "total_active_params": GPT_4.total_active_params,
        "sparsity_factor": GPT_4.sparsity_factor,
    },
}

# Test 2: Model compute methods
results["model_compute"] = {
    "llama_3_70b_arithmetic_cost": Llama_3_70B.arithmetic_cost_flop(1000, 32),
    "llama_3_70b_ffd_flop": Llama_3_70B.ffd_flop(32),
    "llama_3_70b_attn_kv_flop": Llama_3_70B.attn_kv_flop(1000, 32),
    "llama_3_70b_memory_rw": Llama_3_70B.memory_reads_writes_bytes(1000, 32, 1, 8),
    "deepseek_v3_arithmetic_cost": DeepSeek_V3.arithmetic_cost_flop(1000, 32),
    "deepseek_v3_memory_rw": DeepSeek_V3.memory_reads_writes_bytes(1000, 32, 1, 8),
}

# Test 3: NCCL collective latency
results["nccl_latency"] = {
    "allreduce_8_1_LL": float(collective_latency_nccl_seconds(np.array([8]), np.array([1]))[0]),
    "allreduce_8_1_LL128": float(collective_latency_nccl_seconds(np.array([8]), np.array([1]), algo="LL128")[0]),
    "allreduce_64_8_LL": float(collective_latency_nccl_seconds(np.array([64]), np.array([8]))[0]),
    "reducescatter_8_1_LL": float(collective_latency_nccl_seconds(np.array([8]), np.array([1]), coll="reducescatter")[0]),
}

# Test 4: Mean collective time
results["mean_collective_time"] = {
    "allreduce_8gpus_1mb": float(mean_collective_time_nccl_seconds(
        np.array([8]), np.array([1]), 1e6, H100, "allreduce", (1,1))[0]),
    "allreduce_64gpus_1mb": float(mean_collective_time_nccl_seconds(
        np.array([64]), np.array([8]), 1e6, H100, "allreduce", (1,1))[0]),
    "allreduce_8gpus_10mb": float(mean_collective_time_nccl_seconds(
        np.array([8]), np.array([1]), 1e7, H100, "allreduce", (1,1))[0]),
    "p2p_2gpus_1mb": float(mean_collective_time_nccl_seconds(
        np.array([2]), np.array([1]), 1e6, H100, "p2p", (1,1))[0]),
}

# Test 5: Token latency - single GPU
results["token_latency"] = {
    "llama70b_1gpu_bs1": float(new_token_latency_seconds(
        np.array([1.0]), Llama_3_70B, H100, np.array([1.0]), input_len=0)[0]),
    "llama70b_8gpu_bs32": float(new_token_latency_seconds(
        np.array([8.0]), Llama_3_70B, H100, np.array([32.0]), input_len=1000)[0]),
    "llama70b_8gpu_bs1": float(new_token_latency_seconds(
        np.array([8.0]), Llama_3_70B, H100, np.array([1.0]), input_len=0)[0]),
    "llama405b_64gpu_bs32": float(new_token_latency_seconds(
        np.array([64.0]), Llama_3_405B, H100, np.array([32.0]), input_len=4096, use_pp=True)[0]),
    "deepseek_128gpu_bs400": float(new_token_latency_seconds(
        np.array([128.0]), DeepSeek_V3, H100, np.array([400.0]), input_len=100000, use_pp=True)[0]),
}

# Test 6: Speculative decoding
results["spec_dec"] = {
    "llama70b_spec_8b": float(spec_dec_token_latency_seconds(
        np.array([8.0]), Llama_3_70B, Llama_3_8B, H100, np.array([32.0]),
        acceptance_prob=0.8, gamma_max=4, input_len=1000)[0]),
}

# Test 7: matmul_rw_bytes
results["matmul_rw"] = {
    "basic": float(matmul_rw_bytes(1024, 4096, 32)),
    "with_tp": float(matmul_rw_bytes(1024, 4096, 32, tp=8)),
    "fp8": float(matmul_rw_bytes(1024, 4096, 32, wp_bytes=1, ap_bytes=1)),
}

# Convert any remaining numpy types to Python types
def convert(obj):
    if isinstance(obj, dict):
        return {k: convert(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, float) and (obj == float('inf') or obj == float('-inf')):
        return str(obj)
    return obj

results = convert(results)

os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'test'), exist_ok=True)
with open(os.path.join(os.path.dirname(__file__), '..', 'test', 'reference-erdil.json'), 'w') as f:
    json.dump(results, f, indent=2)

print("Generated test/reference-erdil.json")
print(json.dumps(results, indent=2))
