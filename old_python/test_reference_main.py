import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import json
import numpy as np
from main import *

results = {}

# Test 1: Communication functions
results["comm"] = {
    "reduce_scatter_ring_8": list(reduce_scatter_time_s(1e6, 8, 450.0, 1.2)),
    "reduce_scatter_tree_8": list(reduce_scatter_time_s(1e6, 8, 450.0, 1.2, algo="tree")),
    "allgather_ring_8": list(allgather_time_s(1e6, 8, 450.0, 1.2)),
    "allreduce_ring_8": list(allreduce_time_s(1e6, 8, 450.0, 1.2)),
    "allreduce_tree_8": list(allreduce_time_s(1e6, 8, 450.0, 1.2, algo="tree")),
    "allreduce_1gpu": list(allreduce_time_s(1e6, 1, 450.0, 1.2)),
    "p2p_1mb_nvlink": list(p2p_time_s(1e6, 450.0, 2.0)),
    "p2p_1mb_ib": list(p2p_time_s(1e6, 50.0, 10.0)),
    "pipeline_util_4_4": pipeline_utilization(4, 4),
    "pipeline_util_8_16": pipeline_utilization(8, 16),
    "pipeline_util_1_1": pipeline_utilization(1, 1),
}

# Test 2: Hierarchical allreduce
comm = CommParams()
results["hierarchical"] = {
    "intra_node_4gpus": list(hierarchical_allreduce_time_s(1e6, 4, 8, comm)),
    "intra_node_8gpus": list(hierarchical_allreduce_time_s(1e6, 8, 8, comm)),
    "cross_node_16gpus": list(hierarchical_allreduce_time_s(1e6, 16, 8, comm)),
    "cross_node_64gpus": list(hierarchical_allreduce_time_s(1e6, 64, 8, comm)),
}

# Test 3: Model properties
model = Model()  # defaults to Llama 3 70B
results["model_props"] = {
    "weight_size_gb": model.weight_size_gb,
    "weight_per_layer_gb": model.weight_per_layer_gb,
    "largest_matrix_mb": model.largest_matrix_mb,
    "flops_per_token": model.flops_per_token,
    "kv_cache_per_token_kb": model.kv_cache_per_token_kb,
}

# Test 4: Inference throughput
hw = Hardware()
results["inference"] = {}

# Mamba, full HBM
tp, config = calculate_throughput(hw, model, 1.0, 1.0, None, bw_overlap_frac=0.5)
results["inference"]["mamba_full_hbm"] = {
    "throughput": tp,
    "pp": config.get("pp"),
    "tp": config.get("tp"),
}

# Transformer 2k, full HBM
tp, config = calculate_throughput(hw, model, 1.0, 1.0, 2000, bw_overlap_frac=0.5)
results["inference"]["transformer_2k_full_hbm"] = {
    "throughput": tp,
    "pp": config.get("pp"),
    "tp": config.get("tp"),
}

# Mamba, 5% HBM
tp, config = calculate_throughput(hw, model, 0.05, 1.0, None, bw_overlap_frac=0.5)
results["inference"]["mamba_5pct_hbm"] = {
    "throughput": tp,
    "pp": config.get("pp"),
    "tp": config.get("tp"),
}

# Test 5: Training throughput
train_cfg = TrainingConfig(seq_len=8192, context_length=8192)
tp, config = calculate_training_throughput(hw, model, 1.0, 1.0, train_cfg, bw_overlap_frac=0.5)
results["training"] = {
    "transformer_8k_full_hbm": {
        "throughput": tp,
        "pp": config.get("pp"),
        "tp": config.get("tp"),
    }
}

# Mamba training
train_cfg_mamba = TrainingConfig(seq_len=8192, context_length=None)
tp, config = calculate_training_throughput(hw, model, 1.0, 1.0, train_cfg_mamba, bw_overlap_frac=0.5)
results["training"]["mamba_8k_full_hbm"] = {
    "throughput": tp,
    "pp": config.get("pp"),
    "tp": config.get("tp"),
}

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
with open(os.path.join(os.path.dirname(__file__), '..', 'test', 'reference-main.json'), 'w') as f:
    json.dump(results, f, indent=2)

print("Generated test/reference-main.json")
print(json.dumps(results, indent=2))
