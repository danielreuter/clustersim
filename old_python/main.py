import math
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Tuple

# =============================================================================
# Communication Model (α-β with algorithm-aware step counts)
# =============================================================================

@dataclass
class CommParams:
    """
    Communication parameters for α-β model.

    Default values calibrated from NCCL source code (Erdil 2025, arXiv:2506.04645).
    Based on NVIDIA's tuning code in src/graph/tuning.cc.

    The full NCCL latency formula for tree allreduce is:
        t_reduce = 6.8µs + 1.2µs × (intra_ranks - 1) + 10µs × log₂(inter_nodes)

    Note: These are PER-STEP latencies, not per-operation.
    Ring allreduce has 2*(n-1) steps, so per-op latency = steps × alpha + kernel_launch.
    """
    # Kernel/driver launch overhead - paid once per collective (from NCCL)
    kernel_launch_us: float = 6.8    # base latency per collective operation

    # NVLink (intra-node) - collective per step (NCCL: 1.2µs)
    nvlink_alpha_us: float = 1.2     # per-step latency (microseconds)
    nvlink_bw_gb_s: float = 450.0    # bandwidth (GB/s)

    # InfiniBand (inter-node) - collective per step (NCCL: 10µs for tree)
    ib_alpha_us: float = 10.0        # per-step latency (microseconds)
    ib_bw_gb_s: float = 50.0         # bandwidth (GB/s)

    # Point-to-point (for PP transfers) - separate from collectives
    nv_p2p_alpha_us: float = 2.0     # NVLink p2p latency
    nv_p2p_bw_gb_s: float = 450.0    # NVLink p2p bandwidth
    ib_p2p_alpha_us: float = 10.0    # IB p2p latency (higher due to network)
    ib_p2p_bw_gb_s: float = 50.0     # IB p2p bandwidth


def reduce_scatter_time_s(
    bytes_msg: float,
    n: int,
    bw_gb_s: float,
    alpha_us: float,
    algo: str = "ring"
) -> Tuple[float, float]:
    """
    α-β model for reduce-scatter (half of allreduce).
    Returns (latency_s, bandwidth_s) separately.
    """
    if n <= 1:
        return 0.0, 0.0

    if algo == "ring":
        steps = n - 1
        bytes_factor = (n - 1) / n
    elif algo == "tree":
        steps = math.ceil(math.log2(n))
        bytes_factor = (n - 1) / n
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    alpha_s = alpha_us * 1e-6
    bw = bw_gb_s * 1e9

    return steps * alpha_s, bytes_factor * (bytes_msg / bw)


def allgather_time_s(
    bytes_msg: float,
    n: int,
    bw_gb_s: float,
    alpha_us: float,
    algo: str = "ring"
) -> Tuple[float, float]:
    """
    α-β model for allgather (same as reduce-scatter for ring).
    Returns (latency_s, bandwidth_s) separately.
    """
    return reduce_scatter_time_s(bytes_msg, n, bw_gb_s, alpha_us, algo)


def allreduce_time_s(
    bytes_msg: float,
    n: int,
    bw_gb_s: float,
    alpha_us: float,
    algo: str = "ring"
) -> Tuple[float, float]:
    """
    α-β model for allreduce = reduce-scatter + allgather.
    Returns (latency_s, bandwidth_s) separately.
    """
    rs_lat, rs_bw = reduce_scatter_time_s(bytes_msg, n, bw_gb_s, alpha_us, algo)
    ag_lat, ag_bw = allgather_time_s(bytes_msg, n, bw_gb_s, alpha_us, algo)
    return rs_lat + ag_lat, rs_bw + ag_bw


def hierarchical_allreduce_time_s(
    bytes_msg: float,
    tp: int,
    gpus_per_node: int,
    comm: CommParams,
    intra_algo: str = "ring",
    inter_algo: str = "tree"
) -> Tuple[float, float]:
    """
    Hierarchical allreduce: intra-node reduce-scatter + inter-node allreduce + intra-node allgather.

    Includes kernel launch overhead from NCCL (paid once per collective).
    """
    nodes = math.ceil(tp / gpus_per_node)
    gpus_in_tp_per_node = min(tp, gpus_per_node)

    # Kernel launch latency (paid once per collective operation)
    kernel_lat = comm.kernel_launch_us * 1e-6

    if nodes <= 1:
        # Pure intra-node: full allreduce
        lat, bw = allreduce_time_s(bytes_msg, tp, comm.nvlink_bw_gb_s,
                                   comm.nvlink_alpha_us, intra_algo)
        return (lat + kernel_lat, bw)

    # Phase 1: Intra-node reduce-scatter
    rs_lat, rs_bw = reduce_scatter_time_s(
        bytes_msg, gpus_in_tp_per_node,
        comm.nvlink_bw_gb_s, comm.nvlink_alpha_us, intra_algo
    )

    # Phase 2: Inter-node allreduce on sharded data (each GPU has bytes_msg/gpn)
    inter_bytes = bytes_msg / gpus_in_tp_per_node
    ar_lat, ar_bw = allreduce_time_s(
        inter_bytes, nodes,
        comm.ib_bw_gb_s, comm.ib_alpha_us, inter_algo
    )

    # Phase 3: Intra-node allgather
    ag_lat, ag_bw = allgather_time_s(
        bytes_msg, gpus_in_tp_per_node,
        comm.nvlink_bw_gb_s, comm.nvlink_alpha_us, intra_algo
    )

    return (rs_lat + ar_lat + ag_lat + kernel_lat, rs_bw + ar_bw + ag_bw)


def p2p_time_s(
    bytes_msg: float,
    bw_gb_s: float,
    alpha_us: float
) -> Tuple[float, float]:
    """Point-to-point transfer time. Returns (latency_s, bandwidth_s)."""
    return alpha_us * 1e-6, bytes_msg / (bw_gb_s * 1e9)


def pipeline_utilization(pp: int, M: int) -> float:
    """
    Pipeline utilization factor.
    U = M / (M + PP - 1)
    """
    if M <= 0 or pp <= 0:
        return 0.0
    return M / (M + pp - 1)


# =============================================================================
# Hardware Configuration
# =============================================================================

@dataclass
class Hardware:
    """H100 SXM specs"""
    n_gpus: int = 10_000
    gpus_per_node: int = 8

    # Baseline specs (at 100%)
    hbm_capacity_gb: float = 80.0
    hbm_bandwidth_tb_s: float = 2.5  # sustained
    compute_tflops: float = 700.0    # sustained BF16


# =============================================================================
# Model Configuration
# =============================================================================

@dataclass
class Model:
    """Llama 3 70B specs"""
    name: str = "Llama 3 70B"
    n_params: float = 70e9
    n_layers: int = 80
    d_model: int = 8192
    d_ff: int = 28672
    n_heads: int = 64
    n_kv_heads: int = 8

    precision_bytes: int = 2  # BF16

    @property
    def weight_size_gb(self) -> float:
        return self.n_params * self.precision_bytes / 1e9

    @property
    def weight_per_layer_gb(self) -> float:
        return self.weight_size_gb / self.n_layers

    @property
    def largest_matrix_mb(self) -> float:
        return self.d_model * self.d_ff * self.precision_bytes / 1e6

    @property
    def flops_per_token(self) -> float:
        return 2 * self.n_params

    @property
    def kv_cache_per_token_kb(self) -> float:
        d_head = self.d_model // self.n_heads
        return 2 * self.n_layers * self.n_kv_heads * d_head * self.precision_bytes / 1e3

    @property
    def mamba_state_mb(self) -> float:
        d_inner = 16384
        d_state = 64
        return d_inner * d_state * self.n_layers * self.precision_bytes / 1e6


# =============================================================================
# Training Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    """
    Training-specific modeling knobs.

    Defaults assume "nice tricks":
      - 1F1B pipeline schedule
      - Activation checkpointing (recompute forward once)
      - Sequence-parallel activations (sharded by TP)
      - DiLoCo-style infrequent DP sync (amortized)

    Key dials that significantly affect results:
      - param_state_mult: 3.0 = aggressive (ZeRO + 8-bit), 6-8 = conservative (full AdamW)
      - pp_activation_sharded_by_tp: True = optimistic, False = Megatron default
      - tp_comm_seq_parallel: True = smaller TP messages (optimistic), False = full d_model
    """
    seq_len: int
    context_length: Optional[int] = None  # None = Mamba (no attention), int = Transformer

    # Compute multipliers relative to ONE forward pass
    bwd_compute_mult: float = 2.0          # backward ≈ 2x forward
    recompute_fwd_mult: float = 1.0        # 1.0 = full activation checkpointing

    # Parameter-state HBM multiplier relative to BF16 weight shard size.
    # Encodes optimizer choice, ZeRO, 8-bit states, etc.
    #   - AdamW full states (no ZeRO): ~6–8x  (conservative)
    #   - ZeRO-1/2 + standard optimizer: ~4–5x
    #   - ZeRO-3 / 8-bit optimizer: ~2–3x    (aggressive)
    param_state_mult: float = 3.0

    # Activation storage: d_model-sized BF16 tensors per token at stage boundary
    act_tensors_per_token: float = 2.0
    act_shard_by_tp: bool = True           # sequence parallelism for activations

    # Pipeline schedule affects # microbatches whose activations are live
    pipeline_schedule: str = "1f1b"        # "1f1b" or "gpipe"

    # Communication multipliers (relative to inference forward)
    tp_comm_bwd_mult: float = 1.0          # backward comm ~ forward comm
    pp_directions: int = 2                 # forward activations + backward grads

    # TP communication payload sharding under sequence parallelism
    # True = messages are sharded by TP (optimistic: RS/AG instead of full AR)
    # False = full d_model messages (conservative: standard Megatron TP)
    tp_comm_seq_parallel: bool = False

    # Gradient buffer bandwidth (accumulation). 2.0 ~ read+write
    grad_accum_rw_mult: float = 2.0

    # DiLoCo / periodic DP sync
    diloco_sync_every: int = 64            # sync every K optimizer steps
    diloco_payload_frac: float = 1.0       # 1.0 = full payload; <1 = compression
    dp_algo: str = "tree"

    # PP activation sharding at pipeline boundaries
    # True = optimistic (seq-parallel boundaries), False = full activations per TP rank (Megatron default)
    pp_activation_sharded_by_tp: bool = False  # Changed default to conservative

    # Activation checkpointing granularity
    # "stage" = checkpoint only at stage boundaries (optimistic memory, fewer ckpts)
    # "layer" = checkpoint every layer in the stage (conservative, realistic for large models)
    act_ckpt_granularity: str = "layer"

    # Practical guards to keep optimizer from producing unrealistic configs
    hbm_headroom_frac: float = 0.10           # reserve 10% for workspace/fragmentation
    max_microbatch_tokens: Optional[int] = 2_000_000  # cap to prevent absurd microbatches


# =============================================================================
# Inference Throughput Calculator
# =============================================================================

def calculate_throughput(
    hw: Hardware,
    model: Model,
    hbm_fraction: float,
    compute_fraction: float,
    context_length: Optional[int] = None,
    comm: Optional[CommParams] = None,
    bw_overlap_frac: float = 0.5,  # 0 = no overlap, 1 = full bandwidth overlap
    max_tp: int = 128,  # Maximum TP to consider
) -> Tuple[float, dict]:
    """
    Calculate max throughput with explicit α-β comm model.

    Assumes filled/continuous batching pipeline (standard for inference serving).

    Key features:
    - Microbatch count M is a decision variable (searched over)
    - Hierarchical allreduce for cross-node TP
    - PP transfers include latency and count send+recv
    - Overlap factor: only bandwidth overlaps, not latency
    """
    if comm is None:
        comm = CommParams()

    # Scale hardware
    hbm_per_gpu_gb = hw.hbm_capacity_gb * hbm_fraction
    hbm_per_gpu_mb = hbm_per_gpu_gb * 1000
    hbm_bw_tb_s = hw.hbm_bandwidth_tb_s * compute_fraction
    compute_tflops = hw.compute_tflops * compute_fraction

    # FLOPs per token
    base_flops_per_token = 2 * model.flops_per_token
    if context_length is not None:
        attention_flops = 4 * model.n_layers * model.d_model * context_length
        total_flops_per_token = base_flops_per_token + attention_flops
    else:
        total_flops_per_token = base_flops_per_token

    # Compute ceiling
    cluster_tflops = hw.n_gpus * compute_tflops
    compute_ceiling = cluster_tflops * 1e12 / total_flops_per_token

    # State size per sequence
    if context_length is None:
        state_per_seq_mb = model.mamba_state_mb
        kv_read_per_token_mb = 0
    else:
        state_per_seq_mb = context_length * model.kv_cache_per_token_kb / 1000
        kv_read_per_token_mb = state_per_seq_mb

    # Minimum TP required
    min_tp = 1
    if hbm_per_gpu_mb < model.largest_matrix_mb * 1.1:
        min_tp = int(np.ceil(model.largest_matrix_mb * 1.1 / hbm_per_gpu_mb))

    best_throughput = 0
    best_config = None

    # Search over configurations
    pp_values = list(range(1, 81))
    tp_values = [t for t in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512] if t <= max_tp]

    for pp in pp_values:
        if pp > model.n_layers:
            continue
        layers_per_stage = model.n_layers / pp

        for tp in tp_values:
            if tp < min_tp:
                continue

            gpus_per_replica = pp * tp
            if gpus_per_replica > hw.n_gpus:
                continue

            n_replicas = hw.n_gpus // gpus_per_replica
            if n_replicas < 1:
                continue

            # =================================================================
            # Memory constraints
            # =================================================================
            weight_per_gpu_mb = model.weight_size_gb * 1000 / gpus_per_replica
            if weight_per_gpu_mb > hbm_per_gpu_mb * 0.95:
                continue

            # KV cache sharding limited by n_kv_heads for GQA models
            tp_kv_mem = min(tp, model.n_kv_heads) if context_length is not None else tp
            state_per_gpu_per_seq_mb = state_per_seq_mb / (pp * tp_kv_mem)
            activation_per_seq_mb = model.d_model * model.precision_bytes * 2 / 1e6
            activation_per_gpu_per_seq_mb = activation_per_seq_mb / tp

            remaining_mb = hbm_per_gpu_mb - weight_per_gpu_mb
            if remaining_mb <= 0:
                continue

            mem_per_seq = state_per_gpu_per_seq_mb + activation_per_gpu_per_seq_mb
            if mem_per_seq > 0:
                max_B_total = int(remaining_mb / mem_per_seq)
            else:
                max_B_total = 10000
            max_B_total = min(max_B_total, 50000)
            if max_B_total < 1:
                continue

            # =================================================================
            # Search over microbatch counts M
            # =================================================================
            # Expanded search: powers of 2 + PP-relative values
            M_values = set([1, 2, 4, 8, 16, 32, 64, pp // 2, pp, 2 * pp, 4 * pp])
            M_values |= set(range(max(1, pp - 8), pp + 9))  # denser search around pp
            M_values = sorted([M for M in M_values if 1 <= M <= max_B_total])

            for M in M_values:
                # Microbatch size - FIX: don't use max(1, ...) which can violate memory
                m = max_B_total // M
                if m < 1:
                    continue
                B_total = m * M  # Guaranteed <= max_B_total

                # =============================================================
                # Per-microbatch time calculation
                # =============================================================

                # Compute time per microbatch
                flops_per_microbatch = m * total_flops_per_token * layers_per_stage / model.n_layers
                t_compute_per_mb_s = flops_per_microbatch / (tp * compute_tflops * 1e12)

                # Weight read per microbatch
                weight_per_stage_mb = model.weight_size_gb * 1000 / (pp * tp)
                t_weight_read_per_mb_s = weight_per_stage_mb / (hbm_bw_tb_s * 1e6)

                # KV/State read per microbatch
                # Note: KV cache sharding is limited by n_kv_heads (GQA)
                # If tp > n_kv_heads, KV is replicated not sharded further
                if context_length is not None:
                    tp_kv = min(tp, model.n_kv_heads)
                    kv_per_mb_mb = m * kv_read_per_token_mb / (pp * tp_kv)
                    t_kv_read_per_mb_s = kv_per_mb_mb / (hbm_bw_tb_s * 1e6)
                    t_state_rw_per_mb_s = 0
                else:
                    t_kv_read_per_mb_s = 0
                    state_rw_per_mb_mb = m * state_per_seq_mb / (pp * tp) * 2
                    t_state_rw_per_mb_s = state_rw_per_mb_mb / (hbm_bw_tb_s * 1e6)

                t_mem_per_mb_s = t_weight_read_per_mb_s + t_kv_read_per_mb_s + t_state_rw_per_mb_s

                # =============================================================
                # TP communication per microbatch
                # =============================================================
                t_tp_lat_per_mb_s = 0
                t_tp_bw_per_mb_s = 0

                if tp > 1:
                    n_allreduces = 2 * layers_per_stage
                    bytes_per_allreduce = m * model.d_model * model.precision_bytes

                    lat_one, bw_one = hierarchical_allreduce_time_s(
                        bytes_per_allreduce, tp, hw.gpus_per_node, comm
                    )
                    t_tp_lat_per_mb_s = n_allreduces * lat_one
                    t_tp_bw_per_mb_s = n_allreduces * bw_one

                # =============================================================
                # PP communication per microbatch (send + recv for interior stages)
                # =============================================================
                t_pp_lat_per_mb_s = 0
                t_pp_bw_per_mb_s = 0

                if pp > 1:
                    # Note: In Megatron-style TP, after allreduce each rank has FULL hidden state
                    # So PP transfers are full activations, not sharded by TP
                    bytes_per_transfer = m * model.d_model * model.precision_bytes

                    # Determine node crossings
                    if tp <= hw.gpus_per_node:
                        stages_per_node = hw.gpus_per_node // tp
                    else:
                        stages_per_node = 1

                    n_nodes_for_pp = int(np.ceil(pp / stages_per_node)) if stages_per_node > 0 else pp
                    n_crossings = max(0, n_nodes_for_pp - 1)
                    frac_ib = n_crossings / (pp - 1) if pp > 1 else 0

                    # Use proper p2p params (not collective params)
                    alpha_eff = frac_ib * comm.ib_p2p_alpha_us + (1 - frac_ib) * comm.nv_p2p_alpha_us
                    if frac_ib > 0:
                        bw_eff = 1 / (frac_ib / comm.ib_p2p_bw_gb_s + (1 - frac_ib) / comm.nv_p2p_bw_gb_s)
                    else:
                        bw_eff = comm.nv_p2p_bw_gb_s

                    lat, bw = p2p_time_s(bytes_per_transfer, bw_eff, alpha_eff)

                    # Interior stages do send+recv - pessimistic: counts serially
                    # In practice, send/recv can partially overlap with each other and compute
                    p2p_factor = 2 if pp > 2 else 1
                    t_pp_lat_per_mb_s = p2p_factor * lat
                    t_pp_bw_per_mb_s = p2p_factor * bw

                # =============================================================
                # Total comm per microbatch
                # =============================================================
                t_comm_lat_per_mb_s = t_tp_lat_per_mb_s + t_pp_lat_per_mb_s
                t_comm_bw_per_mb_s = t_tp_bw_per_mb_s + t_pp_bw_per_mb_s

                # =============================================================
                # Microbatch time with overlap model
                # Simple interpolation: t = (1-γ)*t_no_overlap + γ*t_full_overlap
                # =============================================================
                t_compute_mem = max(t_compute_per_mb_s, t_mem_per_mb_s)

                t_no_overlap = t_compute_mem + t_comm_lat_per_mb_s + t_comm_bw_per_mb_s
                t_full_overlap = max(t_compute_mem, t_comm_bw_per_mb_s) + t_comm_lat_per_mb_s

                t_microbatch_s = (1 - bw_overlap_frac) * t_no_overlap + bw_overlap_frac * t_full_overlap

                if t_microbatch_s <= 0:
                    continue

                # =============================================================
                # Throughput with pipeline utilization (filled/continuous batching)
                # =============================================================
                # Continuous batching: pipeline stays full as long as M >= PP
                # This is standard for modern inference serving (vLLM, TensorRT-LLM)
                util = min(1.0, M / pp)

                # Time per microbatch * M microbatches, adjusted by utilization
                throughput_per_replica = (m / t_microbatch_s) * util
                cluster_throughput = n_replicas * throughput_per_replica

                # Determine bottleneck
                if t_comm_lat_per_mb_s > max(t_compute_mem, t_comm_bw_per_mb_s):
                    bottleneck = "Comm Latency"
                elif t_comm_bw_per_mb_s > t_compute_mem:
                    bottleneck = "Comm Bandwidth"
                elif t_compute_per_mb_s >= t_mem_per_mb_s:
                    bottleneck = "Compute"
                else:
                    mem_terms = [("Weight Read", t_weight_read_per_mb_s),
                                 ("KV Read", t_kv_read_per_mb_s),
                                 ("State RW", t_state_rw_per_mb_s)]
                    bottleneck = max(mem_terms, key=lambda x: x[1])[0]

                if cluster_throughput > best_throughput:
                    best_throughput = cluster_throughput
                    best_config = {
                        "pp": pp,
                        "tp": tp,
                        "M": M,
                        "m": m,
                        "gpus_per_replica": gpus_per_replica,
                        "n_replicas": n_replicas,
                        "B_total": B_total,
                        "layers_per_stage": layers_per_stage,
                        "weight_per_gpu_mb": weight_per_gpu_mb,
                        "remaining_mb": remaining_mb,
                        "state_per_seq_mb": state_per_seq_mb,
                        "t_compute_per_mb_ms": t_compute_per_mb_s * 1000,
                        "t_weight_read_per_mb_ms": t_weight_read_per_mb_s * 1000,
                        "t_kv_read_per_mb_ms": t_kv_read_per_mb_s * 1000,
                        "t_state_rw_per_mb_ms": t_state_rw_per_mb_s * 1000,
                        "t_tp_lat_per_mb_ms": t_tp_lat_per_mb_s * 1000,
                        "t_tp_bw_per_mb_ms": t_tp_bw_per_mb_s * 1000,
                        "t_pp_lat_per_mb_ms": t_pp_lat_per_mb_s * 1000,
                        "t_pp_bw_per_mb_ms": t_pp_bw_per_mb_s * 1000,
                        "t_microbatch_ms": t_microbatch_s * 1000,
                        "pipeline_util": util,
                        "throughput_per_replica": throughput_per_replica,
                        "hbm_per_gpu_gb": hbm_per_gpu_gb,
                        "bottleneck": bottleneck,
                        "flops_per_token": total_flops_per_token,
                        "bw_overlap_frac": bw_overlap_frac,
                        "schedule": "filled",
                    }

    if best_config is None:
        return 0.0, {"error": "No valid configuration found"}

    best_config["compute_ceiling"] = compute_ceiling
    best_config["efficiency"] = best_throughput / compute_ceiling if compute_ceiling > 0 else 0

    return best_throughput, best_config


# =============================================================================
# Training Throughput Calculator
# =============================================================================

def calculate_training_throughput(
    hw: Hardware,
    model: Model,
    hbm_fraction: float,
    compute_fraction: float,
    train: TrainingConfig,
    comm: Optional[CommParams] = None,
    bw_overlap_frac: float = 0.5,
    max_tp: int = 128,  # Maximum TP to consider
) -> Tuple[float, dict]:
    """
    Training throughput model: forward + backward (+ optional recompute),
    with TP/PP + DiLoCo amortized DP sync.

    Assumes filled pipeline (continuous batching). For training with 1F1B,
    this is optimistic but simplifies the model.

    Returns: (cluster_throughput_tok_s, debug_info)
    """
    if comm is None:
        comm = CommParams()

    # Scale hardware (with headroom for workspace/fragmentation)
    hbm_per_gpu_gb = hw.hbm_capacity_gb * hbm_fraction
    hbm_per_gpu_mb_raw = hbm_per_gpu_gb * 1000
    hbm_per_gpu_mb = hbm_per_gpu_mb_raw * (1.0 - train.hbm_headroom_frac)
    hbm_bw_tb_s = hw.hbm_bandwidth_tb_s * compute_fraction
    compute_tflops = hw.compute_tflops * compute_fraction

    # FLOPs per token (forward only)
    base_fwd_flops_per_token = 2 * model.flops_per_token

    # Training attention FLOPs: per-token ~ 2 * d_model * seq_len per layer
    # Only add attention FLOPs for Transformers (context_length is not None)
    if train.context_length is not None:
        attention_fwd_flops_per_token = 2 * model.n_layers * model.d_model * train.seq_len
        fwd_flops_per_token = base_fwd_flops_per_token + attention_fwd_flops_per_token
    else:
        # Mamba: no attention, just the base MLP/SSM compute
        fwd_flops_per_token = base_fwd_flops_per_token

    # Total training multiplier: forward + backward + recompute-forward
    compute_mult = 1.0 + train.bwd_compute_mult + train.recompute_fwd_mult
    train_flops_per_token = fwd_flops_per_token * compute_mult

    # Compute ceiling (tokens/s)
    cluster_tflops = hw.n_gpus * compute_tflops
    compute_ceiling = cluster_tflops * 1e12 / train_flops_per_token

    # Search space
    pp_values = list(range(1, model.n_layers + 1))
    tp_values = [t for t in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512] if t <= max_tp]

    best_throughput = 0.0
    best_config = None

    for pp in pp_values:
        layers_per_stage = model.n_layers / pp

        for tp in tp_values:
            gpus_per_replica = pp * tp
            if gpus_per_replica > hw.n_gpus:
                continue

            n_replicas = hw.n_gpus // gpus_per_replica
            if n_replicas < 1:
                continue

            # Parameter-state memory (weights + grads + optimizer states)
            weight_shard_mb = model.weight_size_gb * 1000 / gpus_per_replica
            param_state_mb = weight_shard_mb * train.param_state_mult

            if param_state_mb > hbm_per_gpu_mb * 0.95:
                continue

            # Microbatch count search (M)
            M_values = set([1, 2, 4, 8, 16, 32, 64, pp // 2, pp, 2 * pp, 4 * pp])
            M_values |= set(range(max(1, pp - 8), pp + 9))  # denser search around pp
            M_values = sorted([M for M in M_values if M >= 1])

            for M in M_values:
                # How many microbatches' activations are live on a stage?
                if train.pipeline_schedule == "gpipe":
                    in_flight = M
                else:
                    # 1F1B: bounded by pipeline depth
                    in_flight = min(M, pp)

                # Activation memory per sequence per GPU
                # Checkpoint granularity determines how many checkpoint points per stage
                if train.act_ckpt_granularity == "stage":
                    ckpt_points = 1  # Only checkpoint at stage boundaries (optimistic)
                elif train.act_ckpt_granularity == "layer":
                    ckpt_points = int(math.ceil(layers_per_stage))  # Checkpoint every layer (realistic)
                else:
                    raise ValueError(f"Unknown act_ckpt_granularity={train.act_ckpt_granularity}")

                act_per_seq_mb = (
                    train.act_tensors_per_token
                    * ckpt_points  # Scale by number of checkpoint points
                    * train.seq_len
                    * model.d_model
                    * model.precision_bytes
                    / 1e6
                )
                act_shard = tp if train.act_shard_by_tp else 1
                act_per_seq_per_gpu_mb = act_per_seq_mb / act_shard

                # Available memory for activations
                avail_act_mb = hbm_per_gpu_mb - param_state_mb
                if avail_act_mb <= 0:
                    continue

                # Max microbatch size m (sequences per microbatch)
                denom = in_flight * act_per_seq_per_gpu_mb
                if denom <= 0:
                    continue
                m = int(avail_act_mb // denom)
                if m < 1:
                    continue

                # Cap microbatch tokens to prevent absurdly huge microbatches
                tokens_per_microbatch = m * train.seq_len
                if train.max_microbatch_tokens is not None:
                    m_cap = max(1, train.max_microbatch_tokens // train.seq_len)
                    if m > m_cap:
                        m = m_cap
                        tokens_per_microbatch = m * train.seq_len

                # =============================================================
                # Compute time per microbatch (fwd + bwd + recompute)
                # =============================================================
                flops_per_microbatch_stage = (
                    tokens_per_microbatch
                    * train_flops_per_token
                    * (layers_per_stage / model.n_layers)
                )
                t_compute_s = flops_per_microbatch_stage / (tp * compute_tflops * 1e12)

                # =============================================================
                # HBM bandwidth time per microbatch
                # =============================================================
                weight_per_stage_mb = model.weight_size_gb * 1000 / (pp * tp)

                # Weights: fwd + backward dX + recompute fwd
                weight_reads_mult = 2.0 + train.recompute_fwd_mult
                weight_read_mb = weight_reads_mult * weight_per_stage_mb

                # Gradient accumulation buffer
                grad_accum_mb = train.grad_accum_rw_mult * weight_per_stage_mb

                # Activation bandwidth: checkpoint read/write + backward activation reads
                # With activation checkpointing:
                #   - Forward: write boundary activations (1x per layer)
                #   - Recompute: read checkpoint, write intermediate (counted in recompute)
                #   - Backward: read activations for gradient computation
                # Approximate as ~4x activation tensor traffic per layer (conservative)
                act_tensors_per_layer = 4.0  # read + write for fwd, read for bwd, misc
                act_bytes_per_token = model.d_model * model.precision_bytes
                act_shard = tp if train.act_shard_by_tp else 1
                act_bw_mb = (
                    tokens_per_microbatch
                    * act_bytes_per_token
                    * layers_per_stage
                    * act_tensors_per_layer
                    / act_shard
                    / 1e6
                )

                t_mem_s = (weight_read_mb + grad_accum_mb + act_bw_mb) / (hbm_bw_tb_s * 1e6)

                # =============================================================
                # TP communication per microbatch
                # =============================================================
                t_tp_lat_s = 0.0
                t_tp_bw_s = 0.0
                if tp > 1:
                    ar_per_layer_fwd = 2.0
                    ar_per_layer_bwd = ar_per_layer_fwd * train.tp_comm_bwd_mult
                    ar_per_layer_recomp = ar_per_layer_fwd * train.recompute_fwd_mult
                    n_allreduces = layers_per_stage * (ar_per_layer_fwd + ar_per_layer_bwd + ar_per_layer_recomp)

                    # Under sequence parallelism, TP collectives are RS/AG on sharded activations
                    # Conservative (False): full d_model allreduce
                    # Optimistic (True): sharded by TP (like sequence-parallel RS/AG)
                    bytes_per_allreduce = tokens_per_microbatch * model.d_model * model.precision_bytes
                    if train.tp_comm_seq_parallel:
                        bytes_per_allreduce = bytes_per_allreduce / tp

                    lat_one, bw_one = hierarchical_allreduce_time_s(
                        bytes_per_allreduce, tp, hw.gpus_per_node, comm
                    )
                    t_tp_lat_s = n_allreduces * lat_one
                    t_tp_bw_s = n_allreduces * bw_one

                # =============================================================
                # PP communication per microbatch (fwd acts + bwd grads)
                # =============================================================
                t_pp_lat_s = 0.0
                t_pp_bw_s = 0.0
                if pp > 1:
                    bytes_xfer = tokens_per_microbatch * model.d_model * model.precision_bytes

                    if train.pp_activation_sharded_by_tp:
                        bytes_xfer = bytes_xfer / tp

                    # Node crossing heuristic
                    if tp <= hw.gpus_per_node:
                        stages_per_node = hw.gpus_per_node // tp
                    else:
                        stages_per_node = 1
                    n_nodes_for_pp = int(np.ceil(pp / stages_per_node)) if stages_per_node > 0 else pp
                    n_crossings = max(0, n_nodes_for_pp - 1)
                    frac_ib = n_crossings / (pp - 1) if pp > 1 else 0.0

                    alpha_eff = frac_ib * comm.ib_p2p_alpha_us + (1 - frac_ib) * comm.nv_p2p_alpha_us
                    bw_eff = (1 / (frac_ib / comm.ib_p2p_bw_gb_s + (1 - frac_ib) / comm.nv_p2p_bw_gb_s)
                              if frac_ib > 0 else comm.nv_p2p_bw_gb_s)

                    lat_one, bw_one = p2p_time_s(bytes_xfer, bw_eff, alpha_eff)

                    p2p_factor = 2 if pp > 2 else 1  # send+recv for interior stages
                    dir_factor = train.pp_directions  # fwd + bwd
                    t_pp_lat_s = p2p_factor * dir_factor * lat_one
                    t_pp_bw_s = p2p_factor * dir_factor * bw_one

                # =============================================================
                # DiLoCo / DP sync amortized cost
                # =============================================================
                t_dp_lat_s = 0.0
                t_dp_bw_s = 0.0
                if n_replicas > 1 and train.diloco_sync_every > 0:
                    # Payload: shard of weights/grads, possibly compressed
                    bytes_dp = (model.weight_size_gb * 1e9 / gpus_per_replica) * train.diloco_payload_frac

                    dp_lat, dp_bw = allreduce_time_s(
                        bytes_dp, n_replicas, comm.ib_bw_gb_s, comm.ib_alpha_us, algo=train.dp_algo
                    )

                    # Amortize over K steps × M microbatches
                    denom = train.diloco_sync_every * M
                    t_dp_lat_s = dp_lat / denom
                    t_dp_bw_s = dp_bw / denom

                # =============================================================
                # Combine + overlap model
                # =============================================================
                t_comm_lat = t_tp_lat_s + t_pp_lat_s + t_dp_lat_s
                t_comm_bw = t_tp_bw_s + t_pp_bw_s + t_dp_bw_s

                t_compute_mem = max(t_compute_s, t_mem_s)
                t_no_overlap = t_compute_mem + t_comm_lat + t_comm_bw
                t_full_overlap = max(t_compute_mem, t_comm_bw) + t_comm_lat
                t_microbatch_s = (1 - bw_overlap_frac) * t_no_overlap + bw_overlap_frac * t_full_overlap

                if t_microbatch_s <= 0:
                    continue

                # =============================================================
                # Pipeline utilization (filled/continuous batching)
                # =============================================================
                util = min(1.0, M / pp)

                throughput_replica = (tokens_per_microbatch / t_microbatch_s) * util
                cluster_throughput = n_replicas * throughput_replica

                if cluster_throughput > best_throughput:
                    best_throughput = cluster_throughput
                    best_config = {
                        "pp": pp,
                        "tp": tp,
                        "M": M,
                        "m": m,
                        "seq_len": train.seq_len,
                        "gpus_per_replica": gpus_per_replica,
                        "n_replicas": n_replicas,
                        "tokens_per_microbatch": tokens_per_microbatch,
                        "param_state_mb": param_state_mb,
                        "act_in_flight": in_flight,
                        "t_compute_ms": t_compute_s * 1e3,
                        "t_mem_ms": t_mem_s * 1e3,
                        "t_tp_lat_ms": t_tp_lat_s * 1e3,
                        "t_tp_bw_ms": t_tp_bw_s * 1e3,
                        "t_pp_lat_ms": t_pp_lat_s * 1e3,
                        "t_pp_bw_ms": t_pp_bw_s * 1e3,
                        "t_dp_lat_ms": t_dp_lat_s * 1e3,
                        "t_dp_bw_ms": t_dp_bw_s * 1e3,
                        "t_microbatch_ms": t_microbatch_s * 1e3,
                        "pipeline_util": util,
                        "throughput_per_replica": throughput_replica,
                        "hbm_per_gpu_gb": hbm_per_gpu_gb,
                        "train_flops_per_token": train_flops_per_token,
                        "bw_overlap_frac": bw_overlap_frac,
                        "schedule": "filled",
                    }

    if best_config is None:
        return 0.0, {"error": "No valid configuration found (training)"}

    best_config["compute_ceiling"] = compute_ceiling
    best_config["efficiency"] = best_throughput / compute_ceiling if compute_ceiling > 0 else 0.0
    return best_throughput, best_config


# =============================================================================
# Generate Data
# =============================================================================

def generate_data(max_tp: int = 128):
    hw = Hardware()
    model = Model()

    hbm_fractions = np.logspace(np.log10(0.0001), np.log10(1.0), 60)
    hbm_fractions = np.unique(np.sort(np.concatenate([
        hbm_fractions,
        [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
    ])))

    compute_fractions = [1.0, 0.2, 0.05]

    # Three scenarios for comm latency (all assume full bandwidth overlap, γ=1)
    # pessimistic: 2× latency
    # middle: baseline latency
    # optimistic: no latency (α=0)
    scenarios = [
        ("pessimistic", 2.0),   # latency_mult
        ("middle", 1.0),
        ("optimistic", 0.0),
    ]

    # Inference configs
    inference_configs = [
        ("Mamba", None),
        ("Transformer 2k", 2000),
        ("Transformer 8k", 8000),
        ("Transformer 32k", 32000),
    ]

    # Training configs (seq_len matters)
    training_configs = [
        ("Mamba 2k", None, 2048),
        ("Mamba 8k", None, 8192),
        ("Transformer 2k", 2048, 2048),
        ("Transformer 8k", 8192, 8192),
    ]

    base_comm = CommParams()

    # Create CommParams for each latency scenario
    def make_comm(latency_mult):
        return CommParams(
            nvlink_alpha_us=base_comm.nvlink_alpha_us * latency_mult,
            nvlink_bw_gb_s=base_comm.nvlink_bw_gb_s,
            ib_alpha_us=base_comm.ib_alpha_us * latency_mult,
            ib_bw_gb_s=base_comm.ib_bw_gb_s,
            nv_p2p_alpha_us=base_comm.nv_p2p_alpha_us * latency_mult,
            nv_p2p_bw_gb_s=base_comm.nv_p2p_bw_gb_s,
            ib_p2p_alpha_us=base_comm.ib_p2p_alpha_us * latency_mult,
            ib_p2p_bw_gb_s=base_comm.ib_p2p_bw_gb_s,
        )

    # First, compute the unconstrained max for each config (100% compute, 100% HBM, no latency, max TP)
    # Use max_tp for the baseline so efficiency is relative to what's achievable with given TP cap
    inference_max = {}
    for config_name, context_len in inference_configs:
        tp, _ = calculate_throughput(
            hw, model, 1.0, 1.0, context_len,
            comm=make_comm(0.0), bw_overlap_frac=1.0, max_tp=max_tp
        )
        inference_max[config_name] = tp if tp > 0 else 1.0

    training_max = {}
    for config_name, context_len, seq_len in training_configs:
        train_cfg = TrainingConfig(seq_len=seq_len, context_length=context_len)
        tp, _ = calculate_training_throughput(
            hw, model, 1.0, 1.0, train_cfg,
            comm=make_comm(0.0), bw_overlap_frac=1.0, max_tp=max_tp
        )
        training_max[config_name] = tp if tp > 0 else 1.0

    # Inference results: {compute_frac: {config_name: {scenario: {hbm: [], efficiency: []}}}}
    inference_results = {}
    for compute_frac in compute_fractions:
        inference_results[compute_frac] = {}
        for config_name, context_len in inference_configs:
            inference_results[compute_frac][config_name] = {}
            for scenario_name, latency_mult in scenarios:
                inference_results[compute_frac][config_name][scenario_name] = {"hbm": [], "efficiency": []}
                comm = make_comm(latency_mult)
                for hbm_frac in hbm_fractions:
                    tp, info = calculate_throughput(
                        hw, model, hbm_frac, compute_frac, context_len,
                        comm=comm, bw_overlap_frac=1.0, max_tp=max_tp
                    )
                    # Normalize to unconstrained max (100% compute, 100% HBM)
                    efficiency = tp / inference_max[config_name] if tp > 0 else 0
                    inference_results[compute_frac][config_name][scenario_name]["hbm"].append(hbm_frac * 100)
                    inference_results[compute_frac][config_name][scenario_name]["efficiency"].append(efficiency)

    # Training results: {compute_frac: {config_name: {scenario: {hbm: [], efficiency: []}}}}
    training_results = {}
    for compute_frac in compute_fractions:
        training_results[compute_frac] = {}
        for config_name, context_len, seq_len in training_configs:
            training_results[compute_frac][config_name] = {}
            train_cfg = TrainingConfig(seq_len=seq_len, context_length=context_len)
            for scenario_name, latency_mult in scenarios:
                training_results[compute_frac][config_name][scenario_name] = {"hbm": [], "efficiency": []}
                comm = make_comm(latency_mult)
                for hbm_frac in hbm_fractions:
                    tp, info = calculate_training_throughput(
                        hw, model, hbm_frac, compute_frac, train_cfg,
                        comm=comm, bw_overlap_frac=1.0, max_tp=max_tp
                    )
                    # Normalize to unconstrained max (100% compute, 100% HBM)
                    efficiency = tp / training_max[config_name] if tp > 0 else 0
                    training_results[compute_frac][config_name][scenario_name]["hbm"].append(hbm_frac * 100)
                    training_results[compute_frac][config_name][scenario_name]["efficiency"].append(efficiency)

    return inference_results, training_results, hbm_fractions, compute_fractions, inference_configs, training_configs


# =============================================================================
# Plotting
# =============================================================================

def create_plots(inference_results, training_results, hbm_fractions, compute_fractions, inference_configs, training_configs, max_tp: int = 128):
    colors_inference = {
        "Mamba": "#2ecc71",
        "Transformer 2k": "#3498db",
        "Transformer 8k": "#9b59b6",
        "Transformer 32k": "#e74c3c",
    }

    colors_training = {
        "Mamba 2k": "#2ecc71",
        "Mamba 8k": "#27ae60",
        "Transformer 2k": "#3498db",
        "Transformer 8k": "#9b59b6",
    }

    # Line styles for scenarios: pessimistic, middle, optimistic
    scenario_styles = {
        "pessimistic": (":", 1.5),   # dotted, thinner
        "middle": ("-", 2.0),        # solid, normal
        "optimistic": ("--", 1.5),   # dashed, thinner
    }

    # 2x3 grid: rows = inference/training, cols = compute fractions
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"Throughput vs HBM Capacity (Llama 3 70B BF16, 10k H100 SXM, TP≤{max_tp})",
                 fontsize=16, fontweight='bold', y=0.96)

    # Row labels
    fig.text(0.02, 0.68, "Inference\n(decode)", fontsize=12, fontweight='bold',
             rotation=90, va='center', ha='center')
    fig.text(0.02, 0.32, "Training", fontsize=12, fontweight='bold',
             rotation=90, va='center', ha='center')

    for col, compute_frac in enumerate(compute_fractions):
        # --- Inference row (top) ---
        ax_inf = axes[0, col]
        for config_name, _ in inference_configs:
            for scenario_name in ["pessimistic", "middle", "optimistic"]:
                data = inference_results[compute_frac][config_name][scenario_name]
                hbm_share = [h / 100 for h in data["hbm"]]
                efficiency = data["efficiency"]
                style, lw = scenario_styles[scenario_name]
                # Only add label for middle scenario to avoid legend clutter
                label = config_name if scenario_name == "middle" else None
                ax_inf.plot(hbm_share, efficiency, color=colors_inference[config_name],
                           linestyle=style, linewidth=lw, label=label)

        ax_inf.set_xscale('log')
        ax_inf.set_yscale('log')
        ax_inf.set_title(f"{int(compute_frac*100)}% free Compute + HBM BW", fontsize=12, fontweight='bold')
        ax_inf.set_xlim(0.0001, 1.0)
        ax_inf.set_ylim(0.0001, 1.5)
        ax_inf.grid(True, alpha=0.3, which='both')

        if col == 0:
            ax_inf.set_ylabel("Throughput (share of max)", fontsize=10)
            ax_inf.legend(loc='lower right', fontsize=8)

        # --- Training row (bottom) ---
        ax_train = axes[1, col]
        for config_name, _, _ in training_configs:
            for scenario_name in ["pessimistic", "middle", "optimistic"]:
                data = training_results[compute_frac][config_name][scenario_name]
                hbm_share = [h / 100 for h in data["hbm"]]
                efficiency = data["efficiency"]
                style, lw = scenario_styles[scenario_name]
                label = config_name if scenario_name == "middle" else None
                ax_train.plot(hbm_share, efficiency, color=colors_training[config_name],
                             linestyle=style, linewidth=lw, label=label)

        ax_train.set_xscale('log')
        ax_train.set_yscale('log')
        ax_train.set_xlabel("Available HBM (fraction of 80GB)", fontsize=10)
        ax_train.set_xlim(0.0001, 1.0)
        ax_train.set_ylim(0.0001, 1.5)
        ax_train.grid(True, alpha=0.3, which='both')

        if col == 0:
            ax_train.set_ylabel("Throughput (share of max)", fontsize=10)
            ax_train.legend(loc='lower right', fontsize=8)

    plt.tight_layout(rect=[0.04, 0.02, 1, 0.94])
    plt.savefig(f'plots/throughput_vs_hbm_tp{max_tp}.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'plots/throughput_vs_hbm_tp{max_tp}.pdf', bbox_inches='tight')
    print(f"Saved plots/throughput_vs_hbm_tp{max_tp}.png and .pdf")
    plt.close()


# =============================================================================
# Print Summary Tables
# =============================================================================

def print_tables(inference_configs, training_configs):
    hw = Hardware()
    model = Model()

    print("\n" + "="*80)
    print("SUMMARY TABLES")
    print("="*80)

    hbm_points = [1.0, 0.05, 0.01, 0.001]
    hbm_labels = ["100%", "5%", "1%", "0.1%"]

    # Inference table
    print(f"\n{'='*80}")
    print("INFERENCE (100% Compute)")
    print("="*80)

    print(f"\n{'Model':<20}", end="")
    for label in hbm_labels:
        print(f"{'HBM '+label:>15}", end="")
    print()
    print("-"*80)

    for config_name, context_len in inference_configs:
        print(f"{config_name:<20}", end="")
        for hbm_frac in hbm_points:
            tp, info = calculate_throughput(
                hw, model, hbm_frac, 1.0, context_len,
                bw_overlap_frac=0.5
            )
            if tp > 0:
                print(f"{tp/1e6:>12.2f}M  ", end="")
            else:
                print(f"{'N/A':>15}", end="")
        print()

    # Training table
    print(f"\n{'='*80}")
    print("TRAINING (100% Compute)")
    print("="*80)

    print(f"\n{'Model':<20}", end="")
    for label in hbm_labels:
        print(f"{'HBM '+label:>15}", end="")
    print()
    print("-"*80)

    for config_name, context_len, seq_len in training_configs:
        print(f"{config_name:<20}", end="")
        train_cfg = TrainingConfig(seq_len=seq_len, context_length=context_len)
        for hbm_frac in hbm_points:
            tp, info = calculate_training_throughput(
                hw, model, hbm_frac, 1.0, train_cfg,
                bw_overlap_frac=0.5
            )
            if tp > 0:
                print(f"{tp/1e6:>12.2f}M  ", end="")
            else:
                print(f"{'N/A':>15}", end="")
        print()


# =============================================================================
# Detailed Breakdown
# =============================================================================

def print_detailed_breakdown():
    hw = Hardware()
    model = Model()

    print("\n" + "="*80)
    print("DETAILED BREAKDOWN - INFERENCE")
    print("="*80)

    inference_cases = [
        (1.0, 1.0, None, "Mamba, 100% HBM"),
        (0.05, 1.0, None, "Mamba, 5% HBM"),
        (0.01, 1.0, 8000, "Transformer 8k, 1% HBM"),
    ]

    for hbm_frac, compute_frac, context_len, desc in inference_cases:
        print(f"\n{'-'*60}")
        print(f"{desc}")
        print(f"{'-'*60}")

        tp, info = calculate_throughput(
            hw, model, hbm_frac, compute_frac, context_len,
            bw_overlap_frac=0.5
        )

        if "error" in info:
            print(f"ERROR: {info['error']}")
            continue

        print(f"Configuration:")
        print(f"  HBM/GPU: {info['hbm_per_gpu_gb']:.3f} GB ({info['hbm_per_gpu_gb']*1000:.1f} MB)")
        print(f"  PP={info['pp']}, TP={info['tp']}, M={info['M']} -> {info['gpus_per_replica']} GPUs/replica")
        print(f"  Replicas: {info['n_replicas']}")
        print(f"  B_total: {info['B_total']}, m: {info['m']}")

        print(f"\nPer-microbatch time breakdown:")
        print(f"  Compute: {info['t_compute_per_mb_ms']:.3f} ms")
        print(f"  Weight read: {info['t_weight_read_per_mb_ms']:.3f} ms")
        print(f"  TP latency: {info['t_tp_lat_per_mb_ms']:.3f} ms")
        print(f"  TP bandwidth: {info['t_tp_bw_per_mb_ms']:.3f} ms")
        print(f"  PP latency: {info['t_pp_lat_per_mb_ms']:.3f} ms")
        print(f"  PP bandwidth: {info['t_pp_bw_per_mb_ms']:.3f} ms")
        print(f"  Total microbatch: {info['t_microbatch_ms']:.3f} ms")
        print(f"  Pipeline util: {info['pipeline_util']*100:.1f}% (M={info['M']}, PP={info['pp']})")
        print(f"  Bottleneck: {info['bottleneck']}")

        print(f"\nThroughput:")
        print(f"  Per replica: {info['throughput_per_replica']/1e6:.3f} M tok/s")
        print(f"  Cluster: {tp/1e6:.3f} M tok/s")
        print(f"  Efficiency: {info['efficiency']*100:.1f}%")

    # Training breakdown
    print("\n" + "="*80)
    print("DETAILED BREAKDOWN - TRAINING")
    print("="*80)

    training_cases = [
        (1.0, 8192, 8192, "Training 8k, 100% HBM"),  # (hbm_frac, context_len, seq_len, desc)
        (0.05, 8192, 8192, "Training 8k, 5% HBM"),
        (0.01, 8192, 8192, "Training 8k, 1% HBM"),
    ]

    for hbm_frac, context_len, seq_len, desc in training_cases:
        print(f"\n{'-'*60}")
        print(f"{desc}")
        print(f"{'-'*60}")

        train_cfg = TrainingConfig(seq_len=seq_len, context_length=context_len)
        tp, info = calculate_training_throughput(
            hw, model, hbm_frac, 1.0, train_cfg,
            bw_overlap_frac=0.5
        )

        if "error" in info:
            print(f"ERROR: {info['error']}")
            continue

        print(f"Configuration:")
        print(f"  HBM/GPU: {info['hbm_per_gpu_gb']:.3f} GB ({info['hbm_per_gpu_gb']*1000:.1f} MB)")
        print(f"  PP={info['pp']}, TP={info['tp']}, M={info['M']} -> {info['gpus_per_replica']} GPUs/replica")
        print(f"  Replicas: {info['n_replicas']}")
        print(f"  Microbatch: m={info['m']} seqs x {seq_len} = {info['tokens_per_microbatch']} tokens")
        print(f"  In-flight activations: {info['act_in_flight']} microbatches")

        print(f"\nModeling assumptions:")
        print(f"  param_state_mult: {train_cfg.param_state_mult}x (param_state: {info['param_state_mb']:.0f} MB)")
        print(f"  pp_activation_sharded_by_tp: {train_cfg.pp_activation_sharded_by_tp}")
        print(f"  tp_comm_seq_parallel: {train_cfg.tp_comm_seq_parallel}")

        print(f"\nPer-microbatch time breakdown:")
        print(f"  Compute: {info['t_compute_ms']:.3f} ms")
        print(f"  Memory (weights+grads): {info['t_mem_ms']:.3f} ms")
        print(f"  TP latency: {info['t_tp_lat_ms']:.3f} ms")
        print(f"  TP bandwidth: {info['t_tp_bw_ms']:.3f} ms")
        print(f"  PP latency: {info['t_pp_lat_ms']:.3f} ms")
        print(f"  PP bandwidth: {info['t_pp_bw_ms']:.3f} ms")
        print(f"  DP (amortized) latency: {info['t_dp_lat_ms']:.6f} ms")
        print(f"  DP (amortized) bandwidth: {info['t_dp_bw_ms']:.6f} ms")
        print(f"  Total microbatch: {info['t_microbatch_ms']:.3f} ms")
        print(f"  Pipeline util: {info['pipeline_util']*100:.1f}% (M={info['M']}, PP={info['pp']})")

        print(f"\nThroughput:")
        print(f"  Per replica: {info['throughput_per_replica']/1e6:.3f} M tok/s")
        print(f"  Cluster: {tp/1e6:.3f} M tok/s")
        print(f"  Compute ceiling: {info['compute_ceiling']/1e6:.2f} M tok/s")
        print(f"  Efficiency: {info['efficiency']*100:.1f}%")


# =============================================================================
# Diagnostic Tables for Debugging
# =============================================================================

def print_diagnostic_tables():
    """
    Print comprehensive diagnostic tables showing how optimal configurations
    and time breakdowns change across HBM levels. Useful for debugging and
    understanding model behavior.
    """
    hw = Hardware()
    model = Model()

    hbm_points = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001]

    # =========================================================================
    # INFERENCE DIAGNOSTICS
    # =========================================================================
    print("\n" + "="*120)
    print("INFERENCE DIAGNOSTIC TABLE (Decode, 100% Compute)")
    print("="*120)

    inference_configs = [
        ("Mamba", None),
        ("Transformer 2k", 2000),
        ("Transformer 8k", 8000),
    ]

    for config_name, context_len in inference_configs:
        print(f"\n{'─'*120}")
        print(f"  {config_name}")
        print(f"{'─'*120}")

        # Header
        print(f"{'HBM%':>6} │ {'MB/GPU':>8} │ {'PP':>3} {'TP':>3} {'M':>3} {'m':>5} │ "
              f"{'Compute':>8} {'WeightRd':>8} {'TPLat':>7} {'TPBW':>7} {'PPLat':>7} {'PPBW':>7} │ "
              f"{'Total':>8} │ {'Util%':>5} │ {'Eff%':>5} │ {'Bottleneck':>12}")
        print(f"{'':>6} │ {'':>8} │ {'':>3} {'':>3} {'':>3} {'':>5} │ "
              f"{'(ms)':>8} {'(ms)':>8} {'(ms)':>7} {'(ms)':>7} {'(ms)':>7} {'(ms)':>7} │ "
              f"{'(ms)':>8} │ {'':>5} │ {'':>5} │ {'':>12}")
        print("─"*120)

        for hbm_frac in hbm_points:
            tp, info = calculate_throughput(
                hw, model, hbm_frac, 1.0, context_len,
                bw_overlap_frac=1.0, max_tp=1024
            )

            if "error" in info or tp == 0:
                print(f"{hbm_frac*100:>5.1f}% │ {'N/A':>8} │ " + "─"*90)
                continue

            hbm_mb = info['hbm_per_gpu_gb'] * 1000
            print(f"{hbm_frac*100:>5.1f}% │ {hbm_mb:>7.0f}M │ "
                  f"{info['pp']:>3} {info['tp']:>3} {info['M']:>3} {info['m']:>5} │ "
                  f"{info['t_compute_per_mb_ms']:>8.3f} {info['t_weight_read_per_mb_ms']:>8.3f} "
                  f"{info['t_tp_lat_per_mb_ms']:>7.3f} {info['t_tp_bw_per_mb_ms']:>7.3f} "
                  f"{info['t_pp_lat_per_mb_ms']:>7.3f} {info['t_pp_bw_per_mb_ms']:>7.3f} │ "
                  f"{info['t_microbatch_ms']:>8.3f} │ "
                  f"{info['pipeline_util']*100:>5.1f} │ "
                  f"{info['efficiency']*100:>5.1f} │ "
                  f"{info['bottleneck']:>12}")

    # =========================================================================
    # TRAINING DIAGNOSTICS
    # =========================================================================
    print("\n\n" + "="*140)
    print("TRAINING DIAGNOSTIC TABLE (100% Compute)")
    print("="*140)

    training_configs = [
        ("Mamba 8k", None, 8192),
        ("Transformer 8k", 8192, 8192),
    ]

    for config_name, context_len, seq_len in training_configs:
        print(f"\n{'─'*140}")
        print(f"  {config_name}")
        print(f"{'─'*140}")

        # Header
        print(f"{'HBM%':>6} │ {'MB/GPU':>8} │ {'PP':>3} {'TP':>3} {'M':>3} {'m':>3} {'InFlt':>5} │ "
              f"{'Compute':>8} {'Mem':>7} {'TPLat':>6} {'TPBW':>7} {'PPLat':>6} {'PPBW':>7} │ "
              f"{'Total':>8} │ {'Util%':>5} │ {'Eff%':>5} │ {'Replicas':>8} │ {'Throughput':>12}")
        print(f"{'':>6} │ {'':>8} │ {'':>3} {'':>3} {'':>3} {'':>3} {'':>5} │ "
              f"{'(ms)':>8} {'(ms)':>7} {'(ms)':>6} {'(ms)':>7} {'(ms)':>6} {'(ms)':>7} │ "
              f"{'(ms)':>8} │ {'':>5} │ {'':>5} │ {'':>8} │ {'(M tok/s)':>12}")
        print("─"*140)

        train_cfg = TrainingConfig(seq_len=seq_len, context_length=context_len)

        for hbm_frac in hbm_points:
            tp, info = calculate_training_throughput(
                hw, model, hbm_frac, 1.0, train_cfg,
                bw_overlap_frac=1.0, max_tp=1024
            )

            if "error" in info or tp == 0:
                print(f"{hbm_frac*100:>5.1f}% │ {'N/A':>8} │ " + "─"*110)
                continue

            hbm_mb = info['hbm_per_gpu_gb'] * 1000
            print(f"{hbm_frac*100:>5.1f}% │ {hbm_mb:>7.0f}M │ "
                  f"{info['pp']:>3} {info['tp']:>3} {info['M']:>3} {info['m']:>3} {info['act_in_flight']:>5} │ "
                  f"{info['t_compute_ms']:>8.1f} {info['t_mem_ms']:>7.2f} "
                  f"{info['t_tp_lat_ms']:>6.2f} {info['t_tp_bw_ms']:>7.2f} "
                  f"{info['t_pp_lat_ms']:>6.2f} {info['t_pp_bw_ms']:>7.2f} │ "
                  f"{info['t_microbatch_ms']:>8.1f} │ "
                  f"{info['pipeline_util']*100:>5.1f} │ "
                  f"{info['efficiency']*100:>5.1f} │ "
                  f"{info['n_replicas']:>8} │ "
                  f"{tp/1e6:>12.3f}")

    # =========================================================================
    # TIME BREAKDOWN PERCENTAGES
    # =========================================================================
    print("\n\n" + "="*100)
    print("TIME BREAKDOWN PERCENTAGES (what fraction of microbatch time is each component)")
    print("="*100)

    print(f"\n{'─'*100}")
    print("  Inference: Mamba (decode)")
    print(f"{'─'*100}")
    print(f"{'HBM%':>6} │ {'Compute%':>9} │ {'WeightRd%':>10} │ {'TPLat%':>7} │ {'TPBW%':>7} │ {'PPLat%':>7} │ {'PPBW%':>7} │ {'Dominant':>15}")
    print("─"*100)

    for hbm_frac in hbm_points:
        tp, info = calculate_throughput(hw, model, hbm_frac, 1.0, None, bw_overlap_frac=1.0, max_tp=1024)
        if "error" in info or tp == 0:
            continue

        total = info['t_microbatch_ms']
        if total > 0:
            pct_compute = info['t_compute_per_mb_ms'] / total * 100
            pct_weight = info['t_weight_read_per_mb_ms'] / total * 100
            pct_tp_lat = info['t_tp_lat_per_mb_ms'] / total * 100
            pct_tp_bw = info['t_tp_bw_per_mb_ms'] / total * 100
            pct_pp_lat = info['t_pp_lat_per_mb_ms'] / total * 100
            pct_pp_bw = info['t_pp_bw_per_mb_ms'] / total * 100

            # Find dominant
            components = [
                ("Compute", pct_compute),
                ("WeightRd", pct_weight),
                ("TP Lat", pct_tp_lat),
                ("TP BW", pct_tp_bw),
                ("PP Lat", pct_pp_lat),
                ("PP BW", pct_pp_bw),
            ]
            dominant = max(components, key=lambda x: x[1])[0]

            print(f"{hbm_frac*100:>5.1f}% │ {pct_compute:>8.1f}% │ {pct_weight:>9.1f}% │ "
                  f"{pct_tp_lat:>6.1f}% │ {pct_tp_bw:>6.1f}% │ {pct_pp_lat:>6.1f}% │ {pct_pp_bw:>6.1f}% │ "
                  f"{dominant:>15}")

    print(f"\n{'─'*100}")
    print("  Inference: Transformer 8k (decode)")
    print(f"{'─'*100}")
    print(f"{'HBM%':>6} │ {'Compute%':>9} │ {'WeightRd%':>10} │ {'KVRead%':>8} │ {'TPLat%':>7} │ {'TPBW%':>7} │ {'Dominant':>15}")
    print("─"*100)

    for hbm_frac in hbm_points:
        tp, info = calculate_throughput(hw, model, hbm_frac, 1.0, 8000, bw_overlap_frac=1.0, max_tp=1024)
        if "error" in info or tp == 0:
            continue

        total = info['t_microbatch_ms']
        if total > 0:
            pct_compute = info['t_compute_per_mb_ms'] / total * 100
            pct_weight = info['t_weight_read_per_mb_ms'] / total * 100
            pct_kv = info.get('t_kv_read_per_mb_ms', 0) / total * 100
            pct_tp_lat = info['t_tp_lat_per_mb_ms'] / total * 100
            pct_tp_bw = info['t_tp_bw_per_mb_ms'] / total * 100

            components = [
                ("Compute", pct_compute),
                ("WeightRd", pct_weight),
                ("KV Read", pct_kv),
                ("TP Lat", pct_tp_lat),
                ("TP BW", pct_tp_bw),
            ]
            dominant = max(components, key=lambda x: x[1])[0]

            print(f"{hbm_frac*100:>5.1f}% │ {pct_compute:>8.1f}% │ {pct_weight:>9.1f}% │ "
                  f"{pct_kv:>7.1f}% │ {pct_tp_lat:>6.1f}% │ {pct_tp_bw:>6.1f}% │ "
                  f"{dominant:>15}")

    # =========================================================================
    # CONFIGURATION TRANSITIONS
    # =========================================================================
    print("\n\n" + "="*80)
    print("CONFIGURATION TRANSITION POINTS (where PP/TP/M changes)")
    print("="*80)

    print("\nInference Mamba:")
    prev_config = None
    for hbm_frac in np.logspace(np.log10(0.001), 0, 50):
        tp, info = calculate_throughput(hw, model, hbm_frac, 1.0, None, bw_overlap_frac=1.0, max_tp=1024)
        if "error" in info or tp == 0:
            continue
        curr_config = (info['pp'], info['tp'], info['M'])
        if curr_config != prev_config:
            print(f"  HBM {hbm_frac*100:>6.2f}%: PP={info['pp']:>2}, TP={info['tp']:>4}, M={info['M']:>3}, "
                  f"m={info['m']:>4}, util={info['pipeline_util']*100:.0f}%, eff={info['efficiency']*100:.1f}%")
            prev_config = curr_config

    print("\nTraining Transformer 8k:")
    train_cfg = TrainingConfig(seq_len=8192, context_length=8192)
    prev_config = None
    for hbm_frac in np.logspace(np.log10(0.001), 0, 50):
        tp, info = calculate_training_throughput(hw, model, hbm_frac, 1.0, train_cfg, bw_overlap_frac=1.0, max_tp=1024)
        if "error" in info or tp == 0:
            continue
        curr_config = (info['pp'], info['tp'], info['M'])
        if curr_config != prev_config:
            print(f"  HBM {hbm_frac*100:>6.2f}%: PP={info['pp']:>2}, TP={info['tp']:>4}, M={info['M']:>3}, "
                  f"m={info['m']:>3}, in_flight={info['act_in_flight']:>2}, util={info['pipeline_util']*100:.0f}%, eff={info['efficiency']*100:.1f}%")
            prev_config = curr_config


# =============================================================================
# 2D Bandwidth × HBM Sweep Analysis
# =============================================================================

def print_bandwidth_sweep_tables():
    """
    2D sweep: HBM fraction × Bandwidth fraction
    Shows how training/inference degrades as both resources are constrained.
    """
    hw = Hardware()
    model = Model()

    hbm_fracs = [1.0, 0.2, 0.05, 0.01, 0.001]
    bw_fracs = [1.0, 0.5, 0.2, 0.1, 0.05]

    base_comm = CommParams()

    def make_comm_scaled(nvlink_bw_mult=1.0, ib_bw_mult=1.0):
        """Create CommParams with scaled bandwidths (latencies unchanged)."""
        return CommParams(
            nvlink_alpha_us=base_comm.nvlink_alpha_us,
            nvlink_bw_gb_s=base_comm.nvlink_bw_gb_s * nvlink_bw_mult,
            ib_alpha_us=base_comm.ib_alpha_us,
            ib_bw_gb_s=base_comm.ib_bw_gb_s * ib_bw_mult,
            nv_p2p_alpha_us=base_comm.nv_p2p_alpha_us,
            nv_p2p_bw_gb_s=base_comm.nv_p2p_bw_gb_s * nvlink_bw_mult,
            ib_p2p_alpha_us=base_comm.ib_p2p_alpha_us,
            ib_p2p_bw_gb_s=base_comm.ib_p2p_bw_gb_s * ib_bw_mult,
        )

    # =========================================================================
    # TRAINING: HBM × IB Bandwidth (NVLink at 100%)
    # =========================================================================
    print("\n" + "="*140)
    print("TRAINING 2D SWEEP: HBM × IB Bandwidth (NVLink @ 100%)")
    print("Transformer 8k, 100% Compute")
    print("="*140)

    train_cfg = TrainingConfig(seq_len=8192, context_length=8192)

    # Header
    print(f"\n{'':>10} │", end="")
    for ib_frac in bw_fracs:
        print(f"  IB {int(ib_frac*100):>3}%  │", end="")
    print()
    print("─"*10 + "┼" + ("─"*11 + "┼") * len(bw_fracs))

    for hbm_frac in hbm_fracs:
        print(f"HBM {hbm_frac*100:>4.1f}% │", end="")
        for ib_frac in bw_fracs:
            comm = make_comm_scaled(nvlink_bw_mult=1.0, ib_bw_mult=ib_frac)
            tp, info = calculate_training_throughput(
                hw, model, hbm_frac, 1.0, train_cfg,
                comm=comm, bw_overlap_frac=1.0, max_tp=1024
            )
            if "error" in info or tp == 0:
                print(f"   {'N/A':>5}   │", end="")
            else:
                print(f"   {info['efficiency']*100:>5.1f}%  │", end="")
        print()

    # Detailed version with PP/TP
    print(f"\nDetailed (Eff% / PP×TP):")
    print(f"{'':>10} │", end="")
    for ib_frac in bw_fracs:
        print(f"    IB {int(ib_frac*100):>3}%    │", end="")
    print()
    print("─"*10 + "┼" + ("─"*15 + "┼") * len(bw_fracs))

    for hbm_frac in hbm_fracs:
        print(f"HBM {hbm_frac*100:>4.1f}% │", end="")
        for ib_frac in bw_fracs:
            comm = make_comm_scaled(nvlink_bw_mult=1.0, ib_bw_mult=ib_frac)
            tp, info = calculate_training_throughput(
                hw, model, hbm_frac, 1.0, train_cfg,
                comm=comm, bw_overlap_frac=1.0, max_tp=1024
            )
            if "error" in info or tp == 0:
                print(f"      {'N/A':>6}    │", end="")
            else:
                eff = info['efficiency'] * 100
                pp, tp_val = info['pp'], info['tp']
                print(f" {eff:>4.0f}% {pp:>2}×{tp_val:<3} │", end="")
        print()

    # =========================================================================
    # TRAINING: HBM × NVLink Bandwidth (IB at 100%)
    # =========================================================================
    print("\n\n" + "="*140)
    print("TRAINING 2D SWEEP: HBM × NVLink Bandwidth (IB @ 100%)")
    print("Transformer 8k, 100% Compute")
    print("="*140)

    print(f"\n{'':>10} │", end="")
    for nv_frac in bw_fracs:
        print(f"  NV {int(nv_frac*100):>3}%  │", end="")
    print()
    print("─"*10 + "┼" + ("─"*11 + "┼") * len(bw_fracs))

    for hbm_frac in hbm_fracs:
        print(f"HBM {hbm_frac*100:>4.1f}% │", end="")
        for nv_frac in bw_fracs:
            comm = make_comm_scaled(nvlink_bw_mult=nv_frac, ib_bw_mult=1.0)
            tp, info = calculate_training_throughput(
                hw, model, hbm_frac, 1.0, train_cfg,
                comm=comm, bw_overlap_frac=1.0, max_tp=1024
            )
            if "error" in info or tp == 0:
                print(f"   {'N/A':>5}   │", end="")
            else:
                print(f"   {info['efficiency']*100:>5.1f}%  │", end="")
        print()

    # Detailed version with PP/TP
    print(f"\nDetailed (Eff% / PP×TP):")
    print(f"{'':>10} │", end="")
    for nv_frac in bw_fracs:
        print(f"    NV {int(nv_frac*100):>3}%    │", end="")
    print()
    print("─"*10 + "┼" + ("─"*15 + "┼") * len(bw_fracs))

    for hbm_frac in hbm_fracs:
        print(f"HBM {hbm_frac*100:>4.1f}% │", end="")
        for nv_frac in bw_fracs:
            comm = make_comm_scaled(nvlink_bw_mult=nv_frac, ib_bw_mult=1.0)
            tp, info = calculate_training_throughput(
                hw, model, hbm_frac, 1.0, train_cfg,
                comm=comm, bw_overlap_frac=1.0, max_tp=1024
            )
            if "error" in info or tp == 0:
                print(f"      {'N/A':>6}    │", end="")
            else:
                eff = info['efficiency'] * 100
                pp, tp_val = info['pp'], info['tp']
                print(f" {eff:>4.0f}% {pp:>2}×{tp_val:<3} │", end="")
        print()

    # =========================================================================
    # TRAINING: HBM × Both Bandwidths (NVLink = IB scaling)
    # =========================================================================
    print("\n\n" + "="*140)
    print("TRAINING 2D SWEEP: HBM × All Bandwidth (NVLink & IB scaled together)")
    print("Transformer 8k, 100% Compute")
    print("="*140)

    print(f"\n{'':>10} │", end="")
    for bw_frac in bw_fracs:
        print(f"  BW {int(bw_frac*100):>3}%  │", end="")
    print()
    print("─"*10 + "┼" + ("─"*11 + "┼") * len(bw_fracs))

    for hbm_frac in hbm_fracs:
        print(f"HBM {hbm_frac*100:>4.1f}% │", end="")
        for bw_frac in bw_fracs:
            comm = make_comm_scaled(nvlink_bw_mult=bw_frac, ib_bw_mult=bw_frac)
            tp, info = calculate_training_throughput(
                hw, model, hbm_frac, 1.0, train_cfg,
                comm=comm, bw_overlap_frac=1.0, max_tp=1024
            )
            if "error" in info or tp == 0:
                print(f"   {'N/A':>5}   │", end="")
            else:
                print(f"   {info['efficiency']*100:>5.1f}%  │", end="")
        print()

    # Detailed with bottleneck
    print(f"\nDetailed (Eff% / PP×TP / Bottleneck):")
    print(f"{'':>10} │", end="")
    for bw_frac in bw_fracs:
        print(f"      BW {int(bw_frac*100):>3}%       │", end="")
    print()
    print("─"*10 + "┼" + ("─"*20 + "┼") * len(bw_fracs))

    for hbm_frac in hbm_fracs:
        print(f"HBM {hbm_frac*100:>4.1f}% │", end="")
        for bw_frac in bw_fracs:
            comm = make_comm_scaled(nvlink_bw_mult=bw_frac, ib_bw_mult=bw_frac)
            tp, info = calculate_training_throughput(
                hw, model, hbm_frac, 1.0, train_cfg,
                comm=comm, bw_overlap_frac=1.0, max_tp=1024
            )
            if "error" in info or tp == 0:
                print(f"       {'N/A':>10}      │", end="")
            else:
                eff = info['efficiency'] * 100
                pp, tp_val = info['pp'], info['tp']
                # Determine bottleneck
                times = [
                    ("Comp", info['t_compute_ms']),
                    ("Mem", info['t_mem_ms']),
                    ("TPbw", info['t_tp_bw_ms']),
                    ("PPbw", info['t_pp_bw_ms']),
                    ("TPlat", info['t_tp_lat_ms']),
                ]
                bottleneck = max(times, key=lambda x: x[1])[0]
                print(f" {eff:>4.0f}% {pp:>2}×{tp_val:<3} {bottleneck:<5}│", end="")
        print()

    # =========================================================================
    # INFERENCE: HBM × All Bandwidth (Mamba decode)
    # =========================================================================
    print("\n\n" + "="*140)
    print("INFERENCE 2D SWEEP: HBM × All Bandwidth (NVLink & IB scaled together)")
    print("Mamba decode, 100% Compute")
    print("="*140)

    print(f"\n{'':>10} │", end="")
    for bw_frac in bw_fracs:
        print(f"  BW {int(bw_frac*100):>3}%  │", end="")
    print()
    print("─"*10 + "┼" + ("─"*11 + "┼") * len(bw_fracs))

    for hbm_frac in hbm_fracs:
        print(f"HBM {hbm_frac*100:>4.1f}% │", end="")
        for bw_frac in bw_fracs:
            comm = make_comm_scaled(nvlink_bw_mult=bw_frac, ib_bw_mult=bw_frac)
            tp, info = calculate_throughput(
                hw, model, hbm_frac, 1.0, None,  # Mamba
                comm=comm, bw_overlap_frac=1.0, max_tp=1024
            )
            if "error" in info or tp == 0:
                print(f"   {'N/A':>5}   │", end="")
            else:
                print(f"   {info['efficiency']*100:>5.1f}%  │", end="")
        print()

    # Detailed with bottleneck
    print(f"\nDetailed (Eff% / PP×TP / Bottleneck):")
    print(f"{'':>10} │", end="")
    for bw_frac in bw_fracs:
        print(f"      BW {int(bw_frac*100):>3}%       │", end="")
    print()
    print("─"*10 + "┼" + ("─"*20 + "┼") * len(bw_fracs))

    for hbm_frac in hbm_fracs:
        print(f"HBM {hbm_frac*100:>4.1f}% │", end="")
        for bw_frac in bw_fracs:
            comm = make_comm_scaled(nvlink_bw_mult=bw_frac, ib_bw_mult=bw_frac)
            tp, info = calculate_throughput(
                hw, model, hbm_frac, 1.0, None,
                comm=comm, bw_overlap_frac=1.0, max_tp=1024
            )
            if "error" in info or tp == 0:
                print(f"       {'N/A':>10}      │", end="")
            else:
                eff = info['efficiency'] * 100
                pp, tp_val = info['pp'], info['tp']
                bottleneck = info['bottleneck'][:5]
                print(f" {eff:>4.0f}% {pp:>2}×{tp_val:<3} {bottleneck:<5}│", end="")
        print()

    # =========================================================================
    # INFERENCE: Transformer 8k (KV-cache bound)
    # =========================================================================
    print("\n\n" + "="*140)
    print("INFERENCE 2D SWEEP: HBM × All Bandwidth (NVLink & IB scaled together)")
    print("Transformer 8k decode, 100% Compute")
    print("="*140)

    print(f"\n{'':>10} │", end="")
    for bw_frac in bw_fracs:
        print(f"  BW {int(bw_frac*100):>3}%  │", end="")
    print()
    print("─"*10 + "┼" + ("─"*11 + "┼") * len(bw_fracs))

    for hbm_frac in hbm_fracs:
        print(f"HBM {hbm_frac*100:>4.1f}% │", end="")
        for bw_frac in bw_fracs:
            comm = make_comm_scaled(nvlink_bw_mult=bw_frac, ib_bw_mult=bw_frac)
            tp, info = calculate_throughput(
                hw, model, hbm_frac, 1.0, 8000,  # Transformer 8k
                comm=comm, bw_overlap_frac=1.0, max_tp=1024
            )
            if "error" in info or tp == 0:
                print(f"   {'N/A':>5}   │", end="")
            else:
                print(f"   {info['efficiency']*100:>5.1f}%  │", end="")
        print()

    # Detailed with bottleneck
    print(f"\nDetailed (Eff% / PP×TP / Bottleneck):")
    print(f"{'':>10} │", end="")
    for bw_frac in bw_fracs:
        print(f"      BW {int(bw_frac*100):>3}%       │", end="")
    print()
    print("─"*10 + "┼" + ("─"*20 + "┼") * len(bw_fracs))

    for hbm_frac in hbm_fracs:
        print(f"HBM {hbm_frac*100:>4.1f}% │", end="")
        for bw_frac in bw_fracs:
            comm = make_comm_scaled(nvlink_bw_mult=bw_frac, ib_bw_mult=bw_frac)
            tp, info = calculate_throughput(
                hw, model, hbm_frac, 1.0, 8000,
                comm=comm, bw_overlap_frac=1.0, max_tp=1024
            )
            if "error" in info or tp == 0:
                print(f"       {'N/A':>10}      │", end="")
            else:
                eff = info['efficiency'] * 100
                pp, tp_val = info['pp'], info['tp']
                bottleneck = info['bottleneck'][:5]
                print(f" {eff:>4.0f}% {pp:>2}×{tp_val:<3} {bottleneck:<5}│", end="")
        print()

    # =========================================================================
    # SUMMARY: Key observations
    # =========================================================================
    print("\n\n" + "="*100)
    print("KEY SENSITIVITY ANALYSIS")
    print("="*100)

    # Compare IB vs NVLink sensitivity at a few key HBM points
    print("\nBandwidth sensitivity comparison (efficiency drop from 100% BW to 20% BW):")
    print(f"{'HBM':>10} │ {'IB only':>12} │ {'NVLink only':>12} │ {'Both':>12} │ {'More sensitive to':>20}")
    print("─"*75)

    for hbm_frac in [1.0, 0.2, 0.05, 0.01]:
        # Baseline (100% both)
        comm_base = make_comm_scaled(1.0, 1.0)
        _, info_base = calculate_training_throughput(hw, model, hbm_frac, 1.0, train_cfg, comm=comm_base, bw_overlap_frac=1.0, max_tp=1024)
        eff_base = info_base['efficiency'] * 100 if 'efficiency' in info_base else 0

        # IB degraded to 20%
        comm_ib = make_comm_scaled(1.0, 0.2)
        _, info_ib = calculate_training_throughput(hw, model, hbm_frac, 1.0, train_cfg, comm=comm_ib, bw_overlap_frac=1.0, max_tp=1024)
        eff_ib = info_ib['efficiency'] * 100 if 'efficiency' in info_ib else 0

        # NVLink degraded to 20%
        comm_nv = make_comm_scaled(0.2, 1.0)
        _, info_nv = calculate_training_throughput(hw, model, hbm_frac, 1.0, train_cfg, comm=comm_nv, bw_overlap_frac=1.0, max_tp=1024)
        eff_nv = info_nv['efficiency'] * 100 if 'efficiency' in info_nv else 0

        # Both degraded to 20%
        comm_both = make_comm_scaled(0.2, 0.2)
        _, info_both = calculate_training_throughput(hw, model, hbm_frac, 1.0, train_cfg, comm=comm_both, bw_overlap_frac=1.0, max_tp=1024)
        eff_both = info_both['efficiency'] * 100 if 'efficiency' in info_both else 0

        drop_ib = eff_base - eff_ib
        drop_nv = eff_base - eff_nv
        drop_both = eff_base - eff_both

        if drop_ib > drop_nv * 1.5:
            sensitive = "IB (cross-node)"
        elif drop_nv > drop_ib * 1.5:
            sensitive = "NVLink (intra-node)"
        else:
            sensitive = "Both equally"

        print(f"HBM {hbm_frac*100:>4.0f}% │ {drop_ib:>+10.1f}pp │ {drop_nv:>+10.1f}pp │ {drop_both:>+10.1f}pp │ {sensitive:>20}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import os
    os.makedirs("plots", exist_ok=True)

    # Generate plots for different TP caps
    tp_caps = [32, 128, 256, 512]

    for max_tp in tp_caps:
        print(f"\n{'='*80}")
        print(f"Computing throughput data for TP≤{max_tp}...")
        print("="*80)
        inference_results, training_results, hbm_fractions, compute_fractions, inference_configs, training_configs = generate_data(max_tp=max_tp)

        print(f"Creating plots for TP≤{max_tp}...")
        create_plots(inference_results, training_results, hbm_fractions, compute_fractions, inference_configs, training_configs, max_tp=max_tp)

    # Print tables for default TP cap (128)
    print_tables(inference_configs, training_configs)
    print_detailed_breakdown()
    print_diagnostic_tables()
    print_bandwidth_sweep_tables()

    print("\n" + "="*80)
    print("DONE")
    print("="*80)
