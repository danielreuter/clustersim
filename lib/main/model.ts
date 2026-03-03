// =============================================================================
// Data structures for the cluster throughput simulator (alpha-beta comm model)
// =============================================================================

/**
 * Communication parameters for alpha-beta model.
 *
 * Default values calibrated from NCCL source code (Erdil 2025, arXiv:2506.04645).
 * Based on NVIDIA's tuning code in src/graph/tuning.cc.
 *
 * The full NCCL latency formula for tree allreduce is:
 *     t_reduce = 6.8us + 1.2us * (intra_ranks - 1) + 10us * log2(inter_nodes)
 *
 * Note: These are PER-STEP latencies, not per-operation.
 * Ring allreduce has 2*(n-1) steps, so per-op latency = steps * alpha + kernel_launch.
 */
export interface CommParams {
  /** Kernel/driver launch overhead - paid once per collective (from NCCL) */
  kernel_launch_us: number;

  /** NVLink (intra-node) - collective per step (NCCL: 1.2us) */
  nvlink_alpha_us: number;
  /** NVLink bandwidth (GB/s) */
  nvlink_bw_gb_s: number;

  /** InfiniBand (inter-node) - collective per step (NCCL: 10us for tree) */
  ib_alpha_us: number;
  /** IB bandwidth (GB/s) */
  ib_bw_gb_s: number;

  /** NVLink p2p latency (for PP transfers) */
  nv_p2p_alpha_us: number;
  /** NVLink p2p bandwidth */
  nv_p2p_bw_gb_s: number;
  /** IB p2p latency (higher due to network) */
  ib_p2p_alpha_us: number;
  /** IB p2p bandwidth */
  ib_p2p_bw_gb_s: number;
}

export function createCommParams(overrides?: Partial<CommParams>): CommParams {
  return {
    kernel_launch_us: 6.8,
    nvlink_alpha_us: 1.2,
    nvlink_bw_gb_s: 450.0,
    ib_alpha_us: 10.0,
    ib_bw_gb_s: 50.0,
    nv_p2p_alpha_us: 2.0,
    nv_p2p_bw_gb_s: 450.0,
    ib_p2p_alpha_us: 10.0,
    ib_p2p_bw_gb_s: 50.0,
    ...overrides,
  };
}

// =============================================================================
// Hardware Configuration
// =============================================================================

/** H100 SXM specs */
export interface Hardware {
  n_gpus: number;
  gpus_per_node: number;

  /** Baseline HBM capacity (at 100%) in GB */
  hbm_capacity_gb: number;
  /** Sustained HBM bandwidth in TB/s */
  hbm_bandwidth_tb_s: number;
  /** Sustained BF16 compute in TFLOPS */
  compute_tflops: number;
}

export function createHardware(overrides?: Partial<Hardware>): Hardware {
  return {
    n_gpus: 10_000,
    gpus_per_node: 8,
    hbm_capacity_gb: 80.0,
    hbm_bandwidth_tb_s: 2.5,
    compute_tflops: 700.0,
    ...overrides,
  };
}

// =============================================================================
// Model Configuration
// =============================================================================

/** Llama 3 70B specs */
export interface ModelConfig {
  name: string;
  n_params: number;
  n_layers: number;
  d_model: number;
  d_ff: number;
  n_heads: number;
  n_kv_heads: number;
  precision_bytes: number;
}

/**
 * Model with computed properties.
 *
 * Wraps a ModelConfig and provides derived quantities as getters.
 */
export class Model {
  readonly name: string;
  readonly n_params: number;
  readonly n_layers: number;
  readonly d_model: number;
  readonly d_ff: number;
  readonly n_heads: number;
  readonly n_kv_heads: number;
  readonly precision_bytes: number;

  constructor(config?: Partial<ModelConfig>) {
    this.name = config?.name ?? "Llama 3 70B";
    this.n_params = config?.n_params ?? 70e9;
    this.n_layers = config?.n_layers ?? 80;
    this.d_model = config?.d_model ?? 8192;
    this.d_ff = config?.d_ff ?? 28672;
    this.n_heads = config?.n_heads ?? 64;
    this.n_kv_heads = config?.n_kv_heads ?? 8;
    this.precision_bytes = config?.precision_bytes ?? 2; // BF16
  }

  get weight_size_gb(): number {
    return (this.n_params * this.precision_bytes) / 1e9;
  }

  get weight_per_layer_gb(): number {
    return this.weight_size_gb / this.n_layers;
  }

  get largest_matrix_mb(): number {
    return (this.d_model * this.d_ff * this.precision_bytes) / 1e6;
  }

  get flops_per_token(): number {
    return 2 * this.n_params;
  }

  get kv_cache_per_token_kb(): number {
    const d_head = Math.floor(this.d_model / this.n_heads);
    return (
      (2 * this.n_layers * this.n_kv_heads * d_head * this.precision_bytes) /
      1e3
    );
  }

  get mamba_state_mb(): number {
    const d_inner = 16384;
    const d_state = 64;
    return (d_inner * d_state * this.n_layers * this.precision_bytes) / 1e6;
  }
}

// =============================================================================
// Training Configuration
// =============================================================================

/**
 * Training-specific modeling knobs.
 *
 * Defaults assume "nice tricks":
 *   - 1F1B pipeline schedule
 *   - Activation checkpointing (recompute forward once)
 *   - Sequence-parallel activations (sharded by TP)
 *   - DiLoCo-style infrequent DP sync (amortized)
 *
 * Key dials that significantly affect results:
 *   - param_state_mult: 3.0 = aggressive (ZeRO + 8-bit), 6-8 = conservative (full AdamW)
 *   - pp_activation_sharded_by_tp: true = optimistic, false = Megatron default
 *   - tp_comm_seq_parallel: true = smaller TP messages (optimistic), false = full d_model
 */
export interface TrainingConfig {
  seq_len: number;
  /** null = Mamba (no attention), number = Transformer */
  context_length: number | null;

  /** Backward compute multiplier relative to one forward pass */
  bwd_compute_mult: number;
  /** 1.0 = full activation checkpointing */
  recompute_fwd_mult: number;

  /**
   * Parameter-state HBM multiplier relative to BF16 weight shard size.
   * Encodes optimizer choice, ZeRO, 8-bit states, etc.
   *   - AdamW full states (no ZeRO): ~6-8x  (conservative)
   *   - ZeRO-1/2 + standard optimizer: ~4-5x
   *   - ZeRO-3 / 8-bit optimizer: ~2-3x    (aggressive)
   */
  param_state_mult: number;

  /** Activation storage: d_model-sized BF16 tensors per token at stage boundary */
  act_tensors_per_token: number;
  /** Sequence parallelism for activations */
  act_shard_by_tp: boolean;

  /** Pipeline schedule: "1f1b" or "gpipe" */
  pipeline_schedule: string;

  /** Communication multipliers (relative to inference forward) */
  tp_comm_bwd_mult: number;
  /** Forward activations + backward grads */
  pp_directions: number;

  /**
   * TP communication payload sharding under sequence parallelism.
   * true = messages are sharded by TP (optimistic: RS/AG instead of full AR)
   * false = full d_model messages (conservative: standard Megatron TP)
   */
  tp_comm_seq_parallel: boolean;

  /** Gradient buffer bandwidth (accumulation). 2.0 ~ read+write */
  grad_accum_rw_mult: number;

  /** DiLoCo / periodic DP sync: sync every K optimizer steps */
  diloco_sync_every: number;
  /** 1.0 = full payload; <1 = compression */
  diloco_payload_frac: number;
  dp_algo: string;

  /**
   * PP activation sharding at pipeline boundaries.
   * true = optimistic (seq-parallel boundaries)
   * false = full activations per TP rank (Megatron default)
   */
  pp_activation_sharded_by_tp: boolean;

  /**
   * Activation checkpointing granularity.
   * "stage" = checkpoint only at stage boundaries (optimistic memory, fewer ckpts)
   * "layer" = checkpoint every layer in the stage (conservative, realistic for large models)
   */
  act_ckpt_granularity: string;

  /** Reserve fraction for workspace/fragmentation */
  hbm_headroom_frac: number;
  /** Cap to prevent absurd microbatches (null = no cap) */
  max_microbatch_tokens: number | null;
}

export function createTrainingConfig(
  overrides: { seq_len: number } & Partial<Omit<TrainingConfig, "seq_len">>
): TrainingConfig {
  return {
    context_length: null,
    bwd_compute_mult: 2.0,
    recompute_fwd_mult: 1.0,
    param_state_mult: 3.0,
    act_tensors_per_token: 2.0,
    act_shard_by_tp: true,
    pipeline_schedule: "1f1b",
    tp_comm_bwd_mult: 1.0,
    pp_directions: 2,
    tp_comm_seq_parallel: false,
    grad_accum_rw_mult: 2.0,
    diloco_sync_every: 64,
    diloco_payload_frac: 1.0,
    dp_algo: "tree",
    pp_activation_sharded_by_tp: false,
    act_ckpt_granularity: "layer",
    hbm_headroom_frac: 0.1,
    max_microbatch_tokens: 2_000_000,
    ...overrides,
  };
}
