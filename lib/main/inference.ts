// =============================================================================
// Inference Throughput Calculator
// =============================================================================

import type { Hardware, CommParams } from "./model";
import { Model, createCommParams } from "./model";
import {
  hierarchical_allreduce_time_s,
  p2p_time_s,
} from "./comm";

export interface InferenceConfig {
  pp: number;
  tp: number;
  M: number;
  m: number;
  gpus_per_replica: number;
  n_replicas: number;
  B_total: number;
  layers_per_stage: number;
  weight_per_gpu_mb: number;
  remaining_mb: number;
  state_per_seq_mb: number;
  t_compute_per_mb_ms: number;
  t_weight_read_per_mb_ms: number;
  t_kv_read_per_mb_ms: number;
  t_state_rw_per_mb_ms: number;
  t_tp_lat_per_mb_ms: number;
  t_tp_bw_per_mb_ms: number;
  t_pp_lat_per_mb_ms: number;
  t_pp_bw_per_mb_ms: number;
  t_microbatch_ms: number;
  pipeline_util: number;
  throughput_per_replica: number;
  hbm_per_gpu_gb: number;
  bottleneck: string;
  flops_per_token: number;
  bw_overlap_frac: number;
  schedule: string;
  compute_ceiling?: number;
  efficiency?: number;
  error?: string;
}

/**
 * Calculate max throughput with explicit alpha-beta comm model.
 *
 * Assumes filled/continuous batching pipeline (standard for inference serving).
 *
 * Key features:
 * - Microbatch count M is a decision variable (searched over)
 * - Hierarchical allreduce for cross-node TP
 * - PP transfers include latency and count send+recv
 * - Overlap factor: only bandwidth overlaps, not latency
 */
export function calculate_throughput(
  hw: Hardware,
  model: Model,
  hbm_fraction: number,
  compute_fraction: number,
  context_length: number | null = null,
  comm: CommParams | null = null,
  bw_overlap_frac: number = 0.5,
  max_tp: number = 128
): [number, InferenceConfig] {
  if (comm === null) {
    comm = createCommParams();
  }

  // Scale hardware
  const hbm_per_gpu_gb = hw.hbm_capacity_gb * hbm_fraction;
  const hbm_per_gpu_mb = hbm_per_gpu_gb * 1000;
  const hbm_bw_tb_s = hw.hbm_bandwidth_tb_s * compute_fraction;
  const compute_tflops = hw.compute_tflops * compute_fraction;

  // FLOPs per token
  const base_flops_per_token = 2 * model.flops_per_token;
  let total_flops_per_token: number;
  if (context_length !== null) {
    const attention_flops =
      4 * model.n_layers * model.d_model * context_length;
    total_flops_per_token = base_flops_per_token + attention_flops;
  } else {
    total_flops_per_token = base_flops_per_token;
  }

  // Compute ceiling
  const cluster_tflops = hw.n_gpus * compute_tflops;
  const compute_ceiling = (cluster_tflops * 1e12) / total_flops_per_token;

  // State size per sequence
  let state_per_seq_mb: number;
  let kv_read_per_token_mb: number;
  if (context_length === null) {
    state_per_seq_mb = model.mamba_state_mb;
    kv_read_per_token_mb = 0;
  } else {
    state_per_seq_mb = (context_length * model.kv_cache_per_token_kb) / 1000;
    kv_read_per_token_mb = state_per_seq_mb;
  }

  // Minimum TP required
  let min_tp = 1;
  if (hbm_per_gpu_mb < model.largest_matrix_mb * 1.1) {
    min_tp = Math.ceil((model.largest_matrix_mb * 1.1) / hbm_per_gpu_mb);
  }

  let best_throughput = 0;
  let best_config: InferenceConfig | null = null;

  // Search over configurations
  const pp_values: number[] = [];
  for (let i = 1; i <= 80; i++) {
    pp_values.push(i);
  }
  const tp_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512].filter(
    (t) => t <= max_tp
  );

  for (const pp of pp_values) {
    if (pp > model.n_layers) {
      continue;
    }
    const layers_per_stage = model.n_layers / pp;

    for (const tp of tp_values) {
      if (tp < min_tp) {
        continue;
      }

      const gpus_per_replica = pp * tp;
      if (gpus_per_replica > hw.n_gpus) {
        continue;
      }

      const n_replicas = Math.floor(hw.n_gpus / gpus_per_replica);
      if (n_replicas < 1) {
        continue;
      }

      // =================================================================
      // Memory constraints
      // =================================================================
      const weight_per_gpu_mb =
        (model.weight_size_gb * 1000) / gpus_per_replica;
      if (weight_per_gpu_mb > hbm_per_gpu_mb * 0.95) {
        continue;
      }

      // KV cache sharding limited by n_kv_heads for GQA models
      const tp_kv_mem =
        context_length !== null ? Math.min(tp, model.n_kv_heads) : tp;
      const state_per_gpu_per_seq_mb =
        state_per_seq_mb / (pp * tp_kv_mem);
      const activation_per_seq_mb =
        (model.d_model * model.precision_bytes * 2) / 1e6;
      const activation_per_gpu_per_seq_mb = activation_per_seq_mb / tp;

      const remaining_mb = hbm_per_gpu_mb - weight_per_gpu_mb;
      if (remaining_mb <= 0) {
        continue;
      }

      const mem_per_seq =
        state_per_gpu_per_seq_mb + activation_per_gpu_per_seq_mb;
      let max_B_total: number;
      if (mem_per_seq > 0) {
        max_B_total = Math.floor(remaining_mb / mem_per_seq);
      } else {
        max_B_total = 10000;
      }
      max_B_total = Math.min(max_B_total, 50000);
      if (max_B_total < 1) {
        continue;
      }

      // =================================================================
      // Search over microbatch counts M
      // =================================================================
      // Expanded search: powers of 2 + PP-relative values
      const M_set = new Set<number>([
        1, 2, 4, 8, 16, 32, 64,
        Math.floor(pp / 2), pp, 2 * pp, 4 * pp,
      ]);
      // Denser search around pp
      for (
        let i = Math.max(1, pp - 8);
        i <= pp + 8;
        i++
      ) {
        M_set.add(i);
      }
      const M_values = Array.from(M_set)
        .filter((M) => M >= 1 && M <= max_B_total)
        .sort((a, b) => a - b);

      for (const M of M_values) {
        // Microbatch size
        const m = Math.floor(max_B_total / M);
        if (m < 1) {
          continue;
        }
        const B_total = m * M; // Guaranteed <= max_B_total

        // =============================================================
        // Per-microbatch time calculation
        // =============================================================

        // Compute time per microbatch
        const flops_per_microbatch =
          (m * total_flops_per_token * layers_per_stage) / model.n_layers;
        const t_compute_per_mb_s =
          flops_per_microbatch / (tp * compute_tflops * 1e12);

        // Weight read per microbatch
        const weight_per_stage_mb =
          (model.weight_size_gb * 1000) / (pp * tp);
        const t_weight_read_per_mb_s =
          weight_per_stage_mb / (hbm_bw_tb_s * 1e6);

        // KV/State read per microbatch
        let t_kv_read_per_mb_s: number;
        let t_state_rw_per_mb_s: number;
        if (context_length !== null) {
          const tp_kv = Math.min(tp, model.n_kv_heads);
          const kv_per_mb_mb =
            (m * kv_read_per_token_mb) / (pp * tp_kv);
          t_kv_read_per_mb_s = kv_per_mb_mb / (hbm_bw_tb_s * 1e6);
          t_state_rw_per_mb_s = 0;
        } else {
          t_kv_read_per_mb_s = 0;
          const state_rw_per_mb_mb =
            ((m * state_per_seq_mb) / (pp * tp)) * 2;
          t_state_rw_per_mb_s =
            state_rw_per_mb_mb / (hbm_bw_tb_s * 1e6);
        }

        const t_mem_per_mb_s =
          t_weight_read_per_mb_s +
          t_kv_read_per_mb_s +
          t_state_rw_per_mb_s;

        // =============================================================
        // TP communication per microbatch
        // =============================================================
        let t_tp_lat_per_mb_s = 0;
        let t_tp_bw_per_mb_s = 0;

        if (tp > 1) {
          const n_allreduces = 2 * layers_per_stage;
          const bytes_per_allreduce =
            m * model.d_model * model.precision_bytes;

          const [lat_one, bw_one] = hierarchical_allreduce_time_s(
            bytes_per_allreduce,
            tp,
            hw.gpus_per_node,
            comm
          );
          t_tp_lat_per_mb_s = n_allreduces * lat_one;
          t_tp_bw_per_mb_s = n_allreduces * bw_one;
        }

        // =============================================================
        // PP communication per microbatch (send + recv for interior stages)
        // =============================================================
        let t_pp_lat_per_mb_s = 0;
        let t_pp_bw_per_mb_s = 0;

        if (pp > 1) {
          // After allreduce each rank has FULL hidden state
          // So PP transfers are full activations, not sharded by TP
          const bytes_per_transfer =
            m * model.d_model * model.precision_bytes;

          // Determine node crossings
          let stages_per_node: number;
          if (tp <= hw.gpus_per_node) {
            stages_per_node = Math.floor(hw.gpus_per_node / tp);
          } else {
            stages_per_node = 1;
          }

          const n_nodes_for_pp =
            stages_per_node > 0
              ? Math.ceil(pp / stages_per_node)
              : pp;
          const n_crossings = Math.max(0, n_nodes_for_pp - 1);
          const frac_ib =
            pp > 1 ? n_crossings / (pp - 1) : 0;

          // Use proper p2p params (not collective params)
          const alpha_eff =
            frac_ib * comm.ib_p2p_alpha_us +
            (1 - frac_ib) * comm.nv_p2p_alpha_us;
          let bw_eff: number;
          if (frac_ib > 0) {
            bw_eff =
              1 /
              (frac_ib / comm.ib_p2p_bw_gb_s +
                (1 - frac_ib) / comm.nv_p2p_bw_gb_s);
          } else {
            bw_eff = comm.nv_p2p_bw_gb_s;
          }

          const [lat, bw] = p2p_time_s(
            bytes_per_transfer,
            bw_eff,
            alpha_eff
          );

          // Interior stages do send+recv
          const p2p_factor = pp > 2 ? 2 : 1;
          t_pp_lat_per_mb_s = p2p_factor * lat;
          t_pp_bw_per_mb_s = p2p_factor * bw;
        }

        // =============================================================
        // Total comm per microbatch
        // =============================================================
        const t_comm_lat_per_mb_s =
          t_tp_lat_per_mb_s + t_pp_lat_per_mb_s;
        const t_comm_bw_per_mb_s =
          t_tp_bw_per_mb_s + t_pp_bw_per_mb_s;

        // =============================================================
        // Microbatch time with overlap model
        // Simple interpolation: t = (1-gamma)*t_no_overlap + gamma*t_full_overlap
        // =============================================================
        const t_compute_mem = Math.max(
          t_compute_per_mb_s,
          t_mem_per_mb_s
        );

        const t_no_overlap =
          t_compute_mem + t_comm_lat_per_mb_s + t_comm_bw_per_mb_s;
        const t_full_overlap =
          Math.max(t_compute_mem, t_comm_bw_per_mb_s) +
          t_comm_lat_per_mb_s;

        const t_microbatch_s =
          (1 - bw_overlap_frac) * t_no_overlap +
          bw_overlap_frac * t_full_overlap;

        if (t_microbatch_s <= 0) {
          continue;
        }

        // =============================================================
        // Throughput with pipeline utilization (filled/continuous batching)
        // =============================================================
        const util = Math.min(1.0, M / pp);

        // Time per microbatch * M microbatches, adjusted by utilization
        const throughput_per_replica = (m / t_microbatch_s) * util;
        const cluster_throughput = n_replicas * throughput_per_replica;

        // Determine bottleneck
        let bottleneck: string;
        if (
          t_comm_lat_per_mb_s >
          Math.max(t_compute_mem, t_comm_bw_per_mb_s)
        ) {
          bottleneck = "Comm Latency";
        } else if (t_comm_bw_per_mb_s > t_compute_mem) {
          bottleneck = "Comm Bandwidth";
        } else if (t_compute_per_mb_s >= t_mem_per_mb_s) {
          bottleneck = "Compute";
        } else {
          const mem_terms: [string, number][] = [
            ["Weight Read", t_weight_read_per_mb_s],
            ["KV Read", t_kv_read_per_mb_s],
            ["State RW", t_state_rw_per_mb_s],
          ];
          bottleneck = mem_terms.reduce((a, b) =>
            b[1] > a[1] ? b : a
          )[0];
        }

        if (cluster_throughput > best_throughput) {
          best_throughput = cluster_throughput;
          best_config = {
            pp,
            tp,
            M,
            m,
            gpus_per_replica,
            n_replicas,
            B_total,
            layers_per_stage,
            weight_per_gpu_mb,
            remaining_mb,
            state_per_seq_mb,
            t_compute_per_mb_ms: t_compute_per_mb_s * 1000,
            t_weight_read_per_mb_ms: t_weight_read_per_mb_s * 1000,
            t_kv_read_per_mb_ms: t_kv_read_per_mb_s * 1000,
            t_state_rw_per_mb_ms: t_state_rw_per_mb_s * 1000,
            t_tp_lat_per_mb_ms: t_tp_lat_per_mb_s * 1000,
            t_tp_bw_per_mb_ms: t_tp_bw_per_mb_s * 1000,
            t_pp_lat_per_mb_ms: t_pp_lat_per_mb_s * 1000,
            t_pp_bw_per_mb_ms: t_pp_bw_per_mb_s * 1000,
            t_microbatch_ms: t_microbatch_s * 1000,
            pipeline_util: util,
            throughput_per_replica,
            hbm_per_gpu_gb,
            bottleneck,
            flops_per_token: total_flops_per_token,
            bw_overlap_frac,
            schedule: "filled",
          };
        }
      }
    }
  }

  if (best_config === null) {
    return [
      0.0,
      { error: "No valid configuration found" } as unknown as InferenceConfig,
    ];
  }

  best_config.compute_ceiling = compute_ceiling;
  best_config.efficiency =
    compute_ceiling > 0 ? best_throughput / compute_ceiling : 0;

  return [best_throughput, best_config];
}
