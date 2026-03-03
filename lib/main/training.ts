// =============================================================================
// Training Throughput Calculator
// =============================================================================

import type { Hardware, CommParams, TrainingConfig } from "./model";
import { Model, createCommParams, createHardware, createTrainingConfig } from "./model";
import {
  hierarchical_allreduce_time_s,
  allreduce_time_s,
  p2p_time_s,
} from "./comm";
import { calculate_throughput } from "./inference";

export interface TrainingConfigResult {
  pp: number;
  tp: number;
  M: number;
  m: number;
  seq_len: number;
  gpus_per_replica: number;
  n_replicas: number;
  tokens_per_microbatch: number;
  param_state_mb: number;
  act_in_flight: number;
  t_compute_ms: number;
  t_mem_ms: number;
  t_tp_lat_ms: number;
  t_tp_bw_ms: number;
  t_pp_lat_ms: number;
  t_pp_bw_ms: number;
  t_dp_lat_ms: number;
  t_dp_bw_ms: number;
  t_microbatch_ms: number;
  pipeline_util: number;
  throughput_per_replica: number;
  hbm_per_gpu_gb: number;
  train_flops_per_token: number;
  bw_overlap_frac: number;
  schedule: string;
  compute_ceiling?: number;
  efficiency?: number;
  error?: string;
}

/**
 * Training throughput model: forward + backward (+ optional recompute),
 * with TP/PP + DiLoCo amortized DP sync.
 *
 * Assumes filled pipeline (continuous batching). For training with 1F1B,
 * this is optimistic but simplifies the model.
 *
 * Returns: [cluster_throughput_tok_s, debug_info]
 */
export function calculate_training_throughput(
  hw: Hardware,
  model: Model,
  hbm_fraction: number,
  compute_fraction: number,
  train: TrainingConfig,
  comm: CommParams | null = null,
  bw_overlap_frac: number = 0.5,
  max_tp: number = 128
): [number, TrainingConfigResult] {
  if (comm === null) {
    comm = createCommParams();
  }

  // Scale hardware (with headroom for workspace/fragmentation)
  const hbm_per_gpu_gb = hw.hbm_capacity_gb * hbm_fraction;
  const hbm_per_gpu_mb_raw = hbm_per_gpu_gb * 1000;
  const hbm_per_gpu_mb =
    hbm_per_gpu_mb_raw * (1.0 - train.hbm_headroom_frac);
  const hbm_bw_tb_s = hw.hbm_bandwidth_tb_s * compute_fraction;
  const compute_tflops = hw.compute_tflops * compute_fraction;

  // FLOPs per token (forward only)
  const base_fwd_flops_per_token = 2 * model.flops_per_token;

  // Training attention FLOPs: per-token ~ 2 * d_model * seq_len per layer
  // Only add attention FLOPs for Transformers (context_length is not null)
  let fwd_flops_per_token: number;
  if (train.context_length !== null) {
    const attention_fwd_flops_per_token =
      2 * model.n_layers * model.d_model * train.seq_len;
    fwd_flops_per_token =
      base_fwd_flops_per_token + attention_fwd_flops_per_token;
  } else {
    // Mamba: no attention, just the base MLP/SSM compute
    fwd_flops_per_token = base_fwd_flops_per_token;
  }

  // Total training multiplier: forward + backward + recompute-forward
  const compute_mult =
    1.0 + train.bwd_compute_mult + train.recompute_fwd_mult;
  const train_flops_per_token = fwd_flops_per_token * compute_mult;

  // Compute ceiling (tokens/s)
  const cluster_tflops = hw.n_gpus * compute_tflops;
  const compute_ceiling = (cluster_tflops * 1e12) / train_flops_per_token;

  // Search space
  const pp_values: number[] = [];
  for (let i = 1; i <= model.n_layers; i++) {
    pp_values.push(i);
  }
  const tp_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512].filter(
    (t) => t <= max_tp
  );

  let best_throughput = 0.0;
  let best_config: TrainingConfigResult | null = null;

  for (const pp of pp_values) {
    const layers_per_stage = model.n_layers / pp;

    for (const tp of tp_values) {
      const gpus_per_replica = pp * tp;
      if (gpus_per_replica > hw.n_gpus) {
        continue;
      }

      const n_replicas = Math.floor(hw.n_gpus / gpus_per_replica);
      if (n_replicas < 1) {
        continue;
      }

      // Parameter-state memory (weights + grads + optimizer states)
      const weight_shard_mb =
        (model.weight_size_gb * 1000) / gpus_per_replica;
      const param_state_mb = weight_shard_mb * train.param_state_mult;

      if (param_state_mb > hbm_per_gpu_mb * 0.95) {
        continue;
      }

      // Microbatch count search (M)
      const M_set = new Set<number>([
        1, 2, 4, 8, 16, 32, 64,
        Math.floor(pp / 2), pp, 2 * pp, 4 * pp,
      ]);
      // Denser search around pp
      for (let i = Math.max(1, pp - 8); i <= pp + 8; i++) {
        M_set.add(i);
      }
      const M_values = Array.from(M_set)
        .filter((M) => M >= 1)
        .sort((a, b) => a - b);

      for (const M of M_values) {
        // How many microbatches' activations are live on a stage?
        let in_flight: number;
        if (train.pipeline_schedule === "gpipe") {
          in_flight = M;
        } else {
          // 1F1B: bounded by pipeline depth
          in_flight = Math.min(M, pp);
        }

        // Activation memory per sequence per GPU
        // Checkpoint granularity determines how many checkpoint points per stage
        let ckpt_points: number;
        if (train.act_ckpt_granularity === "stage") {
          ckpt_points = 1; // Only checkpoint at stage boundaries (optimistic)
        } else if (train.act_ckpt_granularity === "layer") {
          ckpt_points = Math.ceil(layers_per_stage); // Checkpoint every layer (realistic)
        } else {
          throw new Error(
            `Unknown act_ckpt_granularity=${train.act_ckpt_granularity}`
          );
        }

        const act_per_seq_mb =
          (train.act_tensors_per_token *
            ckpt_points *
            train.seq_len *
            model.d_model *
            model.precision_bytes) /
          1e6;
        const act_shard = train.act_shard_by_tp ? tp : 1;
        const act_per_seq_per_gpu_mb = act_per_seq_mb / act_shard;

        // Available memory for activations
        const avail_act_mb = hbm_per_gpu_mb - param_state_mb;
        if (avail_act_mb <= 0) {
          continue;
        }

        // Max microbatch size m (sequences per microbatch)
        const denom = in_flight * act_per_seq_per_gpu_mb;
        if (denom <= 0) {
          continue;
        }
        let m = Math.floor(avail_act_mb / denom);
        if (m < 1) {
          continue;
        }

        // Cap microbatch tokens to prevent absurdly huge microbatches
        let tokens_per_microbatch = m * train.seq_len;
        if (train.max_microbatch_tokens !== null) {
          const m_cap = Math.max(
            1,
            Math.floor(train.max_microbatch_tokens / train.seq_len)
          );
          if (m > m_cap) {
            m = m_cap;
            tokens_per_microbatch = m * train.seq_len;
          }
        }

        // =============================================================
        // Compute time per microbatch (fwd + bwd + recompute)
        // =============================================================
        const flops_per_microbatch_stage =
          tokens_per_microbatch *
          train_flops_per_token *
          (layers_per_stage / model.n_layers);
        const t_compute_s =
          flops_per_microbatch_stage / (tp * compute_tflops * 1e12);

        // =============================================================
        // HBM bandwidth time per microbatch
        // =============================================================
        const weight_per_stage_mb =
          (model.weight_size_gb * 1000) / (pp * tp);

        // Weights: fwd + backward dX + recompute fwd
        const weight_reads_mult = 2.0 + train.recompute_fwd_mult;
        const weight_read_mb = weight_reads_mult * weight_per_stage_mb;

        // Gradient accumulation buffer
        const grad_accum_mb =
          train.grad_accum_rw_mult * weight_per_stage_mb;

        // Activation bandwidth
        const act_tensors_per_layer = 4.0;
        const act_bytes_per_token =
          model.d_model * model.precision_bytes;
        const act_shard_bw = train.act_shard_by_tp ? tp : 1;
        const act_bw_mb =
          (tokens_per_microbatch *
            act_bytes_per_token *
            layers_per_stage *
            act_tensors_per_layer) /
          act_shard_bw /
          1e6;

        const t_mem_s =
          (weight_read_mb + grad_accum_mb + act_bw_mb) /
          (hbm_bw_tb_s * 1e6);

        // =============================================================
        // TP communication per microbatch
        // =============================================================
        let t_tp_lat_s = 0.0;
        let t_tp_bw_s = 0.0;
        if (tp > 1) {
          const ar_per_layer_fwd = 2.0;
          const ar_per_layer_bwd =
            ar_per_layer_fwd * train.tp_comm_bwd_mult;
          const ar_per_layer_recomp =
            ar_per_layer_fwd * train.recompute_fwd_mult;
          const n_allreduces =
            layers_per_stage *
            (ar_per_layer_fwd + ar_per_layer_bwd + ar_per_layer_recomp);

          // Under sequence parallelism, TP collectives are RS/AG on sharded activations
          let bytes_per_allreduce =
            tokens_per_microbatch *
            model.d_model *
            model.precision_bytes;
          if (train.tp_comm_seq_parallel) {
            bytes_per_allreduce = bytes_per_allreduce / tp;
          }

          const [lat_one, bw_one] = hierarchical_allreduce_time_s(
            bytes_per_allreduce,
            tp,
            hw.gpus_per_node,
            comm
          );
          t_tp_lat_s = n_allreduces * lat_one;
          t_tp_bw_s = n_allreduces * bw_one;
        }

        // =============================================================
        // PP communication per microbatch (fwd acts + bwd grads)
        // =============================================================
        let t_pp_lat_s = 0.0;
        let t_pp_bw_s = 0.0;
        if (pp > 1) {
          let bytes_xfer =
            tokens_per_microbatch *
            model.d_model *
            model.precision_bytes;

          if (train.pp_activation_sharded_by_tp) {
            bytes_xfer = bytes_xfer / tp;
          }

          // Node crossing heuristic
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
            pp > 1 ? n_crossings / (pp - 1) : 0.0;

          const alpha_eff =
            frac_ib * comm.ib_p2p_alpha_us +
            (1 - frac_ib) * comm.nv_p2p_alpha_us;
          const bw_eff =
            frac_ib > 0
              ? 1 /
                (frac_ib / comm.ib_p2p_bw_gb_s +
                  (1 - frac_ib) / comm.nv_p2p_bw_gb_s)
              : comm.nv_p2p_bw_gb_s;

          const [lat_one, bw_one] = p2p_time_s(
            bytes_xfer,
            bw_eff,
            alpha_eff
          );

          const p2p_factor = pp > 2 ? 2 : 1; // send+recv for interior stages
          const dir_factor = train.pp_directions; // fwd + bwd
          t_pp_lat_s = p2p_factor * dir_factor * lat_one;
          t_pp_bw_s = p2p_factor * dir_factor * bw_one;
        }

        // =============================================================
        // DiLoCo / DP sync amortized cost
        // =============================================================
        let t_dp_lat_s = 0.0;
        let t_dp_bw_s = 0.0;
        if (n_replicas > 1 && train.diloco_sync_every > 0) {
          // Payload: shard of weights/grads, possibly compressed
          const bytes_dp =
            ((model.weight_size_gb * 1e9) / gpus_per_replica) *
            train.diloco_payload_frac;

          const [dp_lat, dp_bw] = allreduce_time_s(
            bytes_dp,
            n_replicas,
            comm.ib_bw_gb_s,
            comm.ib_alpha_us,
            train.dp_algo
          );

          // Amortize over K steps * M microbatches
          const dp_denom = train.diloco_sync_every * M;
          t_dp_lat_s = dp_lat / dp_denom;
          t_dp_bw_s = dp_bw / dp_denom;
        }

        // =============================================================
        // Combine + overlap model
        // =============================================================
        const t_comm_lat = t_tp_lat_s + t_pp_lat_s + t_dp_lat_s;
        const t_comm_bw = t_tp_bw_s + t_pp_bw_s + t_dp_bw_s;

        const t_compute_mem = Math.max(t_compute_s, t_mem_s);
        const t_no_overlap = t_compute_mem + t_comm_lat + t_comm_bw;
        const t_full_overlap =
          Math.max(t_compute_mem, t_comm_bw) + t_comm_lat;
        const t_microbatch_s =
          (1 - bw_overlap_frac) * t_no_overlap +
          bw_overlap_frac * t_full_overlap;

        if (t_microbatch_s <= 0) {
          continue;
        }

        // =============================================================
        // Pipeline utilization (filled/continuous batching)
        // =============================================================
        const util = Math.min(1.0, M / pp);

        const throughput_replica =
          (tokens_per_microbatch / t_microbatch_s) * util;
        const cluster_throughput = n_replicas * throughput_replica;

        if (cluster_throughput > best_throughput) {
          best_throughput = cluster_throughput;
          best_config = {
            pp,
            tp,
            M,
            m,
            seq_len: train.seq_len,
            gpus_per_replica,
            n_replicas,
            tokens_per_microbatch,
            param_state_mb,
            act_in_flight: in_flight,
            t_compute_ms: t_compute_s * 1e3,
            t_mem_ms: t_mem_s * 1e3,
            t_tp_lat_ms: t_tp_lat_s * 1e3,
            t_tp_bw_ms: t_tp_bw_s * 1e3,
            t_pp_lat_ms: t_pp_lat_s * 1e3,
            t_pp_bw_ms: t_pp_bw_s * 1e3,
            t_dp_lat_ms: t_dp_lat_s * 1e3,
            t_dp_bw_ms: t_dp_bw_s * 1e3,
            t_microbatch_ms: t_microbatch_s * 1e3,
            pipeline_util: util,
            throughput_per_replica: throughput_replica,
            hbm_per_gpu_gb,
            train_flops_per_token,
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
      {
        error: "No valid configuration found (training)",
      } as unknown as TrainingConfigResult,
    ];
  }

  best_config.compute_ceiling = compute_ceiling;
  best_config.efficiency =
    compute_ceiling > 0 ? best_throughput / compute_ceiling : 0.0;
  return [best_throughput, best_config];
}

// =============================================================================
// Generate Data
// =============================================================================

export interface ScenarioData {
  hbm: number[];
  efficiency: number[];
}

export interface GenerateDataResult {
  inference_results: Record<
    number,
    Record<string, Record<string, ScenarioData>>
  >;
  training_results: Record<
    number,
    Record<string, Record<string, ScenarioData>>
  >;
  hbm_fractions: number[];
  compute_fractions: number[];
  inference_configs: [string, number | null][];
  training_configs: [string, number | null, number][];
}

/**
 * Helper to generate a logspace array (equivalent to np.logspace).
 */
function logspace(
  log10_start: number,
  log10_stop: number,
  num: number
): number[] {
  const result: number[] = [];
  const step = (log10_stop - log10_start) / (num - 1);
  for (let i = 0; i < num; i++) {
    result.push(Math.pow(10, log10_start + step * i));
  }
  return result;
}

/**
 * Sweep HBM fractions and compute fractions across inference and training configs.
 */
export function generate_data(max_tp: number = 128): GenerateDataResult {
  const hw = createHardware();
  const model = new Model();

  // Generate HBM fractions: logspace + specific points, sorted and unique
  let hbm_fractions_raw = logspace(
    Math.log10(0.0001),
    Math.log10(1.0),
    60
  );
  const specific_points = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0];
  hbm_fractions_raw = hbm_fractions_raw.concat(specific_points);
  // Deduplicate and sort
  const hbm_set = new Set(hbm_fractions_raw.map((v) => +v.toPrecision(15)));
  const hbm_fractions = Array.from(hbm_set).sort((a, b) => a - b);

  const compute_fractions = [1.0, 0.2, 0.05];

  // Three scenarios for comm latency (all assume full bandwidth overlap, gamma=1)
  const scenarios: [string, number][] = [
    ["pessimistic", 2.0],
    ["middle", 1.0],
    ["optimistic", 0.0],
  ];

  // Inference configs
  const inference_configs: [string, number | null][] = [
    ["Mamba", null],
    ["Transformer 2k", 2000],
    ["Transformer 8k", 8000],
    ["Transformer 32k", 32000],
  ];

  // Training configs (seq_len matters)
  const training_configs: [string, number | null, number][] = [
    ["Mamba 2k", null, 2048],
    ["Mamba 8k", null, 8192],
    ["Transformer 2k", 2048, 2048],
    ["Transformer 8k", 8192, 8192],
  ];

  const base_comm = createCommParams();

  function make_comm(latency_mult: number): CommParams {
    return createCommParams({
      nvlink_alpha_us: base_comm.nvlink_alpha_us * latency_mult,
      nvlink_bw_gb_s: base_comm.nvlink_bw_gb_s,
      ib_alpha_us: base_comm.ib_alpha_us * latency_mult,
      ib_bw_gb_s: base_comm.ib_bw_gb_s,
      nv_p2p_alpha_us: base_comm.nv_p2p_alpha_us * latency_mult,
      nv_p2p_bw_gb_s: base_comm.nv_p2p_bw_gb_s,
      ib_p2p_alpha_us: base_comm.ib_p2p_alpha_us * latency_mult,
      ib_p2p_bw_gb_s: base_comm.ib_p2p_bw_gb_s,
    });
  }

  // Compute unconstrained max for each config (100% compute, 100% HBM, no latency, max TP)
  const inference_max: Record<string, number> = {};
  for (const [config_name, context_len] of inference_configs) {
    const [tp_val] = calculate_throughput(
      hw,
      model,
      1.0,
      1.0,
      context_len,
      make_comm(0.0),
      1.0,
      max_tp
    );
    inference_max[config_name] = tp_val > 0 ? tp_val : 1.0;
  }

  const training_max: Record<string, number> = {};
  for (const [config_name, context_len, seq_len] of training_configs) {
    const train_cfg = createTrainingConfig({
      seq_len,
      context_length: context_len,
    });
    const [tp_val] = calculate_training_throughput(
      hw,
      model,
      1.0,
      1.0,
      train_cfg,
      make_comm(0.0),
      1.0,
      max_tp
    );
    training_max[config_name] = tp_val > 0 ? tp_val : 1.0;
  }

  // Inference results
  const inference_results: Record<
    number,
    Record<string, Record<string, ScenarioData>>
  > = {};
  for (const compute_frac of compute_fractions) {
    inference_results[compute_frac] = {};
    for (const [config_name, context_len] of inference_configs) {
      inference_results[compute_frac][config_name] = {};
      for (const [scenario_name, latency_mult] of scenarios) {
        const data: ScenarioData = { hbm: [], efficiency: [] };
        const comm_params = make_comm(latency_mult);
        for (const hbm_frac of hbm_fractions) {
          const [tp_val] = calculate_throughput(
            hw,
            model,
            hbm_frac,
            compute_frac,
            context_len,
            comm_params,
            1.0,
            max_tp
          );
          const efficiency =
            tp_val > 0 ? tp_val / inference_max[config_name] : 0;
          data.hbm.push(hbm_frac * 100);
          data.efficiency.push(efficiency);
        }
        inference_results[compute_frac][config_name][scenario_name] = data;
      }
    }
  }

  // Training results
  const training_results: Record<
    number,
    Record<string, Record<string, ScenarioData>>
  > = {};
  for (const compute_frac of compute_fractions) {
    training_results[compute_frac] = {};
    for (const [config_name, context_len, seq_len] of training_configs) {
      training_results[compute_frac][config_name] = {};
      const train_cfg = createTrainingConfig({
        seq_len,
        context_length: context_len,
      });
      for (const [scenario_name, latency_mult] of scenarios) {
        const data: ScenarioData = { hbm: [], efficiency: [] };
        const comm_params = make_comm(latency_mult);
        for (const hbm_frac of hbm_fractions) {
          const [tp_val] = calculate_training_throughput(
            hw,
            model,
            hbm_frac,
            compute_frac,
            train_cfg,
            comm_params,
            1.0,
            max_tp
          );
          const efficiency =
            tp_val > 0 ? tp_val / training_max[config_name] : 0;
          data.hbm.push(hbm_frac * 100);
          data.efficiency.push(efficiency);
        }
        training_results[compute_frac][config_name][scenario_name] = data;
      }
    }
  }

  return {
    inference_results,
    training_results,
    hbm_fractions,
    compute_fractions,
    inference_configs,
    training_configs,
  };
}
