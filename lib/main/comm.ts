// =============================================================================
// Communication Model (alpha-beta with algorithm-aware step counts)
// =============================================================================

import type { CommParams } from "./model";
import { createCommParams } from "./model";

/**
 * Alpha-beta model for reduce-scatter (half of allreduce).
 * Returns [latency_s, bandwidth_s] separately.
 */
export function reduce_scatter_time_s(
  bytes_msg: number,
  n: number,
  bw_gb_s: number,
  alpha_us: number,
  algo: string = "ring"
): [number, number] {
  if (n <= 1) {
    return [0.0, 0.0];
  }

  let steps: number;
  let bytes_factor: number;

  if (algo === "ring") {
    steps = n - 1;
    bytes_factor = (n - 1) / n;
  } else if (algo === "tree") {
    steps = Math.ceil(Math.log2(n));
    bytes_factor = (n - 1) / n;
  } else {
    throw new Error(`Unknown algorithm: ${algo}`);
  }

  const alpha_s = alpha_us * 1e-6;
  const bw = bw_gb_s * 1e9;

  return [steps * alpha_s, bytes_factor * (bytes_msg / bw)];
}

/**
 * Alpha-beta model for allgather (same as reduce-scatter for ring).
 * Returns [latency_s, bandwidth_s] separately.
 */
export function allgather_time_s(
  bytes_msg: number,
  n: number,
  bw_gb_s: number,
  alpha_us: number,
  algo: string = "ring"
): [number, number] {
  return reduce_scatter_time_s(bytes_msg, n, bw_gb_s, alpha_us, algo);
}

/**
 * Alpha-beta model for allreduce = reduce-scatter + allgather.
 * Returns [latency_s, bandwidth_s] separately.
 */
export function allreduce_time_s(
  bytes_msg: number,
  n: number,
  bw_gb_s: number,
  alpha_us: number,
  algo: string = "ring"
): [number, number] {
  const [rs_lat, rs_bw] = reduce_scatter_time_s(
    bytes_msg,
    n,
    bw_gb_s,
    alpha_us,
    algo
  );
  const [ag_lat, ag_bw] = allgather_time_s(
    bytes_msg,
    n,
    bw_gb_s,
    alpha_us,
    algo
  );
  return [rs_lat + ag_lat, rs_bw + ag_bw];
}

/**
 * Hierarchical allreduce: intra-node reduce-scatter + inter-node allreduce + intra-node allgather.
 *
 * Includes kernel launch overhead from NCCL (paid once per collective).
 */
export function hierarchical_allreduce_time_s(
  bytes_msg: number,
  tp: number,
  gpus_per_node: number,
  comm: CommParams,
  intra_algo: string = "ring",
  inter_algo: string = "tree"
): [number, number] {
  const nodes = Math.ceil(tp / gpus_per_node);
  const gpus_in_tp_per_node = Math.min(tp, gpus_per_node);

  // Kernel launch latency (paid once per collective operation)
  const kernel_lat = comm.kernel_launch_us * 1e-6;

  if (nodes <= 1) {
    // Pure intra-node: full allreduce
    const [lat, bw] = allreduce_time_s(
      bytes_msg,
      tp,
      comm.nvlink_bw_gb_s,
      comm.nvlink_alpha_us,
      intra_algo
    );
    return [lat + kernel_lat, bw];
  }

  // Phase 1: Intra-node reduce-scatter
  const [rs_lat, rs_bw] = reduce_scatter_time_s(
    bytes_msg,
    gpus_in_tp_per_node,
    comm.nvlink_bw_gb_s,
    comm.nvlink_alpha_us,
    intra_algo
  );

  // Phase 2: Inter-node allreduce on sharded data (each GPU has bytes_msg/gpn)
  const inter_bytes = bytes_msg / gpus_in_tp_per_node;
  const [ar_lat, ar_bw] = allreduce_time_s(
    inter_bytes,
    nodes,
    comm.ib_bw_gb_s,
    comm.ib_alpha_us,
    inter_algo
  );

  // Phase 3: Intra-node allgather
  const [ag_lat, ag_bw] = allgather_time_s(
    bytes_msg,
    gpus_in_tp_per_node,
    comm.nvlink_bw_gb_s,
    comm.nvlink_alpha_us,
    intra_algo
  );

  return [rs_lat + ar_lat + ag_lat + kernel_lat, rs_bw + ar_bw + ag_bw];
}

/**
 * Point-to-point transfer time. Returns [latency_s, bandwidth_s].
 */
export function p2p_time_s(
  bytes_msg: number,
  bw_gb_s: number,
  alpha_us: number
): [number, number] {
  return [alpha_us * 1e-6, bytes_msg / (bw_gb_s * 1e9)];
}

/**
 * Pipeline utilization factor.
 * U = M / (M + PP - 1)
 */
export function pipeline_utilization(pp: number, M: number): number {
  if (M <= 0 || pp <= 0) {
    return 0.0;
  }
  return M / (M + pp - 1);
}
