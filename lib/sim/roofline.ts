import { Model, MODEL_MAP } from "@/lib/erdil/models"
import { GPU, GPU_MAP } from "@/lib/erdil/gpus"
import type { ThroughputEstimate, TrainingSyncPolicy } from "./types"

export type RooflineResult = {
  throughput: ThroughputEstimate
  nPersist: number
  workspace: number
  optBatch: number
}

/**
 * Roofline-lite backend for inference.
 *
 * Assumes TP = nGpu, no PP/EP. Searches over batch sizes to find
 * the optimal throughput given available FLOP and HBM.
 */
export function rooflineLite(
  model: Model,
  gpu: GPU,
  nGpu: number,
  ctx: number,
  availFlops: number,
  availHbmBytes: number,
): RooflineResult {
  const nPersist = model.totalParams * model.weightPrecisionBytes
  const perGpuFlops = availFlops / nGpu
  const perGpuHbm = availHbmBytes / nGpu

  let bestThroughput = 0
  let bestBatch = 0
  let bestWorkspace = 0
  let bestRegime: ThroughputEstimate["regime"] = "under-batched"

  for (let exp = 0; exp <= 18; exp++) {
    const batch = 2 ** exp
    const workspace = model.kvCacheSizePerInputBytes * ctx * batch
    const perGpuPersist = nPersist / nGpu

    if (perGpuPersist + workspace > perGpuHbm) break

    const flop = model.arithmeticCostFlop(ctx, batch, 1)
    const tCompute = flop / (nGpu * perGpuFlops)
    const memBytes = model.memoryReadsWritesBytes(ctx, batch, 1, nGpu)
    const tMem = memBytes / (nGpu * gpu.hbmBandwidthBps)
    const latency = Math.max(tCompute, tMem)
    const throughput = batch / latency

    if (throughput > bestThroughput) {
      bestThroughput = throughput
      bestBatch = batch
      bestWorkspace = workspace
      bestRegime = tCompute >= tMem ? "compute" : "memory-bandwidth"
    }
  }

  if (bestBatch <= 2 && bestThroughput > 0) {
    bestRegime = "under-batched"
  }

  return {
    throughput: {
      unitsPerSecond: bestThroughput,
      regime: bestRegime,
    },
    nPersist,
    workspace: bestWorkspace,
    optBatch: bestBatch,
  }
}

/**
 * Roofline-lite backend for training.
 *
 * Multiplies FLOP by 3 (fwd+bwd), adds optimizer state to nPersist.
 * Optimizer state ≈ 12 bytes per param (Adam: param + momentum + variance in FP32).
 */
export function rooflineLiteTraining(
  model: Model,
  gpu: GPU,
  nGpu: number,
  availFlops: number,
  availHbmBytes: number,
): RooflineResult {
  const weightBytes = model.totalParams * model.weightPrecisionBytes
  const optimizerBytes = model.totalParams * 12 // Adam state in FP32
  const nPersist = weightBytes + optimizerBytes
  const perGpuFlops = availFlops / nGpu
  const perGpuHbm = availHbmBytes / nGpu

  let bestThroughput = 0
  let bestBatch = 0
  let bestWorkspace = 0
  let bestRegime: ThroughputEstimate["regime"] = "under-batched"

  // For training, context isn't KV-cache bound; use activations as workspace
  // Approximate activation memory as 2 * dModel * layers * batch * activationPrecisionBytes
  const activBytesPerToken = 2 * model.dModel * model.layers * model.activationPrecisionBytes

  for (let exp = 0; exp <= 18; exp++) {
    const batch = 2 ** exp
    const workspace = activBytesPerToken * batch
    const perGpuPersist = nPersist / nGpu

    if (perGpuPersist + workspace > perGpuHbm) break

    // Training FLOP = 3× inference FLOP (fwd + bwd)
    const flop = 3 * model.arithmeticCostFlop(1, batch, 1)
    const tCompute = flop / (nGpu * perGpuFlops)
    const memBytes = 3 * model.memoryReadsWritesBytes(1, batch, 1, nGpu)
    const tMem = memBytes / (nGpu * gpu.hbmBandwidthBps)
    const latency = Math.max(tCompute, tMem)
    const throughput = batch / latency

    if (throughput > bestThroughput) {
      bestThroughput = throughput
      bestBatch = batch
      bestWorkspace = workspace
      bestRegime = tCompute >= tMem ? "compute" : "memory-bandwidth"
    }
  }

  if (bestBatch <= 2 && bestThroughput > 0) {
    bestRegime = "under-batched"
  }

  return {
    throughput: {
      unitsPerSecond: bestThroughput,
      regime: bestRegime,
    },
    nPersist,
    workspace: bestWorkspace,
    optBatch: bestBatch,
  }
}

/**
 * Derive d_in/d_out from a training sync policy.
 */
export function deriveSyncIO(policy: TrainingSyncPolicy): { dIn: number; dOut: number } {
  switch (policy.mode) {
    case "none":
      return { dIn: 0, dOut: 0 }
    case "checkpoint":
      return {
        dIn: 0,
        dOut: policy.bytesOutPerSync / policy.tokensPerSync,
      }
    case "periodic-updates":
      return {
        dIn: policy.bytesInPerSync / policy.tokensPerSync,
        dOut: policy.bytesOutPerSync / policy.tokensPerSync,
      }
  }
}

export function resolveModel(key: string): Model {
  const m = MODEL_MAP[key]
  if (!m) throw new Error(`Unknown model: ${key}`)
  return m
}

export function resolveGpu(key: string): GPU {
  const g = GPU_MAP[key]
  if (!g) throw new Error(`Unknown GPU: ${key}`)
  return g
}

export { MODEL_MAP, GPU_MAP }
