// ---------------------------------------------------------------------------
// Speculative decoding functions – ported from erdil.py (Erdil 2025)
// ---------------------------------------------------------------------------

import { GPU } from "./gpus";
import { Model, scaleModel } from "./models";
import { tokenLatencySecondsDefault } from "./model";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Generate values similar to np.logspace(start, stop, num, base). */
function logspace(start: number, stop: number, num: number, base: number = 10): number[] {
  if (num <= 0) return [];
  if (num === 1) return [base ** start];
  const result: number[] = [];
  const step = (stop - start) / (num - 1);
  for (let i = 0; i < num; i++) {
    result.push(base ** (start + i * step));
  }
  return result;
}

// ---------------------------------------------------------------------------
// spec_dec_token_latency_seconds
// ---------------------------------------------------------------------------

/**
 * Token latency with speculative decoding.
 *
 * For each candidate gamma (draft length), computes:
 *   total = latency(target, seq_len=gamma) + gamma * latency(draft)
 *   expected_tokens = (1 - alpha^gamma) / (1 - alpha)
 *   effective = total / expected_tokens
 *
 * Returns the minimum over all gamma values.
 */
export function specDecTokenLatencySeconds(
  nGpu: number,
  model: Model,
  approxModel: Model,
  gpu: GPU,
  batchSize: number,
  acceptanceProb: number,
  gammaMax: number,
  inputLen: number = 0,
  usePp: boolean = false,
): number {
  let finalLatency = tokenLatencySecondsDefault(
    nGpu,
    model,
    gpu,
    batchSize,
    inputLen,
    1,
    usePp,
  );

  const approxTokenLatency = tokenLatencySecondsDefault(
    nGpu,
    approxModel,
    gpu,
    batchSize,
    inputLen,
    1,
    usePp,
  );

  for (let gamma = 2; gamma <= gammaMax; gamma++) {
    const baseTokenLatency = tokenLatencySecondsDefault(
      nGpu,
      model,
      gpu,
      batchSize,
      inputLen,
      gamma,
      usePp,
    );

    const totalLatency = baseTokenLatency + approxTokenLatency * gamma;
    const expectedTokensGenerated =
      (1 - acceptanceProb ** gamma) / (1 - acceptanceProb);
    finalLatency = Math.min(finalLatency, totalLatency / expectedTokensGenerated);
  }

  return finalLatency;
}

// ---------------------------------------------------------------------------
// spec_dec_mfu_target_opt
// ---------------------------------------------------------------------------

/**
 * Find the (N_GPU, batch_size) that minimizes latency while achieving
 * approximately the target MFU, using speculative decoding.
 *
 * Scalar grid search version (no numpy 2D arrays).
 */
export function specDecMfuTargetOpt(
  targetMfu: number,
  model: Model,
  approxModel: Model,
  gpu: GPU,
  acceptanceProb: number,
  gammaMax: number,
  inputLen: number = 0,
): { nGpu: number; batchSize: number; latencySeconds: number } {
  const minNumOfGpus =
    (model.totalParams * model.weightPrecisionBytes) / gpu.hbmSizeBytes;

  const batchSizeRange = logspace(0, 14 + Math.log2(model.sparsityFactor), 100, 2);
  const nGpuRange = logspace(Math.log2(minNumOfGpus), 10, 100, 2);

  let bestLatency = Infinity;
  let bestNGpu = nGpuRange[0];
  let bestBatchSize = batchSizeRange[0];

  for (const nGpuVal of nGpuRange) {
    for (const bsVal of batchSizeRange) {
      const latency = specDecTokenLatencySeconds(
        nGpuVal,
        model,
        approxModel,
        gpu,
        bsVal,
        acceptanceProb,
        gammaMax,
        inputLen,
      );

      if (!isFinite(latency)) continue;

      const flopsPerSecond =
        gpu.theoreticalFlopPerSecond[model.weightPrecisionBytes * 8];
      const mfu =
        model.arithmeticCostFlop(inputLen, bsVal) /
        (latency * flopsPerSecond * nGpuVal);

      // Only consider if within 5% of target MFU (in log space)
      if (Math.abs(Math.log(mfu) - Math.log(targetMfu)) > 0.05) continue;

      if (latency < bestLatency) {
        bestLatency = latency;
        bestNGpu = nGpuVal;
        bestBatchSize = bsVal;
      }
    }
  }

  // Re-evaluate at the best point
  const finalLatency = specDecTokenLatencySeconds(
    bestNGpu,
    model,
    approxModel,
    gpu,
    bestBatchSize,
    acceptanceProb,
    gammaMax,
    inputLen,
  );

  return { nGpu: bestNGpu, batchSize: bestBatchSize, latencySeconds: finalLatency };
}

// ---------------------------------------------------------------------------
// search_for_model
// ---------------------------------------------------------------------------

/**
 * Binary search for a model scale factor such that, at the target MFU,
 * the model achieves the target tokens/second.
 */
export function searchForModel(
  baseModel: Model,
  approxModel: Model,
  gpu: GPU,
  acceptanceProb: number,
  gammaMax: number,
  inputLen: number,
  targetMfu: number,
  targetTokPerSec: number,
): { model: Model; nGpu: number; batchSize: number } {
  let low = -4;
  let high = 4;
  let currentModel = baseModel;
  let nGpu = 1;
  let batchSize = 1;

  while (high - low > 1e-2) {
    const mid = low + (high - low) / 2;
    currentModel = scaleModel(baseModel, 10 ** mid);

    const result = specDecMfuTargetOpt(
      targetMfu,
      currentModel,
      approxModel,
      gpu,
      acceptanceProb,
      gammaMax,
      inputLen,
    );
    nGpu = result.nGpu;
    batchSize = result.batchSize;
    const tokPerSec = 1 / result.latencySeconds;

    if (tokPerSec < targetTokPerSec) {
      high = mid;
    } else {
      low = mid;
    }
  }

  return { model: currentModel, nGpu, batchSize };
}
