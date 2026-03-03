// ---------------------------------------------------------------------------
// Token economics / Pareto front analysis – ported from erdil.py (Erdil 2025)
// ---------------------------------------------------------------------------

import { GPU, H100, H200, A100, V100, H800, H20 } from "./gpus";
import {
  Model,
  GPT_4,
  GPT_4_2x,
  GPT_4_4x,
  GPT_4_8x,
  GPT_4_16x,
  GPT_4_100T_param,
  Llama_3_8B,
  Llama_3_8B_MQA,
  Llama_3_70B,
  Llama_3_70B_8_bit,
  Llama_3_70B_4_bit,
  Llama_3_405B,
  Llama_3_405B_8_bit,
  Llama_3_gpt4_size,
  Llama_3_gpt4_size_2x,
  Llama_3_gpt4_size_4x,
  Llama_3_gpt4_size_8x,
  Llama_3_gpt4_size_16x,
  Mixtral_8x22B,
  DeepSeek_V3,
  Mistral_Large_2,
} from "./models";
import { specDecTokenLatencySeconds } from "./spec-dec";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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
// TokenEconSettings
// ---------------------------------------------------------------------------

export class TokenEconSettings {
  name: string;
  gpu: GPU;
  model: Model;
  approxModel: Model | null;
  inputLen: number;
  maxThroughputTokensPerSecond: number;
  observedPerf: [number, number][] | null;
  color: string;
  specDec: boolean;
  acceptanceProb: number;

  constructor(opts: {
    name: string;
    gpu: GPU;
    model: Model;
    inputLen?: number;
    maxThroughputTokensPerSecond?: number;
    observedPerf?: [number, number][] | null;
    color?: string;
    specDec?: boolean;
    approxModel?: Model | null;
    acceptanceProb?: number;
  }) {
    this.name = opts.name;
    this.gpu = opts.gpu;
    this.model = opts.model;
    this.inputLen = opts.inputLen ?? 0;
    this.maxThroughputTokensPerSecond = opts.maxThroughputTokensPerSecond ?? Infinity;
    this.observedPerf = opts.observedPerf ?? null;
    this.color = opts.color ?? "";
    this.specDec = opts.specDec ?? false;
    this.acceptanceProb = opts.acceptanceProb ?? 1;
    this.approxModel = opts.approxModel ?? null;
  }
}

// ---------------------------------------------------------------------------
// ComparisonSettings
// ---------------------------------------------------------------------------

export class ComparisonSettings {
  comparisonList: TokenEconSettings[];
  fileName: string;
  plotTitle: string;

  constructor(
    comparisonList: TokenEconSettings[],
    fileName: string,
    plotTitle: string,
  ) {
    this.comparisonList = comparisonList;
    this.fileName = fileName;
    this.plotTitle = plotTitle;
  }
}

// ---------------------------------------------------------------------------
// Comparison presets
// ---------------------------------------------------------------------------

export const llamaComparison = new ComparisonSettings(
  [
    new TokenEconSettings({ name: "Llama 3 70B (1M tok/sec)", gpu: H100, model: Llama_3_70B, inputLen: 1, maxThroughputTokensPerSecond: 1e6, color: "darkblue" }),
    new TokenEconSettings({ name: "Llama 3 70B (10K tok/sec)", gpu: H100, model: Llama_3_70B, inputLen: 1, maxThroughputTokensPerSecond: 1e4, color: "blue" }),
    new TokenEconSettings({ name: "Llama 3 8B (1M tok/sec)", gpu: H100, model: Llama_3_8B, inputLen: 1, maxThroughputTokensPerSecond: 1e6, color: "darkred" }),
    new TokenEconSettings({ name: "Llama 3 8B (10K tok/sec)", gpu: H100, model: Llama_3_8B, inputLen: 1, maxThroughputTokensPerSecond: 1e4, color: "red" }),
  ],
  "token_economics_llama_comparison",
  "Token economics of Llama models on the H100 SXM",
);

export const allModelsComparison = new ComparisonSettings(
  [
    new TokenEconSettings({ name: "GPT-4 16-bit (speculative)", gpu: H100, model: GPT_4, color: "blue" }),
    new TokenEconSettings({ name: "Llama 3.1 405B 8-bit", gpu: H100, model: Llama_3_405B, color: "purple" }),
    new TokenEconSettings({ name: "Llama 3 70B 8-bit", gpu: H100, model: Llama_3_70B, color: "red" }),
    new TokenEconSettings({ name: "Mixtral 8x22B 16-bit", gpu: H100, model: Mixtral_8x22B, color: "green" }),
    new TokenEconSettings({ name: "DeepSeek-V3 8-bit", gpu: H100, model: DeepSeek_V3, color: "black" }),
  ],
  "token_economics_all_models_comparison",
  "Token economics of all models on the H100 SXM",
);

export const allModelsComparisonWithSpecDec = new ComparisonSettings(
  [
    new TokenEconSettings({ name: "GPT-4 16-bit (speculative)", gpu: H100, model: GPT_4, approxModel: Llama_3_8B, specDec: true, acceptanceProb: 0.8, color: "blue" }),
    new TokenEconSettings({ name: "Llama 3.1 405B 8-bit", gpu: H100, model: Llama_3_405B_8_bit, approxModel: Llama_3_8B, specDec: true, acceptanceProb: 0.8, color: "purple" }),
    new TokenEconSettings({ name: "Llama 3 70B 8-bit", gpu: H100, model: Llama_3_70B_8_bit, approxModel: Llama_3_8B, specDec: true, acceptanceProb: 0.8, color: "red" }),
    new TokenEconSettings({ name: "Mixtral 8x22B 16-bit", gpu: H100, model: Mixtral_8x22B, approxModel: Llama_3_8B, specDec: true, acceptanceProb: 0.8, color: "green" }),
    new TokenEconSettings({ name: "DeepSeek-V3 8-bit", gpu: H100, model: DeepSeek_V3, approxModel: Llama_3_8B, specDec: true, acceptanceProb: 0.8, color: "black" }),
  ],
  "token_economics_all_models_comparison_with_spec_dec",
  "Token economics of all models on the H100 SXM with speculative decoding",
);

export const gpusComparison = new ComparisonSettings(
  [
    new TokenEconSettings({ name: "H100", gpu: H100, model: Llama_3_70B_8_bit, inputLen: 0, maxThroughputTokensPerSecond: Infinity, color: "blue" }),
    new TokenEconSettings({ name: "A100", gpu: A100, model: Llama_3_70B_8_bit, inputLen: 0, maxThroughputTokensPerSecond: Infinity, color: "green" }),
    new TokenEconSettings({ name: "V100", gpu: V100, model: Llama_3_70B_8_bit, inputLen: 0, maxThroughputTokensPerSecond: Infinity, color: "red" }),
  ],
  "token_economics_gpus_comparison",
  "Token economics of Llama 3 70B (8-bit quantization) on different GPUs",
);

export const gpusLongContextComparison = new ComparisonSettings(
  [
    new TokenEconSettings({ name: "H100", gpu: H100, model: Llama_3_405B_8_bit, inputLen: 1e5, maxThroughputTokensPerSecond: Infinity, color: "blue" }),
    new TokenEconSettings({ name: "H800", gpu: H800, model: Llama_3_405B_8_bit, inputLen: 1e5, maxThroughputTokensPerSecond: Infinity, color: "cyan" }),
    new TokenEconSettings({ name: "A100", gpu: A100, model: Llama_3_405B_8_bit, inputLen: 1e5, maxThroughputTokensPerSecond: Infinity, color: "green" }),
    new TokenEconSettings({ name: "V100", gpu: V100, model: Llama_3_405B_8_bit, inputLen: 1e5, maxThroughputTokensPerSecond: Infinity, color: "red" }),
    new TokenEconSettings({ name: "H20", gpu: H20, model: Llama_3_405B_8_bit, inputLen: 1e5, maxThroughputTokensPerSecond: Infinity, color: "gray" }),
    new TokenEconSettings({ name: "H200", gpu: H200, model: Llama_3_405B_8_bit, inputLen: 1e5, maxThroughputTokensPerSecond: Infinity, color: "black" }),
  ],
  "token_economics_gpus_long_context_comparison",
  "Token economics of Llama 3 405B on different GPUs (long context)",
);

export const contextLengthComparison = new ComparisonSettings(
  [
    new TokenEconSettings({ name: "Empty context", gpu: H100, model: Llama_3_70B_8_bit, inputLen: 0, maxThroughputTokensPerSecond: Infinity, color: "blue" }),
    new TokenEconSettings({ name: "Context of 1K tokens", gpu: H100, model: Llama_3_70B_8_bit, inputLen: 1000, maxThroughputTokensPerSecond: Infinity, color: "green" }),
    new TokenEconSettings({ name: "Context of 10K tokens", gpu: H100, model: Llama_3_70B_8_bit, inputLen: 10000, maxThroughputTokensPerSecond: Infinity, color: "red" }),
  ],
  "token_economics_context_length_comparison",
  "Token economics of Llama 3 70B on different context lengths",
);

export const quantizationComparison = new ComparisonSettings(
  [
    new TokenEconSettings({ name: "Llama 3 70B 16-bit", gpu: H100, model: Llama_3_70B, inputLen: 0, color: "red" }),
    new TokenEconSettings({ name: "Llama 3 70B 8-bit", gpu: H100, model: Llama_3_70B_8_bit, inputLen: 0, color: "green" }),
    new TokenEconSettings({ name: "Llama 3 70B 4-bit", gpu: H100, model: Llama_3_70B_4_bit, inputLen: 0, color: "blue" }),
  ],
  "token_economics_quantization_comparison",
  "Token economics of Llama 3 70B on different precisions",
);

export const specDecComparison = new ComparisonSettings(
  [
    new TokenEconSettings({ name: "Llama 3 70B (8B acceptance=0.8)", gpu: H100, model: Llama_3_70B, approxModel: Llama_3_8B, specDec: true, acceptanceProb: 0.8, color: "red" }),
    new TokenEconSettings({ name: "Llama 3 70B (no spec dec)", gpu: H100, model: Llama_3_70B, approxModel: Llama_3_8B, specDec: true, acceptanceProb: 0, color: "gray" }),
    new TokenEconSettings({ name: "Llama 3.1 405B (8B acceptance=0.8)", gpu: H100, model: Llama_3_405B, approxModel: Llama_3_8B, specDec: true, acceptanceProb: 0.8, color: "blue" }),
    new TokenEconSettings({ name: "Llama 3.1 405B (no spec dec)", gpu: H100, model: Llama_3_405B, approxModel: Llama_3_70B, specDec: true, acceptanceProb: 0, color: "black" }),
  ],
  "token_economics_spec_dec_comparison",
  "Token economics of Llama models on different spec dec settings",
);

export const mistralLargeComparison = new ComparisonSettings(
  [
    new TokenEconSettings({ name: "Short context", gpu: H100, model: Mistral_Large_2, inputLen: 0, maxThroughputTokensPerSecond: Infinity, color: "red" }),
    new TokenEconSettings({ name: "Short context with speculative decoding", gpu: H100, model: Mistral_Large_2, approxModel: Llama_3_8B_MQA, specDec: true, acceptanceProb: 0.8, inputLen: 0, maxThroughputTokensPerSecond: Infinity, color: "darkred" }),
    new TokenEconSettings({ name: "Long context", gpu: H100, model: Mistral_Large_2, inputLen: 100000, maxThroughputTokensPerSecond: Infinity, color: "blue" }),
    new TokenEconSettings({ name: "Long context with speculative decoding", gpu: H100, model: Mistral_Large_2, approxModel: Llama_3_8B_MQA, specDec: true, acceptanceProb: 0.8, inputLen: 100000, maxThroughputTokensPerSecond: Infinity, color: "darkblue" }),
  ],
  "token_economics_mistral_large_comparison",
  "Token economics of Mistral Large 2 on different context lengths",
);

export const deepseekV3ContextLenComparison = new ComparisonSettings(
  [
    new TokenEconSettings({ name: "Short context", gpu: H100, model: DeepSeek_V3, inputLen: 0, maxThroughputTokensPerSecond: Infinity, color: "red" }),
    new TokenEconSettings({ name: "Short context with speculative decoding", gpu: H100, model: DeepSeek_V3, approxModel: Llama_3_8B_MQA, specDec: true, acceptanceProb: 0.8, inputLen: 0, maxThroughputTokensPerSecond: Infinity, color: "darkred" }),
    new TokenEconSettings({ name: "Long context", gpu: H100, model: DeepSeek_V3, inputLen: 100000, maxThroughputTokensPerSecond: Infinity, color: "blue" }),
    new TokenEconSettings({ name: "Long context with speculative decoding", gpu: H100, model: DeepSeek_V3, approxModel: Llama_3_8B_MQA, specDec: true, acceptanceProb: 0.8, inputLen: 100000, maxThroughputTokensPerSecond: Infinity, color: "darkblue" }),
  ],
  "token_economics_deepseek_v3_comparison",
  "Token economics of DeepSeek-V3 (8-bits) on different context lengths",
);

export const gpt4ComparisonWithSpecDec = new ComparisonSettings(
  [
    new TokenEconSettings({ name: "GPT-4 (1.8T params)", gpu: H200, model: GPT_4, approxModel: Llama_3_8B, specDec: true, acceptanceProb: 0.8, color: "#440154" }),
    new TokenEconSettings({ name: "Half-size GPT-4 (900B params)", gpu: H200, model: GPT_4_2x, approxModel: Llama_3_8B, specDec: true, acceptanceProb: 0.8, color: "#31688e" }),
    new TokenEconSettings({ name: "Quarter-size GPT-4 (450B params)", gpu: H200, model: GPT_4_4x, approxModel: Llama_3_8B, specDec: true, acceptanceProb: 0.8, color: "#35b779" }),
    new TokenEconSettings({ name: "Eighth-size GPT-4 (225B params)", gpu: H200, model: GPT_4_8x, approxModel: Llama_3_8B, specDec: true, acceptanceProb: 0.8, color: "#90d743" }),
    new TokenEconSettings({ name: "Sixteenth-size GPT-4 (112B params)", gpu: H200, model: GPT_4_16x, approxModel: Llama_3_8B, specDec: true, acceptanceProb: 0.8, color: "#fde725" }),
    new TokenEconSettings({ name: "Llama 3 70B", gpu: H200, model: Llama_3_70B, approxModel: Llama_3_8B, specDec: true, acceptanceProb: 0.8, color: "red" }),
  ],
  "token_economics_gpt_4_comparison_with_spec_dec",
  "Models on the scale of GPT-4 can't be served as quickly or cheaply as GPT-4o",
);

export const gpt4LlamaComparisonWithSpecDec = new ComparisonSettings(
  [
    new TokenEconSettings({ name: "GPT-4 (1.8T params)", gpu: H200, model: GPT_4, approxModel: Llama_3_8B, specDec: true, acceptanceProb: 0.8, color: "#440154" }),
    new TokenEconSettings({ name: "Half-size GPT-4 (900B params)", gpu: H200, model: GPT_4_2x, approxModel: Llama_3_8B, specDec: true, acceptanceProb: 0.8, color: "#31688e" }),
    new TokenEconSettings({ name: "Quarter-size GPT-4 (450B params)", gpu: H200, model: GPT_4_4x, approxModel: Llama_3_8B, specDec: true, acceptanceProb: 0.8, color: "#35b779" }),
    new TokenEconSettings({ name: "Eighth-size GPT-4 (225B params)", gpu: H200, model: GPT_4_8x, approxModel: Llama_3_8B, specDec: true, acceptanceProb: 0.8, color: "#90d743" }),
    new TokenEconSettings({ name: "Sixteenth-size GPT-4 (112B params)", gpu: H200, model: GPT_4_16x, approxModel: Llama_3_8B, specDec: true, acceptanceProb: 0.8, color: "#fde725" }),
    new TokenEconSettings({ name: "Llama 3 1.8T", gpu: H200, model: Llama_3_gpt4_size, approxModel: Llama_3_8B, specDec: true, acceptanceProb: 0.8, color: "#7c0d0d" }),
    new TokenEconSettings({ name: "Llama 3 900B", gpu: H200, model: Llama_3_gpt4_size_2x, approxModel: Llama_3_8B, specDec: true, acceptanceProb: 0.8, color: "#bc3d22" }),
    new TokenEconSettings({ name: "Llama 3 450B", gpu: H200, model: Llama_3_gpt4_size_4x, approxModel: Llama_3_8B, specDec: true, acceptanceProb: 0.8, color: "#e9724c" }),
    new TokenEconSettings({ name: "Llama 3 225B", gpu: H200, model: Llama_3_gpt4_size_8x, approxModel: Llama_3_8B, specDec: true, acceptanceProb: 0.8, color: "#faa476" }),
    new TokenEconSettings({ name: "Llama 3 112B", gpu: H200, model: Llama_3_gpt4_size_16x, approxModel: Llama_3_8B, specDec: true, acceptanceProb: 0.8, color: "#fcdeac" }),
  ],
  "token_economics_gpt_4_comparison_with_spec_dec",
  "Models on the scale of GPT-4 can't be served as quickly or cheaply as GPT-4o",
);

export const gpt4ComparisonWithSpecDecLongContext = new ComparisonSettings(
  [
    new TokenEconSettings({ name: "GPT-4 (1.8T params)", gpu: H200, model: GPT_4, approxModel: Llama_3_8B, specDec: true, inputLen: 1e5, acceptanceProb: 0.8, color: "#440154" }),
    new TokenEconSettings({ name: "Half-size GPT-4 (900B params)", gpu: H200, model: GPT_4_2x, approxModel: Llama_3_8B, specDec: true, inputLen: 1e5, acceptanceProb: 0.8, color: "#31688e" }),
    new TokenEconSettings({ name: "Quarter-size GPT-4 (450B params)", gpu: H200, model: GPT_4_4x, approxModel: Llama_3_8B, specDec: true, inputLen: 1e5, acceptanceProb: 0.8, color: "#35b779" }),
    new TokenEconSettings({ name: "Eighth-size GPT-4 (225B params)", gpu: H200, model: GPT_4_8x, approxModel: Llama_3_8B, specDec: true, inputLen: 1e5, acceptanceProb: 0.8, color: "#90d743" }),
    new TokenEconSettings({ name: "Sixteenth-size GPT-4 (112B params)", gpu: H200, model: GPT_4_16x, approxModel: Llama_3_8B, specDec: true, inputLen: 1e5, acceptanceProb: 0.8, color: "#fde725" }),
  ],
  "token_economics_gpt_4_comparison_with_spec_dec_long_context",
  "Token economics of scaled down GPT-4 versions on the H200 SXM (100K context length)",
);

// ---------------------------------------------------------------------------
// Pareto front computation
// ---------------------------------------------------------------------------

export type TokenLatencyFn = (
  nGpu: number,
  model: Model,
  gpu: GPU,
  batchSize: number,
  inputLen?: number,
  seqLen?: number,
  usePp?: boolean,
) => number;

export interface ParetoFrontResult {
  xCoords: number[];        // tokens/sec/request
  yCoords: number[];        // $/M output tokens
  gpuCounts: number[];
  batchSizes: number[];
  mfuValues: number[];
}

/**
 * Compute Pareto fronts for a list of comparison settings.
 *
 * This is a scalar port of the numpy 2D grid approach. It iterates
 * over all (n_gpu, batch_size) combinations and builds the frontier.
 */
export function paretoFronts(
  comparisonList: TokenEconSettings[],
  tokenLatencySecondsFunc: TokenLatencyFn,
  usePp: boolean = false,
): ParetoFrontResult[] {
  const results: ParetoFrontResult[] = [];

  for (const cs of comparisonList) {
    const gpu = cs.gpu;
    const model = cs.model;
    const inputLen = cs.inputLen;
    const maxThroughput = cs.maxThroughputTokensPerSecond;

    const minNumOfGpus =
      (model.totalParams * model.weightPrecisionBytes) / gpu.hbmSizeBytes;

    const batchSizeRange = logspace(0, 18 + Math.log2(model.sparsityFactor), 400, 2);
    const nGpuRange = logspace(Math.log2(minNumOfGpus), 18, 400, 2);

    // Compute latency for every (nGpu, batchSize) pair
    interface GridPoint {
      nGpu: number;
      batchSize: number;
      latency: number;
      gpuSecondsPerToken: number;
    }

    const grid: GridPoint[] = [];

    for (const nGpuVal of nGpuRange) {
      for (const bsVal of batchSizeRange) {
        let latency: number;
        if (cs.specDec && cs.approxModel) {
          latency = specDecTokenLatencySeconds(
            nGpuVal,
            model,
            cs.approxModel,
            gpu,
            bsVal,
            cs.acceptanceProb,
            5,
            inputLen,
            usePp,
          );
        } else {
          latency = tokenLatencySecondsFunc(nGpuVal, model, gpu, bsVal, inputLen, 1, usePp);
        }

        if (!isFinite(latency)) continue;

        const gpuSecPerTok = (nGpuVal * latency) / bsVal;
        grid.push({ nGpu: nGpuVal, batchSize: bsVal, latency, gpuSecondsPerToken: gpuSecPerTok });
      }
    }

    if (grid.length === 0) {
      results.push({ xCoords: [], yCoords: [], gpuCounts: [], batchSizes: [], mfuValues: [] });
      continue;
    }

    // Sort by latency to iterate latency thresholds
    grid.sort((a, b) => a.latency - b.latency);

    const minLatency = grid[0].latency;
    const latencyRange = logspace(0, 2, 1000, 10).map((v) => v * minLatency);

    const xCoords: number[] = [];
    const yCoords: number[] = [];
    const gpuCounts: number[] = [];
    const batchSizes: number[] = [];
    const mfuValues: number[] = [];

    const seenLatencies = new Set<number>();

    for (const latencyThreshold of latencyRange) {
      // Find points within threshold AND satisfying throughput cap
      let bestCost = Infinity;
      let bestPoint: GridPoint | null = null;

      for (const pt of grid) {
        if (pt.latency > latencyThreshold) continue;
        if (pt.batchSize / pt.latency > maxThroughput) continue;

        if (pt.gpuSecondsPerToken < bestCost) {
          bestCost = pt.gpuSecondsPerToken;
          bestPoint = pt;
        }
      }

      if (bestPoint === null) continue;

      // Deduplicate by latency value
      if (seenLatencies.has(bestPoint.latency)) continue;
      seenLatencies.add(bestPoint.latency);

      const gpuPriceDollarsPerSecond = gpu.priceDollarsPerHour / 3600;

      xCoords.push(1 / bestPoint.latency);
      yCoords.push(1e6 * bestPoint.gpuSecondsPerToken * gpuPriceDollarsPerSecond);
      gpuCounts.push(bestPoint.nGpu);
      batchSizes.push(bestPoint.batchSize);

      const mfu =
        model.arithmeticCostFlop(inputLen, bestPoint.batchSize) /
        (bestPoint.nGpu *
          gpu.theoreticalFlopPerSecond[model.weightPrecisionBytes * 8] *
          bestPoint.latency);
      mfuValues.push(mfu);
    }

    results.push({ xCoords, yCoords, gpuCounts, batchSizes, mfuValues });
  }

  return results;
}

// ---------------------------------------------------------------------------
// User preference intensity
// ---------------------------------------------------------------------------

export function userPreferenceIntensity(
  tokensPerSecondPerRequest: number,
  pricePerMillionTokens: number,
): number {
  return tokensPerSecondPerRequest ** 3 / pricePerMillionTokens;
}

// ---------------------------------------------------------------------------
// preference_maximizing_settings
// ---------------------------------------------------------------------------

export interface PreferenceResult {
  tokensPerSecondPerRequest: number;
  priceDollarsPerMillionTokens: number;
  gpusPerInstance: number;
  batchSize: number;
  utilizationRate: number;
}

export function preferenceMaximizingSettings(
  comparison: TokenEconSettings[],
  tokenEconomicsResults: ParetoFrontResult[],
): Record<string, PreferenceResult> {
  const results: Record<string, PreferenceResult> = {};

  for (let i = 0; i < comparison.length; i++) {
    const cs = comparison[i];
    const { xCoords, yCoords, gpuCounts, batchSizes, mfuValues } = tokenEconomicsResults[i];

    if (xCoords.length === 0) continue;

    let maxIntensity = -Infinity;
    let maxIdx = 0;

    for (let j = 0; j < xCoords.length; j++) {
      const intensity = userPreferenceIntensity(xCoords[j], yCoords[j]);
      if (intensity > maxIntensity) {
        maxIntensity = intensity;
        maxIdx = j;
      }
    }

    results[cs.name] = {
      tokensPerSecondPerRequest: xCoords[maxIdx],
      priceDollarsPerMillionTokens: yCoords[maxIdx],
      gpusPerInstance: gpuCounts[maxIdx],
      batchSize: batchSizes[maxIdx],
      utilizationRate: mfuValues[maxIdx],
    };
  }

  return results;
}
