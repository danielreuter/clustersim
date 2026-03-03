// ---------------------------------------------------------------------------
// Token latency & throughput functions – ported from erdil.py (Erdil 2025)
// ---------------------------------------------------------------------------

import { GPU } from "./gpus";
import { Model, matmulRwBytes } from "./models";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Compute integer divisors of n. */
function divisors(n: number): number[] {
  const result: number[] = [];
  const absN = Math.abs(Math.round(n));
  if (absN === 0) return result;
  for (let i = 1; i * i <= absN; i++) {
    if (absN % i === 0) {
      result.push(i);
      if (i !== absN / i) result.push(absN / i);
    }
  }
  return result.sort((a, b) => a - b);
}

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
// Feed-forward and attention read/write bytes (extracted helpers)
// ---------------------------------------------------------------------------

/**
 * Only the *feed-forward* rw-bytes part of Model.memoryReadsWritesBytes.
 */
export function ffRwBytes(model: Model, batchTokens: number, tp: number): number {
  const usedExpertsFraction = 1 - (1 - 1 / model.sparsityFactor) ** batchTokens;
  const wp = model.weightPrecisionBytes;
  const ap = model.activationPrecisionBytes;
  return (
    usedExpertsFraction *
    model.nExperts *
    (model.ffMatrixCount[0] + model.ffMatrixCount[1]) *
    matmulRwBytes(model.dModel, model.dFf, batchTokens / model.sparsityFactor, wp, ap, tp)
  );
}

/**
 * Only the *attention* rw-bytes part of Model.memoryReadsWritesBytes.
 */
export function attnRwBytes(model: Model, batchTokens: number, tp: number): number {
  const wp = model.weightPrecisionBytes;
  const ap = model.activationPrecisionBytes;

  if (model.mla) {
    const qkvC = matmulRwBytes(
      model.mlaKvCompressionDim! + model.mlaQueryCompressionDim!,
      model.dModel,
      batchTokens,
      wp,
      ap,
      tp,
    );
    const qkv = matmulRwBytes(
      model.dAllAttnHeads,
      model.mlaKvCompressionDim! + model.mlaQueryCompressionDim!,
      batchTokens,
      wp,
      ap,
      tp,
    );
    const proj = matmulRwBytes(model.dModel, model.dHead * model.numQueryHeads, batchTokens, wp, ap, tp);
    return qkvC + qkv + proj;
  }

  const qkv = matmulRwBytes(model.dAllAttnHeads, model.dModel, batchTokens, wp, ap, tp);
  const proj = matmulRwBytes(model.dModel, model.dHead * model.numQueryHeads, batchTokens, wp, ap, tp);
  return qkv + proj;
}

// ---------------------------------------------------------------------------
// TP layout helper
// ---------------------------------------------------------------------------

/**
 * Return (nRanks, nParallelReduces, nNodes) for the given TP degree.
 */
export function tpLayout(
  tpDegree: number,
  twoD: boolean,
  gpuNodeSize: number,
): [number, number, number] {
  if (twoD) {
    const sqrtTp = Math.sqrt(tpDegree);
    const nRanks = sqrtTp;
    const nParRed = tpDegree / nRanks;
    const nNodes = Math.ceil(tpDegree / gpuNodeSize) ** (1 / 2);
    return [nRanks, nParRed, nNodes];
  }
  return [tpDegree, 1, Math.ceil(tpDegree / gpuNodeSize)];
}

// ---------------------------------------------------------------------------
// token_latency_seconds_as_presented_in_paper  (Section 3.5)
// ---------------------------------------------------------------------------

/**
 * Token latency exactly as presented in Section 3.5 of the paper.
 * Scalar version – single (nGpu, batchSize) pair.
 */
export function tokenLatencySecondsAsPresentedInPaper(
  nGpu: number,
  model: Model,
  gpu: GPU,
  batchSize: number,
  inputLen: number = 0,
  seqLen: number = 1,
  _usePp: boolean = false,
): number {
  const serialMatmulsPerLayer = 4;
  const nRanks = nGpu ** (1 / 2);
  const nNodes = Math.ceil(nGpu / gpu.nodeSize) ** (1 / 2);
  const nParallelReduces = nGpu / nRanks;

  const wordsReduced = [
    model.dHead * (model.numKvHeads + model.numQueryHeads) / nParallelReduces,
    model.dModel / nParallelReduces,
    (model.nActiveExperts * model.ffMatrixCount[0] * model.dFf) / nParallelReduces,
    (model.nActiveExperts * model.ffMatrixCount[1] * model.dModel) / nParallelReduces,
  ];

  const arithmeticCostFlop = model.arithmeticCostFlop(inputLen, batchSize * seqLen);
  const memoryRwBytes_ = model.memoryReadsWritesBytes(inputLen, batchSize, seqLen, nGpu);

  let networkCommTimeSec = 0;

  for (const wr of wordsReduced) {
    const bytesReduced = wr * batchSize * seqLen * model.activationPrecisionBytes;

    networkCommTimeSec +=
      model.layers *
      gpu.collectiveTimeSeconds(nRanks, nNodes, bytesReduced, gpu, "allreduce", [1, 1]);
  }

  let result =
    model.layers * serialMatmulsPerLayer * gpu.kernelLaunchLatencySeconds +
    networkCommTimeSec +
    Math.max(
      memoryRwBytes_ / (nGpu * gpu.hbmBandwidthBps),
      arithmeticCostFlop / (nGpu * gpu.flopPerSecond[8 * model.weightPrecisionBytes]),
    );

  const kvCacheSizeBytes = model.kvCacheSizePerInputBytes * inputLen * batchSize;
  if (nGpu * gpu.hbmSizeBytes < kvCacheSizeBytes + model.weightPrecisionBytes * model.totalParams) {
    result = Infinity;
  }

  return result;
}

// ---------------------------------------------------------------------------
// final_token_latency_seconds  (2D TP + PP but no EP)
// ---------------------------------------------------------------------------

/**
 * Token latency with 2D TP on/off and pipeline parallelism.
 * Does not support expert parallelism.
 */
export function finalTokenLatencySeconds(
  nGpu: number,
  model: Model,
  gpu: GPU,
  batchSize: number,
  inputLen: number = 0,
  seqLen: number = 1,
  usePp: boolean = false,
): number {
  let best = Infinity;

  const ppDegreeList: number[] = [1];
  if (usePp) {
    for (const v of logspace(0, Math.log2(model.layers), 10, 2)) {
      if (!ppDegreeList.includes(v)) ppDegreeList.push(v);
    }
  }

  for (const twoD of [true, false]) {
    for (const nPP of ppDegreeList) {
      const nTP = nGpu / nPP;
      const numOfMicrobatches = nPP;
      const microbatchSize = batchSize / numOfMicrobatches;

      const serialMatmulsPerLayer = 4;

      let nRanks: number;
      let nNodes: number;
      let wordsReduced: number[];

      if (twoD) {
        nRanks = nTP ** (1 / 2);
        const nParallelReduces = nTP / nRanks;
        nNodes = Math.ceil(nTP / gpu.nodeSize) ** (1 / 2);
        wordsReduced = [
          model.dHead * (model.numKvHeads + model.numQueryHeads) / nParallelReduces,
          model.dModel / nParallelReduces,
          (model.nActiveExperts * model.ffMatrixCount[0] * model.dFf) / nParallelReduces,
          (model.nActiveExperts * model.ffMatrixCount[1] * model.dModel) / nParallelReduces,
        ];
      } else {
        nRanks = nTP;
        nNodes = Math.ceil(nTP / gpu.nodeSize);
        wordsReduced = [
          model.dModel,
          model.nActiveExperts * model.ffMatrixCount[1] * model.dModel,
        ];
      }

      const arithmeticCostFlop = model.arithmeticCostFlop(inputLen, batchSize * seqLen);
      const memoryRwBytes_ =
        numOfMicrobatches * model.memoryReadsWritesBytes(inputLen, microbatchSize, seqLen, nTP);

      let networkCommTimeSec = 0;

      for (const wr of wordsReduced) {
        const bytesReduced =
          (numOfMicrobatches / nPP) * wr * microbatchSize * seqLen * model.activationPrecisionBytes;

        networkCommTimeSec +=
          model.layers *
          gpu.collectiveTimeSeconds(nRanks, nNodes, bytesReduced, gpu, "allreduce", [1, 0]);
        networkCommTimeSec +=
          model.layers *
          gpu.collectiveTimeSeconds(nRanks, nNodes, bytesReduced, gpu, "allreduce", [0, 1]);
      }

      const ppWordsRead = (model.dModel * microbatchSize * seqLen) / nTP;
      const ppBytesRead = ppWordsRead * model.activationPrecisionBytes;

      networkCommTimeSec +=
        (nPP - 1) * gpu.collectiveTimeSeconds(2, 2, ppBytesRead, gpu, "p2p", [1, 0]);
      networkCommTimeSec +=
        (nPP - 1) *
        (numOfMicrobatches / nPP) *
        gpu.collectiveTimeSeconds(2, 2, ppBytesRead, gpu, "p2p", [0, 1]);

      let currResult =
        model.layers * serialMatmulsPerLayer * gpu.kernelLaunchLatencySeconds +
        networkCommTimeSec +
        Math.max(
          memoryRwBytes_ / (nGpu * gpu.hbmBandwidthBps),
          arithmeticCostFlop / (nGpu * gpu.flopPerSecond[8 * model.weightPrecisionBytes]),
        );

      const kvCacheSizeBytes = model.kvCacheSizePerInputBytes * inputLen * batchSize;
      if (nGpu * gpu.hbmSizeBytes < kvCacheSizeBytes + model.weightPrecisionBytes * model.totalParams) {
        currResult = Infinity;
      }
      if (nPP > Math.min(nGpu, batchSize)) {
        currResult = Infinity;
      }

      best = Math.min(best, currResult);
    }
  }

  return best;
}

// ---------------------------------------------------------------------------
// new_token_latency_seconds  (v2: EP + distinct TP for attn vs FF)
// ---------------------------------------------------------------------------

/**
 * Token latency with Expert-Parallelism + distinct TP degrees for
 * attention vs feed-forward.
 *
 * This is the default latency function (`token_latency_seconds_default`).
 */
export function newTokenLatencySeconds(
  nGpu: number,
  model: Model,
  gpu: GPU,
  batchSize: number,
  inputLen: number = 0,
  seqLen: number = 1,
  usePp: boolean = false,
  allowAttnScaledown: boolean = true,
): number {
  let best = Infinity;

  const apBytes = model.activationPrecisionBytes;

  // PP candidates
  const ppList: number[] = [1];
  if (usePp) {
    const maxPp = Math.min(batchSize, model.layers);
    for (const v of logspace(0, Math.log2(Math.max(1, maxPp)), 10, 2)) {
      if (!ppList.includes(v)) ppList.push(v);
    }
  }

  for (const twoD of [true, false]) {
    for (const nPP of ppList) {
      const microBs = batchSize / nPP;
      const nTPTotal = nGpu / nPP;

      // Expert parallelism
      let nEP = Math.min(nTPTotal, model.nExperts);
      if (microBs < 2 * model.sparsityFactor) {
        nEP = 1;
      }

      const nTPFf = Math.max(1, nTPTotal / Math.max(1, nEP));

      // Attention scale-down candidates
      const attnScaleFactors: number[] = allowAttnScaledown
        ? logspace(0, Math.log2(Math.max(1, nTPTotal)), 6, 2)
        : [1];

      for (const attnScaledownFactor of attnScaleFactors) {
        const nTPAttn = nTPTotal / attnScaledownFactor;

        // TP layouts
        const [nRanksAttn, nParRedAttn, nNodesAttn] = tpLayout(nTPAttn, twoD, gpu.nodeSize);
        const [nRanksFf, _nParRedFf, nNodesFf] = tpLayout(nTPFf, twoD, gpu.nodeSize);

        // Words reduced – attention
        let wordsAttn: number[];
        if (twoD) {
          wordsAttn = [
            model.dModel / nParRedAttn,
            (model.dHead * (model.numKvHeads + model.numQueryHeads)) / nParRedAttn,
          ];
        } else {
          wordsAttn = [model.dModel];
        }

        // Words reduced – feed-forward
        const wordsFf: number[] = [];
        if (twoD) {
          wordsFf.push(
            (model.nActiveExperts * model.ffMatrixCount[0] * model.dFf) / (nTPTotal / nRanksFf),
            (model.nActiveExperts * model.ffMatrixCount[1] * model.dModel) / (nTPTotal / nRanksFf),
          );
        } else {
          wordsFf.push(
            (model.nActiveExperts * model.ffMatrixCount[1] * model.dModel) / (nTPTotal / nRanksFf),
          );
        }

        // Collective helper
        const collectiveSum = (
          words: number[],
          nRanks: number,
          nNodes: number,
        ): number => {
          let total = 0;
          for (const w of words) {
            const bytesR = w * microBs * seqLen * apBytes;
            total += model.layers * gpu.collectiveTimeSeconds(nRanks, nNodes, bytesR, gpu, "allreduce");
          }
          return total;
        };

        let net =
          collectiveSum(wordsAttn, nRanksAttn, nNodesAttn) +
          collectiveSum(wordsFf, nRanksFf, nNodesFf);

        // EP communication
        if (model.nExperts > 1) {
          const partEp = Math.min(model.nActiveExperts, nEP);
          const bytesPerRank =
            (model.dModel * microBs * partEp * seqLen * apBytes) / nTPTotal;
          const nodesEp = Math.ceil(partEp / gpu.nodeSize);
          net +=
            model.layers *
            2 *
            gpu.collectiveTimeSeconds(partEp, nodesEp, bytesPerRank, gpu, "all-to-all");
        }

        // PP p2p latency
        const ppWords = (model.dModel * microBs * seqLen) / nTPTotal;
        const ppBytes = ppWords * model.activationPrecisionBytes;
        net += (nPP - 1) * gpu.collectiveTimeSeconds(2, 2, ppBytes, gpu, "p2p");

        // Memory & compute
        const batchTok = microBs * seqLen;
        const ffRw = ffRwBytes(model, batchTok, nTPFf);
        const attnRw = attnRwBytes(model, batchTok, nTPAttn);
        const kvCache = model.kvCacheSizePerInputBytes * inputLen * microBs;
        const unembed = model.weightPrecisionBytes * model.vocabSize * model.dModel;

        const memRwTimeSeconds =
          nPP *
          (kvCache / (nGpu * gpu.hbmBandwidthBps) +
            model.layers *
              (ffRw / (nGpu * gpu.hbmBandwidthBps) +
                attnRw / (nGpu * gpu.hbmBandwidthBps / attnScaledownFactor)) +
            unembed / (nGpu * gpu.hbmBandwidthBps));

        const flopTimeSeconds =
          (model.ffdFlop(batchSize * seqLen) + model.attnKvFlop(inputLen, batchSize * seqLen)) /
            (nGpu * gpu.flopPerSecond[8 * model.weightPrecisionBytes]) +
          model.attnProjFlop(inputLen, batchSize * seqLen) /
            (nGpu * gpu.flopPerSecond[8 * model.weightPrecisionBytes] / attnScaledownFactor);

        const serialKernels = model.layers * 4 * gpu.kernelLaunchLatencySeconds;

        let candidate = serialKernels + net + Math.max(memRwTimeSeconds, flopTimeSeconds);

        // Capacity checks
        if (
          nGpu * gpu.hbmSizeBytes <
          kvCache + model.weightPrecisionBytes * model.totalParams
        ) {
          candidate = Infinity;
        }
        if (nPP > Math.min(nGpu, batchSize)) {
          candidate = Infinity;
        }

        best = Math.min(best, candidate);
      }
    }
  }

  return best;
}

// The default token latency function
export const tokenLatencySecondsDefault = newTokenLatencySeconds;

// ---------------------------------------------------------------------------
// Token latency with gamma-overlap breakdown (for UI)
// ---------------------------------------------------------------------------

export interface LatencyBreakdown {
  pp: number;
  tp: number;
  twoDTp: boolean;
  batch: number;
  microbatches: number;
  microbatchSize: number;
  tComputeS: number;
  tMemS: number;
  tCommLatS: number;
  tCommBwS: number;
  tTotalS: number;
  gamma: number;
  error?: string;
}

/**
 * Returns (latencySeconds, infoDict) for the best PP/TP layout,
 * with comm split into latency vs bandwidth and combined via overlap gamma.
 */
export function tokenLatencyWithGammaBreakdown(
  nGpu: number,
  model: Model,
  gpu: GPU,
  batchSize: number,
  inputLen: number = 0,
  seqLen: number = 1,
  usePp: boolean = true,
  gamma: number = 0.5,
): [number, LatencyBreakdown] {
  // Candidate PP degrees
  const ppList: number[] = [1];
  if (usePp) {
    const maxPp = Math.max(1, Math.min(model.layers, nGpu, batchSize));
    const raw = logspace(0, Math.log2(maxPp), 10, 2);
    for (const x of raw) {
      const rounded = Math.max(1, Math.round(x));
      if (!ppList.includes(rounded)) ppList.push(rounded);
    }
    ppList.sort((a, b) => a - b);
  }

  let bestTotal = Infinity;
  let bestInfo: LatencyBreakdown | null = null;

  const apBytes = model.activationPrecisionBytes;

  for (const twoD of [true, false]) {
    for (const nPP of ppList) {
      if (nPP < 1 || nPP > Math.min(nGpu, batchSize)) continue;
      if (nGpu % nPP !== 0) continue;

      const nTP = Math.floor(nGpu / nPP);
      if (nTP < 1) continue;

      const microCt = nPP;
      const microBs = batchSize / microCt;
      if (microBs <= 0) continue;

      // TP layout
      let nRanks: number;
      let nNodes: number;
      let wordsReduced: number[];

      if (twoD) {
        nRanks = Math.sqrt(nTP);
        const nParallelReduces = nTP / nRanks;
        nNodes = Math.sqrt(Math.ceil(nTP / gpu.nodeSize));
        wordsReduced = [
          (model.dHead * (model.numKvHeads + model.numQueryHeads)) / nParallelReduces,
          model.dModel / nParallelReduces,
          (model.nActiveExperts * model.ffMatrixCount[0] * model.dFf) / nParallelReduces,
          (model.nActiveExperts * model.ffMatrixCount[1] * model.dModel) / nParallelReduces,
        ];
      } else {
        nRanks = nTP;
        nNodes = Math.ceil(nTP / gpu.nodeSize);
        wordsReduced = [
          model.dModel,
          model.nActiveExperts * model.ffMatrixCount[1] * model.dModel,
        ];
      }

      // Compute time
      const flop = model.arithmeticCostFlop(inputLen, batchSize * seqLen, seqLen);
      const tCompute = flop / (nGpu * (gpu.flopPerSecond[8 * model.weightPrecisionBytes] ?? 1e15));

      // Memory time
      const memRwBytes_ = microCt * model.memoryReadsWritesBytes(inputLen, microBs, seqLen, nTP);
      const tMem = memRwBytes_ / (nGpu * gpu.hbmBandwidthBps);

      // Kernel launch overhead
      const serialKernels = model.layers * 4 * gpu.kernelLaunchLatencySeconds;
      const tComputeEff = tCompute + serialKernels;

      // Comm split: latency vs bandwidth
      let tCommLat = 0;
      let tCommBw = 0;

      for (const w of wordsReduced) {
        const bytesReduced = w * microBs * seqLen * apBytes;

        tCommLat +=
          model.layers *
          gpu.collectiveTimeSeconds(nRanks, nNodes, bytesReduced, gpu, "allreduce", [1, 0]);
        tCommBw +=
          model.layers *
          gpu.collectiveTimeSeconds(nRanks, nNodes, bytesReduced, gpu, "allreduce", [0, 1]);
      }

      // PP p2p transfers
      if (nPP > 1) {
        const ppWords = (model.dModel * microBs * seqLen) / nTP;
        const ppBytes = ppWords * apBytes;
        const hops = nPP - 1;

        tCommLat += hops * gpu.collectiveTimeSeconds(2, 2, ppBytes, gpu, "p2p", [1, 0]);
        tCommBw += hops * gpu.collectiveTimeSeconds(2, 2, ppBytes, gpu, "p2p", [0, 1]);
      }

      // Overlap model
      const tComputeMem = Math.max(tComputeEff, tMem);
      const tNoOverlap = tComputeMem + tCommLat + tCommBw;
      const tFullOverlap = Math.max(tComputeMem, tCommBw) + tCommLat;
      const tTotal = (1 - gamma) * tNoOverlap + gamma * tFullOverlap;

      // Capacity check
      const kvCacheBytes = model.kvCacheSizePerInputBytes * inputLen * batchSize;
      const weightsBytes = model.weightPrecisionBytes * model.totalParams;
      if (nGpu * gpu.hbmSizeBytes < kvCacheBytes + weightsBytes) {
        continue;
      }

      if (tTotal < bestTotal) {
        bestTotal = tTotal;
        bestInfo = {
          pp: nPP,
          tp: nTP,
          twoDTp: twoD,
          batch: batchSize,
          microbatches: microCt,
          microbatchSize: microBs,
          tComputeS: tComputeEff,
          tMemS: tMem,
          tCommLatS: tCommLat,
          tCommBwS: tCommBw,
          tTotalS: tTotal,
          gamma,
        };
      }
    }
  }

  if (bestInfo === null) {
    return [Infinity, { error: "No feasible config" } as LatencyBreakdown];
  }

  return [bestTotal, bestInfo];
}

// ---------------------------------------------------------------------------
// scaled_gpu helper
// ---------------------------------------------------------------------------

export function scaledGpu(
  baseGpu: GPU,
  hbmFraction: number = 1.0,
  computeFraction: number = 1.0,
): GPU {
  // Deep-clone via reconstruction
  const g = new GPU({
    name: baseGpu.name,
    flopPerSecond: { ...baseGpu.theoreticalFlopPerSecond },
    hbmBandwidthBps: baseGpu.theoreticalHbmBandwidthBps,
    hbmSizeBytes: baseGpu.hbmSizeBytes,
    l2CacheSizeBytes: baseGpu.l2CacheSizeBytes,
    l2BandwidthBps: baseGpu.l2BandwidthBps,
    intranodeAllreduceBandwidthBps: baseGpu.intranodeAllreduceBandwidthBps,
    internodeAllreduceBandwidthBps: baseGpu.internodeAllreduceBandwidthBps,
    nodeSize: baseGpu.nodeSize,
    priceDollarsPerHour: baseGpu.priceDollarsPerHour,
    kernelLaunchLatencySeconds: baseGpu.kernelLaunchLatencySeconds,
    collectiveTimeSeconds: baseGpu.collectiveTimeSeconds,
    arithmeticUtilizationCap: baseGpu.arithmeticUtilizationCap,
    memoryBwdUtilizationCap: baseGpu.memoryBwdUtilizationCap,
  });

  g.hbmSizeBytes *= hbmFraction;

  if (computeFraction !== 1.0) {
    g.hbmBandwidthBps *= computeFraction;
    for (const k of Object.keys(g.flopPerSecond)) {
      g.flopPerSecond[Number(k)] *= computeFraction;
    }
    g.intranodeAllreduceBandwidthBps *= computeFraction;
    g.internodeAllreduceBandwidthBps *= computeFraction;
  }

  return g;
}

// ---------------------------------------------------------------------------
// maximize_cluster_throughput
// ---------------------------------------------------------------------------

export interface ClusterThroughputResult {
  clusterGpus: number;
  gpusPerInstance: number;
  replicas: number;
  throughputTokS: number;
  bottleneck: string;
  computeCeilingTokS: number;
  efficiency: number;
  hbmFraction: number;
  gammaOverlap: number;
  pp: number;
  tp: number;
  twoDTp: boolean;
  batch: number;
  microbatches: number;
  microbatchSize: number;
  tComputeS: number;
  tMemS: number;
  tCommLatS: number;
  tCommBwS: number;
  tTotalS: number;
  gamma: number;
  error?: string;
}

/**
 * Maximize cluster throughput for a fixed cluster size.
 *
 * Searches over instance sizes (GPUs per replica) and batch sizes.
 */
export function maximizeClusterThroughput(
  clusterGpus: number,
  model: Model,
  gpu: GPU,
  inputLen: number,
  hbmFraction: number = 1.0,
  gammaOverlap: number = 0.5,
  computeFraction: number = 1.0,
  usePp: boolean = true,
  seqLen: number = 1,
  maxInstanceGpus: number = 4096,
): [number, ClusterThroughputResult] {
  const g = scaledGpu(gpu, hbmFraction, computeFraction);

  // Candidate instance sizes (powers of 2)
  const maxInst = Math.min(clusterGpus, maxInstanceGpus);
  const instSizes: number[] = [1];
  for (let k = 0; k <= Math.floor(Math.log2(Math.max(1, maxInst))); k++) {
    const s = 2 ** k;
    if (!instSizes.includes(s)) instSizes.push(s);
  }

  let bestTp = 0;
  let best: ClusterThroughputResult | null = null;

  for (const nInst of instSizes) {
    if (nInst > clusterGpus) continue;
    const replicas = Math.floor(clusterGpus / nInst);
    if (replicas < 1) continue;

    // Batch search cap based on HBM capacity
    const weightsBytes = model.weightPrecisionBytes * model.totalParams;
    const freeBytes = nInst * g.hbmSizeBytes - weightsBytes;
    if (freeBytes <= 0) continue;

    let batchMax: number;
    if (inputLen > 0) {
      const kvPerBatch = model.kvCacheSizePerInputBytes * inputLen;
      batchMax = Math.max(1, Math.floor(freeBytes / kvPerBatch));
    } else {
      batchMax = 2 ** 16;
    }

    // Batch candidates: powers of 2 up to cap
    const maxPow = Math.floor(Math.log2(Math.max(1, Math.min(batchMax, 2 ** 18))));
    const batchCandidates: number[] = [];
    for (let k = 0; k <= maxPow; k++) {
      batchCandidates.push(2 ** k);
    }

    for (const batch of batchCandidates) {
      const [latS, info] = tokenLatencyWithGammaBreakdown(
        nInst,
        model,
        g,
        batch,
        inputLen,
        seqLen,
        usePp,
        gammaOverlap,
      );
      if (!isFinite(latS) || info.error) continue;

      const clusterTp = (replicas * batch * seqLen) / latS;

      if (clusterTp > bestTp) {
        // Bottleneck classification
        const tComputeMem = Math.max(info.tComputeS, info.tMemS);
        let bottleneck: string;
        if (info.tCommLatS > Math.max(tComputeMem, info.tCommBwS)) {
          bottleneck = "Comm Latency";
        } else if (info.tCommBwS > tComputeMem) {
          bottleneck = "Comm BW";
        } else if (info.tComputeS >= info.tMemS) {
          bottleneck = "Compute";
        } else {
          bottleneck = "Memory";
        }

        // Compute ceiling + efficiency
        const flopsTotal = model.arithmeticCostFlop(inputLen, batch, seqLen);
        const flopsPerToken = batch > 0 ? flopsTotal / batch : flopsTotal;
        const clusterFlops =
          clusterGpus * (g.flopPerSecond[8 * model.weightPrecisionBytes] ?? 1e15);
        const computeCeiling = flopsPerToken > 0 ? clusterFlops / flopsPerToken : 0;
        const efficiency = computeCeiling > 0 ? clusterTp / computeCeiling : 0;

        bestTp = clusterTp;
        best = {
          ...info,
          clusterGpus,
          gpusPerInstance: nInst,
          replicas,
          throughputTokS: clusterTp,
          bottleneck,
          computeCeilingTokS: computeCeiling,
          efficiency,
          hbmFraction,
          gammaOverlap,
        };
      }
    }
  }

  if (best === null) {
    return [0, { error: "No valid configuration found" } as unknown as ClusterThroughputResult];
  }

  return [bestTp, best];
}
