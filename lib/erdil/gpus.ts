// ---------------------------------------------------------------------------
// GPU definitions – ported from erdil.py (Erdil 2025)
// ---------------------------------------------------------------------------

/**
 * Signature for the collective‑time callback stored on each GPU.
 *
 * @param nRanks    Number of ranks participating in the collective
 * @param nNodes    Number of nodes spanned
 * @param bytes     Payload in bytes
 * @param gpu       The GPU object (for bandwidth look‑ups)
 * @param coll      Collective type: "allreduce" | "all-to-all" | "p2p"
 * @param latencyBwWeights  Tuple [latWeight, bwWeight] – allows callers to
 *                          request only the latency or bandwidth component
 */
export type CollectiveTimeFn = (
  nRanks: number,
  nNodes: number,
  bytes: number,
  gpu: GPU,
  coll: string,
  latencyBwWeights?: [number, number],
) => number;

// ---------------------------------------------------------------------------
// GPU class
// ---------------------------------------------------------------------------

export class GPU {
  name: string;
  theoreticalFlopPerSecond: Record<number, number>;
  theoreticalHbmBandwidthBps: number;
  flopPerSecond: Record<number, number>;
  hbmBandwidthBps: number;
  hbmSizeBytes: number;
  l2BandwidthBps: number;
  l2CacheSizeBytes: number;
  intranodeAllreduceBandwidthBps: number;
  internodeAllreduceBandwidthBps: number;
  nodeSize: number;
  priceDollarsPerHour: number;
  kernelLaunchLatencySeconds: number;
  collectiveTimeSeconds: CollectiveTimeFn;
  arithmeticUtilizationCap: number;
  memoryBwdUtilizationCap: number;

  constructor(opts: {
    name: string;
    flopPerSecond: Record<number, number>;
    hbmBandwidthBps: number;
    hbmSizeBytes: number;
    l2CacheSizeBytes: number;
    l2BandwidthBps: number;
    intranodeAllreduceBandwidthBps: number;
    internodeAllreduceBandwidthBps: number;
    nodeSize: number;
    priceDollarsPerHour: number;
    kernelLaunchLatencySeconds: number;
    collectiveTimeSeconds: CollectiveTimeFn;
    arithmeticUtilizationCap?: number;
    memoryBwdUtilizationCap?: number;
  }) {
    this.name = opts.name;
    this.theoreticalFlopPerSecond = { ...opts.flopPerSecond };
    this.theoreticalHbmBandwidthBps = opts.hbmBandwidthBps;

    const auCap = opts.arithmeticUtilizationCap ?? 1;
    const muCap = opts.memoryBwdUtilizationCap ?? 1;

    this.flopPerSecond = {} as Record<number, number>;
    for (const p of Object.keys(opts.flopPerSecond)) {
      this.flopPerSecond[Number(p)] = opts.flopPerSecond[Number(p)] * auCap;
    }

    this.hbmBandwidthBps = opts.hbmBandwidthBps * muCap;
    this.hbmSizeBytes = opts.hbmSizeBytes;
    this.l2BandwidthBps = opts.l2BandwidthBps;
    this.l2CacheSizeBytes = opts.l2CacheSizeBytes;
    this.intranodeAllreduceBandwidthBps = opts.intranodeAllreduceBandwidthBps;
    this.internodeAllreduceBandwidthBps = opts.internodeAllreduceBandwidthBps;
    this.nodeSize = opts.nodeSize;
    this.priceDollarsPerHour = opts.priceDollarsPerHour;
    this.kernelLaunchLatencySeconds = opts.kernelLaunchLatencySeconds;
    this.collectiveTimeSeconds = opts.collectiveTimeSeconds;
    this.arithmeticUtilizationCap = auCap;
    this.memoryBwdUtilizationCap = muCap;
  }
}

// ---------------------------------------------------------------------------
// Collective latency / time helpers (NCCL model)
// ---------------------------------------------------------------------------

export function collectiveLatencyNcclSeconds(
  nRanks: number,
  nNodes: number,
  coll: string = "allreduce",
  algo: string = "LL",
  llBaseLatencySeconds: number = 6.8e-6,
  ll128BaseLatencySeconds: number = 14e-6,
): number {
  const m = coll === "allreduce" ? 2 : 1;

  if (algo === "LL") {
    return m * ((nRanks / nNodes - 1) * 0.6e-6 + 5e-6 * Math.log2(nNodes)) + llBaseLatencySeconds;
  } else if (algo === "LL128") {
    return m * ((nRanks / nNodes - 1) * 1.25e-6 + 8.5e-6 * Math.log2(nNodes)) + ll128BaseLatencySeconds;
  } else if (algo === "Simple") {
    return m * ((nRanks / nNodes - 1) * 28e-6 + 28e-6 * Math.log2(nNodes));
  }
  throw new Error("Algorithm code given to collectiveLatencyNcclSeconds is invalid.");
}

/**
 * Mean collective time using the NCCL model – scalar version.
 *
 * Picks the best algorithm (Simple, LL128, LL) and returns the
 * weighted (latency, bandwidth) result.
 */
export function meanCollectiveTimeNcclSeconds(
  nRanks: number,
  nNodes: number,
  bytesReduced: number,
  gpu: GPU,
  coll: string = "allreduce",
  latencyBwWeights: [number, number] = [1, 1],
  llBaseLatencySeconds: number = 6.8e-6,
  ll128BaseLatencySeconds: number = 14e-6,
  overlapComms: boolean = true,
): number {
  if (nRanks <= 1) return 0;

  const [latWeight, bwWeight] = latencyBwWeights;

  const algorithmBandwidthFactors: Record<string, number> = {
    Simple: 1,
    LL128: 0.95,
    LL: 0.5,
  };

  let bestResult = Infinity;
  let bestWeighted = Infinity;

  for (const algo of Object.keys(algorithmBandwidthFactors)) {
    const bwFactor = algorithmBandwidthFactors[algo];

    const currLatencyTime = collectiveLatencyNcclSeconds(
      nRanks,
      nNodes,
      coll,
      algo,
      llBaseLatencySeconds,
      ll128BaseLatencySeconds,
    );

    let currBwTime: number;
    if (overlapComms) {
      currBwTime = Math.max(
        nNodes * Math.max(0, nRanks / nNodes - 1) * bytesReduced / (nRanks * gpu.intranodeAllreduceBandwidthBps * bwFactor),
        (nNodes - 1) * bytesReduced / (nRanks * gpu.internodeAllreduceBandwidthBps * bwFactor),
      );
    } else {
      currBwTime =
        nNodes * Math.max(0, nRanks / nNodes - 1) * bytesReduced / (nRanks * gpu.intranodeAllreduceBandwidthBps * bwFactor) +
        (nNodes - 1) * bytesReduced / (nRanks * gpu.internodeAllreduceBandwidthBps * bwFactor);
    }

    if (coll !== "allreduce") {
      currBwTime /= 2;
    }

    const currResult = currLatencyTime + currBwTime;

    if (currResult < bestResult) {
      bestResult = currResult;
      bestWeighted = latWeight * currLatencyTime + bwWeight * currBwTime;
    }
  }

  return bestWeighted;
}

/**
 * TPU collective time model – scalar version.
 */
export function tpuCollectiveTimeSeconds(
  nRanks: number,
  nNodes: number,
  bytesReduced: number,
  gpu: GPU,
  coll: string = "allreduce",
  latencyBwWeights: [number, number] = [1, 1],
): number {
  if (nRanks <= 1) return 0;

  const m = coll === "allreduce" ? 2 : 1;
  const [latWeight, bwWeight] = latencyBwWeights;

  const currLatencyTime = m * (nRanks - 1) * 1e-6;
  let currBwTime = (nRanks - 1) * bytesReduced / (nRanks * gpu.internodeAllreduceBandwidthBps);

  if (coll !== "allreduce") {
    currBwTime /= 2;
  }

  return latWeight * currLatencyTime + bwWeight * currBwTime;
}

// ---------------------------------------------------------------------------
// GPU definitions
// ---------------------------------------------------------------------------

export const H100 = new GPU({
  name: "H100",
  flopPerSecond: { 4: 2e15, 8: 2e15, 16: 1e15 },
  hbmBandwidthBps: 3.3e12,
  hbmSizeBytes: 8e10,
  l2CacheSizeBytes: 2.5e7,
  l2BandwidthBps: 1.2e13,
  intranodeAllreduceBandwidthBps: 9e11 / 4,
  internodeAllreduceBandwidthBps: 5e10 / 2,
  nodeSize: 8,
  priceDollarsPerHour: (3.15 * 2) / 3,
  kernelLaunchLatencySeconds: 4e-6,
  collectiveTimeSeconds: meanCollectiveTimeNcclSeconds,
  arithmeticUtilizationCap: 0.7,
  memoryBwdUtilizationCap: 0.75,
});

export const H800 = new GPU({
  name: "H800",
  flopPerSecond: { 4: 2e15, 8: 2e15, 16: 1e15 },
  hbmBandwidthBps: 3.3e12,
  hbmSizeBytes: 8e10,
  l2CacheSizeBytes: 2.5e7,
  l2BandwidthBps: 1.2e13,
  intranodeAllreduceBandwidthBps: 4e11 / 4,
  internodeAllreduceBandwidthBps: 5e10 / 2,
  nodeSize: 8,
  priceDollarsPerHour: (3.15 * 2) / 3,
  kernelLaunchLatencySeconds: 4e-6,
  collectiveTimeSeconds: meanCollectiveTimeNcclSeconds,
  arithmeticUtilizationCap: 0.7,
  memoryBwdUtilizationCap: 0.75,
});

export const H20 = new GPU({
  name: "H20",
  flopPerSecond: { 4: 2 * 148e12, 8: 2 * 148e12, 16: 148e12 },
  hbmBandwidthBps: 4e12,
  hbmSizeBytes: 96e9,
  l2CacheSizeBytes: 2.5e7,
  l2BandwidthBps: 1.2e13,
  intranodeAllreduceBandwidthBps: 9e11 / 4,
  internodeAllreduceBandwidthBps: 5e10 / 2,
  nodeSize: 8,
  priceDollarsPerHour: (13.5 / 24) * ((3.15 * 2) / 3),
  kernelLaunchLatencySeconds: 4e-6,
  collectiveTimeSeconds: meanCollectiveTimeNcclSeconds,
  arithmeticUtilizationCap: 1,
  memoryBwdUtilizationCap: 0.75,
});

export const H100_ZL = new GPU({
  name: "H100 ZL",
  flopPerSecond: { 4: 2e15, 8: 2e15, 16: 1e15 },
  hbmBandwidthBps: 3.3e12,
  hbmSizeBytes: 8e10,
  l2CacheSizeBytes: 2.5e7,
  l2BandwidthBps: 1.2e13,
  intranodeAllreduceBandwidthBps: 9e11 / 4,
  internodeAllreduceBandwidthBps: 5e10 / 2,
  nodeSize: 8,
  priceDollarsPerHour: (3.15 * 2) / 3,
  kernelLaunchLatencySeconds: 0,
  collectiveTimeSeconds: (nRanks, nNodes, bytes, gpu, coll, latencyBwWeights) =>
    meanCollectiveTimeNcclSeconds(
      nRanks,
      nNodes,
      bytes,
      gpu,
      coll,
      latencyBwWeights,
      0,   // llBaseLatencySeconds = 0
      0,   // ll128BaseLatencySeconds = 0
    ),
  arithmeticUtilizationCap: 1,
  memoryBwdUtilizationCap: 1,
});

export const H200 = new GPU({
  name: "H200",
  flopPerSecond: { 4: 2e15, 8: 2e15, 16: 1e15 },
  hbmBandwidthBps: 4.8e12,
  hbmSizeBytes: 141e9,
  l2CacheSizeBytes: 2.5e7,
  l2BandwidthBps: 1.2e13,
  intranodeAllreduceBandwidthBps: 9e11 / 4,
  internodeAllreduceBandwidthBps: 5e10 / 2,
  nodeSize: 8,
  priceDollarsPerHour: (3.5 / 2.5) * ((3.15 * 2) / 3),
  kernelLaunchLatencySeconds: 4e-6,
  collectiveTimeSeconds: meanCollectiveTimeNcclSeconds,
  arithmeticUtilizationCap: 0.6,
  memoryBwdUtilizationCap: 0.75,
});

export const A100 = new GPU({
  name: "A100",
  flopPerSecond: { 8: 6.24e14, 16: 3.12e14 },
  hbmBandwidthBps: 2e12,
  hbmSizeBytes: 8e10,
  l2CacheSizeBytes: 2.5e7,
  l2BandwidthBps: 1.2e13,
  intranodeAllreduceBandwidthBps: 6e11 / 4,
  internodeAllreduceBandwidthBps: 2.5e10 / 2,
  nodeSize: 8,
  priceDollarsPerHour: (2.26 * 2) / 3,
  kernelLaunchLatencySeconds: 4e-6,
  collectiveTimeSeconds: meanCollectiveTimeNcclSeconds,
  arithmeticUtilizationCap: 0.8,
  memoryBwdUtilizationCap: 0.75,
});

export const V100 = new GPU({
  name: "V100",
  flopPerSecond: { 8: 1e14, 16: 1e14 },
  hbmBandwidthBps: 9e11,
  hbmSizeBytes: 1.6e10,
  l2CacheSizeBytes: 2.5e7,
  l2BandwidthBps: 3e12,
  intranodeAllreduceBandwidthBps: 3e11 / 4,
  internodeAllreduceBandwidthBps: 1.25e10 / 2,
  nodeSize: 8,
  priceDollarsPerHour: (0.63 * 2) / 3,
  kernelLaunchLatencySeconds: 4e-6,
  collectiveTimeSeconds: meanCollectiveTimeNcclSeconds,
  arithmeticUtilizationCap: 0.8,
  memoryBwdUtilizationCap: 0.75,
});

export const TPU_v4 = new GPU({
  name: "TPU v4",
  flopPerSecond: { 8: 2.6e14, 16: 2.6e14 },
  hbmBandwidthBps: 1.6e12,
  hbmSizeBytes: 32e9,
  l2CacheSizeBytes: 5e4,
  l2BandwidthBps: 6.44e12,
  intranodeAllreduceBandwidthBps: Infinity,
  internodeAllreduceBandwidthBps: 2.5e11,
  nodeSize: 1,
  priceDollarsPerHour: 1,
  kernelLaunchLatencySeconds: 4e-6,
  collectiveTimeSeconds: tpuCollectiveTimeSeconds,
});

export const Groq_LPU = new GPU({
  name: "Groq LPU",
  flopPerSecond: { 8: 750e12, 16: 188e12 },
  hbmBandwidthBps: 8e13,
  hbmSizeBytes: 230e6,
  l2CacheSizeBytes: 0,
  l2BandwidthBps: Infinity,
  intranodeAllreduceBandwidthBps: Infinity,
  internodeAllreduceBandwidthBps: 330e9 / 2,
  nodeSize: 1,
  priceDollarsPerHour: 1,
  kernelLaunchLatencySeconds: 4e-6,
  collectiveTimeSeconds: tpuCollectiveTimeSeconds,
});

/** Lookup map for UI selectors. */
export const GPU_MAP: Record<string, GPU> = {
  H100,
  H200,
  A100,
  H20,
};
