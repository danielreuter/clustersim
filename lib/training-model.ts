export interface HardwareProfile {
  name: string
  hbmPerGpu: number // GB
  peakFlops: number // TFLOPS (BF16)
  nvlinkBandwidth: number // GB/s bidirectional
  infinibandBandwidth: number // GB/s per GPU
  gpusPerNode: number
}

export const HARDWARE_PROFILES: Record<string, HardwareProfile> = {
  h100: {
    name: "H100 SXM",
    hbmPerGpu: 80,
    peakFlops: 2000,
    nvlinkBandwidth: 900,
    infinibandBandwidth: 50,
    gpusPerNode: 8,
  },
  mi300x: {
    name: "MI300X",
    hbmPerGpu: 192,
    peakFlops: 2600,
    nvlinkBandwidth: 896,
    infinibandBandwidth: 50,
    gpusPerNode: 8,
  },
  b200: {
    name: "B200",
    hbmPerGpu: 192,
    peakFlops: 4500,
    nvlinkBandwidth: 1800,
    infinibandBandwidth: 100,
    gpusPerNode: 8,
  },
}

export interface ResourceLevels {
  compute: number // % free (0.01-100)
  hbmCapacity: number // % free
  nvlinkBandwidth: number // % free
  infinibandBandwidth: number // % free
}

export interface ModelConfig {
  totalParams: number // in billions
  hiddenDim: number
  numLayers: number
  attentionHeads: number
  sequenceLength: number
  microbatchSize: number // tokens
}

export interface AdversaryConfig {
  maxPipelineStages: number
  overlapFactor: number // 0-1
  bytesPerParam: number // 12 or 16
}

export interface DerivedQuantities {
  minGpus: number
  minTpDegree: number
  communicationRegime: "in-node" | "cross-node"
  pipelineStages: number
  microbatches: number
  maxMicrobatches: number
  pipelineEfficiency: number
  effectiveBandwidth: number
  activationMemoryPerMicrobatch: number
  hbmHeadroomPerGpu: number
  stateSize: number
}

export interface TimeBreakdown {
  computePlusBubble: number
  communication: number
  totalSlowdown: number
  bottleneck: "compute" | "pipeline bubbles (HBM capacity)" | "network (NVLink)" | "network (InfiniBand)"
}

export function calculateDerivedQuantities(
  resources: ResourceLevels,
  model: ModelConfig,
  adversary: AdversaryConfig,
  hardware: HardwareProfile,
): DerivedQuantities {
  const { hbmCapacity } = resources
  const { totalParams, hiddenDim, numLayers, microbatchSize } = model
  const { maxPipelineStages, bytesPerParam } = adversary
  const { hbmPerGpu, gpusPerNode, nvlinkBandwidth, infinibandBandwidth } = hardware

  // Total state size in GB
  const stateSize = (bytesPerParam * totalParams * 1e9) / 1e9 // GB

  // Minimum GPUs from HBM capacity
  const availableHbm = (hbmCapacity / 100) * hbmPerGpu
  const minGpus = Math.max(1, Math.ceil(stateSize / availableHbm))

  // Minimum tensor parallelism degree
  const minTpDegree = Math.max(1, Math.ceil(minGpus / maxPipelineStages))

  // Communication regime
  const communicationRegime: "in-node" | "cross-node" = minTpDegree <= gpusPerNode ? "in-node" : "cross-node"

  // Effective bandwidth
  const effectiveBandwidth =
    communicationRegime === "in-node"
      ? (resources.nvlinkBandwidth / 100) * nvlinkBandwidth
      : (resources.infinibandBandwidth / 100) * infinibandBandwidth

  // Pipeline stages
  const pipelineStages = Math.min(maxPipelineStages, Math.ceil(minGpus / minTpDegree))

  // Available HBM per GPU for activations
  const hbmForState = stateSize / minGpus
  const hbmAvailableForActivations = availableHbm - hbmForState
  const hbmHeadroomPerGpu = 0.2 * availableHbm + hbmAvailableForActivations

  // Activation memory per microbatch per GPU (in GB)
  const layersPerGpu = numLayers / pipelineStages
  const activationMemoryPerMicrobatch = (2 * hiddenDim * layersPerGpu * microbatchSize * 2) / 1e9

  // Maximum microbatches
  const maxMicrobatches = Math.max(1, Math.floor(hbmHeadroomPerGpu / Math.max(activationMemoryPerMicrobatch, 0.001)))

  // Actual microbatches (cap at reasonable global batch)
  const microbatches = Math.min(maxMicrobatches, 64)

  // Pipeline efficiency
  const pipelineEfficiency = microbatches / (microbatches + pipelineStages - 1)

  return {
    minGpus,
    minTpDegree,
    communicationRegime,
    pipelineStages,
    microbatches,
    maxMicrobatches,
    pipelineEfficiency,
    effectiveBandwidth,
    activationMemoryPerMicrobatch,
    hbmHeadroomPerGpu,
    stateSize,
  }
}

export function calculateTimeBreakdown(
  resources: ResourceLevels,
  model: ModelConfig,
  adversary: AdversaryConfig,
  hardware: HardwareProfile,
  derived: DerivedQuantities,
): TimeBreakdown {
  const { compute } = resources
  const { hiddenDim, microbatchSize } = model
  const { overlapFactor } = adversary
  const { peakFlops } = hardware
  const { pipelineEfficiency, effectiveBandwidth, communicationRegime } = derived

  // Compute time (normalized, baseline = 1)
  const tCompute = 1 / (compute / 100)

  // Compute time with pipeline bubbles
  const tComputePlusBubble = tCompute / pipelineEfficiency

  // Communication time
  // Bytes transferred per layer for TP all-reduce (forward + backward)
  const bytesPerLayer = 2 * microbatchSize * hiddenDim * 2 * 2 // fwd+bwd, bf16

  // FLOPS per layer (forward + backward)
  const flopsPerLayer = 2 * microbatchSize * hiddenDim * hiddenDim * 4

  // Communication time relative to compute at 100%
  const tComm = (bytesPerLayer * peakFlops * 1e12) / (effectiveBandwidth * 1e9 * flopsPerLayer)

  // Combined step time
  const tStep = Math.max(tComputePlusBubble, overlapFactor * tComm) + (1 - overlapFactor) * tComm

  // Bottleneck classification
  let bottleneck: TimeBreakdown["bottleneck"]
  if (tComputePlusBubble > tComm) {
    if (pipelineEfficiency < 0.5) {
      bottleneck = "pipeline bubbles (HBM capacity)"
    } else {
      bottleneck = "compute"
    }
  } else {
    if (communicationRegime === "cross-node") {
      bottleneck = "network (InfiniBand)"
    } else {
      bottleneck = "network (NVLink)"
    }
  }

  return {
    computePlusBubble: tComputePlusBubble,
    communication: tComm,
    totalSlowdown: tStep,
    bottleneck,
  }
}

export interface SensitivityPoint {
  resourceValue: number
  slowdown: number
}

export function calculateSensitivityCurve(
  baseResources: ResourceLevels,
  model: ModelConfig,
  adversary: AdversaryConfig,
  hardware: HardwareProfile,
  resourceKey: keyof ResourceLevels,
  points = 50,
): SensitivityPoint[] {
  const results: SensitivityPoint[] = []

  // Log scale from 0.1% to 100%
  const minLog = Math.log10(0.1)
  const maxLog = Math.log10(100)
  const step = (maxLog - minLog) / (points - 1)

  for (let i = 0; i < points; i++) {
    const resourceValue = Math.pow(10, minLog + i * step)
    const testResources = { ...baseResources, [resourceKey]: resourceValue }

    const derived = calculateDerivedQuantities(testResources, model, adversary, hardware)
    const time = calculateTimeBreakdown(testResources, model, adversary, hardware, derived)

    results.push({
      resourceValue,
      slowdown: time.totalSlowdown,
    })
  }

  return results
}
