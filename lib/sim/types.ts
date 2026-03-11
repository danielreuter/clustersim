// All units: FLOP/s, bytes, bytes/s, seconds

import type { GPU } from "@/lib/erdil/gpus"

export type Hardware = {
  name: string
  gpuKey: string
  nGpu: number
}

/** Hardware with compute/memory derived from GPU specs + model precision */
export type ResolvedHardware = {
  name: string
  gpu: GPU
  nGpu: number
  computeFlops: number   // Ĝ  — nGpu × gpu FLOP/s at model precision
  hbmBytes: number       // M̂  — nGpu × gpu HBM capacity
}

export type HonestLoad = {
  claimedComputeFlops: number  // f* — honest compute consumption (FLOP/s)
  claimedMemoryBytes: number   // m* — honest memory consumption (bytes)
}

export type Verifier = {
  alpha: number               // fraction of claimed compute proven real [0,1]
  alphaMemory?: number        // fraction of claimed memory proven real [0,1] (defaults to 1)
  covertIngressBps: number    // b_in  — usable covert input rate (bytes/s)
  covertEgressBps: number     // b_out — usable covert output rate (bytes/s)
  survivingStateBytes: number // C — covert state surviving sanitization (bytes)
  epochSeconds: number        // τ — sanitization epoch length (seconds)
  downtimeSeconds: number     // T — sanitization downtime per epoch (seconds)
  sanitizationEnabled: boolean
}

// Internal workload type used by composeGamma (constructed by roofline backends)
export type CovertWorkloadV1 = {
  label: string
  kind: "inference" | "training"
  unit: string                  // "token", "train-token", etc.
  stateBytes: number            // n   — total covert HBM footprint (persist + workspace)
  persistBytes?: number         // n_persist — covert state reloaded after sanitization (defaults to stateBytes)
  flopPerUnit: number           // g   — FLOP per unit of covert output
  ingressBytesPerUnit: number   // d_in
  egressBytesPerUnit: number    // d_out
}

// Model-based inference workload
export type CovertWorkloadInference = {
  label: string
  kind: "inference"
  unit: "token"
  backend: "roofline-lite"
  modelKey: string
  contextLength: number
}

// Model-based training workload
export type TrainingSyncPolicy =
  | { mode: "none" }
  | { mode: "checkpoint"; bytesOutPerSync: number; tokensPerSync: number }
  | { mode: "periodic-updates"; bytesInPerSync: number; bytesOutPerSync: number; tokensPerSync: number }

export type CovertWorkloadTraining = {
  label: string
  kind: "training"
  unit: "train-token"
  backend: "roofline-lite"
  modelKey: string
  syncPolicy: TrainingSyncPolicy
}

export type CovertWorkload = CovertWorkloadV1 | CovertWorkloadInference | CovertWorkloadTraining

export function isV2Workload(w: CovertWorkload): w is CovertWorkloadInference | CovertWorkloadTraining {
  return "backend" in w
}

export type Scenario = {
  hardware: Hardware
  honest: HonestLoad
  verifier: Verifier
  covert: CovertWorkload
}

/** Direct scenario with raw numeric fields — no GPU/model lookup required */
export type DirectScenario = {
  hardware: { computeFlops: number; hbmBytes: number }
  honest: HonestLoad
  verifier: Verifier
  covert: CovertWorkloadV1
}

export type ThroughputEstimate = {
  unitsPerSecond: number
  regime?: "compute" | "memory" | "latency" | "comm" | "memory-bandwidth" | "under-batched"
  details?: Record<string, number | string>
}

export type GammaResult = {
  gamma: number
  dominant: "memory-fit" | "duty" | "compute" | "ingress" | "egress"
  theta0: number
  thetaVerified: number
  gammaDuty: number
  gammaCompute: number
  gammaIngress: number
  gammaEgress: number
  tReload: number
  tCovert: number
  fitMarginBytes: number
  finite: boolean
  reason?: string
  v2?: {
    regime: string
    optimalBatchSize: number
    nPersistBytes: number
    workspaceBytes: number
  }
}

export type SimulationSnapshot = {
  input: Scenario
  output: GammaResult
}
