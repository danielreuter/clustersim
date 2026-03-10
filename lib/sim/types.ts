// All units: FLOP/s, bytes, bytes/s, seconds

export type Hardware = {
  name: string
  computeFlops: number   // Ĝ  — system peak FLOP/s
  hbmBytes: number       // M̂  — system HBM capacity in bytes
}

export type HonestLoad = {
  claimedComputeFlops: number  // f* — honest compute consumption (FLOP/s)
  claimedMemoryBytes: number   // m* — honest memory consumption (bytes)
}

export type Verifier = {
  alpha: number               // fraction of honest compute proven real [0,1]
  covertIngressBps: number    // b_in  — usable covert input rate (bytes/s)
  covertEgressBps: number     // b_out — usable covert output rate (bytes/s)
  survivingStateBytes: number // C — covert state surviving sanitization (bytes)
  epochSeconds: number        // τ — sanitization epoch length (seconds)
  downtimeSeconds: number     // T — sanitization downtime per epoch (seconds)
  sanitizationEnabled: boolean
}

// v1 workload: abstract, user-specified parameters
export type CovertWorkloadV1 = {
  label: string
  kind: "inference" | "training"
  unit: string                  // "token", "train-token", etc.
  stateBytes: number            // n   — covert state that must reside in HBM
  flopPerUnit: number           // g   — FLOP per unit of covert output
  ingressBytesPerUnit: number   // d_in
  egressBytesPerUnit: number    // d_out
}

// v2 inference: model-based, batch-optimized via roofline
export type CovertWorkloadInference = {
  label: string
  kind: "inference"
  unit: "token"
  backend: "roofline-lite"
  modelKey: string
  gpuKey: string
  nGpu: number
  contextLength: number
}

// v2 training: sync-aware
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
  gpuKey: string
  nGpu: number
  syncPolicy: TrainingSyncPolicy
}

// Union — v1 objects lack `backend` field
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
  gammaV1?: number
}

export type SimulationSnapshot = {
  input: Scenario
  output: GammaResult
}
