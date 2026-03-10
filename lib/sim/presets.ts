import type { Hardware, CovertWorkload, CovertWorkloadV1, CovertWorkloadInference, CovertWorkloadTraining, Verifier } from "./types"
import { MODEL_MAP } from "@/lib/erdil/models"
import { GPU_MAP } from "@/lib/erdil/gpus"

const TB = 1e12
const GB = 1e9
const KB = 1e3
const PFLOPS = 1e15

// ---------------------------------------------------------------------------
// Hardware presets (from WTS doc Table 1)
// ---------------------------------------------------------------------------

export const HARDWARE: Record<string, Hardware> = {
  "4x-dgx-h100": {
    name: "4× DGX H100",
    computeFlops: 31.7 * PFLOPS,
    hbmBytes: 2.56 * TB,
  },
  "4x-dgx-h200": {
    name: "4× DGX H200",
    computeFlops: 31.7 * PFLOPS,
    hbmBytes: 4.512 * TB,
  },
  "gb200-nvl72": {
    name: "GB200 NVL72",
    computeFlops: 180 * PFLOPS,
    hbmBytes: 13.4 * TB,
  },
  "vera-rubin-nvl72": {
    name: "Vera Rubin NVL72",
    computeFlops: 288 * PFLOPS,
    hbmBytes: 20.7 * TB,
  },
}

// ---------------------------------------------------------------------------
// Covert workload presets (from WTS doc Table 2)
// ---------------------------------------------------------------------------

export const WORKLOADS: Record<string, CovertWorkloadV1> = {
  "inf-1t": {
    label: "Inference 1T dense (BF16)",
    kind: "inference",
    unit: "token",
    stateBytes: 2.0 * TB,
    flopPerUnit: 2e12,
    ingressBytesPerUnit: 4,
    egressBytesPerUnit: 4,
  },
  "inf-200b": {
    label: "Inference 200B dense (BF16)",
    kind: "inference",
    unit: "token",
    stateBytes: 0.4 * TB,
    flopPerUnit: 4e11,
    ingressBytesPerUnit: 4,
    egressBytesPerUnit: 4,
  },
  "train-70b": {
    label: "Training 70B dense (BF16)",
    kind: "training",
    unit: "train-token",
    stateBytes: 1.3 * TB,
    flopPerUnit: 4.2e11,
    ingressBytesPerUnit: 0,
    egressBytesPerUnit: 0,
  },
}

// ---------------------------------------------------------------------------
// Verifier presets (from WTS doc reference parameters)
// ---------------------------------------------------------------------------

export const VERIFIER_FULL: Verifier = {
  alpha: 1.0,
  covertIngressBps: 100 * KB,    // 100 KB/s
  covertEgressBps: 20 * KB,      // 20 KB/s
  survivingStateBytes: 17 * GB,  // C ≈ 17 GB
  epochSeconds: 5,               // τ = 5s
  downtimeSeconds: 0.25,         // T = 0.25s
  sanitizationEnabled: true,
}

export const VERIFIER_NO_SANITIZATION: Verifier = {
  ...VERIFIER_FULL,
  sanitizationEnabled: false,
}

// ---------------------------------------------------------------------------
// Helpers for constructing honest loads at a given utilization fraction
// ---------------------------------------------------------------------------

export function honestLoadFromFractions(
  hw: Hardware,
  computeFrac: number,
  memoryFrac: number,
) {
  return {
    claimedComputeFlops: computeFrac * hw.computeFlops,
    claimedMemoryBytes: memoryFrac * hw.hbmBytes,
  }
}

// ---------------------------------------------------------------------------
// V2 workload presets (model-based, roofline-lite backend)
// ---------------------------------------------------------------------------

export const WORKLOADS_V2_INFERENCE: Record<string, CovertWorkloadInference> = {
  "v2-inf-llama70b-h100x8": {
    label: "Llama 3 70B on 8×H100",
    kind: "inference",
    unit: "token",
    backend: "roofline-lite",
    modelKey: "Llama 3 70B",
    gpuKey: "H100",
    nGpu: 8,
    contextLength: 2048,
  },
  "v2-inf-llama405b-h200x8": {
    label: "Llama 3 405B on 8×H200",
    kind: "inference",
    unit: "token",
    backend: "roofline-lite",
    modelKey: "Llama 3 405B",
    gpuKey: "H200",
    nGpu: 8,
    contextLength: 2048,
  },
  "v2-inf-deepseekv3-h100x8": {
    label: "DeepSeek V3 on 8×H100",
    kind: "inference",
    unit: "token",
    backend: "roofline-lite",
    modelKey: "DeepSeek V3",
    gpuKey: "H100",
    nGpu: 8,
    contextLength: 4096,
  },
  "v2-inf-llama8b-h100x1": {
    label: "Llama 3 8B on 1×H100",
    kind: "inference",
    unit: "token",
    backend: "roofline-lite",
    modelKey: "Llama 3 8B",
    gpuKey: "H100",
    nGpu: 1,
    contextLength: 4096,
  },
}

export const WORKLOADS_V2_TRAINING: Record<string, CovertWorkloadTraining> = {
  "v2-train-llama70b-h100x8-nosync": {
    label: "Train Llama 70B 8×H100 (no sync)",
    kind: "training",
    unit: "train-token",
    backend: "roofline-lite",
    modelKey: "Llama 3 70B",
    gpuKey: "H100",
    nGpu: 8,
    syncPolicy: { mode: "none" },
  },
  "v2-train-llama70b-h100x8-ckpt": {
    label: "Train Llama 70B 8×H100 (checkpoint)",
    kind: "training",
    unit: "train-token",
    backend: "roofline-lite",
    modelKey: "Llama 3 70B",
    gpuKey: "H100",
    nGpu: 8,
    syncPolicy: {
      mode: "checkpoint",
      bytesOutPerSync: 140e9, // ~70B params × 2 bytes
      tokensPerSync: 1e6,
    },
  },
  "v2-train-llama70b-h100x8-periodic": {
    label: "Train Llama 70B 8×H100 (periodic sync)",
    kind: "training",
    unit: "train-token",
    backend: "roofline-lite",
    modelKey: "Llama 3 70B",
    gpuKey: "H100",
    nGpu: 8,
    syncPolicy: {
      mode: "periodic-updates",
      bytesInPerSync: 140e9,
      bytesOutPerSync: 140e9,
      tokensPerSync: 1e6,
    },
  },
}

export const WORKLOADS_V2: Record<string, CovertWorkloadInference | CovertWorkloadTraining> = {
  ...WORKLOADS_V2_INFERENCE,
  ...WORKLOADS_V2_TRAINING,
}

export { MODEL_MAP, GPU_MAP }
