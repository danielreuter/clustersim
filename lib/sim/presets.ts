import type { Hardware, CovertWorkloadInference, CovertWorkloadTraining, Verifier } from "./types"
import { MODEL_MAP } from "@/lib/erdil/models"
import { GPU_MAP } from "@/lib/erdil/gpus"

const GB = 1e9
const KB = 1e3

// ---------------------------------------------------------------------------
// Hardware presets (GPU type + count)
// ---------------------------------------------------------------------------

export const HARDWARE: Record<string, Hardware> = {
  "8xH100": { name: "8× H100", gpuKey: "H100", nGpu: 8 },
  "32xH100": { name: "32× H100 (4 nodes)", gpuKey: "H100", nGpu: 32 },
  "8xH200": { name: "8× H200", gpuKey: "H200", nGpu: 8 },
  "32xH200": { name: "32× H200 (4 nodes)", gpuKey: "H200", nGpu: 32 },
  "8xA100": { name: "8× A100", gpuKey: "A100", nGpu: 8 },
  "8xH20": { name: "8× H20", gpuKey: "H20", nGpu: 8 },
  "72xB200": { name: "GB200 NVL72", gpuKey: "B200", nGpu: 72 },
}

// ---------------------------------------------------------------------------
// Covert workload presets (model-based, hardware-agnostic)
// ---------------------------------------------------------------------------

export const WORKLOADS_V2_INFERENCE: Record<string, CovertWorkloadInference> = {
  "inf-llama8b": {
    label: "Llama 3 8B",
    kind: "inference",
    unit: "token",
    backend: "roofline-lite",
    modelKey: "Llama 3 8B",
    contextLength: 4096,
  },
  "inf-llama70b": {
    label: "Llama 3 70B",
    kind: "inference",
    unit: "token",
    backend: "roofline-lite",
    modelKey: "Llama 3 70B",
    contextLength: 2048,
  },
  "inf-llama405b": {
    label: "Llama 3 405B",
    kind: "inference",
    unit: "token",
    backend: "roofline-lite",
    modelKey: "Llama 3 405B",
    contextLength: 2048,
  },
  "inf-deepseekv3": {
    label: "DeepSeek V3",
    kind: "inference",
    unit: "token",
    backend: "roofline-lite",
    modelKey: "DeepSeek V3",
    contextLength: 4096,
  },
}

export const WORKLOADS_V2_TRAINING: Record<string, CovertWorkloadTraining> = {
  "train-llama70b-nosync": {
    label: "Train Llama 70B (no sync)",
    kind: "training",
    unit: "train-token",
    backend: "roofline-lite",
    modelKey: "Llama 3 70B",
    syncPolicy: { mode: "none" },
  },
  "train-llama70b-ckpt": {
    label: "Train Llama 70B (checkpoint)",
    kind: "training",
    unit: "train-token",
    backend: "roofline-lite",
    modelKey: "Llama 3 70B",
    syncPolicy: {
      mode: "checkpoint",
      bytesOutPerSync: 140e9,
      tokensPerSync: 1e6,
    },
  },
  "train-llama70b-periodic": {
    label: "Train Llama 70B (periodic sync)",
    kind: "training",
    unit: "train-token",
    backend: "roofline-lite",
    modelKey: "Llama 3 70B",
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

// ---------------------------------------------------------------------------
// Verifier presets
// ---------------------------------------------------------------------------

export const VERIFIER_FULL: Verifier = {
  alpha: 1.0,
  covertIngressBps: 100 * KB,
  covertEgressBps: 20 * KB,
  survivingStateBytes: 17 * GB,
  epochSeconds: 5,
  downtimeSeconds: 0.25,
  sanitizationEnabled: true,
}

export const VERIFIER_NO_SANITIZATION: Verifier = {
  ...VERIFIER_FULL,
  sanitizationEnabled: false,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

export function honestLoadFromFractions(
  computeFlops: number,
  hbmBytes: number,
  computeFrac: number,
  memoryFrac: number,
) {
  return {
    claimedComputeFlops: computeFrac * computeFlops,
    claimedMemoryBytes: memoryFrac * hbmBytes,
  }
}

export { MODEL_MAP, GPU_MAP }
