import type { Hardware, CovertWorkloadV1, CovertWorkloadInference, CovertWorkloadTraining, Verifier } from "./types"
import { MODEL_MAP } from "@/lib/erdil/models"
import { GPU_MAP } from "@/lib/erdil/gpus"
import { rooflineLite, rooflineLiteTraining, deriveSyncIO, resolveHardware } from "./roofline"

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

// ---------------------------------------------------------------------------
// Preset snap helpers for DirectScenario
// ---------------------------------------------------------------------------

/** Compute hardware FLOP/s and HBM from GPU key + count */
export function computeHardwarePreset(
  gpuKey: string,
  nGpu: number,
  precisionBytes = 2,
): { computeFlops: number; hbmBytes: number } {
  const rh = resolveHardware({ name: "", gpuKey, nGpu }, precisionBytes)
  return { computeFlops: rh.computeFlops, hbmBytes: rh.hbmBytes }
}

export type CovertPresetConfig = {
  modelKey: string
  kind: "inference" | "training"
  contextLength?: number
  syncPolicy?: import("./types").TrainingSyncPolicy
}

/** Compute covert workload V1 params from model + hardware via roofline */
export function computeCovertPreset(
  hardware: { computeFlops: number; hbmBytes: number },
  config: CovertPresetConfig,
): CovertWorkloadV1 {
  const model = MODEL_MAP[config.modelKey]
  if (!model) throw new Error(`Unknown model: ${config.modelKey}`)

  const gpu = GPU_MAP["H100"] // dummy GPU for bandwidth calc (only used for hbmBandwidthBps)
  // Estimate nGpu from total HBM
  const nGpu = Math.max(1, Math.round(hardware.hbmBytes / gpu.hbmSizeBytes))

  const isInference = config.kind === "inference"
  const ctx = isInference ? (config.contextLength ?? 2048) : 1

  const roofline = isInference
    ? rooflineLite(model, gpu, nGpu, ctx, hardware.computeFlops, hardware.hbmBytes)
    : rooflineLiteTraining(model, gpu, nGpu, hardware.computeFlops, hardware.hbmBytes)

  const stateBytes = roofline.nPersist + roofline.workspace
  const g = roofline.throughput.unitsPerSecond > 0
    ? hardware.computeFlops / roofline.throughput.unitsPerSecond
    : Infinity

  let dIn = 0
  let dOut = isInference ? 4 : 0
  if (!isInference && config.syncPolicy) {
    const sync = deriveSyncIO(config.syncPolicy)
    dIn = sync.dIn
    dOut = sync.dOut
  }

  return {
    label: isInference ? `${config.modelKey} inference` : `Train ${config.modelKey}`,
    kind: config.kind,
    unit: isInference ? "token" : "train-token",
    stateBytes,
    persistBytes: roofline.nPersist,
    flopPerUnit: g,
    ingressBytesPerUnit: dIn,
    egressBytesPerUnit: dOut,
  }
}

export type HardwarePresetConfig = { gpuKey: string; nGpu: number }

export const HARDWARE_PRESETS: Record<string, HardwarePresetConfig> = {
  "8xH100": { gpuKey: "H100", nGpu: 8 },
  "32xH100": { gpuKey: "H100", nGpu: 32 },
  "8xH200": { gpuKey: "H200", nGpu: 8 },
  "32xH200": { gpuKey: "H200", nGpu: 32 },
  "8xA100": { gpuKey: "A100", nGpu: 8 },
  "8xH20": { gpuKey: "H20", nGpu: 8 },
}

export const COVERT_PRESETS: Record<string, CovertPresetConfig> = {
  "inf-llama8b": { modelKey: "Llama 3 8B", kind: "inference", contextLength: 4096 },
  "inf-llama70b": { modelKey: "Llama 3 70B", kind: "inference", contextLength: 2048 },
  "inf-llama405b": { modelKey: "Llama 3 405B", kind: "inference", contextLength: 2048 },
  "inf-deepseekv3": { modelKey: "DeepSeek V3", kind: "inference", contextLength: 4096 },
  "train-llama70b-nosync": { modelKey: "Llama 3 70B", kind: "training", syncPolicy: { mode: "none" } },
  "train-llama70b-ckpt": { modelKey: "Llama 3 70B", kind: "training", syncPolicy: { mode: "checkpoint", bytesOutPerSync: 140e9, tokensPerSync: 1e6 } },
  "train-llama70b-periodic": { modelKey: "Llama 3 70B", kind: "training", syncPolicy: { mode: "periodic-updates", bytesInPerSync: 140e9, bytesOutPerSync: 140e9, tokensPerSync: 1e6 } },
}

export { MODEL_MAP, GPU_MAP }
