import type { Hardware, CovertWorkload, Verifier } from "./types"

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

export const WORKLOADS: Record<string, CovertWorkload> = {
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
