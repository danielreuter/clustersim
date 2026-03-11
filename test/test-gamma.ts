// Verify Γ engine with GPU-based hardware
import {
  simulate,
  resolveHardware,
  honestLoadFromFractions,
  VERIFIER_FULL,
  VERIFIER_NO_SANITIZATION,
  type Hardware,
  type Scenario,
  type CovertWorkloadInference,
  type CovertWorkloadTraining,
} from "../lib/sim"

let passed = 0
let failed = 0

function assert(
  label: string,
  actual: number,
  expected: number,
  tolerance = 0.05,
) {
  const pass = expected === Infinity
    ? actual === Infinity
    : expected === 0
      ? actual === 0
      : Math.abs(actual - expected) / expected < tolerance
  if (pass) {
    passed++
  } else {
    failed++
    console.error(`  FAIL: ${label} — expected ${expected}, got ${actual}`)
  }
}

// Hardware presets
const HW_8xH100: Hardware = { name: "8× H100", gpuKey: "H100", nGpu: 8 }
const HW_32xH100: Hardware = { name: "32× H100", gpuKey: "H100", nGpu: 32 }
const HW_8xH200: Hardware = { name: "8× H200", gpuKey: "H200", nGpu: 8 }

function inferenceScenario(
  hw: Hardware,
  modelKey: string,
  ctx: number,
  sanitization: boolean,
  alpha = 1.0,
  computeFrac = 0.5,
  memoryFrac = 0.5,
): Scenario {
  const rh = resolveHardware(hw, 2) // BF16 for honest load fractions
  return {
    hardware: hw,
    honest: honestLoadFromFractions(rh.computeFlops, rh.hbmBytes, computeFrac, memoryFrac),
    verifier: sanitization ? VERIFIER_FULL : VERIFIER_NO_SANITIZATION,
    covert: {
      label: `${modelKey} inference`,
      kind: "inference",
      unit: "token",
      backend: "roofline-lite",
      modelKey,
      contextLength: ctx,
    },
  }
}

// ===================================================================
// Basic inference: Llama 70B on 8×H100
// ===================================================================
console.log("\n=== Llama 70B on 8×H100 (no sanitization) ===")
{
  const s = inferenceScenario(HW_8xH100, "Llama 3 70B", 2048, false)
  const r = simulate(s)
  console.log(`  Γ = ${r.gamma.toFixed(1)}, dominant = ${r.dominant}`)
  console.log(`  regime: ${r.v2?.regime}, optBatch: ${r.v2?.optimalBatchSize}`)
  assert("70B: has v2 metadata", r.v2 ? 1 : 0, 1)
  assert("70B: finite", r.finite ? 1 : 0, 1)
  // With 50% honest compute and α=1, gammaCompute should be ~2×
  assert("70B: gammaCompute ≈ 2", r.gammaCompute, 2, 0.3)
}

// ===================================================================
// 98% honest compute should cause large overhead
// ===================================================================
console.log("\n=== Llama 70B on 8×H100, 98% honest compute ===")
{
  const s = inferenceScenario(HW_8xH100, "Llama 3 70B", 2048, false, 1.0, 0.98, 0.1)
  const r = simulate(s)
  console.log(`  Γ = ${r.gamma.toFixed(1)}, dominant = ${r.dominant}`)
  // With only 2% compute remaining, should see significant overhead
  // (may be <50× because smaller batches shift to memory-bandwidth-bound regime)
  assert("98%: gamma > 5", r.gamma > 5 ? 1 : 0, 1)
}

// ===================================================================
// Hardware resolution: GPU specs should determine compute
// ===================================================================
console.log("\n=== Hardware resolution ===")
{
  const rh = resolveHardware(HW_8xH100, 2) // BF16 = precision key 16
  // H100 FP16: 1e15 * 0.7 (util cap) = 7e14 per GPU
  // 8 GPUs = 5.6e15
  assert("8×H100 BF16 compute", rh.computeFlops, 8 * 1e15 * 0.7, 0.01)
  // H100 HBM: 80 GB per GPU, 8 GPUs = 640 GB
  assert("8×H100 HBM", rh.hbmBytes, 8 * 80e9, 0.01)
  console.log(`  computeFlops = ${rh.computeFlops.toExponential(2)}, hbmBytes = ${rh.hbmBytes.toExponential(2)}`)
}

// ===================================================================
// Memory fit: large model on small hardware
// ===================================================================
console.log("\n=== Llama 405B on 8×H100 (memory check) ===")
{
  const s = inferenceScenario(HW_8xH100, "Llama 3 405B", 2048, false, 1.0, 0.1, 0.1)
  const r = simulate(s)
  console.log(`  Γ = ${r.gamma.toFixed(1)}, dominant = ${r.dominant}, fitMargin = ${(r.fitMarginBytes / 1e9).toFixed(1)} GB`)
  // 405B at BF16 = ~810 GB weights. 8×H100 = 640 GB total HBM. Should not fit.
  assert("405B on 8×H100: infeasible", r.gamma, Infinity)
  assert("405B on 8×H100: memory-fit", r.dominant === "memory-fit" ? 1 : 0, 1)
}

// ===================================================================
// Llama 405B on 8×H200 should fit (8×141GB = 1128 GB)
// ===================================================================
console.log("\n=== Llama 405B on 8×H200 ===")
{
  const s = inferenceScenario(HW_8xH200, "Llama 3 405B", 2048, false, 1.0, 0.1, 0.1)
  const r = simulate(s)
  console.log(`  Γ = ${r.gamma.toFixed(1)}, dominant = ${r.dominant}`)
  assert("405B on 8×H200: finite", r.finite ? 1 : 0, 1)
}

// ===================================================================
// Sanitization: should have very high overhead with small epoch
// ===================================================================
console.log("\n=== Llama 70B with sanitization ===")
{
  const s = inferenceScenario(HW_8xH100, "Llama 3 70B", 2048, true)
  const r = simulate(s)
  console.log(`  Γ = ${r.gamma.toFixed(1)}, dominant = ${r.dominant}`)
  console.log(`  tReload = ${r.tReload.toFixed(1)}s, tCovert = ${r.tCovert.toFixed(1)}s`)
  // With 5s epoch and large model to reload, should have significant duty overhead
  assert("sanitization: gammaDuty > 1", r.gammaDuty > 1 ? 1 : 0, 1)
}

// ===================================================================
// Training workload with periodic sync
// ===================================================================
console.log("\n=== Training: Llama 8B periodic sync ===")
{
  const hw = HW_8xH100
  const rh = resolveHardware(hw, 2)
  const wl: CovertWorkloadTraining = {
    label: "Train 8B periodic",
    kind: "training",
    unit: "train-token",
    backend: "roofline-lite",
    modelKey: "Llama 3 8B",
    syncPolicy: {
      mode: "periodic-updates",
      bytesInPerSync: 16e9,
      bytesOutPerSync: 16e9,
      tokensPerSync: 1e6,
    },
  }
  const s: Scenario = {
    hardware: hw,
    honest: honestLoadFromFractions(rh.computeFlops, rh.hbmBytes, 0.5, 0.1),
    verifier: { ...VERIFIER_NO_SANITIZATION, covertEgressBps: 1e6 },
    covert: wl,
  }
  const r = simulate(s)
  console.log(`  Γ = ${r.gamma.toFixed(1)}, dominant = ${r.dominant}`)
  assert("train: egress factor > 1", r.gammaEgress > 1 ? 1 : 0, 1)
}

// ===================================================================
// Small model on large hardware: 8B on 8×H100 should be fast
// ===================================================================
console.log("\n=== Llama 8B on 8×H100 ===")
{
  const s = inferenceScenario(HW_8xH100, "Llama 3 8B", 4096, false, 0, 0, 0)
  // No honest load, no verification → γ should be 1
  s.verifier = { ...s.verifier, alpha: 0, covertIngressBps: Infinity, covertEgressBps: Infinity }
  const r = simulate(s)
  console.log(`  Γ = ${r.gamma.toFixed(1)}, regime: ${r.v2?.regime}, optBatch: ${r.v2?.optimalBatchSize}`)
  assert("8B unverified: Γ = 1", r.gamma, 1, 0.01)
}

// ===================================================================
console.log(
  `\n${"=".repeat(50)}\nResults: ${passed} passed, ${failed} failed out of ${passed + failed} tests`,
)
if (failed > 0) process.exit(1)
else console.log("All tests passed!")
