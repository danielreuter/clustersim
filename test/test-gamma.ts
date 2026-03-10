// Verify Γ engine against WTS doc worked examples
import {
  simulate,
  HARDWARE,
  WORKLOADS,
  VERIFIER_FULL,
  VERIFIER_NO_SANITIZATION,
  honestLoadFromFractions,
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

function scenario(
  hwKey: string,
  wlKey: string,
  sanitization: boolean,
  alpha = 1.0,
  computeFrac = 0.5,
  memoryFrac = 0.5,
): Scenario {
  const hw = HARDWARE[hwKey]
  return {
    hardware: hw,
    honest: honestLoadFromFractions(hw, computeFrac, memoryFrac),
    verifier: sanitization ? VERIFIER_FULL : VERIFIER_NO_SANITIZATION,
    covert: WORKLOADS[wlKey],
  }
}

// ===================================================================
// GB200 NVL72 + W_inf,1T  (WTS doc Table: progressive layers)
// ===================================================================
console.log("\n=== GB200 NVL72 + Inference 1T ===")

// No verification (α=0, no network transparency, no sanitization)
{
  const s = scenario("gb200-nvl72", "inf-1t", false, 0)
  // Override verifier to have no I/O caps
  s.verifier = { ...s.verifier, alpha: 0, covertIngressBps: Infinity, covertEgressBps: Infinity }
  const r = simulate(s)
  assert("I/O only: Γ", r.gamma, 1)
}

// + matmul transparency (α=1, no network, no sanitization)
{
  const s = scenario("gb200-nvl72", "inf-1t", false, 1)
  s.verifier = { ...s.verifier, covertIngressBps: Infinity, covertEgressBps: Infinity }
  const r = simulate(s)
  assert("+ matmul: Γ_compute", r.gammaCompute, 2)
  assert("+ matmul: Γ", r.gamma, 2)
}

// + network transparency (α=1, b_in=100KB/s, b_out=20KB/s, no sanitization)
{
  const r = simulate(scenario("gb200-nvl72", "inf-1t", false))
  assert("+ network: Γ_compute", r.gammaCompute, 2)
  assert("+ network: Γ_ingress", r.gammaIngress, 3.6, 0.1)
  assert("+ network: Γ_egress", r.gammaEgress, 18, 0.1)
  assert("+ network: Γ", r.gamma, 18, 0.1)
  assert("+ network: dominant is egress", r.dominant === "egress" ? 1 : 0, 1)
}

// + memory sanitization
{
  const r = simulate(scenario("gb200-nvl72", "inf-1t", true))
  assert("+ sanitization: Γ", r.gamma, Infinity)
  assert("+ sanitization: dominant is duty", r.dominant === "duty" ? 1 : 0, 1)
}

// ===================================================================
// 4× DGX H100 + W_inf,1T — should not fit (memory)
// ===================================================================
console.log("\n=== 4× DGX H100 + Inference 1T (memory fit fail) ===")
{
  const r = simulate(scenario("4x-dgx-h100", "inf-1t", false))
  assert("memory fit: Γ", r.gamma, Infinity)
  assert("memory fit: dominant", r.dominant === "memory-fit" ? 1 : 0, 1)
}

// ===================================================================
// 4× DGX H100 + W_inf,200B — should fit, egress-bound
// ===================================================================
console.log("\n=== 4× DGX H100 + Inference 200B ===")
{
  const r = simulate(scenario("4x-dgx-h100", "inf-200b", false))
  assert("200B: Γ_compute", r.gammaCompute, 2)
  assert("200B: finite", r.finite ? 1 : 0, 1)
  // Γ_egress = Ĝ*d_out / (g*b_out) = 31.7e15*4 / (4e11*20e3) = 15.85
  assert("200B: Γ_egress", r.gammaEgress, 15.85, 0.1)
  assert("200B: dominant is egress", r.dominant === "egress" ? 1 : 0, 1)
}

// ===================================================================
// GB200 NVL72 + W_train,70B — d_in=d_out=0, compute only
// ===================================================================
console.log("\n=== GB200 NVL72 + Training 70B ===")
{
  const r = simulate(scenario("gb200-nvl72", "train-70b", false))
  assert("train: Γ_compute", r.gammaCompute, 2)
  assert("train: Γ_ingress", r.gammaIngress, 1)
  assert("train: Γ_egress", r.gammaEgress, 1)
  assert("train: Γ", r.gamma, 2)
  assert("train: dominant is compute", r.dominant === "compute" ? 1 : 0, 1)
}

// ===================================================================
// GB200 NVL72 + W_inf,200B — highest egress overhead
// ===================================================================
console.log("\n=== GB200 NVL72 + Inference 200B ===")
{
  const r = simulate(scenario("gb200-nvl72", "inf-200b", false))
  // Γ_egress = 180e15*4 / (4e11*20e3) = 90
  assert("200B on GB200: Γ_egress", r.gammaEgress, 90, 0.1)
  assert("200B on GB200: dominant is egress", r.dominant === "egress" ? 1 : 0, 1)
}

// ===================================================================
// V2: inference workload
// ===================================================================
console.log("\n=== Inference: Llama 70B on 8×H100 ===")
{
  const hw = HARDWARE["4x-dgx-h100"]
  const wl: CovertWorkloadInference = {
    label: "Llama 70B on 8×H100",
    kind: "inference",
    unit: "token",
    backend: "roofline-lite",
    modelKey: "Llama 3 70B",
    gpuKey: "H100",
    nGpu: 8,
    contextLength: 2048,
  }
  const s: Scenario = {
    hardware: hw,
    honest: honestLoadFromFractions(hw, 0.5, 0.5),
    verifier: VERIFIER_FULL,
    covert: wl,
  }
  const r = simulate(s)
  console.log(`  Γ = ${r.gamma.toFixed(1)}, dominant = ${r.dominant}`)
  console.log(`  regime: ${r.v2?.regime}, optBatch: ${r.v2?.optimalBatchSize}`)
  assert("inf: has v2 metadata", r.v2 ? 1 : 0, 1)
}

// ===================================================================
// V2: training workload with sync (use 8B to fit in 8×H100)
// ===================================================================
console.log("\n=== Training: Llama 8B periodic sync ===")
{
  const hw = HARDWARE["4x-dgx-h100"]
  const wl: CovertWorkloadTraining = {
    label: "Train 8B periodic",
    kind: "training",
    unit: "train-token",
    backend: "roofline-lite",
    modelKey: "Llama 3 8B",
    gpuKey: "H100",
    nGpu: 8,
    syncPolicy: {
      mode: "periodic-updates",
      bytesInPerSync: 16e9,
      bytesOutPerSync: 16e9,
      tokensPerSync: 1e6,
    },
  }
  const s: Scenario = {
    hardware: hw,
    honest: honestLoadFromFractions(hw, 0.5, 0.1),
    verifier: { ...VERIFIER_NO_SANITIZATION, covertEgressBps: 1e6 },
    covert: wl,
  }
  const r = simulate(s)
  console.log(`  Γ = ${r.gamma.toFixed(1)}, dominant = ${r.dominant}`)
  // With periodic sync, egress should become a factor
  assert("train: egress factor > 1", r.gammaEgress > 1 ? 1 : 0, 1)
}

// ===================================================================
console.log(
  `\n${"=".repeat(50)}\nResults: ${passed} passed, ${failed} failed out of ${passed + failed} tests`,
)
if (failed > 0) process.exit(1)
else console.log("All tests passed!")
