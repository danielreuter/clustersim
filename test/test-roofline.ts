import {
  rooflineLite,
  rooflineLiteTraining,
  deriveSyncIO,
  resolveModel,
  resolveGpu,
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
      : Math.abs(actual - expected) / Math.abs(expected) < tolerance
  if (pass) {
    passed++
  } else {
    failed++
    console.error(`  FAIL: ${label} — expected ${expected}, got ${actual}`)
  }
}

function assertRange(
  label: string,
  actual: number,
  low: number,
  high: number,
) {
  const pass = actual >= low && actual <= high
  if (pass) {
    passed++
  } else {
    failed++
    console.error(`  FAIL: ${label} — expected [${low}, ${high}], got ${actual}`)
  }
}

function assertEqual(label: string, actual: string, expected: string) {
  if (actual === expected) {
    passed++
  } else {
    failed++
    console.error(`  FAIL: ${label} — expected "${expected}", got "${actual}"`)
  }
}

// ===================================================================
// Llama 70B on 8×H100 — should produce positive throughput
// ===================================================================
console.log("\n=== Roofline: Llama 70B on 8×H100 ===")
{
  const model = resolveModel("Llama 3 70B")
  const gpu = resolveGpu("H100")
  const nGpu = 8
  const ctx = 2048
  const totalFlops = nGpu * gpu.flopPerSecond[16]
  const totalHbm = nGpu * gpu.hbmSizeBytes

  const result = rooflineLite(model, gpu, nGpu, ctx, totalFlops, totalHbm)

  console.log(`  throughput: ${result.throughput.unitsPerSecond.toFixed(0)} tok/s`)
  console.log(`  regime: ${result.throughput.regime}`)
  console.log(`  optBatch: ${result.optBatch}`)
  console.log(`  nPersist: ${(result.nPersist / 1e9).toFixed(1)} GB`)

  // Should be positive throughput
  assert("throughput > 0", result.throughput.unitsPerSecond > 0 ? 1 : 0, 1)

  // Batch should be > 1 (model fits well in 640 GB)
  assert("optBatch > 1", result.optBatch > 1 ? 1 : 0, 1)

  // nPersist for 70B model at 2 bytes/param ≈ 140 GB
  assert("nPersist ≈ 140GB", result.nPersist, 70e9 * 2, 0.3)
}

// ===================================================================
// Llama 405B on single H100 — should not fit (80GB HBM, ~810GB weights)
// ===================================================================
console.log("\n=== Roofline: Llama 405B on 1×H100 (doesn't fit) ===")
{
  const model = resolveModel("Llama 3 405B")
  const gpu = resolveGpu("H100")
  const result = rooflineLite(model, gpu, 1, 2048, gpu.flopPerSecond[16], gpu.hbmSizeBytes)

  assert("405B single H100: throughput = 0", result.throughput.unitsPerSecond, 0)
  console.log(`  throughput: ${result.throughput.unitsPerSecond} (expected 0)`)
}

// ===================================================================
// Small batch → under-batched regime
// ===================================================================
console.log("\n=== Roofline: Llama 8B on 1×H100 — regime check ===")
{
  const model = resolveModel("Llama 3 8B")
  const gpu = resolveGpu("H100")
  // Give only enough HBM for batch=1 or 2 by limiting available memory
  const weightBytes = model.totalParams * model.weightPrecisionBytes
  const kvPerToken = model.kvCacheSizePerInputBytes
  // Allow batch=2 but not batch=4
  const tightHbm = weightBytes + kvPerToken * 4096 * 3

  const result = rooflineLite(model, gpu, 1, 4096, gpu.flopPerSecond[16], tightHbm)

  console.log(`  regime: ${result.throughput.regime}, optBatch: ${result.optBatch}`)
  assertEqual("tight memory → under-batched", result.throughput.regime ?? "none", "under-batched")
}

// ===================================================================
// Large batch capacity → should get reasonable throughput (may be mem-bw bound)
// ===================================================================
console.log("\n=== Roofline: Llama 70B 8×H100, large batch capacity ===")
{
  const model = resolveModel("Llama 3 70B")
  const gpu = resolveGpu("H100")
  const nGpu = 8
  const result = rooflineLite(model, gpu, nGpu, 512, nGpu * gpu.flopPerSecond[16], nGpu * gpu.hbmSizeBytes)

  console.log(`  regime: ${result.throughput.regime}, optBatch: ${result.optBatch}`)
  // Should achieve high batch and non-zero throughput
  assert("large batch capacity → optBatch > 4", result.optBatch > 4 ? 1 : 0, 1)
  assert("large batch capacity → throughput > 0", result.throughput.unitsPerSecond > 0 ? 1 : 0, 1)
}

// ===================================================================
// Training roofline — use 8×H200 (1.1TB HBM) to fit optimizer state
// ===================================================================
console.log("\n=== Roofline Training: Llama 8B on 8×H100 ===")
{
  const model = resolveModel("Llama 3 8B")
  const gpu = resolveGpu("H100")
  const nGpu = 8

  const result = rooflineLiteTraining(model, gpu, nGpu, nGpu * gpu.flopPerSecond[16], nGpu * gpu.hbmSizeBytes)

  console.log(`  throughput: ${result.throughput.unitsPerSecond.toFixed(0)} train-tok/s`)
  console.log(`  regime: ${result.throughput.regime}`)
  console.log(`  nPersist: ${(result.nPersist / 1e9).toFixed(1)} GB (weights + optimizer)`)

  assert("training throughput > 0", result.throughput.unitsPerSecond > 0 ? 1 : 0, 1)
  // nPersist should include optimizer state
  assert("training nPersist includes optimizer",
    result.nPersist > model.totalParams * model.weightPrecisionBytes ? 1 : 0, 1)
}

// Training 70B doesn't fit on 8×H100 (optimizer=840GB > 640GB)
console.log("\n=== Roofline Training: Llama 70B on 8×H100 (doesn't fit) ===")
{
  const model = resolveModel("Llama 3 70B")
  const gpu = resolveGpu("H100")
  const result = rooflineLiteTraining(model, gpu, 8, 8 * gpu.flopPerSecond[16], 8 * gpu.hbmSizeBytes)

  console.log(`  throughput: ${result.throughput.unitsPerSecond} (expected 0, optimizer too large)`)
  assert("70B train 8×H100: throughput = 0", result.throughput.unitsPerSecond, 0)
}

// ===================================================================
// deriveSyncIO
// ===================================================================
console.log("\n=== deriveSyncIO ===")
{
  const none = deriveSyncIO({ mode: "none" })
  assert("none: dIn", none.dIn, 0)
  assert("none: dOut", none.dOut, 0)

  const ckpt = deriveSyncIO({ mode: "checkpoint", bytesOutPerSync: 140e9, tokensPerSync: 1e6 })
  assert("checkpoint: dIn", ckpt.dIn, 0)
  assert("checkpoint: dOut", ckpt.dOut, 140e9 / 1e6)

  const periodic = deriveSyncIO({
    mode: "periodic-updates",
    bytesInPerSync: 100e9,
    bytesOutPerSync: 140e9,
    tokensPerSync: 1e6,
  })
  assert("periodic: dIn", periodic.dIn, 100e9 / 1e6)
  assert("periodic: dOut", periodic.dOut, 140e9 / 1e6)
}

// ===================================================================
console.log(
  `\n${"=".repeat(50)}\nResults: ${passed} passed, ${failed} failed out of ${passed + failed} tests`,
)
if (failed > 0) process.exit(1)
else console.log("All tests passed!")
