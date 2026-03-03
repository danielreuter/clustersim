// Test runner for main.py TypeScript port
// Usage: npx tsx test/test-main.ts

import {
  createCommParams,
  createHardware,
  Model,
  createTrainingConfig,
  reduce_scatter_time_s,
  allgather_time_s,
  allreduce_time_s,
  hierarchical_allreduce_time_s,
  p2p_time_s,
  pipeline_utilization,
  calculate_throughput,
  calculate_training_throughput,
} from "../lib/main";

import * as fs from "fs";
import * as path from "path";

const ref = JSON.parse(
  fs.readFileSync(path.join(__dirname, "reference-main.json"), "utf-8")
);

let passed = 0;
let failed = 0;

function assert(
  name: string,
  actual: number,
  expected: number,
  tolerance: number = 1e-6
) {
  if (expected === 0 && actual === 0) {
    passed++;
    return;
  }
  if (!isFinite(expected) && !isFinite(actual)) {
    passed++;
    return;
  }

  const relErr = Math.abs(actual - expected) / Math.max(Math.abs(expected), 1e-30);
  if (relErr < tolerance) {
    passed++;
  } else {
    console.error(
      `FAIL: ${name} — expected ${expected}, got ${actual} (relErr=${relErr.toExponential(3)})`
    );
    failed++;
  }
}

function assertTuple(
  name: string,
  actual: [number, number],
  expected: [number, number],
  tolerance: number = 1e-6
) {
  assert(`${name}[0]`, actual[0], expected[0], tolerance);
  assert(`${name}[1]`, actual[1], expected[1], tolerance);
}

// ==========================================================================
// Test 1: Communication functions
// ==========================================================================
console.log("\n=== Communication Functions ===");

const c = ref.comm;
assertTuple("reduceScatter.ring8", reduce_scatter_time_s(1e6, 8, 450.0, 1.2), c.reduce_scatter_ring_8);
assertTuple("reduceScatter.tree8", reduce_scatter_time_s(1e6, 8, 450.0, 1.2, "tree"), c.reduce_scatter_tree_8);
assertTuple("allgather.ring8", allgather_time_s(1e6, 8, 450.0, 1.2), c.allgather_ring_8);
assertTuple("allreduce.ring8", allreduce_time_s(1e6, 8, 450.0, 1.2), c.allreduce_ring_8);
assertTuple("allreduce.tree8", allreduce_time_s(1e6, 8, 450.0, 1.2, "tree"), c.allreduce_tree_8);
assertTuple("allreduce.1gpu", allreduce_time_s(1e6, 1, 450.0, 1.2), c.allreduce_1gpu);
assertTuple("p2p.nvlink", p2p_time_s(1e6, 450.0, 2.0), c.p2p_1mb_nvlink);
assertTuple("p2p.ib", p2p_time_s(1e6, 50.0, 10.0), c.p2p_1mb_ib);
assert("pipelineUtil.4_4", pipeline_utilization(4, 4), c.pipeline_util_4_4);
assert("pipelineUtil.8_16", pipeline_utilization(8, 16), c.pipeline_util_8_16);
assert("pipelineUtil.1_1", pipeline_utilization(1, 1), c.pipeline_util_1_1);

// ==========================================================================
// Test 2: Hierarchical allreduce
// ==========================================================================
console.log("\n=== Hierarchical Allreduce ===");

const comm = createCommParams();
const h = ref.hierarchical;
assertTuple("hier.intra4", hierarchical_allreduce_time_s(1e6, 4, 8, comm), h.intra_node_4gpus);
assertTuple("hier.intra8", hierarchical_allreduce_time_s(1e6, 8, 8, comm), h.intra_node_8gpus);
assertTuple("hier.cross16", hierarchical_allreduce_time_s(1e6, 16, 8, comm), h.cross_node_16gpus);
assertTuple("hier.cross64", hierarchical_allreduce_time_s(1e6, 64, 8, comm), h.cross_node_64gpus);

// ==========================================================================
// Test 3: Model properties
// ==========================================================================
console.log("\n=== Model Properties ===");

const model = new Model();
const mp = ref.model_props;
assert("model.weightSizeGb", model.weight_size_gb, mp.weight_size_gb);
assert("model.weightPerLayerGb", model.weight_per_layer_gb, mp.weight_per_layer_gb);
assert("model.largestMatrixMb", model.largest_matrix_mb, mp.largest_matrix_mb);
assert("model.flopsPerToken", model.flops_per_token, mp.flops_per_token);
assert("model.kvCachePerTokenKb", model.kv_cache_per_token_kb, mp.kv_cache_per_token_kb);

// ==========================================================================
// Test 4: Inference throughput
// ==========================================================================
console.log("\n=== Inference Throughput ===");

const hw = createHardware();
const inf = ref.inference;

// Mamba, full HBM
const [tp1, cfg1] = calculate_throughput(hw, model, 1.0, 1.0, null, undefined, 0.5);
assert("inf.mamba_full.throughput", tp1, inf.mamba_full_hbm.throughput, 0.1);

// Transformer 2k, full HBM
const [tp2, cfg2] = calculate_throughput(hw, model, 1.0, 1.0, 2000, undefined, 0.5);
assert("inf.t2k_full.throughput", tp2, inf.transformer_2k_full_hbm.throughput, 0.1);

// Mamba, 5% HBM
const [tp3, cfg3] = calculate_throughput(hw, model, 0.05, 1.0, null, undefined, 0.5);
assert("inf.mamba_5pct.throughput", tp3, inf.mamba_5pct_hbm.throughput, 0.15);

// ==========================================================================
// Test 5: Training throughput
// ==========================================================================
console.log("\n=== Training Throughput ===");

const tr = ref.training;

// Transformer 8k
const trainCfg = createTrainingConfig({ seq_len: 8192, context_length: 8192 });
const [ttp1, tcfg1] = calculate_training_throughput(hw, model, 1.0, 1.0, trainCfg, undefined, 0.5);
assert("train.t8k_full.throughput", ttp1, tr.transformer_8k_full_hbm.throughput, 0.15);

// Mamba 8k
const trainCfgMamba = createTrainingConfig({ seq_len: 8192, context_length: null });
const [ttp2, tcfg2] = calculate_training_throughput(hw, model, 1.0, 1.0, trainCfgMamba, undefined, 0.5);
assert("train.mamba_8k.throughput", ttp2, tr.mamba_8k_full_hbm.throughput, 0.15);

// ==========================================================================
// Results
// ==========================================================================
console.log(`\n${"=".repeat(50)}`);
console.log(`Results: ${passed} passed, ${failed} failed out of ${passed + failed} tests`);
if (failed > 0) {
  process.exit(1);
} else {
  console.log("All tests passed!");
}
