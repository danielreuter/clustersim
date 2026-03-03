// Test runner for erdil.py TypeScript port
// Usage: npx tsx test/test-erdil.ts

import {
  Model,
  matmulRwBytes,
  Llama_3_70B,
  Llama_3_8B,
  Llama_3_405B,
  DeepSeek_V3,
  GPT_4,
  GPU,
  H100,
  collectiveLatencyNcclSeconds,
  meanCollectiveTimeNcclSeconds,
  newTokenLatencySeconds,
  specDecTokenLatencySeconds,
} from "../lib/erdil";

import * as fs from "fs";
import * as path from "path";

const ref = JSON.parse(
  fs.readFileSync(path.join(__dirname, "reference-erdil.json"), "utf-8")
);

let passed = 0;
let failed = 0;

function assert(
  name: string,
  actual: number,
  expected: number | string,
  tolerance: number = 1e-6
) {
  // Handle "inf" strings from Python JSON
  if (expected === "inf" || expected === "Infinity") {
    if (actual === Infinity || !isFinite(actual)) {
      passed++;
      return;
    }
    console.error(`FAIL: ${name} — expected Infinity, got ${actual}`);
    failed++;
    return;
  }

  const exp = typeof expected === "string" ? parseFloat(expected) : expected;

  if (exp === 0 && actual === 0) {
    passed++;
    return;
  }

  const relErr = Math.abs(actual - exp) / Math.max(Math.abs(exp), 1e-30);
  if (relErr < tolerance) {
    passed++;
  } else {
    console.error(
      `FAIL: ${name} — expected ${exp}, got ${actual} (relErr=${relErr.toExponential(3)})`
    );
    failed++;
  }
}

// ==========================================================================
// Test 1: Model properties
// ==========================================================================
console.log("\n=== Model Properties ===");

const llama70b = ref.model_properties.llama_3_70b;
assert("llama70b.totalParams", Llama_3_70B.totalParams, llama70b.total_params);
assert("llama70b.totalActiveParams", Llama_3_70B.totalActiveParams, llama70b.total_active_params);
assert("llama70b.totalFfParams", Llama_3_70B.totalFfParams, llama70b.total_ff_params);
assert("llama70b.totalAttnParams", Llama_3_70B.totalAttnParams, llama70b.total_attn_params);
assert("llama70b.kvCacheSizePerInputBytes", Llama_3_70B.kvCacheSizePerInputBytes, llama70b.kv_cache_size_per_input_bytes);
assert("llama70b.dHead", Llama_3_70B.dHead, llama70b.d_head);
assert("llama70b.numKvHeads", Llama_3_70B.numKvHeads, llama70b.num_kv_heads);
assert("llama70b.sparsityFactor", Llama_3_70B.sparsityFactor, llama70b.sparsity_factor);
assert("llama70b.embeddingParams", Llama_3_70B.embeddingParams, llama70b.embedding_params);

const dsv3 = ref.model_properties.deepseek_v3;
assert("dsv3.totalParams", DeepSeek_V3.totalParams, dsv3.total_params);
assert("dsv3.totalActiveParams", DeepSeek_V3.totalActiveParams, dsv3.total_active_params);
assert("dsv3.kvCacheSizePerInputBytes", DeepSeek_V3.kvCacheSizePerInputBytes, dsv3.kv_cache_size_per_input_bytes);
assert("dsv3.sparsityFactor", DeepSeek_V3.sparsityFactor, dsv3.sparsity_factor);

const gpt4 = ref.model_properties.gpt_4;
assert("gpt4.totalParams", GPT_4.totalParams, gpt4.total_params);
assert("gpt4.totalActiveParams", GPT_4.totalActiveParams, gpt4.total_active_params);
assert("gpt4.sparsityFactor", GPT_4.sparsityFactor, gpt4.sparsity_factor);

// ==========================================================================
// Test 2: Model compute methods
// ==========================================================================
console.log("\n=== Model Compute ===");

const mc = ref.model_compute;
assert("llama70b.arithmeticCostFlop", Llama_3_70B.arithmeticCostFlop(1000, 32), mc.llama_3_70b_arithmetic_cost);
assert("llama70b.ffdFlop", Llama_3_70B.ffdFlop(32), mc.llama_3_70b_ffd_flop);
assert("llama70b.attnKvFlop", Llama_3_70B.attnKvFlop(1000, 32), mc.llama_3_70b_attn_kv_flop);
assert("llama70b.memoryRwBytes", Llama_3_70B.memoryReadsWritesBytes(1000, 32, 1, 8), mc.llama_3_70b_memory_rw, 1e-4);
assert("dsv3.arithmeticCostFlop", DeepSeek_V3.arithmeticCostFlop(1000, 32), mc.deepseek_v3_arithmetic_cost);
assert("dsv3.memoryRwBytes", DeepSeek_V3.memoryReadsWritesBytes(1000, 32, 1, 8), mc.deepseek_v3_memory_rw, 1e-4);

// ==========================================================================
// Test 3: NCCL collective latency
// ==========================================================================
console.log("\n=== NCCL Latency ===");

const nl = ref.nccl_latency;
assert("nccl.allreduce_8_1_LL", collectiveLatencyNcclSeconds(8, 1, "allreduce", "LL"), nl.allreduce_8_1_LL);
assert("nccl.allreduce_8_1_LL128", collectiveLatencyNcclSeconds(8, 1, "allreduce", "LL128"), nl.allreduce_8_1_LL128);
assert("nccl.allreduce_64_8_LL", collectiveLatencyNcclSeconds(64, 8, "allreduce", "LL"), nl.allreduce_64_8_LL);
assert("nccl.reducescatter_8_1_LL", collectiveLatencyNcclSeconds(8, 1, "reducescatter", "LL"), nl.reducescatter_8_1_LL);

// ==========================================================================
// Test 4: Mean collective time
// ==========================================================================
console.log("\n=== Mean Collective Time ===");

const mct = ref.mean_collective_time;
assert("mct.allreduce_8gpus_1mb", meanCollectiveTimeNcclSeconds(8, 1, 1e6, H100, "allreduce", [1, 1]), mct.allreduce_8gpus_1mb, 1e-4);
assert("mct.allreduce_64gpus_1mb", meanCollectiveTimeNcclSeconds(64, 8, 1e6, H100, "allreduce", [1, 1]), mct.allreduce_64gpus_1mb, 1e-4);
assert("mct.allreduce_8gpus_10mb", meanCollectiveTimeNcclSeconds(8, 1, 1e7, H100, "allreduce", [1, 1]), mct.allreduce_8gpus_10mb, 1e-4);
assert("mct.p2p_2gpus_1mb", meanCollectiveTimeNcclSeconds(2, 1, 1e6, H100, "p2p", [1, 1]), mct.p2p_2gpus_1mb, 1e-4);

// ==========================================================================
// Test 5: Token latency
// ==========================================================================
console.log("\n=== Token Latency ===");

const tl = ref.token_latency;
assert("tl.llama70b_1gpu_bs1", newTokenLatencySeconds(1, Llama_3_70B, H100, 1, 0), tl.llama70b_1gpu_bs1);
assert("tl.llama70b_8gpu_bs32", newTokenLatencySeconds(8, Llama_3_70B, H100, 32, 1000), tl.llama70b_8gpu_bs32, 0.05);
assert("tl.llama70b_8gpu_bs1", newTokenLatencySeconds(8, Llama_3_70B, H100, 1, 0), tl.llama70b_8gpu_bs1, 0.05);
assert("tl.llama405b_64gpu_bs32", newTokenLatencySeconds(64, Llama_3_405B, H100, 32, 4096, 1, true), tl.llama405b_64gpu_bs32, 0.15);
assert("tl.deepseek_128gpu_bs400", newTokenLatencySeconds(128, DeepSeek_V3, H100, 400, 100000, 1, true), tl.deepseek_128gpu_bs400, 0.15);

// ==========================================================================
// Test 6: Speculative decoding
// ==========================================================================
console.log("\n=== Speculative Decoding ===");

const sd = ref.spec_dec;
assert("sd.llama70b_spec_8b", specDecTokenLatencySeconds(8, Llama_3_70B, Llama_3_8B, H100, 32, 0.8, 4, 1000), sd.llama70b_spec_8b, 0.15);

// ==========================================================================
// Test 7: matmul_rw_bytes
// ==========================================================================
console.log("\n=== matmulRwBytes ===");

const mr = ref.matmul_rw;
assert("matmul.basic", matmulRwBytes(1024, 4096, 32), mr.basic);
assert("matmul.with_tp", matmulRwBytes(1024, 4096, 32, 1, 1, 8), mr.with_tp, 1e-4);
assert("matmul.fp8", matmulRwBytes(1024, 4096, 32, 1, 1), mr.fp8);

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
