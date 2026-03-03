// ---------------------------------------------------------------------------
// lib/erdil – GPU cluster inference throughput simulator
// Based on the Erdil 2025 paper
// ---------------------------------------------------------------------------

// GPU definitions and collective-time functions
export {
  GPU,
  collectiveLatencyNcclSeconds,
  meanCollectiveTimeNcclSeconds,
  tpuCollectiveTimeSeconds,
  H100,
  H800,
  H20,
  H100_ZL,
  H200,
  A100,
  V100,
  TPU_v4,
  Groq_LPU,
  GPU_MAP,
} from "./gpus";
export type { CollectiveTimeFn } from "./gpus";

// Model definitions
export {
  Model,
  matmulRwBytes,
  scaleModel,
  GPT_4,
  GPT_3_5,
  GPT_3,
  PaLM_540B,
  PaLM_8B,
  Falcon,
  GPT_J_6B,
  Mixtral_8x22B,
  Mixtral_8x22B_MQA,
  Mixtral_8x7B,
  Mistral_Large_2,
  Llama_3_8B,
  Llama_3_8B_MQA,
  Llama_3_8B_8_bit,
  Llama_3_70B,
  Llama_3_70B_MQA,
  Llama_3_70B_8_bit,
  Llama_3_70B_4_bit,
  Llama_3_405B,
  Llama_3_405B_8_bit,
  DeepSeek_V3,
  GPT_4_100T_param,
  GPT_4_2x,
  GPT_4_4x,
  GPT_4_8x,
  GPT_4_16x,
  Llama_3_gpt4_size,
  Llama_3_gpt4_size_2x,
  Llama_3_gpt4_size_4x,
  Llama_3_gpt4_size_8x,
  Llama_3_gpt4_size_16x,
  MODEL_MAP,
} from "./models";

// Communication helpers (re-exports)
export {
  collectiveLatencyNcclSeconds as commCollectiveLatency,
  meanCollectiveTimeNcclSeconds as commMeanCollectiveTime,
  tpuCollectiveTimeSeconds as commTpuCollectiveTime,
} from "./comm";

// Core token latency and throughput functions
export {
  ffRwBytes,
  attnRwBytes,
  tpLayout,
  tokenLatencySecondsAsPresentedInPaper,
  finalTokenLatencySeconds,
  newTokenLatencySeconds,
  tokenLatencySecondsDefault,
  tokenLatencyWithGammaBreakdown,
  scaledGpu,
  maximizeClusterThroughput,
} from "./model";
export type {
  LatencyBreakdown,
  ClusterThroughputResult,
} from "./model";

// Speculative decoding
export {
  specDecTokenLatencySeconds,
  specDecMfuTargetOpt,
  searchForModel,
} from "./spec-dec";

// Token economics / Pareto front analysis
export {
  TokenEconSettings,
  ComparisonSettings,
  llamaComparison,
  allModelsComparison,
  allModelsComparisonWithSpecDec,
  gpusComparison,
  gpusLongContextComparison,
  contextLengthComparison,
  quantizationComparison,
  specDecComparison,
  mistralLargeComparison,
  deepseekV3ContextLenComparison,
  gpt4ComparisonWithSpecDec,
  gpt4LlamaComparisonWithSpecDec,
  gpt4ComparisonWithSpecDecLongContext,
  paretoFronts,
  userPreferenceIntensity,
  preferenceMaximizingSettings,
} from "./token-economics";
export type {
  TokenLatencyFn,
  ParetoFrontResult,
  PreferenceResult,
} from "./token-economics";
