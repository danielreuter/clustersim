// =============================================================================
// Re-export everything from the main simulator modules
// =============================================================================

// Data structures
export type {
  CommParams,
  Hardware,
  ModelConfig,
  TrainingConfig,
} from "./model";

export {
  createCommParams,
  createHardware,
  Model,
  createTrainingConfig,
} from "./model";

// Communication model
export {
  reduce_scatter_time_s,
  allgather_time_s,
  allreduce_time_s,
  hierarchical_allreduce_time_s,
  p2p_time_s,
  pipeline_utilization,
} from "./comm";

// Inference throughput calculator
export type { InferenceConfig } from "./inference";
export { calculate_throughput } from "./inference";

// Training throughput calculator
export type { TrainingConfigResult, ScenarioData, GenerateDataResult } from "./training";
export { calculate_training_throughput, generate_data } from "./training";
