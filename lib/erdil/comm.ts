// ---------------------------------------------------------------------------
// Communication helpers – re-exports from gpus.ts
// ---------------------------------------------------------------------------
//
// All collective-time functions live in gpus.ts next to the GPU class.
// This module re-exports them for consumers that prefer an explicit
// "comm" import path.
// ---------------------------------------------------------------------------

export {
  collectiveLatencyNcclSeconds,
  meanCollectiveTimeNcclSeconds,
  tpuCollectiveTimeSeconds,
} from "./gpus";
export type { CollectiveTimeFn } from "./gpus";
