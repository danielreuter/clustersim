export type AuditInputs = {
  requestCount: number
  replayShare: number
  costMultiplier: number
  missProbability: number
  targetBadShare: number
  maxBatchSize: number
  halfBatchUtilization: number
  manualBatchSize: number
}

export type BatchEvaluation = {
  batchSize: number
  utilization: number
  requestCapacity: number
  auditedRequests: number
  batchCount: number
  failuresNeeded: number
  cleanUpperBound: number
  targetDetection: number
  targetExpectedBadRequests: number
}

export type AuditResult = AuditInputs & {
  confidence: number
  utilizationGamma: number
  fullUtilizationCapacity: number
  evaluations: BatchEvaluation[]
  certificateOptimal: BatchEvaluation
  targetOptimal: BatchEvaluation
  manual: BatchEvaluation
  fullBatch: BatchEvaluation
}

const MAX_SUPPORTED_BATCH = 512
const LOG_FACTORIALS = (() => {
  const values = new Array<number>(MAX_SUPPORTED_BATCH + 1).fill(0)
  for (let i = 2; i <= MAX_SUPPORTED_BATCH; i += 1) {
    values[i] = values[i - 1] + Math.log(i)
  }
  return values
})()

function clamp(n: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, n))
}

function logAddExp(a: number, b: number): number {
  if (a === -Infinity) return b
  if (b === -Infinity) return a
  const hi = Math.max(a, b)
  const lo = Math.min(a, b)
  return hi + Math.log1p(Math.exp(lo - hi))
}

function logOneMinusExp(logX: number): number {
  if (logX === -Infinity) return 0
  if (logX >= 0) return -Infinity
  if (logX < Math.log(0.5)) return Math.log1p(-Math.exp(logX))
  return Math.log(-Math.expm1(logX))
}

function logBinomialTerm(n: number, h: number, q: number): number {
  return LOG_FACTORIALS[n]
    - LOG_FACTORIALS[h]
    - LOG_FACTORIALS[n - h]
    + h * Math.log(q)
    + (n - h) * Math.log1p(-q)
}

/** log P[Binomial(n,q) < threshold], evaluated using the smaller tail. */
function logBatchPassProbability(n: number, threshold: number, q: number): number {
  if (threshold <= 0) return -Infinity
  if (threshold > n || q <= 0) return 0
  if (q >= 1) return -Infinity

  const k = threshold - 1
  const mode = Math.floor((n + 1) * q)

  if (k < mode) {
    let logLower = -Infinity
    for (let h = k; h >= 0; h -= 1) {
      const term = logBinomialTerm(n, h, q)
      logLower = logAddExp(logLower, term)
      if (term < logLower - 50) break
    }
    return Math.min(0, logLower)
  }

  let logUpper = -Infinity
  for (let h = threshold; h <= n; h += 1) {
    const term = logBinomialTerm(n, h, q)
    logUpper = logAddExp(logUpper, term)
    if (term < logUpper - 50) break
  }
  return Math.min(0, logOneMinusExp(logUpper))
}

function utilization(batchSize: number, maxBatchSize: number, gamma: number): number {
  if (maxBatchSize <= 1 || gamma <= 0) return 1
  return clamp((batchSize / maxBatchSize) ** gamma, 0, 1)
}

function failuresNeeded(batchSize: number, costMultiplier: number): number {
  return Math.floor(batchSize / costMultiplier) + 1
}

function logCleanProbability(
  batchSize: number,
  batchCount: number,
  costMultiplier: number,
  badShare: number,
): number {
  if (batchCount <= 0) return 0
  const threshold = failuresNeeded(batchSize, costMultiplier)
  const logOneBatchPass = logBatchPassProbability(batchSize, threshold, badShare)
  return logOneBatchPass === -Infinity ? -Infinity : batchCount * logOneBatchPass
}

function detectionProbability(logClean: number): number {
  if (logClean === -Infinity || logClean < -745) return 1
  if (logClean >= 0) return 0
  return clamp(-Math.expm1(logClean), 0, 1)
}

function cleanUpperBound(
  batchSize: number,
  batchCount: number,
  costMultiplier: number,
  missProbability: number,
): number {
  if (batchCount <= 0) return 1
  const threshold = failuresNeeded(batchSize, costMultiplier)
  const target = Math.log(missProbability)

  if (threshold === 1) {
    return clamp(-Math.expm1(target / (batchSize * batchCount)), 0, 1)
  }

  let lo = 0
  let hi = 1
  for (let iteration = 0; iteration < 56; iteration += 1) {
    const mid = (lo + hi) / 2
    const logClean = logCleanProbability(batchSize, batchCount, costMultiplier, mid)
    if (logClean > target) lo = mid
    else hi = mid
  }
  return hi
}

function evaluateBatch(inputs: AuditInputs, batchSize: number, gamma: number): BatchEvaluation {
  const relativeUtilization = utilization(batchSize, inputs.maxBatchSize, gamma)
  const requestCapacity = Math.floor(inputs.requestCount * inputs.replayShare * relativeUtilization)
  const batchCount = Math.floor(requestCapacity / batchSize)
  const auditedRequests = batchCount * batchSize
  const logCleanAtTarget = logCleanProbability(
    batchSize,
    batchCount,
    inputs.costMultiplier,
    inputs.targetBadShare,
  )

  return {
    batchSize,
    utilization: relativeUtilization,
    requestCapacity,
    auditedRequests,
    batchCount,
    failuresNeeded: failuresNeeded(batchSize, inputs.costMultiplier),
    cleanUpperBound: cleanUpperBound(
      batchSize,
      batchCount,
      inputs.costMultiplier,
      inputs.missProbability,
    ),
    targetDetection: detectionProbability(logCleanAtTarget),
    targetExpectedBadRequests: auditedRequests * inputs.targetBadShare,
  }
}

export function computeAudit(raw: AuditInputs): AuditResult {
  const inputs: AuditInputs = {
    requestCount: Math.max(1, Math.round(raw.requestCount)),
    replayShare: clamp(raw.replayShare, 1e-12, 1),
    costMultiplier: Math.max(1.001, raw.costMultiplier),
    missProbability: clamp(raw.missProbability, 1e-9, 0.5),
    targetBadShare: clamp(raw.targetBadShare, 1e-12, 0.999999),
    maxBatchSize: clamp(Math.round(raw.maxBatchSize), 1, MAX_SUPPORTED_BATCH),
    halfBatchUtilization: clamp(raw.halfBatchUtilization, 0.05, 1),
    manualBatchSize: 1,
  }
  inputs.manualBatchSize = clamp(Math.round(raw.manualBatchSize), 1, inputs.maxBatchSize)

  const gamma = inputs.halfBatchUtilization >= 0.999999
    ? 0
    : Math.log(inputs.halfBatchUtilization) / Math.log(0.5)

  const evaluations = Array.from(
    { length: inputs.maxBatchSize },
    (_, index) => evaluateBatch(inputs, index + 1, gamma),
  )

  const certificateOptimal = evaluations.reduce((best, candidate) => {
    if (candidate.cleanUpperBound < best.cleanUpperBound) return candidate
    if (candidate.cleanUpperBound > best.cleanUpperBound) return best
    return candidate.auditedRequests > best.auditedRequests ? candidate : best
  })

  const targetOptimal = evaluations.reduce((best, candidate) => {
    if (candidate.targetDetection > best.targetDetection) return candidate
    if (candidate.targetDetection < best.targetDetection) return best
    return candidate.cleanUpperBound < best.cleanUpperBound ? candidate : best
  })

  return {
    ...inputs,
    confidence: 1 - inputs.missProbability,
    utilizationGamma: gamma,
    fullUtilizationCapacity: Math.floor(inputs.requestCount * inputs.replayShare),
    evaluations,
    certificateOptimal,
    targetOptimal,
    manual: evaluations[inputs.manualBatchSize - 1],
    fullBatch: evaluations[inputs.maxBatchSize - 1],
  }
}
