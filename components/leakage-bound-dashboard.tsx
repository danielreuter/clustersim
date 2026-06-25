"use client"

import { useMemo, useState } from "react"
import { Check, Clipboard, Link as LinkIcon, Plus, Trash2 } from "lucide-react"
import { ThemeToggle } from "@/components/theme-toggle"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"

type AuditCheckState = {
  id: string
  name: string
  overhead: number
  residualShare: number
}

type BoundState = {
  tasksPerEpoch: number
  horizonEpochs: number
  detectionMiss: number
  outputCap: number
  ledgerEntropy: number
  auditShare: number
  checks: AuditCheckState[]
}

type ProcessedCheck = AuditCheckState & {
  order: number
  overheadCost: number
  residualBytes: number
}

type AllocationEvaluation = {
  KBytes: number
  challenges: number[]
  pChecks: number[]
  pBands: number[]
  usedAuditBudget: number
  unusedAuditBudget: number
  uncaughtEntropyBytes: number
  finitePartialBytes: number
  terms: {
    proofPassingTaskEntropyBytes: number
    missedProofFailingTaskBytes: number
    ledgerEntropyBytes: number
  }
  bands: Array<{
    label: string
    entropyBytes: number
    residualAfterBytes: number
    samplingProbability: number
    rawFailuresForDetection: number
    detectionBudgetedFailingOutputs: number
    horizonCappedFailingOutputs: number
    contributionBytes: number
  }>
}

const ZERO_EXP = -12
const DEFAULT_CHECKS: AuditCheckState[] = [
  {
    id: "approximate-replay",
    name: "Approximate proof",
    overhead: Math.log10(500),
    residualShare: 0.1,
  },
  {
    id: "exact-replay",
    name: "Exact proof",
    overhead: Math.log10(500000),
    residualShare: 0,
  },
]

const DEFAULT_STATE: BoundState = {
  tasksPerEpoch: 9,
  horizonEpochs: Math.log10(365),
  detectionMiss: -2,
  outputCap: Math.log10(50e3),
  ledgerEntropy: Math.log10(1e6),
  auditShare: Math.log10(0.01),
  checks: DEFAULT_CHECKS,
}

function clamp(n: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, n))
}

function floorStable(n: number): number {
  const rounded = Math.round(n)
  const tolerance = Math.max(1, Math.abs(n)) * 1e-12
  return Math.abs(n - rounded) <= tolerance ? rounded : Math.floor(n)
}

function ceilStable(n: number): number {
  const rounded = Math.round(n)
  const tolerance = Math.max(1, Math.abs(n)) * 1e-12
  return Math.abs(n - rounded) <= tolerance ? rounded : Math.ceil(n)
}

function countFromLog(exp: number): number {
  return Math.max(1, Math.round(10 ** exp))
}

function bytesFromLog(exp: number): number {
  return Math.max(1, Math.round(10 ** exp))
}

function zeroableBytesFromLog(exp: number): number {
  if (exp <= ZERO_EXP + 1e-9) return 0
  return Math.max(1, Math.round(10 ** exp))
}

function fmtCount(n: number, decimals = 1): string {
  if (n === Infinity) return "infinite"
  if (!Number.isFinite(n)) return "--"
  if (n >= 1e12) return `${(n / 1e12).toFixed(decimals)}T`
  if (n >= 1e9) return `${(n / 1e9).toFixed(decimals)}B`
  if (n >= 1e6) return `${(n / 1e6).toFixed(decimals)}M`
  if (n >= 1e3) return `${(n / 1e3).toFixed(decimals)}K`
  if (n >= 10) return Math.round(n).toLocaleString("en-US")
  if (Number.isInteger(n)) return n.toLocaleString("en-US")
  return n.toFixed(decimals)
}

function fmtBytes(n: number): string {
  if (n === Infinity) return "infinite"
  if (!Number.isFinite(n)) return "--"
  if (n >= 1e18) return `${(n / 1e18).toFixed(2)} EB`
  if (n >= 1e15) return `${(n / 1e15).toFixed(2)} PB`
  if (n >= 1e12) return `${(n / 1e12).toFixed(2)} TB`
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)} GB`
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)} MB`
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)} KB`
  return `${Math.round(n).toLocaleString("en-US")} B`
}

function fmtPct(frac: number): string {
  if (!Number.isFinite(frac)) return "--"
  const pct = frac * 100
  if (pct >= 10) return `${pct.toFixed(1)}%`
  if (pct >= 1) return `${pct.toFixed(2)}%`
  if (pct >= 0.01) return `${pct.toPrecision(2)}%`
  if (pct === 0) return "0%"
  return `${pct.toExponential(1)}%`
}

function fmtRatio(n: number): string {
  if (!Number.isFinite(n)) return "--"
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B times`
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M times`
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K times`
  if (n >= 100) return `${n.toFixed(0)} times`
  if (n >= 10) return `${n.toFixed(1)} times`
  return `${n.toFixed(2)} times`
}

function multiplyBudget(count: number, bytes: number): number {
  if (bytes <= 0) return 0
  if (count === Infinity) return Infinity
  return count * bytes
}

function addTerms(...terms: number[]): number {
  return terms.some(term => term === Infinity) ? Infinity : terms.reduce((sum, term) => sum + term, 0)
}

const PROOF_LAYER_STYLES = [
  {
    bar: "bg-rose-500",
    bg: "bg-rose-500/10",
    border: "border-rose-500/30",
    text: "text-rose-700 dark:text-rose-300",
  },
  {
    bar: "bg-sky-500",
    bg: "bg-sky-500/10",
    border: "border-sky-500/30",
    text: "text-sky-700 dark:text-sky-300",
  },
  {
    bar: "bg-amber-500",
    bg: "bg-amber-500/10",
    border: "border-amber-500/30",
    text: "text-amber-700 dark:text-amber-300",
  },
  {
    bar: "bg-violet-500",
    bg: "bg-violet-500/10",
    border: "border-violet-500/30",
    text: "text-violet-700 dark:text-violet-300",
  },
]

const PASSING_LAYER_STYLE = {
  bar: "bg-blue-500",
  bg: "bg-blue-500/10",
  border: "border-blue-500/30",
  text: "text-blue-700 dark:text-blue-300",
}

const LEDGER_LAYER_STYLE = {
  bar: "bg-emerald-500",
  bg: "bg-emerald-500/10",
  border: "border-emerald-500/30",
  text: "text-emerald-700 dark:text-emerald-300",
}

function lowerFirst(text: string): string {
  return text ? `${text.charAt(0).toLowerCase()}${text.slice(1)}` : text
}

function stripProofSuffix(name: string): string {
  return lowerFirst(name.trim().replace(/\s+proof$/i, ""))
}

function proofBandLabel(checks: ProcessedCheck[], index: number): string {
  const current = lowerFirst(checks[index]?.name ?? `Proof ${index + 1}`)
  if (index === 0) return `Fails ${current}`

  const prior = checks.slice(0, index).map(check => stripProofSuffix(check.name)).filter(Boolean)
  if (prior.length === 1) return `Passes ${prior[0]}, fails ${current}`
  return `Passes ${prior.length} weaker proofs, fails ${current}`
}

function logScalePct(value: number, max: number): number {
  if (value <= 1 || max <= 1) return 0
  return clamp(Math.log10(value) / Math.log10(max), 0, 1) * 100
}

function logTicks(max: number): number[] {
  if (!Number.isFinite(max) || max <= 1) return [1]
  const maxPower = Math.floor(Math.log10(max))
  const ticks = [1]
  for (let power = 3; power <= maxPower; power += 3) {
    ticks.push(10 ** power)
  }
  const roundedMax = Math.round(max)
  if (ticks[ticks.length - 1] !== roundedMax) ticks.push(roundedMax)
  return ticks
}

function encodeState(state: BoundState): string {
  try {
    return btoa(JSON.stringify(state))
  } catch {
    return ""
  }
}

function readNumber(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback
}

function newCheckId() {
  return `check-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`
}

function normalizeCheck(check: Partial<AuditCheckState>, index: number): AuditCheckState {
  const fallback = DEFAULT_CHECKS[index] ?? {
    id: `check-${index + 1}`,
    name: `Proof ${index + 1}`,
    overhead: 3,
    residualShare: 0.01,
  }

  const name = typeof check.name === "string" && check.name.trim()
    ? check.name.trim()
    : fallback.name

  return {
    id: typeof check.id === "string" && check.id ? check.id : fallback.id,
    name,
    overhead: readNumber(check.overhead, fallback.overhead),
    residualShare: clamp(readNumber(check.residualShare, fallback.residualShare), 0, 1),
  }
}

function normalizeChecks(checks: unknown): AuditCheckState[] {
  if (!Array.isArray(checks) || checks.length === 0) return DEFAULT_CHECKS
  const normalized = checks
    .map((check, index) => normalizeCheck(check as Partial<AuditCheckState>, index))
  return normalized.length > 0 ? normalized : DEFAULT_CHECKS
}

function normalizeState(state: BoundState): BoundState {
  return {
    ...state,
    checks: normalizeChecks(state.checks),
  }
}

function residualShareFromLegacy(residualExp: number | undefined, outputExp: number | undefined, fallback: number) {
  if (typeof residualExp !== "number" || typeof outputExp !== "number") return fallback
  return clamp((10 ** residualExp) / Math.max(1, 10 ** outputExp), 0, 1)
}

function decodeState(hash: string): BoundState | null {
  try {
    const raw = hash.startsWith("#") ? hash.slice(1) : hash
    if (!raw) return null
    const parsed = JSON.parse(atob(raw)) as Partial<BoundState> & {
      mode?: "one" | "two"
      n?: number
      beta?: number
      h?: number
      r?: number
      o?: number
      b?: number
      p?: number
      replayOverhead?: number
      hTask?: number
      exactOverhead?: number
      approxOverhead?: number
      hApprox?: number
    }

    if (typeof parsed.tasksPerEpoch === "number") {
      return normalizeState({
        tasksPerEpoch: readNumber(parsed.tasksPerEpoch, DEFAULT_STATE.tasksPerEpoch),
        horizonEpochs: readNumber(parsed.horizonEpochs, DEFAULT_STATE.horizonEpochs),
        detectionMiss: readNumber(parsed.detectionMiss, DEFAULT_STATE.detectionMiss),
        outputCap: readNumber(parsed.outputCap, DEFAULT_STATE.outputCap),
        ledgerEntropy: readNumber(parsed.ledgerEntropy, DEFAULT_STATE.ledgerEntropy),
        auditShare: readNumber(parsed.auditShare, DEFAULT_STATE.auditShare),
        checks: normalizeChecks(parsed.checks),
      })
    }

    if (typeof parsed.mode === "string") {
      const outputCap = readNumber(parsed.outputCap, DEFAULT_STATE.outputCap)
      const checks = parsed.mode === "one"
        ? [
            {
              id: "replay",
              name: "Replay proof",
              overhead: readNumber(parsed.replayOverhead, DEFAULT_STATE.checks[0].overhead),
              residualShare: residualShareFromLegacy(parsed.hTask, outputCap, 0),
            },
          ]
        : [
            {
              id: "approximate-replay",
              name: "Approximate proof",
              overhead: readNumber(parsed.approxOverhead, DEFAULT_CHECKS[0].overhead),
              residualShare: residualShareFromLegacy(parsed.hApprox, outputCap, DEFAULT_CHECKS[0].residualShare),
            },
            {
              id: "exact-replay",
              name: "Exact proof",
              overhead: readNumber(parsed.exactOverhead, DEFAULT_CHECKS[1].overhead),
              residualShare: 0,
            },
          ]
      return normalizeState({
        tasksPerEpoch: readNumber(parsed.tasksPerEpoch, DEFAULT_STATE.tasksPerEpoch),
        horizonEpochs: readNumber(parsed.horizonEpochs, DEFAULT_STATE.horizonEpochs),
        detectionMiss: readNumber(parsed.detectionMiss, DEFAULT_STATE.detectionMiss),
        outputCap,
        ledgerEntropy: readNumber(parsed.ledgerEntropy, DEFAULT_STATE.ledgerEntropy),
        auditShare: readNumber(parsed.auditShare, DEFAULT_STATE.auditShare),
        checks,
      })
    }

    if (
      typeof parsed.n === "number" &&
      typeof parsed.beta === "number" &&
      typeof parsed.h === "number" &&
      typeof parsed.r === "number" &&
      typeof parsed.o === "number"
    ) {
      return normalizeState({
        ...DEFAULT_STATE,
        tasksPerEpoch: parsed.n,
        auditShare: parsed.beta,
        detectionMiss: parsed.r,
        outputCap: parsed.o,
        checks: [
          {
            id: "replay",
            name: "Replay proof",
            overhead: parsed.h,
            residualShare: 0,
          },
        ],
      })
    }

    if (
      typeof parsed.n === "number" &&
      typeof parsed.b === "number" &&
      typeof parsed.h === "number" &&
      typeof parsed.r === "number" &&
      typeof parsed.o === "number"
    ) {
      const legacyBudgetExp = parsed.b
      const auditShare = legacyBudgetExp <= 0
        ? 10 ** legacyBudgetExp
        : (10 ** legacyBudgetExp) / (10 ** parsed.n)
      return normalizeState({
        ...DEFAULT_STATE,
        tasksPerEpoch: parsed.n,
        auditShare: Math.log10(clamp(auditShare, 1e-12, 1)),
        detectionMiss: parsed.r,
        outputCap: parsed.o,
        checks: [
          {
            id: "replay",
            name: "Replay proof",
            overhead: parsed.h,
            residualShare: 0,
          },
        ],
      })
    }

    if (
      typeof parsed.n === "number" &&
      typeof parsed.p === "number" &&
      typeof parsed.r === "number" &&
      typeof parsed.o === "number"
    ) {
      const legacyChallenges = Math.max(0, (10 ** parsed.p) * (10 ** parsed.n))
      const auditOverhead = 10 ** DEFAULT_CHECKS[0].overhead
      const auditShare = (legacyChallenges * auditOverhead) / (10 ** parsed.n)
      return normalizeState({
        ...DEFAULT_STATE,
        tasksPerEpoch: parsed.n,
        auditShare: Math.log10(clamp(auditShare, 1e-12, 1)),
        detectionMiss: parsed.r,
        outputCap: parsed.o,
      })
    }

    return null
  } catch {
    return null
  }
}

function quotaForDetection(p: number, detectionThreshold: number) {
  if (p <= 0) {
    return {
      rawFailuresForDetection: Infinity,
      firstFailuresDetectedAtThreshold: Infinity,
      budgetedFailingOutputs: Infinity,
      approximation: Infinity,
    }
  }
  if (p >= 1) {
    return {
      rawFailuresForDetection: 1,
      firstFailuresDetectedAtThreshold: 1,
      budgetedFailingOutputs: 0,
      approximation: -Math.log1p(-detectionThreshold),
    }
  }
  const rawFailuresForDetection = Math.log1p(-detectionThreshold) / Math.log1p(-p)
  const firstFailuresDetectedAtThreshold = Math.max(1, ceilStable(rawFailuresForDetection))
  return {
    rawFailuresForDetection,
    firstFailuresDetectedAtThreshold,
    budgetedFailingOutputs: Math.max(0, firstFailuresDetectedAtThreshold - 1),
    approximation: -Math.log1p(-detectionThreshold) / p,
  }
}

function processChecks(checks: AuditCheckState[], outputBytes: number): ProcessedCheck[] {
  return checks
    .map((check, order) => ({
      ...check,
      order,
      overheadCost: Math.max(1, 10 ** check.overhead),
      residualBytes: clamp(check.residualShare, 0, 1) * outputBytes,
    }))
    .sort((a, b) => {
      if (b.residualBytes !== a.residualBytes) return b.residualBytes - a.residualBytes
      return a.order - b.order
    })
}

function evaluateAllocation({
  checks,
  challenges,
  tasksPerEpoch,
  detectionThreshold,
  outputBytes,
  taskCount,
  ledgerTermBytes,
  totalAuditBudget,
}: {
  checks: ProcessedCheck[]
  challenges: number[]
  tasksPerEpoch: number
  detectionThreshold: number
  outputBytes: number
  taskCount: number
  ledgerTermBytes: number
  totalAuditBudget: number
}): AllocationEvaluation {
  const pChecks = challenges.map(challengeCount => (
    tasksPerEpoch <= 0 ? 0 : clamp(challengeCount / tasksPerEpoch, 0, 1)
  ))
  const pBands = new Array(checks.length).fill(0)
  let noCheckProbability = 1

  for (let index = checks.length - 1; index >= 0; index -= 1) {
    noCheckProbability *= 1 - pChecks[index]
    pBands[index] = 1 - noCheckProbability
  }

  let previousResidual = outputBytes
  let missedProofFailingTaskBytes = 0
  let uncaughtEntropyBytes = 0
  let finitePartialBytes = 0
  const bands: AllocationEvaluation["bands"] = []

  checks.forEach((check, index) => {
    const bandEntropyBytes = Math.max(0, previousResidual - check.residualBytes)
    const quota = quotaForDetection(pBands[index], detectionThreshold)
    const horizonCappedFailingOutputs = Math.min(taskCount, quota.budgetedFailingOutputs)
    const term = multiplyBudget(horizonCappedFailingOutputs, bandEntropyBytes)

    missedProofFailingTaskBytes = addTerms(missedProofFailingTaskBytes, term)
    if (term === Infinity) {
      uncaughtEntropyBytes += bandEntropyBytes
    } else {
      finitePartialBytes += term
    }
    bands.push({
      label: check.name,
      entropyBytes: bandEntropyBytes,
      residualAfterBytes: check.residualBytes,
      samplingProbability: pBands[index],
      rawFailuresForDetection: quota.rawFailuresForDetection,
      detectionBudgetedFailingOutputs: quota.budgetedFailingOutputs,
      horizonCappedFailingOutputs,
      contributionBytes: term,
    })
    previousResidual = check.residualBytes
  })

  const strongestResidualBytes = checks.length > 0 ? checks[checks.length - 1].residualBytes : outputBytes
  const proofPassingTaskEntropyBytes = taskCount * strongestResidualBytes
  const KBytes = addTerms(proofPassingTaskEntropyBytes, missedProofFailingTaskBytes, ledgerTermBytes)
  const usedAuditBudget = checks.reduce((sum, check, index) => (
    sum + challenges[index] * check.overheadCost
  ), 0)

  return {
    KBytes,
    challenges,
    pChecks,
    pBands,
    usedAuditBudget,
    unusedAuditBudget: Math.max(0, totalAuditBudget - usedAuditBudget),
    uncaughtEntropyBytes,
    finitePartialBytes,
    terms: {
      proofPassingTaskEntropyBytes,
      missedProofFailingTaskBytes,
      ledgerEntropyBytes: ledgerTermBytes,
    },
    bands,
  }
}

function isBetterEvaluation(candidate: AllocationEvaluation, incumbent: AllocationEvaluation | null) {
  if (incumbent == null) return true
  const candidateFinite = Number.isFinite(candidate.KBytes)
  const incumbentFinite = Number.isFinite(incumbent.KBytes)
  if (candidateFinite && !incumbentFinite) return true
  if (!candidateFinite && incumbentFinite) return false
  if (candidateFinite && incumbentFinite) {
    if (candidate.KBytes < incumbent.KBytes) return true
    if (candidate.KBytes > incumbent.KBytes) return false
  } else {
    if (candidate.uncaughtEntropyBytes < incumbent.uncaughtEntropyBytes) return true
    if (candidate.uncaughtEntropyBytes > incumbent.uncaughtEntropyBytes) return false
  }
  if (candidate.finitePartialBytes < incumbent.finitePartialBytes) return true
  if (candidate.finitePartialBytes > incumbent.finitePartialBytes) return false
  return candidate.usedAuditBudget > incumbent.usedAuditBudget
}

function maxChallengesForBudget(tasksPerEpoch: number, budget: number, cost: number) {
  return Math.min(tasksPerEpoch, Math.max(0, floorStable(budget / Math.max(1, cost))))
}

function allocationFromShares(checks: ProcessedCheck[], tasksPerEpoch: number, totalBudget: number, shares: number[]) {
  const totalShare = shares.reduce((sum, share) => sum + Math.max(0, share), 0)
  if (totalShare <= 0) return checks.map(() => 0)
  const allocation = checks.map((check, index) => {
    const budget = totalBudget * (Math.max(0, shares[index] ?? 0) / totalShare)
    return maxChallengesForBudget(tasksPerEpoch, budget, check.overheadCost)
  })
  return clampAllocationToBudget(allocation, checks, totalBudget)
}

function clampAllocationToBudget(challenges: number[], checks: ProcessedCheck[], totalBudget: number) {
  const allocation = challenges.map((challengeCount, index) => (
    Math.min(
      Math.max(0, Math.round(challengeCount)),
      maxChallengesForBudget(Infinity, totalBudget, checks[index].overheadCost)
    )
  ))

  let used = checks.reduce((sum, check, index) => sum + allocation[index] * check.overheadCost, 0)
  while (used > totalBudget) {
    let bestIndex = -1
    let bestCost = -1
    checks.forEach((check, index) => {
      if (allocation[index] > 0 && check.overheadCost > bestCost) {
        bestIndex = index
        bestCost = check.overheadCost
      }
    })
    if (bestIndex < 0) break
    allocation[bestIndex] -= 1
    used -= checks[bestIndex].overheadCost
  }

  return allocation
}

function optimizePair({
  current,
  firstIndex,
  secondIndex,
  checks,
  tasksPerEpoch,
  totalAuditBudget,
  evaluate,
}: {
  current: number[]
  firstIndex: number
  secondIndex: number
  checks: ProcessedCheck[]
  tasksPerEpoch: number
  totalAuditBudget: number
  evaluate: (allocation: number[]) => AllocationEvaluation
}) {
  const fixedSpend = checks.reduce((sum, check, index) => {
    if (index === firstIndex || index === secondIndex) return sum
    return sum + current[index] * check.overheadCost
  }, 0)
  const pairBudget = Math.max(0, totalAuditBudget - fixedSpend)
  const firstCost = checks[firstIndex].overheadCost
  const secondCost = checks[secondIndex].overheadCost
  const maxFirst = Math.min(
    tasksPerEpoch,
    Math.max(0, floorStable(pairBudget / firstCost))
  )
  let bestAllocation = current
  let bestEvaluation = evaluate(current)

  const considerFirstCount = (firstCountRaw: number) => {
    const firstCount = Math.max(0, Math.min(maxFirst, Math.round(firstCountRaw)))
    const remainingBudget = Math.max(0, pairBudget - firstCount * firstCost)
    const secondCount = Math.min(
      tasksPerEpoch,
      Math.max(0, floorStable(remainingBudget / secondCost))
    )
    const candidate = [...current]
    candidate[firstIndex] = firstCount
    candidate[secondIndex] = secondCount
    const candidateEvaluation = evaluate(candidate)
    if (isBetterEvaluation(candidateEvaluation, bestEvaluation)) {
      bestAllocation = candidate
      bestEvaluation = candidateEvaluation
    }
  }

  if (maxFirst <= 50000) {
    for (let firstCount = 0; firstCount <= maxFirst; firstCount += 1) {
      considerFirstCount(firstCount)
    }
    return bestAllocation
  }

  const samples = 240
  for (let sample = 0; sample <= samples; sample += 1) {
    considerFirstCount((maxFirst * sample) / samples)
  }

  const currentBest = bestAllocation[firstIndex]
  const window = Math.max(2000, Math.floor(maxFirst / samples))
  const lo = Math.max(0, currentBest - window)
  const hi = Math.min(maxFirst, currentBest + window)
  for (let firstCount = lo; firstCount <= hi; firstCount += 1) {
    considerFirstCount(firstCount)
  }

  return bestAllocation
}

function optimizeAllocation({
  checks,
  tasksPerEpoch,
  auditShare,
  detectionThreshold,
  outputBytes,
  taskCount,
  ledgerTermBytes,
}: {
  checks: ProcessedCheck[]
  tasksPerEpoch: number
  auditShare: number
  detectionThreshold: number
  outputBytes: number
  taskCount: number
  ledgerTermBytes: number
}) {
  const totalAuditBudget = tasksPerEpoch * auditShare
  const evaluate = (allocation: number[]) => evaluateAllocation({
    checks,
    challenges: clampAllocationToBudget(allocation, checks, totalAuditBudget),
    tasksPerEpoch,
    detectionThreshold,
    outputBytes,
    taskCount,
    ledgerTermBytes,
    totalAuditBudget,
  })

  if (checks.length === 0) {
    return evaluate([])
  }

  if (checks.length === 1) {
    return evaluate([maxChallengesForBudget(tasksPerEpoch, totalAuditBudget, checks[0].overheadCost)])
  }

  const bandWeights = checks.map((check, index) => {
    const previous = index === 0 ? outputBytes : checks[index - 1].residualBytes
    return Math.max(0, previous - check.residualBytes)
  })
  const starts: number[][] = [
    checks.map(() => 0),
    allocationFromShares(checks, tasksPerEpoch, totalAuditBudget, checks.map(() => 1)),
    allocationFromShares(checks, tasksPerEpoch, totalAuditBudget, checks.map(check => 1 / check.overheadCost)),
    allocationFromShares(checks, tasksPerEpoch, totalAuditBudget, bandWeights),
  ]

  checks.forEach((check, index) => {
    const allocation = checks.map(() => 0)
    allocation[index] = maxChallengesForBudget(tasksPerEpoch, totalAuditBudget, check.overheadCost)
    starts.push(allocation)
  })

  let best: AllocationEvaluation | null = null

  starts.forEach(start => {
    let current = clampAllocationToBudget(start, checks, totalAuditBudget)
    let currentEval = evaluate(current)
    let improved = true
    let passes = 0

    while (improved && passes < 8) {
      improved = false
      passes += 1
      for (let firstIndex = 0; firstIndex < checks.length; firstIndex += 1) {
        for (let secondIndex = firstIndex + 1; secondIndex < checks.length; secondIndex += 1) {
          const candidate = optimizePair({
            current,
            firstIndex,
            secondIndex,
            checks,
            tasksPerEpoch,
            totalAuditBudget,
            evaluate,
          })
          const candidateEval = evaluate(candidate)
          if (isBetterEvaluation(candidateEval, currentEval)) {
            current = candidate
            currentEval = candidateEval
            improved = true
          }
        }
      }
    }

    if (isBetterEvaluation(currentEval, best)) {
      best = currentEval
    }
  })

  return best ?? evaluate(checks.map(() => 0))
}

function computeBound(state: BoundState) {
  const tasksPerEpoch = countFromLog(state.tasksPerEpoch)
  const horizonEpochs = countFromLog(state.horizonEpochs)
  const taskCount = tasksPerEpoch * horizonEpochs
  const detectionMiss = clamp(10 ** state.detectionMiss, 1e-6, 0.5)
  const detectionThreshold = 1 - detectionMiss
  const outputBytes = bytesFromLog(state.outputCap)
  const ledgerEntropyBytes = zeroableBytesFromLog(state.ledgerEntropy)
  const auditShare = clamp(10 ** state.auditShare, 0, 1)
  const checks = processChecks(state.checks, outputBytes)
  const ledgerTermBytes = horizonEpochs * ledgerEntropyBytes
  const allocation = optimizeAllocation({
    checks,
    tasksPerEpoch,
    auditShare,
    detectionThreshold,
    outputBytes,
    taskCount,
    ledgerTermBytes,
  })

  return {
    KBytes: allocation.KBytes,
    tasksPerEpoch,
    horizonEpochs,
    taskCount,
    detectionThreshold,
    detectionMiss,
    outputBytes,
    ledgerEntropyBytes,
    auditShare,
    checks,
    allocation: allocation.challenges,
    pChecks: allocation.pChecks,
    pBands: allocation.pBands,
    usedAuditBudget: allocation.usedAuditBudget,
    unusedAuditBudget: allocation.unusedAuditBudget,
    bands: allocation.bands,
    terms: allocation.terms,
    totalAuditBudget: tasksPerEpoch * auditShare,
  }
}

function jsonReplacer(_key: string, value: unknown) {
  if (typeof value === "number" && !Number.isFinite(value)) {
    if (value === Infinity) return "Infinity"
    if (value === -Infinity) return "-Infinity"
    return "NaN"
  }
  return value
}

function InfoTip({ text }: { text: string }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          className="ml-1 inline-flex h-3.5 w-3.5 items-center justify-center rounded-full bg-muted text-[9px] font-medium leading-none text-muted-foreground transition-colors hover:bg-muted-foreground/20"
        >
          i
        </button>
      </TooltipTrigger>
      <TooltipContent side="top" sideOffset={4} className="max-w-xs">
        {text}
      </TooltipContent>
    </Tooltip>
  )
}

function IconButton({
  copied,
  title,
  onClick,
  icon,
}: {
  copied: boolean
  title: string
  onClick: () => void
  icon: "link" | "copy"
}) {
  const Icon = icon === "link" ? LinkIcon : Clipboard
  return (
    <button
      type="button"
      onClick={onClick}
      className="inline-flex h-8 w-8 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
      title={title}
      aria-label={title}
    >
      {copied ? <Check className="h-4 w-4" /> : <Icon className="h-4 w-4" />}
    </button>
  )
}

function LogSlider({
  label,
  tooltip,
  valueExp,
  onValueExpChange,
  min,
  max,
  step,
  formatValue,
}: {
  label: string
  tooltip?: string
  valueExp: number
  onValueExpChange: (v: number) => void
  min: number
  max: number
  step: number
  formatValue: (v: number) => string
}) {
  return (
    <div>
      <div className="flex items-center justify-between gap-3">
        <Label className="text-xs text-muted-foreground">
          {label}
          {tooltip && <InfoTip text={tooltip} />}
        </Label>
        <span className="shrink-0 font-mono text-xs">{formatValue(10 ** valueExp)}</span>
      </div>
      <Slider
        value={[valueExp]}
        onValueChange={([v]) => onValueExpChange(v)}
        min={min}
        max={max}
        step={step}
        className="mt-2"
      />
    </div>
  )
}

function ZeroableByteSlider({
  label,
  tooltip,
  valueExp,
  onValueExpChange,
  max,
}: {
  label: string
  tooltip?: string
  valueExp: number
  onValueExpChange: (v: number) => void
  max: number
}) {
  const value = zeroableBytesFromLog(valueExp)
  return (
    <div>
      <div className="flex items-center justify-between gap-3">
        <Label className="text-xs text-muted-foreground">
          {label}
          {tooltip && <InfoTip text={tooltip} />}
        </Label>
        <span className="shrink-0 font-mono text-xs">{fmtBytes(value)}</span>
      </div>
      <Slider
        value={[valueExp]}
        onValueChange={([v]) => onValueExpChange(v)}
        min={ZERO_EXP}
        max={max}
        step={0.05}
        className="mt-2"
      />
    </div>
  )
}

function PercentSlider({
  label,
  tooltip,
  value,
  onValueChange,
  detail,
}: {
  label: string
  tooltip?: string
  value: number
  onValueChange: (v: number) => void
  detail: string
}) {
  return (
    <div>
      <div className="flex items-center justify-between gap-3">
        <Label className="text-xs text-muted-foreground">
          {label}
          {tooltip && <InfoTip text={tooltip} />}
        </Label>
        <span className="shrink-0 font-mono text-xs">{detail}</span>
      </div>
      <Slider
        value={[value]}
        onValueChange={([v]) => onValueChange(v)}
        min={0}
        max={1}
        step={0.005}
        className="mt-2"
      />
    </div>
  )
}

function CheckEditor({
  check,
  outputBytes,
  canRemove,
  onChange,
  onRemove,
}: {
  check: AuditCheckState
  outputBytes: number
  canRemove: boolean
  onChange: (next: AuditCheckState) => void
  onRemove: () => void
}) {
  const residualBytes = outputBytes * check.residualShare
  return (
    <div className="rounded-lg border border-border bg-background px-3 py-3">
      <div className="mb-4 flex items-center gap-2">
        <input
          value={check.name}
          onChange={event => onChange({ ...check, name: event.target.value })}
          className="h-8 min-w-0 flex-1 rounded-md border border-input bg-background px-2 text-sm outline-none transition-shadow focus-visible:ring-2 focus-visible:ring-ring"
          aria-label="Proof name"
        />
        <button
          type="button"
          onClick={onRemove}
          disabled={!canRemove}
          className="inline-flex h-8 w-8 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-muted hover:text-foreground disabled:pointer-events-none disabled:opacity-30"
          aria-label={`Remove ${check.name}`}
          title="Remove proof"
        >
          <Trash2 className="h-4 w-4" />
        </button>
      </div>
      <div className="space-y-4">
        <LogSlider
          label="Per-task audit overhead"
          tooltip="Compute cost of producing one proof, measured in ordinary task equivalents."
          valueExp={check.overhead}
          onValueExpChange={value => onChange({ ...check, overhead: value })}
          min={0}
          max={9}
          step={0.05}
          formatValue={fmtRatio}
        />
        <PercentSlider
          label="Residual entropy share"
          tooltip="Share of the task output size that can still be chosen after passing this proof."
          value={check.residualShare}
          onValueChange={value => onChange({ ...check, residualShare: clamp(value, 0, 1) })}
          detail={`${fmtPct(check.residualShare)} (${fmtBytes(residualBytes)})`}
        />
      </div>
    </div>
  )
}

export function LeakageBoundDashboard() {
  const initial = typeof window !== "undefined" ? decodeState(window.location.hash) : null
  const [state, setState] = useState<BoundState>(normalizeState(initial ?? DEFAULT_STATE))
  const [copiedJson, setCopiedJson] = useState(false)
  const [copiedLink, setCopiedLink] = useState(false)

  const result = useMemo(() => computeBound(state), [state])

  const setField = <K extends keyof BoundState>(key: K, value: BoundState[K]) => {
    setState(prev => normalizeState({ ...prev, [key]: value }))
  }

  const updateCheck = (id: string, next: AuditCheckState) => {
    setState(prev => normalizeState({
      ...prev,
      checks: prev.checks.map(check => check.id === id ? next : check),
    }))
  }

  const addCheck = () => {
    setState(prev => normalizeState({
      ...prev,
      checks: [
        ...prev.checks,
        {
          id: newCheckId(),
          name: `Proof ${prev.checks.length + 1}`,
          overhead: 3,
          residualShare: 0.01,
        },
      ],
    }))
  }

  const removeCheck = (id: string) => {
    setState(prev => normalizeState({
      ...prev,
      checks: prev.checks.length <= 1 ? prev.checks : prev.checks.filter(check => check.id !== id),
    }))
  }

  const handleCopyJson = () => {
    navigator.clipboard.writeText(JSON.stringify({ input: state, derived: result }, jsonReplacer, 2))
    setCopiedJson(true)
    setTimeout(() => setCopiedJson(false), 1800)
  }

  const handleCopyLink = () => {
    const encoded = encodeState(state)
    const url = `${window.location.origin}${window.location.pathname}#${encoded}`
    navigator.clipboard.writeText(url)
    setCopiedLink(true)
    setTimeout(() => setCopiedLink(false), 1800)
  }

  const strongestResidualBytes = result.checks[result.checks.length - 1]?.residualBytes ?? result.outputBytes
  const proofLayers = result.bands.map((band, index) => ({
    id: `proof-band-${index}`,
    label: proofBandLabel(result.checks, index),
    accountedLabel: `Accounted for by ${stripProofSuffix(band.label)} proof`,
    proofName: band.label,
    style: PROOF_LAYER_STYLES[index % PROOF_LAYER_STYLES.length],
    contributionBytes: band.contributionBytes,
    bytesPerOutput: band.entropyBytes,
    missedOutputs: band.horizonCappedFailingOutputs,
    proofsProduced: (result.allocation[index] ?? 0) * result.horizonEpochs,
    accountedOutputs: Math.max(0, result.taskCount - band.horizonCappedFailingOutputs),
    coveredBytes: Math.max(0, result.taskCount - band.horizonCappedFailingOutputs) * band.entropyBytes,
    residualAfterBytes: band.residualAfterBytes,
    accountedBottomBytes: result.outputBytes - band.residualAfterBytes - band.entropyBytes,
    accountedTopBytes: result.outputBytes - band.residualAfterBytes,
  }))

  const compositionLayers = [
    {
      id: "proof-passing",
      label: "Passes every proof",
      style: PASSING_LAYER_STYLE,
      contributionBytes: result.terms.proofPassingTaskEntropyBytes,
      detail: "Bytes that remain possible even after every configured proof accepts the output.",
    },
    ...proofLayers.map(layer => ({
      id: layer.id,
      label: layer.label,
      style: layer.style,
      contributionBytes: layer.contributionBytes,
      detail: `Bytes from proof-failing outputs below the detection threshold for ${lowerFirst(layer.proofName)}.`,
    })),
    {
      id: "ledger",
      label: "Ledger choices",
      style: LEDGER_LAYER_STYLE,
      contributionBytes: result.terms.ledgerEntropyBytes,
      detail: "Per-epoch ledger or task-choice freedom.",
    },
  ]
  const taskTicks = logTicks(result.taskCount)
  const rectangleYTicks = Array.from(new Set([
    0,
    ...proofLayers.map(layer => Math.round(layer.accountedTopBytes)),
    Math.round(result.outputBytes),
  ])).sort((a, b) => a - b)

  const allocationRows = result.checks.map((check, index) => {
    const proofsPerEpoch = result.allocation[index] ?? 0
    const computeUsed = proofsPerEpoch * check.overheadCost
    return {
      check,
      proofsProduced: proofsPerEpoch * result.horizonEpochs,
      computeShare: result.usedAuditBudget > 0 ? computeUsed / result.usedAuditBudget : 0,
    }
  })

  return (
    <div className="min-h-screen bg-background">
      <div className="mx-auto max-w-5xl px-4 py-6">
        <div className="mb-6 flex items-start justify-between gap-4">
          <div>
            <div className="mb-1.5 text-[11px] font-mono uppercase tracking-[0.22em] text-muted-foreground">
              Interactive · Compartmentalization bound
            </div>
            <h1 className="text-2xl font-bold tracking-tight text-foreground">
              Covert capacity bound
            </h1>
            <p className="mt-1 max-w-2xl text-sm text-muted-foreground">
              Estimate how many bytes of dishonest communication fit below a selected detection threshold.
            </p>
          </div>
          <div className="flex shrink-0 items-center gap-1">
            <IconButton copied={copiedLink} title="Copy shareable link" onClick={handleCopyLink} icon="link" />
            <IconButton copied={copiedJson} title="Copy input and derived JSON" onClick={handleCopyJson} icon="copy" />
            <ThemeToggle />
          </div>
        </div>

        <Card className="mb-4 overflow-hidden border-foreground/20">
          <CardContent className="pt-6">
            <div>
              <div className="text-sm font-medium text-foreground">
                Dishonest communication below detection threshold
              </div>
              <div className="mt-2 font-mono text-5xl font-bold tracking-tight md:text-6xl">
                {fmtBytes(result.KBytes)}
              </div>
              <div className="mt-2 text-sm text-muted-foreground">
                Over {fmtCount(result.horizonEpochs, 1)} epochs at {fmtPct(result.detectionThreshold)} detection.
              </div>
            </div>

            <div className="mt-5 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
              {compositionLayers.map(layer => (
                <div
                  key={layer.id}
                  className={`min-h-[116px] rounded-lg border px-3 py-3 ${layer.style.border} ${layer.style.bg}`}
                >
                  <div className="flex items-center gap-1.5 text-xs font-medium text-foreground">
                    <span className={`h-2.5 w-2.5 shrink-0 rounded-sm ${layer.style.bar}`} />
                    <span className="min-w-0 truncate">{layer.label}</span>
                    <InfoTip text={layer.detail} />
                  </div>
                  <div className={`mt-4 font-mono text-2xl font-semibold leading-none ${layer.style.text}`}>
                    {fmtBytes(layer.contributionBytes)}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card className="mb-4">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-semibold">Proof coverage</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <div className="min-w-[780px]">
                <div className="grid grid-cols-[6.25rem_1fr] gap-3">
                  <div className="relative h-72 overflow-visible">
                    {rectangleYTicks.map(tick => {
                      const pct = (tick / result.outputBytes) * 100
                      const positionClass = pct >= 99
                        ? "top-0"
                        : pct <= 1
                          ? "bottom-0"
                          : "translate-y-1/2"
                      const positionStyle = pct >= 99
                        ? { top: 0 }
                        : pct <= 1
                          ? { bottom: 0 }
                          : { bottom: `${pct}%` }
                      return (
                        <div
                          key={`y-${tick}`}
                          className={`absolute right-0 font-mono text-[10px] text-muted-foreground ${positionClass}`}
                          style={positionStyle}
                        >
                          {fmtBytes(tick)}
                        </div>
                      )
                    })}
                  </div>

                  <div>
                    <div className="relative h-72 overflow-hidden rounded-lg border border-border bg-muted/30">
                      {taskTicks.map(tick => (
                        <div
                          key={`grid-${tick}`}
                          className="absolute bottom-0 top-0 border-l border-border/60"
                          style={{ left: `${logScalePct(tick, result.taskCount)}%` }}
                        />
                      ))}

                      {strongestResidualBytes > 0 && (
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <div
                              className={`absolute left-0 ${PASSING_LAYER_STYLE.bar} opacity-70`}
                              style={{
                                bottom: `${((result.outputBytes - strongestResidualBytes) / result.outputBytes) * 100}%`,
                                height: `${(strongestResidualBytes / result.outputBytes) * 100}%`,
                                width: "100%",
                              }}
                            />
                          </TooltipTrigger>
                          <TooltipContent side="top" sideOffset={6} className="max-w-xs">
                            <div className="text-xs font-medium">Passes every proof</div>
                            <div className="mt-1 text-xs text-muted-foreground">
                              {fmtBytes(strongestResidualBytes)} across {fmtCount(result.taskCount, 1)} outputs.
                            </div>
                          </TooltipContent>
                        </Tooltip>
                      )}

                      {proofLayers.map(layer => {
                        const bottom = (Math.max(0, layer.accountedBottomBytes) / result.outputBytes) * 100
                        const height = (layer.bytesPerOutput / result.outputBytes) * 100
                        const boundary = logScalePct(layer.missedOutputs, result.taskCount)
                        const width = Math.max(0, 100 - boundary)
                        return (
                          <div key={`${layer.id}-rectangle`}>
                            {boundary > 0 && (
                              <Tooltip>
                                <TooltipTrigger asChild>
                                  <div
                                    className={`absolute left-0 ${layer.style.bg} ring-1 ring-background/40`}
                                    style={{
                                      bottom: `${bottom}%`,
                                      height: `${height}%`,
                                      width: `${boundary}%`,
                                      minWidth: layer.missedOutputs > 0 ? 2 : 0,
                                      minHeight: layer.bytesPerOutput > 0 ? 2 : 0,
                                    }}
                                  >
                                    {height >= 8 && boundary >= 10 && (
                                      <div className={`flex h-full items-center justify-center px-2 font-mono text-xs font-semibold ${layer.style.text}`}>
                                        {fmtBytes(layer.contributionBytes)}
                                      </div>
                                    )}
                                  </div>
                                </TooltipTrigger>
                                <TooltipContent side="top" sideOffset={6} className="max-w-xs">
                                  <div className="text-xs font-medium">Proof-failing outputs below detection threshold</div>
                                  <div className="mt-1 font-mono text-xs">{fmtBytes(layer.contributionBytes)}</div>
                                  <div className="mt-1 text-xs text-muted-foreground">
                                    {fmtCount(layer.missedOutputs, 1)} outputs in {layer.label}
                                  </div>
                                </TooltipContent>
                              </Tooltip>
                            )}
                            <Tooltip>
                              <TooltipTrigger asChild>
                                <div
                                  className={`absolute left-0 ${layer.style.bar} opacity-80 ring-1 ring-background/40`}
                                  style={{
                                    left: `${boundary}%`,
                                    bottom: `${bottom}%`,
                                    height: `${height}%`,
                                    width: `${width}%`,
                                    minWidth: width > 0 ? 2 : 0,
                                    minHeight: layer.bytesPerOutput > 0 ? 2 : 0,
                                  }}
                                >
                                  {height >= 8 && width >= 14 && (
                                    <div className="flex h-full items-center justify-center px-2 font-mono text-xs font-semibold text-white">
                                      {fmtBytes(layer.coveredBytes)}
                                    </div>
                                  )}
                                </div>
                              </TooltipTrigger>
                              <TooltipContent side="top" sideOffset={6} className="max-w-xs">
                                <div className="text-xs font-medium">{layer.accountedLabel}</div>
                                <div className="mt-1 font-mono text-xs">{fmtBytes(layer.coveredBytes)}</div>
                                <div className="mt-1 text-xs text-muted-foreground">
                                  boundary: {fmtCount(layer.missedOutputs, 1)} proof-failing outputs below threshold
                                </div>
                              </TooltipContent>
                            </Tooltip>
                          </div>
                        )
                      })}
                    </div>

                    <div className="relative mt-2 h-10">
                      {taskTicks.map(tick => (
                        <div
                          key={`axis-${tick}`}
                          className="absolute top-0 -translate-x-1/2 text-center"
                          style={{ left: `${logScalePct(tick, result.taskCount)}%` }}
                        >
                          <div className="mx-auto h-2 border-l border-border" />
                          <div className="mt-1 whitespace-nowrap font-mono text-[10px] text-muted-foreground">
                            {fmtCount(tick, 1)}
                          </div>
                        </div>
                      ))}
                      <div className="absolute bottom-0 left-0 right-0 text-center text-[10px] font-mono uppercase tracking-wider text-muted-foreground">
                        proof-failing outputs below detection threshold, log scale
                      </div>
                    </div>
                  </div>
                </div>

                <div className="mt-3 flex flex-wrap gap-2">
                  {proofLayers.map(layer => (
                    <div
                      key={`${layer.id}-legend`}
                      className="inline-flex items-center gap-1.5 rounded-md border border-border bg-background px-2 py-1 text-xs"
                    >
                      <span className={`h-2.5 w-2.5 rounded-sm ${layer.style.bar}`} />
                      <span className="font-medium">{layer.accountedLabel}</span>
                      <span className="font-mono text-muted-foreground">{fmtBytes(layer.coveredBytes)}</span>
                    </div>
                  ))}
                  {strongestResidualBytes > 0 && (
                    <div className="inline-flex items-center gap-1.5 rounded-md border border-border bg-background px-2 py-1 text-xs">
                      <span className={`h-2.5 w-2.5 rounded-sm ${PASSING_LAYER_STYLE.bar}`} />
                      <span className="font-medium">Passes every proof</span>
                      <span className="font-mono text-muted-foreground">{fmtBytes(strongestResidualBytes)}</span>
                    </div>
                  )}
                  <div className="inline-flex items-center gap-1.5 rounded-md border border-border bg-background px-2 py-1 text-xs">
                    <span className="h-2.5 w-2.5 rounded-sm bg-muted" />
                    <span className="font-medium">Proof-failing outputs below detection threshold</span>
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        <div className="mt-4 grid grid-cols-1 gap-4 md:grid-cols-2">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-semibold">Shared parameters</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <LogSlider
                label="Tasks per epoch"
                tooltip="Declared task outputs eligible for proof sampling in one epoch."
                valueExp={state.tasksPerEpoch}
                onValueExpChange={value => setField("tasksPerEpoch", value)}
                min={3}
                max={12}
                step={0.05}
                formatValue={value => fmtCount(Math.round(value), 1)}
              />
              <LogSlider
                label="Number of epochs"
                tooltip="Number of epochs included in the cumulative security bound."
                valueExp={state.horizonEpochs}
                onValueExpChange={value => setField("horizonEpochs", value)}
                min={0}
                max={4}
                step={0.01}
                formatValue={value => `${fmtCount(Math.round(value), 1)} epochs`}
              />
              <LogSlider
                label="Detection threshold"
                tooltip="Probability of catching the developer once failing outputs exceed the bound."
                valueExp={state.detectionMiss}
                onValueExpChange={value => setField("detectionMiss", value)}
                min={-6}
                max={Math.log10(0.5)}
                step={0.01}
                formatValue={miss => `${fmtPct(1 - miss)} detection`}
              />
              <LogSlider
                label="Task output size limit"
                tooltip="Maximum dishonest payload in one replay-failing task output."
                valueExp={state.outputCap}
                onValueExpChange={value => setField("outputCap", value)}
                min={0}
                max={12}
                step={0.05}
                formatValue={fmtBytes}
              />
              <ZeroableByteSlider
                label="Ledger entropy per epoch"
                tooltip="Entropy from task choice, ledger freedom, or other per-epoch degrees of freedom."
                valueExp={state.ledgerEntropy}
                onValueExpChange={value => setField("ledgerEntropy", value)}
                max={12}
              />
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between gap-3">
                <CardTitle className="text-sm font-semibold">Proof parameters</CardTitle>
                <button
                  type="button"
                  onClick={addCheck}
                  className="inline-flex h-8 items-center gap-1.5 rounded-md border border-border px-2 text-xs font-medium text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
                >
                  <Plus className="h-3.5 w-3.5" />
                  Add proof
                </button>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              <LogSlider
                label="Total audit compute share"
                tooltip="Share of epoch task-equivalent compute reserved for proofs."
                valueExp={state.auditShare}
                onValueExpChange={value => setField("auditShare", value)}
                min={-8}
                max={0}
                step={0.05}
                formatValue={fmtPct}
              />
              <div className="space-y-3">
                {state.checks.map(check => (
                  <CheckEditor
                    key={check.id}
                    check={check}
                    outputBytes={result.outputBytes}
                    canRemove={state.checks.length > 1}
                    onChange={next => updateCheck(check.id, next)}
                    onRemove={() => removeCheck(check.id)}
                  />
                ))}
              </div>
              <div className="rounded-lg border border-border bg-background px-3 py-3">
                <div className="text-xs font-medium">Compute allocation</div>
                <div className="mt-1 text-xs text-muted-foreground">
                  Rows show allocated audit compute and total proofs over the accounting horizon.
                </div>
                <div className="mt-3 space-y-2">
                  {allocationRows.map(row => (
                    <div key={row.check.id} className="rounded-md bg-muted/40 px-2 py-2 text-xs">
                      <div className="grid grid-cols-[1fr_auto_auto] items-center gap-3">
                        <div className="font-medium">{row.check.name}</div>
                        <div className="font-mono text-muted-foreground">{fmtPct(row.computeShare)}</div>
                        <div className="min-w-[7rem] text-right font-mono">
                          {fmtCount(row.proofsProduced, 1)} proofs
                        </div>
                      </div>
                      <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-background">
                        <div
                          className="h-full rounded-full bg-foreground/65"
                          style={{ width: `${clamp(row.computeShare, 0, 1) * 100}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        <p className="mt-5 max-w-3xl text-xs text-muted-foreground">
          All capacity terms are measured in bytes. Residual entropy is set as a share of the task
          output size limit. The optimizer assigns integer proofs per epoch under the total audit
          compute budget.
        </p>
      </div>
    </div>
  )
}
