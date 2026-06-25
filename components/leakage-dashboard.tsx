"use client"

import { useMemo, useState } from "react"
import { Check, Clipboard, Link as LinkIcon } from "lucide-react"
import { ThemeToggle } from "@/components/theme-toggle"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"

type LeakageState = {
  // Policy parameters
  n: number
  beta: number
  h: number
  sLimit: number
  r: number
  o: number
  // Attack parameters
  work: number
  carriedState: number
  dirtyStride: number
}

type StrategyName = "clean" | "dirty" | "none"

const DEFAULT_STATE: LeakageState = {
  n: 9,
  beta: Math.log10(0.005),
  h: 6,
  sLimit: Math.log10(500e3),
  r: -2,
  o: Math.log10(50e3),
  work: 9,
  carriedState: Math.log10(500e3),
  dirtyStride: 6,
}

const SURVIVAL_PRESETS = [0.5, 0.9, 0.95, 0.99, 0.999]
const RISK_MAX = Math.log10(0.5)

function encodeState(s: LeakageState): string {
  try {
    return btoa(JSON.stringify(s))
  } catch {
    return ""
  }
}

function decodeState(hash: string): LeakageState | null {
  try {
    const raw = hash.startsWith("#") ? hash.slice(1) : hash
    if (!raw) return null
    const parsed = JSON.parse(atob(raw)) as Partial<LeakageState> & {
      cleanState?: number
      dirtyState?: number
      b?: number
      mc?: number
      md?: number
      p?: number
    }
    if (
      typeof parsed.n === "number" &&
      typeof parsed.beta === "number" &&
      typeof parsed.h === "number" &&
      typeof parsed.sLimit === "number" &&
      typeof parsed.r === "number" &&
      typeof parsed.o === "number" &&
      typeof parsed.work === "number" &&
      typeof parsed.carriedState === "number" &&
      typeof parsed.dirtyStride === "number"
    ) {
      return {
        n: parsed.n,
        beta: parsed.beta,
        h: parsed.h,
        sLimit: parsed.sLimit,
        r: parsed.r,
        o: parsed.o,
        work: parsed.work,
        carriedState: parsed.carriedState,
        dirtyStride: parsed.dirtyStride,
      }
    }
    if (
      typeof parsed.n === "number" &&
      typeof parsed.beta === "number" &&
      typeof parsed.h === "number" &&
      typeof parsed.r === "number" &&
      typeof parsed.o === "number" &&
      typeof parsed.cleanState === "number" &&
      typeof parsed.dirtyState === "number"
    ) {
      const carriedState = Math.max(parsed.cleanState, parsed.dirtyState)
      return {
        n: parsed.n,
        beta: parsed.beta,
        h: parsed.h,
        sLimit: typeof parsed.sLimit === "number" ? parsed.sLimit : DEFAULT_STATE.sLimit,
        r: parsed.r,
        o: parsed.o,
        work: typeof parsed.work === "number" ? parsed.work : DEFAULT_STATE.work,
        carriedState,
        dirtyStride: typeof parsed.dirtyStride === "number" ? parsed.dirtyStride : DEFAULT_STATE.dirtyStride,
      }
    }
    if (
      typeof parsed.n === "number" &&
      typeof parsed.b === "number" &&
      typeof parsed.h === "number" &&
      typeof parsed.r === "number" &&
      typeof parsed.o === "number" &&
      typeof parsed.mc === "number" &&
      typeof parsed.md === "number"
    ) {
      const legacyBudgetExp = parsed.b
      const carriedState = Math.max(parsed.mc, parsed.md)
      const beta = legacyBudgetExp <= 0 ? 10 ** legacyBudgetExp : (10 ** legacyBudgetExp) / (10 ** parsed.n)
      return {
        n: parsed.n,
        beta: Math.log10(clamp(beta, 1e-9, 1)),
        h: parsed.h,
        sLimit: DEFAULT_STATE.sLimit,
        r: parsed.r,
        o: parsed.o,
        work: DEFAULT_STATE.work,
        carriedState,
        dirtyStride: DEFAULT_STATE.dirtyStride,
      }
    }
    if (
      typeof parsed.n === "number" &&
      typeof parsed.p === "number" &&
      typeof parsed.r === "number" &&
      typeof parsed.o === "number" &&
      typeof parsed.mc === "number" &&
      typeof parsed.md === "number"
    ) {
      const legacyP = parsed.p
      const carriedState = Math.max(parsed.mc, parsed.md)
      const legacyChallenges = Math.max(0, (10 ** legacyP) * (10 ** parsed.n))
      const auditOverhead = 10 ** DEFAULT_STATE.h
      const beta = (legacyChallenges * auditOverhead) / (10 ** parsed.n)
      return {
        n: parsed.n,
        beta: Math.log10(clamp(beta, 1e-9, 1)),
        h: DEFAULT_STATE.h,
        sLimit: DEFAULT_STATE.sLimit,
        r: parsed.r,
        o: parsed.o,
        work: DEFAULT_STATE.work,
        carriedState,
        dirtyStride: DEFAULT_STATE.dirtyStride,
      }
    }
    return null
  } catch {
    return null
  }
}

function fmtBytes(n: number): string {
  if (!Number.isFinite(n)) return "--"
  if (n >= 1e12) return `${(n / 1e12).toFixed(1)} TB`
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)} GB`
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)} MB`
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)} KB`
  return `${Math.round(n).toLocaleString("en-US")} B`
}

function fmtCount(n: number, decimals = 1): string {
  if (!Number.isFinite(n)) return "--"
  if (n >= 1e12) return `${(n / 1e12).toFixed(decimals)}T`
  if (n >= 1e9) return `${(n / 1e9).toFixed(decimals)}B`
  if (n >= 1e6) return `${(n / 1e6).toFixed(decimals)}M`
  if (n >= 1e3) return `${(n / 1e3).toFixed(decimals)}K`
  if (n >= 10) return Math.round(n).toLocaleString("en-US")
  return n.toFixed(decimals)
}

function fmtRatio(n: number): string {
  if (!Number.isFinite(n)) return "--"
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M x`
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K x`
  if (n >= 100) return `${n.toFixed(0)}x`
  if (n >= 10) return `${n.toFixed(1)}x`
  if (n >= 1) return `${n.toFixed(2)}x`
  return `${n.toPrecision(2)}x`
}

function fmtSlowdown(n: number | null): string {
  if (n == null || !Number.isFinite(n)) return "∞×"
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M×`
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K×`
  if (n >= 100) return `${n.toFixed(0)}×`
  if (n >= 10) return `${n.toFixed(1)}×`
  return `${n.toFixed(2)}×`
}

function fmtPct(frac: number): string {
  if (!Number.isFinite(frac)) return "--"
  const p = frac * 100
  if (p >= 10) return `${p.toFixed(1)}%`
  if (p >= 1) return `${p.toFixed(2)}%`
  if (p >= 0.01) return `${p.toPrecision(2)}%`
  if (p === 0) return "0%"
  return `${p.toExponential(1)}%`
}

function fmtEpochs(n: number | null): string {
  if (n == null || !Number.isFinite(n)) return "∞ epochs"
  if (n >= 1000) return `${fmtCount(n, 1)} epochs`
  if (n >= 10) return `${n.toFixed(1)} epochs`
  if (n >= 1) return `${n.toFixed(2)} epochs`
  if (n === 0) return "0 epochs"
  if (n >= 0.0001) return `${n.toPrecision(2)} epochs`
  return `${n.toExponential(2)} epochs`
}

function clamp(n: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, n))
}

function floorStable(n: number): number {
  const rounded = Math.round(n)
  const tolerance = Math.max(1, Math.abs(n)) * 1e-12
  return Math.abs(n - rounded) <= tolerance ? rounded : Math.floor(n)
}

function dirtyCommitOutputsForWork(work: number, stride: number, outputsPerCommit: number): number {
  if (work <= 0) return 0
  return Math.ceil(work / stride) * outputsPerCommit
}

function maxDirtyCommitsForBudget(budget: number, outputsPerCommit: number): number {
  if (!Number.isFinite(budget)) return Infinity
  if (outputsPerCommit <= 0) return Infinity
  return Math.floor(budget / outputsPerCommit)
}

function currentQ(riskExp: number): number {
  const risk = clamp(10 ** riskExp, 1e-4, 0.5)
  return 1 - risk
}

function computeForQ(state: LeakageState, q: number) {
  const N = Math.max(1, Math.round(10 ** state.n))
  const auditShare = clamp(10 ** state.beta, 0, 1)
  const auditBudget = auditShare * N
  const auditOverhead = Math.max(1, 10 ** state.h)
  const maxChallenges = Math.min(N, floorStable(auditBudget / auditOverhead))
  const p = maxChallenges / N
  const o = Math.max(1, Math.round(10 ** state.o))
  const sLimit = Math.max(1, Math.round(10 ** state.sLimit))
  const totalWork = Math.max(1, Math.round(10 ** state.work))
  const carriedState = Math.max(1, Math.round(10 ** state.carriedState))
  const dirtyStride = Math.max(1, Math.round(10 ** state.dirtyStride))
  const kRaw = p <= 0 ? Infinity : Math.log(q) / Math.log1p(-p)
  const kWhole = Math.max(0, Math.floor(kRaw))
  const epsilon = Math.min(1, kRaw / N)
  const s = p * N
  const epsilonApprox = s <= 0 ? Infinity : Math.min(1, -Math.log(q) / s)
  const dirtyExchangeBudgetBytes = kWhole * o

  const cleanOutputsPerTask = Math.max(1, Math.ceil(carriedState / o))
  const cleanCandidateOutputs = totalWork * cleanOutputsPerTask
  const cleanWorks = carriedState <= sLimit
  const cleanStrategy = {
    name: "clean" as const,
    works: cleanWorks,
    slowdown: cleanWorks ? cleanOutputsPerTask : null,
    candidateSlowdown: cleanOutputsPerTask,
    totalOutputs: cleanCandidateOutputs,
    epochs: cleanWorks ? cleanCandidateOutputs / N : null,
    outputsPerTask: cleanOutputsPerTask,
    overheadOutputs: totalWork * Math.max(0, cleanOutputsPerTask - 1),
    reason: cleanWorks
      ? `carried state fits S (${fmtBytes(carriedState)} <= ${fmtBytes(sLimit)})`
      : `carried state exceeds S (${fmtBytes(carriedState)} > ${fmtBytes(sLimit)})`,
  }

  const dirtyOutputsPerCommit = Math.max(1, Math.ceil(carriedState / o))
  const dirtyCommits = Math.ceil(totalWork / dirtyStride)
  const dirtyRiskyOutputs = dirtyCommitOutputsForWork(totalWork, dirtyStride, dirtyOutputsPerCommit)
  const dirtySurvival = p <= 0 ? 1 : Math.exp(dirtyRiskyOutputs * Math.log1p(-p))
  const dirtyWorks = dirtySurvival >= q
  const dirtyCandidateOutputs = totalWork + dirtyRiskyOutputs
  const dirtyCandidateSlowdown = dirtyCandidateOutputs / totalWork
  const maxSafeDirtyCommits = maxDirtyCommitsForBudget(kWhole, dirtyOutputsPerCommit)
  const qSafeDirtyCapacity = Number.isFinite(maxSafeDirtyCommits)
    ? Math.min(totalWork, maxSafeDirtyCommits * dirtyStride)
    : totalWork
  const dirtyStrategy = {
    name: "dirty" as const,
    works: dirtyWorks,
    slowdown: dirtyWorks ? dirtyCandidateSlowdown : null,
    candidateSlowdown: dirtyCandidateSlowdown,
    totalOutputs: dirtyCandidateOutputs,
    epochs: dirtyWorks ? dirtyCandidateOutputs / N : null,
    stride: dirtyStride,
    commits: dirtyCommits,
    outputsPerCommit: dirtyOutputsPerCommit,
    riskyOutputs: dirtyRiskyOutputs,
    survival: dirtySurvival,
    maxSafeCommits: maxSafeDirtyCommits,
    qSafeCapacity: qSafeDirtyCapacity,
    referenceFits: qSafeDirtyCapacity >= totalWork,
    reason: dirtyWorks
      ? `survival meets q (${fmtPct(dirtySurvival)} >= ${fmtPct(q)})`
      : `survival below q (${fmtPct(dirtySurvival)} < ${fmtPct(q)})`,
  }

  const bestStrategy = (() => {
    if (cleanStrategy.works && dirtyStrategy.works) {
      return cleanStrategy.candidateSlowdown <= dirtyStrategy.candidateSlowdown
        ? { name: "clean" as StrategyName, label: "Clean", slowdown: cleanStrategy.candidateSlowdown }
        : { name: "dirty" as StrategyName, label: "Dirty", slowdown: dirtyStrategy.candidateSlowdown }
    }
    if (cleanStrategy.works) {
      return { name: "clean" as StrategyName, label: "Clean", slowdown: cleanStrategy.candidateSlowdown }
    }
    if (dirtyStrategy.works) {
      return { name: "dirty" as StrategyName, label: "Dirty", slowdown: dirtyStrategy.candidateSlowdown }
    }
    return { name: "none" as StrategyName, label: "None", slowdown: null }
  })()

  return {
    N,
    p,
    q,
    auditShare,
    auditBudget,
    auditOverhead,
    maxChallenges,
    o,
    sLimit,
    totalWork,
    carriedState,
    dirtyStride,
    kRaw,
    kWhole,
    epsilon,
    epsilonApprox,
    s,
    baselineOutputs: totalWork,
    baselineEpochs: totalWork / N,
    cleanFeasible: cleanWorks,
    dirtyFeasible: dirtyWorks,
    cleanOutputsPerTask,
    dirtyOutputsPerCommit,
    dirtySurvival,
    dirtyRiskyOutputs,
    dirtyExchangeBudgetBytes,
    qSafeDirtyCapacity,
    maxSafeDirtyCommits,
    cleanStrategy,
    dirtyStrategy,
    bestStrategy,
  }
}

function InfoTip({ text }: { text: string }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          className="inline-flex h-3.5 w-3.5 items-center justify-center rounded-full bg-muted text-[9px] font-medium leading-none text-muted-foreground transition-colors hover:bg-muted-foreground/20"
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

function Metric({
  label,
  value,
  detail,
  accent,
}: {
  label: string
  value: string
  detail?: string
  accent?: boolean
}) {
  return (
    <div className={`rounded-lg border px-3 py-2.5 ${accent ? "border-foreground/30 bg-muted/50" : "border-border bg-background"}`}>
      <div className="text-[10px] font-mono uppercase tracking-wider text-muted-foreground">
        {label}
      </div>
      <div className="mt-1 font-mono text-lg font-semibold leading-none">{value}</div>
      {detail && <div className="mt-1 text-xs text-muted-foreground">{detail}</div>}
    </div>
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

function StrategyCard({
  title,
  subtitle,
  tone,
  works,
  isBest,
  headline,
  detail,
  stats,
}: {
  title: string
  subtitle: string
  tone: "clean" | "dirty"
  works: boolean
  isBest: boolean
  headline: string
  detail: string
  stats: Array<{ label: string; value: string; detail?: string }>
}) {
  const accent = tone === "clean"
    ? "border-blue-500/35 bg-blue-500/5"
    : "border-red-500/35 bg-red-500/5"
  const blocked = "border-amber-500/40 bg-amber-500/5"
  return (
    <Card className={`min-h-[328px] ${works ? accent : blocked}`}>
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between gap-3">
          <div>
            <CardTitle className="text-sm font-semibold">{title}</CardTitle>
            <div className="mt-1 text-xs text-muted-foreground">{subtitle}</div>
          </div>
          <div className="flex shrink-0 items-center gap-1.5">
            {isBest && (
              <span className="rounded-full bg-foreground px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-background">
                Best
              </span>
            )}
            <span className={`rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide ${
              works
                ? "bg-emerald-500/10 text-emerald-700 dark:text-emerald-300"
                : "bg-amber-500/10 text-amber-700 dark:text-amber-300"
            }`}>
              {works ? "Works" : "Doesn't work"}
            </span>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="min-h-[86px]">
          <div className={`font-mono text-4xl font-bold leading-none tracking-tight ${works ? "text-foreground" : "text-amber-700 dark:text-amber-300"}`}>
            {headline}
          </div>
          <div className="mt-2 max-w-xl text-sm text-muted-foreground">{detail}</div>
        </div>
        <div className="mt-5 grid grid-cols-2 gap-2">
          {stats.map(stat => (
            <div key={stat.label} className="min-h-[72px] rounded-md border border-border bg-background/75 px-3 py-2">
              <div className="text-[9.5px] font-mono uppercase tracking-wider text-muted-foreground">
                {stat.label}
              </div>
              <div className="mt-1 font-mono text-base font-semibold leading-none">{stat.value}</div>
              {stat.detail && <div className="mt-1 text-[11px] text-muted-foreground">{stat.detail}</div>}
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}

export function LeakageDashboard() {
  const initial = typeof window !== "undefined" ? decodeState(window.location.hash) : null
  const [nLog, setNLog] = useState(initial?.n ?? DEFAULT_STATE.n)
  const [auditShareLog, setAuditShareLog] = useState(initial?.beta ?? DEFAULT_STATE.beta)
  const [auditOverheadLog, setAuditOverheadLog] = useState(initial?.h ?? DEFAULT_STATE.h)
  const [sLimitLog, setSLimitLog] = useState(initial?.sLimit ?? DEFAULT_STATE.sLimit)
  const [riskLog, setRiskLog] = useState(initial?.r ?? DEFAULT_STATE.r)
  const [oLog, setOLog] = useState(initial?.o ?? DEFAULT_STATE.o)
  const [workLog, setWorkLog] = useState(initial?.work ?? DEFAULT_STATE.work)
  const [carriedStateLog, setCarriedStateLog] = useState(initial?.carriedState ?? DEFAULT_STATE.carriedState)
  const [dirtyStrideLog, setDirtyStrideLog] = useState(initial?.dirtyStride ?? DEFAULT_STATE.dirtyStride)
  const [copiedJson, setCopiedJson] = useState(false)
  const [copiedLink, setCopiedLink] = useState(false)

  const state: LeakageState = useMemo(() => ({
    n: nLog,
    beta: auditShareLog,
    h: auditOverheadLog,
    sLimit: sLimitLog,
    r: riskLog,
    o: oLog,
    work: workLog,
    carriedState: carriedStateLog,
    dirtyStride: dirtyStrideLog,
  }), [nLog, auditShareLog, auditOverheadLog, sLimitLog, riskLog, oLog, workLog, carriedStateLog, dirtyStrideLog])

  const q = currentQ(riskLog)
  const result = useMemo(() => computeForQ(state, q), [state, q])
  const rows = useMemo(() => SURVIVAL_PRESETS.map(targetQ => {
    const r = computeForQ(state, targetQ)
    return {
      targetQ,
      dirtyWorks: r.dirtyStrategy.works,
      dirtySurvival: r.dirtyStrategy.survival,
      qSafeDirtyCapacity: r.dirtyStrategy.qSafeCapacity,
      referenceFits: r.dirtyStrategy.referenceFits,
      kWhole: r.kWhole,
    }
  }), [state])

  const handleCopyJson = () => {
    navigator.clipboard.writeText(JSON.stringify({ input: state, derived: result }, null, 2))
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

  const clean = result.cleanStrategy
  const dirty = result.dirtyStrategy

  return (
    <div className="min-h-screen bg-background">
      <div className="mx-auto max-w-5xl px-4 py-6">
        <div className="mb-6 flex items-start justify-between gap-4">
          <div>
            <div className="mb-1.5 text-[11px] font-mono uppercase tracking-[0.22em] text-muted-foreground">
              Interactive · Endpoint leakage
            </div>
            <h1 className="text-2xl font-bold tracking-tight text-foreground">
              Clean vs dirty endpoint simulator
            </h1>
            <p className="mt-1 max-w-2xl text-sm text-muted-foreground">
              Compare the clean-only and dirty-only strategies for the same reference training job.
            </p>
          </div>
          <div className="flex shrink-0 items-center gap-1">
            <IconButton copied={copiedLink} title="Copy shareable link" onClick={handleCopyLink} icon="link" />
            <IconButton copied={copiedJson} title="Copy input and derived JSON" onClick={handleCopyJson} icon="copy" />
            <ThemeToggle />
          </div>
        </div>

        <div className="mb-4 grid grid-cols-1 gap-4 md:grid-cols-2">
          <StrategyCard
            title="Clean strategy"
            subtitle="Carry state in every clean task output"
            tone="clean"
            works={clean.works}
            isBest={result.bestStrategy.name === "clean"}
            headline={clean.works ? fmtSlowdown(clean.slowdown) : "Doesn't work"}
            detail={clean.works
              ? `Completes the reference job in ${fmtEpochs(clean.epochs)}.`
              : clean.reason}
            stats={[
              {
                label: "State traffic",
                value: `${fmtCount(clean.outputsPerTask, 1)} outputs/task`,
                detail: `${fmtBytes(result.carriedState)} carried`,
              },
              {
                label: "Total outputs",
                value: fmtCount(clean.totalOutputs, 1),
                detail: `${fmtCount(clean.overheadOutputs, 1)} overhead`,
              },
              {
                label: "Input gate",
                value: clean.works ? "within S" : "blocked",
                detail: `S = ${fmtBytes(result.sLimit)}`,
              },
              {
                label: "Candidate cost",
                value: fmtSlowdown(clean.candidateSlowdown),
                detail: clean.works ? "valid clean run" : "if S were relaxed",
              },
            ]}
          />

          <StrategyCard
            title="Dirty strategy"
            subtitle="Run locally, then make fixed-stride state commits"
            tone="dirty"
            works={dirty.works}
            isBest={result.bestStrategy.name === "dirty"}
            headline={dirty.works ? fmtSlowdown(dirty.slowdown) : "Doesn't work"}
            detail={dirty.works
              ? `Survival is ${fmtPct(dirty.survival)} against target ${fmtPct(result.q)}.`
              : `${dirty.reason}; candidate slowdown ${fmtSlowdown(dirty.candidateSlowdown)}.`}
            stats={[
              {
                label: "Commits",
                value: fmtCount(dirty.commits, 1),
                detail: `every ${fmtCount(dirty.stride, 1)} task-eq`,
              },
              {
                label: "Risky outputs",
                value: fmtCount(dirty.riskyOutputs, 1),
                detail: `${fmtCount(dirty.outputsPerCommit, 1)} per commit`,
              },
              {
                label: "Survival",
                value: fmtPct(dirty.survival),
                detail: `target ${fmtPct(result.q)}`,
              },
              {
                label: "q-safe capacity",
                value: fmtCount(dirty.qSafeCapacity, 1),
                detail: dirty.referenceFits ? "reference fits" : "reference too large",
              },
            ]}
          />
        </div>

        <Card className="mb-4">
          <CardContent className="pt-6">
            <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
              <Metric
                label="Best strategy"
                value={result.bestStrategy.label}
                detail={result.bestStrategy.slowdown == null ? "no valid strategy" : fmtSlowdown(result.bestStrategy.slowdown)}
                accent
              />
              <Metric
                label="Baseline epochs"
                value={fmtEpochs(result.baselineEpochs)}
                detail={`${fmtCount(result.baselineOutputs, 1)} task-eq`}
              />
              <Metric
                label="Challenge probability"
                value={fmtPct(result.p)}
                detail={`${fmtCount(result.maxChallenges, 1)} challenges`}
              />
              <Metric
                label="Run risk budget"
                value={fmtCount(result.kWhole, 1)}
                detail="dirty commit outputs"
              />
              <Metric
                label="Dirty exchange budget"
                value={fmtBytes(result.dirtyExchangeBudgetBytes)}
                detail={`${fmtCount(result.kWhole, 1)} q-safe outputs`}
              />
            </div>
          </CardContent>
        </Card>

        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-semibold">Policy parameters</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <LogSlider
                label="Task outputs per epoch N"
                tooltip="Total task outputs eligible for audit during the epoch."
                valueExp={nLog}
                onValueExpChange={setNLog}
                min={3}
                max={12}
                step={0.05}
                formatValue={v => fmtCount(Math.round(v), 1)}
              />
              <LogSlider
                label="Audit compute share beta"
                tooltip="Share of total epoch task compute reserved for exact audits."
                valueExp={auditShareLog}
                onValueExpChange={setAuditShareLog}
                min={-6}
                max={0}
                step={0.05}
                formatValue={fmtPct}
              />
              <LogSlider
                label="Overhead per audit"
                tooltip="Compute cost of one exact audit, measured as a multiple of one ordinary task output."
                valueExp={auditOverheadLog}
                onValueExpChange={setAuditOverheadLog}
                min={0}
                max={9}
                step={0.05}
                formatValue={fmtRatio}
              />
              <LogSlider
                label="Non-whitelisted input limit S"
                tooltip="Maximum non-whitelisted input/state bytes allowed in a clean PoCR-valid task."
                valueExp={sLimitLog}
                onValueExpChange={setSLimitLog}
                min={3}
                max={10}
                step={0.05}
                formatValue={fmtBytes}
              />
              <LogSlider
                label="Max bytes per output o"
                tooltip="Payload that one output can carry if it is used for communication."
                valueExp={oLog}
                onValueExpChange={setOLog}
                min={0}
                max={9}
                step={0.05}
                formatValue={fmtBytes}
              />
              <LogSlider
                label="Whole-run survival target q"
                tooltip="Whole-run survival target; tolerated detection risk is 1 - q."
                valueExp={riskLog}
                onValueExpChange={setRiskLog}
                min={-4}
                max={RISK_MAX}
                step={0.01}
                formatValue={risk => `${fmtPct(1 - risk)} survival`}
              />
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-semibold">Attack parameters</CardTitle>
            </CardHeader>
            <CardContent className="space-y-5">
              <LogSlider
                label="Training compute required"
                tooltip="Total task-equivalent compute needed to complete the covert training job."
                valueExp={workLog}
                onValueExpChange={setWorkLog}
                min={0}
                max={12}
                step={0.05}
                formatValue={v => `${fmtCount(Math.round(v), 1)} task-eq`}
              />
              <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
                <LogSlider
                  label="Carried state size"
                  tooltip="State carried by the attack. Clean carries this every task; dirty carries it at each dirty commit."
                  valueExp={carriedStateLog}
                  onValueExpChange={setCarriedStateLog}
                  min={0}
                  max={13}
                  step={0.05}
                  formatValue={fmtBytes}
                />
                <LogSlider
                  label="Dirty stride"
                  tooltip="Fixed task-equivalent compute between dirty state commits."
                  valueExp={dirtyStrideLog}
                  onValueExpChange={setDirtyStrideLog}
                  min={0}
                  max={9}
                  step={0.05}
                  formatValue={v => `${fmtCount(Math.round(v), 1)} task-eq`}
                />
              </div>
              <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
                <Metric
                  label="Clean state traffic"
                  value={`${fmtCount(result.cleanOutputsPerTask, 1)} outputs/task`}
                  detail={`clean valid iff state <= S ${fmtBytes(result.sLimit)}`}
                />
                <Metric
                  label="Dirty commit traffic"
                  value={`${fmtCount(result.dirtyOutputsPerCommit, 1)} outputs/commit`}
                  detail={`commit every ${fmtCount(result.dirtyStride, 1)} task-eq`}
                />
              </div>
            </CardContent>
          </Card>
        </div>

        <Card className="mt-4">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-semibold">Dirty survival sweep</CardTitle>
          </CardHeader>
          <CardContent>
            <table className="w-full border-collapse text-sm">
              <thead>
                <tr>
                  <th className="border-b border-border px-2 py-1.5 text-left text-[9.5px] font-mono font-normal uppercase tracking-wider text-muted-foreground">
                    Whole-run q
                  </th>
                  <th className="border-b border-border px-2 py-1.5 text-right text-[9.5px] font-mono font-normal uppercase tracking-wider text-muted-foreground">
                    q-safe dirty capacity
                  </th>
                  <th className="border-b border-border px-2 py-1.5 text-right text-[9.5px] font-mono font-normal uppercase tracking-wider text-muted-foreground">
                    Reference job
                  </th>
                  <th className="border-b border-border px-2 py-1.5 text-right text-[9.5px] font-mono font-normal uppercase tracking-wider text-muted-foreground">
                    Commit-output budget
                  </th>
                  <th className="border-b border-border px-2 py-1.5 text-left text-[9.5px] font-mono font-normal uppercase tracking-wider text-muted-foreground">
                    Dirty strategy
                  </th>
                </tr>
              </thead>
              <tbody>
                {rows.map(row => (
                  <tr
                    key={row.targetQ}
                    className={Math.abs(row.targetQ - q) < 1e-9 ? "bg-muted/50" : "hover:bg-muted/30"}
                  >
                    <td className="border-b border-border/60 px-2 py-2 font-mono">
                      {fmtPct(row.targetQ)}
                    </td>
                    <td className="border-b border-border/60 px-2 py-2 text-right font-mono">
                      {fmtCount(row.qSafeDirtyCapacity, 1)}
                    </td>
                    <td className={`border-b border-border/60 px-2 py-2 text-right font-medium ${row.referenceFits ? "text-emerald-600 dark:text-emerald-400" : "text-amber-600 dark:text-amber-400"}`}>
                      {row.referenceFits ? "fits" : "too large"}
                    </td>
                    <td className="border-b border-border/60 px-2 py-2 text-right font-mono">
                      {fmtCount(row.kWhole, 1)}
                    </td>
                    <td className={`border-b border-border/60 px-2 py-2 font-medium ${row.dirtyWorks ? "text-emerald-600 dark:text-emerald-400" : "text-amber-600 dark:text-amber-400"}`}>
                      {row.dirtyWorks ? `works at ${fmtPct(row.dirtySurvival)}` : `doesn't work at ${fmtPct(row.dirtySurvival)}`}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </CardContent>
        </Card>

        <p className="mt-5 max-w-3xl text-xs text-muted-foreground">
          The audit budget is beta times total epoch task compute. The page chooses the maximum
          number of exact challenges affordable under that share, then uses p = challenges / N.
          q is the survival target for the whole training run. Clean is valid only when the carried
          state fits under S. Dirty uses the selected fixed stride and only dirty commit outputs
          count against the run-wide detection budget. This view compares clean-only and dirty-only
          strategies directly; it does not add a hybrid optimizer.
        </p>
      </div>
    </div>
  )
}
