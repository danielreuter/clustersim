"use client"

import { useMemo, useState } from "react"
import { Check, Clipboard, Link as LinkIcon } from "lucide-react"
import { ThemeToggle } from "@/components/theme-toggle"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { computeAudit, type AuditResult, type BatchEvaluation } from "@/lib/audit-batching"

type DashboardState = {
  requestCountExp: number
  replayShareExp: number
  costMultiplierExp: number
  outputSizeExp: number
  detectionMissExp: number
  maxBatchSize: number
  halfBatchUtilization: number
  manualBatchSize: number
}

const DEFAULT_STATE: DashboardState = {
  requestCountExp: 11,
  replayShareExp: -4,
  costMultiplierExp: 1,
  outputSizeExp: 6,
  detectionMissExp: -2,
  maxBatchSize: 200,
  halfBatchUtilization: 0.5,
  manualBatchSize: 100,
}

// Detection-power view is intentionally omitted; the lib still needs a value.
const UNUSED_TARGET_SHARE = 1e-3
const STANDARD_COMPUTE_TIERS = [2, 5, 10, 20, 50, 100]

function clamp(n: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, n))
}

function fmtCount(n: number, decimals = 1): string {
  if (!Number.isFinite(n)) return "--"
  if (n >= 1e15) return `${(n / 1e15).toFixed(decimals)}Q`
  if (n >= 1e12) return `${(n / 1e12).toFixed(decimals)}T`
  if (n >= 1e9) return `${(n / 1e9).toFixed(decimals)}B`
  if (n >= 1e6) return `${(n / 1e6).toFixed(decimals)}M`
  if (n >= 1e3) return `${(n / 1e3).toFixed(decimals)}K`
  if (n >= 10) return Math.round(n).toLocaleString("en-US")
  if (Number.isInteger(n)) return n.toLocaleString("en-US")
  return n.toFixed(decimals)
}

function fmtPct(frac: number): string {
  if (!Number.isFinite(frac)) return "--"
  if (frac <= 0) return "0%"
  const pct = frac * 100
  if (pct >= 10) return `${pct.toFixed(1)}%`
  if (pct >= 1) return `${pct.toFixed(2)}%`
  if (pct >= 0.01) return `${pct.toPrecision(3)}%`
  if (pct >= 1e-4) return `${pct.toPrecision(3)}%`
  return `${pct.toExponential(2)}%`
}

function fmtBytes(n: number): string {
  if (!Number.isFinite(n)) return "--"
  if (n >= 1e18) return `${(n / 1e18).toFixed(2)} EB`
  if (n >= 1e15) return `${(n / 1e15).toFixed(2)} PB`
  if (n >= 1e12) return `${(n / 1e12).toFixed(2)} TB`
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)} GB`
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)} MB`
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)} KB`
  return `${Math.round(n).toLocaleString("en-US")} B`
}

function fmtMultiplier(a: number): string {
  if (a >= 100) return `${a.toFixed(0)}×C`
  if (a >= 10) return `${a.toFixed(1)}×C`
  return `${a.toFixed(2)}×C`
}

function fmtOneIn(q: number): string {
  return q > 0 ? `1 in ${fmtCount(1 / q, 2)}` : "--"
}

function sameMultiplier(a: number, b: number): boolean {
  return Math.abs(Math.log(a / b)) < 1e-6
}

function computeTierMultipliers(selected: number): number[] {
  return [...STANDARD_COMPUTE_TIERS, selected]
    .filter(multiplier => multiplier >= 1.01 && multiplier <= 1000)
    .sort((a, b) => a - b)
    .filter((multiplier, index, values) => (
      index === 0 || !sameMultiplier(multiplier, values[index - 1])
    ))
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
      <TooltipContent side="top" sideOffset={4} className="max-w-xs">{text}</TooltipContent>
    </Tooltip>
  )
}

function IconButton({ copied, title, onClick, icon }: {
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
      title={title}
      aria-label={title}
      className="inline-flex h-8 w-8 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
    >
      {copied ? <Check className="h-4 w-4" /> : <Icon className="h-4 w-4" />}
    </button>
  )
}

function LogSlider({ label, tooltip, valueExp, onChange, min, max, step, formatValue }: {
  label: string
  tooltip: string
  valueExp: number
  onChange: (value: number) => void
  min: number
  max: number
  step: number
  formatValue: (value: number) => string
}) {
  return (
    <div>
      <div className="flex items-center justify-between gap-3">
        <Label className="text-xs text-muted-foreground">{label}<InfoTip text={tooltip} /></Label>
        <span className="shrink-0 font-mono text-xs">{formatValue(10 ** valueExp)}</span>
      </div>
      <Slider value={[valueExp]} onValueChange={([value]: number[]) => onChange(value)} min={min} max={max} step={step} className="mt-2" />
    </div>
  )
}

function LinearSlider({ label, tooltip, value, onChange, min, max, step, formatValue }: {
  label: string
  tooltip: string
  value: number
  onChange: (value: number) => void
  min: number
  max: number
  step: number
  formatValue: (value: number) => string
}) {
  return (
    <div>
      <div className="flex items-center justify-between gap-3">
        <Label className="text-xs text-muted-foreground">{label}<InfoTip text={tooltip} /></Label>
        <span className="shrink-0 font-mono text-xs">{formatValue(value)}</span>
      </div>
      <Slider value={[value]} onValueChange={([next]: number[]) => onChange(next)} min={min} max={max} step={step} className="mt-2" />
    </div>
  )
}

function encodeState(state: DashboardState): string {
  try { return btoa(JSON.stringify(state)) } catch { return "" }
}

function decodeState(hash: string): DashboardState | null {
  try {
    const raw = hash.startsWith("#") ? hash.slice(1) : hash
    if (!raw) return null
    const value = JSON.parse(atob(raw)) as Partial<DashboardState>
    const number = (candidate: unknown, fallback: number) =>
      typeof candidate === "number" && Number.isFinite(candidate) ? candidate : fallback
    const maxBatchSize = clamp(Math.round(number(value.maxBatchSize, 200)), 1, 512)
    return {
      requestCountExp: clamp(number(value.requestCountExp, 11), 6, 14),
      replayShareExp: clamp(number(value.replayShareExp, -4), -10, -0.3),
      costMultiplierExp: clamp(number(value.costMultiplierExp, 1), Math.log10(1.01), 3),
      outputSizeExp: clamp(number(value.outputSizeExp, 6), 0, 12),
      detectionMissExp: clamp(number(value.detectionMissExp, -2), -8, Math.log10(0.5)),
      maxBatchSize,
      halfBatchUtilization: clamp(number(value.halfBatchUtilization, 0.5), 0.05, 1),
      manualBatchSize: clamp(Math.round(number(value.manualBatchSize, 100)), 1, maxBatchSize),
    }
  } catch { return null }
}

type CardStyle = { bar: string; bg: string; border: string; text: string }

const OPTIMAL_STYLE: CardStyle = {
  bar: "bg-emerald-500",
  bg: "bg-emerald-500/10",
  border: "border-emerald-500/30",
  text: "text-emerald-700 dark:text-emerald-300",
}
const MANUAL_STYLE: CardStyle = {
  bar: "bg-amber-500",
  bg: "bg-amber-500/10",
  border: "border-amber-500/30",
  text: "text-amber-700 dark:text-amber-300",
}
const FULL_STYLE: CardStyle = {
  bar: "bg-rose-500",
  bg: "bg-rose-500/10",
  border: "border-rose-500/30",
  text: "text-rose-700 dark:text-rose-300",
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between gap-3">
      <span className="text-muted-foreground">{label}</span>
      <span className="font-mono">{value}</span>
    </div>
  )
}

function BatchPolicyCard({ style, label, tooltip, policy, result, outputBytes }: {
  style: CardStyle
  label: string
  tooltip: string
  policy: BatchEvaluation
  result: AuditResult
  outputBytes: number
}) {
  const unruledOutOutputs = policy.cleanUpperBound * result.requestCount

  return (
    <div className={`rounded-lg border px-4 py-4 ${style.border} ${style.bg}`}>
      <div className="flex items-center gap-1.5 text-xs font-medium text-foreground">
        <span className={`h-2.5 w-2.5 shrink-0 rounded-sm ${style.bar}`} />
        <span className="min-w-0 truncate">{label}</span>
        <InfoTip text={tooltip} />
        <span className="ml-auto shrink-0 rounded-md border bg-background px-1.5 py-0.5 font-mono text-[11px] font-semibold">
          b={policy.batchSize}
        </span>
      </div>
      <div className={`mt-4 font-mono text-3xl font-semibold leading-none ${style.text}`}>
        {fmtPct(policy.cleanUpperBound)}
      </div>
      <div className="mt-1.5 text-xs text-muted-foreground">prevalence cap · {fmtOneIn(policy.cleanUpperBound)}</div>
      <div className="mt-4 space-y-1.5 text-xs">
        <Row label="Utilization" value={fmtPct(policy.utilization)} />
        <Row label="Requests audited" value={fmtCount(policy.auditedRequests, 2)} />
        <Row label="Unruled-out outputs" value={fmtCount(unruledOutOutputs, 2)} />
        <Row label="Output payload" value={fmtBytes(unruledOutOutputs * outputBytes)} />
      </div>
    </div>
  )
}

function TierBoundsCard({ rows, selectedMultiplier, requestCount, outputBytes }: {
  rows: Array<{
    multiplier: number
    policy: BatchEvaluation
  }>
  selectedMultiplier: number
  requestCount: number
  outputBytes: number
}) {
  return (
    <Card className="mb-4">
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-semibold">Compute tier bounds</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[680px] text-left text-xs">
            <thead className="border-b text-muted-foreground">
              <tr>
                <th className="pb-2 font-medium">Tier</th>
                <th className="pb-2 font-medium">Best replay batch</th>
                <th className="pb-2 font-medium">Prevalence cap</th>
                <th className="pb-2 font-medium">Unruled-out outputs</th>
                <th className="pb-2 font-medium">Output payload</th>
                <th className="pb-2 font-medium">Utilization</th>
              </tr>
            </thead>
            <tbody>
              {rows.map(({ multiplier, policy }) => {
                const unruledOutOutputs = policy.cleanUpperBound * requestCount
                const selected = sameMultiplier(multiplier, selectedMultiplier)
                return (
                  <tr key={multiplier} className={selected ? "bg-muted/45" : undefined}>
                    <td className="py-2 pr-3">
                      <span className="inline-flex items-center gap-1.5">
                        <span className={selected ? "h-2.5 w-2.5 rounded-sm bg-emerald-500" : "h-2.5 w-2.5 rounded-sm bg-muted-foreground/30"} />
                        <span className="font-mono font-semibold">{fmtMultiplier(multiplier)}</span>
                      </span>
                    </td>
                    <td className="py-2 pr-3 font-mono">b={policy.batchSize}</td>
                    <td className="py-2 pr-3 font-mono">{fmtPct(policy.cleanUpperBound)}</td>
                    <td className="py-2 pr-3 font-mono">{fmtCount(unruledOutOutputs, 2)}</td>
                    <td className="py-2 pr-3 font-mono">{fmtBytes(unruledOutOutputs * outputBytes)}</td>
                    <td className="py-2 font-mono">{fmtPct(policy.utilization)}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
        <p className="mt-3 text-xs leading-relaxed text-muted-foreground">
          Each row asks how many outputs could have required at least that compute depth. The byte column multiplies the
          prevalence cap by the committed population and the task output size limit.
        </p>
      </CardContent>
    </Card>
  )
}

function PrevalenceChart({ evaluations, optimalBatch, manualBatch }: {
  evaluations: BatchEvaluation[]
  optimalBatch: number
  manualBatch: number
}) {
  const width = 820
  const height = 260
  const left = 76
  const right = 24
  const top = 18
  const bottom = 46
  const plotWidth = width - left - right
  const plotHeight = height - top - bottom
  const maxBatch = evaluations.length
  const value = (item: BatchEvaluation) => item.cleanUpperBound
  const transform = (v: number) => Math.log10(Math.max(v, 1e-14))
  const transformed = evaluations.map(item => transform(value(item)))
  const yMin = Math.floor(Math.min(...transformed))
  const yMax = Math.max(0, Math.ceil(Math.max(...transformed)))
  const span = Math.max(1e-9, yMax - yMin)
  const x = (batch: number) => left + ((batch - 1) / Math.max(1, maxBatch - 1)) * plotWidth
  const y = (v: number) => top + ((yMax - transform(v)) / span) * plotHeight
  const path = evaluations
    .map((item, index) => `${index === 0 ? "M" : "L"}${x(item.batchSize).toFixed(2)},${y(value(item)).toFixed(2)}`)
    .join(" ")
  const yTicks = Array.from({ length: yMax - yMin + 1 }, (_, index) => yMin + index)
  const xTicks = [1, Math.round(maxBatch / 4), Math.round(maxBatch / 2), Math.round((3 * maxBatch) / 4), maxBatch]
    .filter((batch, index, all) => batch >= 1 && all.indexOf(batch) === index)

  return (
    <div className="overflow-x-auto">
      <svg viewBox={`0 0 ${width} ${height}`} className="min-w-[700px]" role="img" aria-label="Prevalence cap across physical batch sizes">
        {yTicks.map(tick => {
          const position = top + ((yMax - tick) / span) * plotHeight
          return (
            <g key={tick}>
              <line x1={left} x2={width - right} y1={position} y2={position} className="stroke-border" />
              <text x={left - 10} y={position + 4} textAnchor="end" className="fill-muted-foreground font-mono text-[10px]">{fmtPct(10 ** tick)}</text>
            </g>
          )
        })}
        <line x1={x(manualBatch)} x2={x(manualBatch)} y1={top} y2={height - bottom} className="stroke-amber-500/60" strokeDasharray="5 4" />
        <line x1={x(optimalBatch)} x2={x(optimalBatch)} y1={top} y2={height - bottom} className="stroke-emerald-500/70" strokeDasharray="5 4" />
        <path d={path} fill="none" className="stroke-foreground" strokeWidth="2" />
        <circle cx={x(optimalBatch)} cy={y(value(evaluations[optimalBatch - 1]))} r="4" className="fill-emerald-500" />
        <circle cx={x(manualBatch)} cy={y(value(evaluations[manualBatch - 1]))} r="4" className="fill-amber-500" />
        {xTicks.map(batch => (
          <g key={batch}>
            <line x1={x(batch)} x2={x(batch)} y1={height - bottom} y2={height - bottom + 5} className="stroke-border" />
            <text x={x(batch)} y={height - bottom + 20} textAnchor="middle" className="fill-muted-foreground font-mono text-[10px]">{batch}</text>
          </g>
        ))}
        <text x={left + plotWidth / 2} y={height - 8} textAnchor="middle" className="fill-muted-foreground text-[11px]">physical batch size b</text>
      </svg>
    </div>
  )
}

export function AuditBatchingDashboard() {
  const initial = typeof window !== "undefined" ? decodeState(window.location.hash) : null
  const [state, setState] = useState<DashboardState>(initial ?? DEFAULT_STATE)
  const [copiedLink, setCopiedLink] = useState(false)
  const [copiedJson, setCopiedJson] = useState(false)

  const result = useMemo(() => computeAudit({
    requestCount: 10 ** state.requestCountExp,
    replayShare: 10 ** state.replayShareExp,
    costMultiplier: 10 ** state.costMultiplierExp,
    missProbability: 10 ** state.detectionMissExp,
    targetBadShare: UNUSED_TARGET_SHARE,
    maxBatchSize: state.maxBatchSize,
    halfBatchUtilization: state.halfBatchUtilization,
    manualBatchSize: state.manualBatchSize,
  }), [state])
  const outputBytes = 10 ** state.outputSizeExp
  const tierBounds = useMemo(() => (
    computeTierMultipliers(10 ** state.costMultiplierExp).map(multiplier => ({
      multiplier,
      policy: computeAudit({
        requestCount: 10 ** state.requestCountExp,
        replayShare: 10 ** state.replayShareExp,
        costMultiplier: multiplier,
        missProbability: 10 ** state.detectionMissExp,
        targetBadShare: UNUSED_TARGET_SHARE,
        maxBatchSize: state.maxBatchSize,
        halfBatchUtilization: state.halfBatchUtilization,
        manualBatchSize: state.manualBatchSize,
      }).prevalenceOptimal,
    }))
  ), [state])

  const setField = <K extends keyof DashboardState>(key: K, value: DashboardState[K]) => {
    setState(previous => {
      const next = { ...previous, [key]: value }
      if (key === "maxBatchSize") next.manualBatchSize = clamp(next.manualBatchSize, 1, Number(value))
      return next
    })
  }

  const copyLink = () => {
    navigator.clipboard.writeText(`${window.location.origin}${window.location.pathname}#${encodeState(state)}`)
    setCopiedLink(true)
    setTimeout(() => setCopiedLink(false), 1800)
  }

  const copyJson = () => {
    navigator.clipboard.writeText(JSON.stringify({ input: state, derived: result }, null, 2))
    setCopiedJson(true)
    setTimeout(() => setCopiedJson(false), 1800)
  }

  const best = result.prevalenceOptimal
  const permitted = best.cleanUpperBound * result.requestCount
  const permittedBytes = permitted * outputBytes

  return (
    <div className="min-h-screen bg-background">
      <div className="mx-auto max-w-5xl px-4 py-6">
        <div className="mb-6 flex items-start justify-between gap-4">
          <div>
            <div className="mb-1.5 text-[11px] font-mono uppercase tracking-[0.22em] text-muted-foreground">
              Interactive · Isolated replay audit
            </div>
            <h1 className="text-2xl font-bold tracking-tight text-foreground">Compute tier prevalence bounds</h1>
            <p className="mt-1 max-w-2xl text-sm text-muted-foreground">
              Estimate what a clean temporal replay audit rules out when the replay compartment is a physical batch with one aggregate compute budget.
            </p>
          </div>
          <div className="flex shrink-0 items-center gap-1">
            <IconButton copied={copiedLink} title="Copy shareable link" onClick={copyLink} icon="link" />
            <IconButton copied={copiedJson} title="Copy input and derived JSON" onClick={copyJson} icon="copy" />
            <ThemeToggle />
          </div>
        </div>

        <Card className="mb-4 overflow-hidden border-foreground/20">
          <CardContent className="pt-6">
            <div className="text-sm font-medium text-foreground">Selected compute tier prevalence cap</div>
            <div className="mt-2 font-mono text-5xl font-bold tracking-tight md:text-6xl">{fmtPct(best.cleanUpperBound)}</div>
            <p className="mt-3 max-w-3xl text-sm text-muted-foreground">
              At {fmtPct(result.confidence)} confidence, temporal isolated replay with best batch b={best.batchSize} rules out any population where
              this share or more of the {fmtCount(result.requestCount)} committed requests cost at least {fmtMultiplier(result.costMultiplier)} to
              replay. That is about {fmtOneIn(best.cleanUpperBound)}, or up to {fmtCount(permitted, 2)} unruled-out outputs carrying
              {fmtBytes(permittedBytes)} at the selected output limit.
            </p>
            <div className="mt-4 rounded-lg border border-amber-500/30 bg-amber-500/10 px-4 py-3 text-sm leading-relaxed text-amber-900 dark:text-amber-100">
              The guarantee is compartmentalized at the replay batch, not necessarily at the individual request. A population sitting
              exactly at this cap would still pass a clean audit {fmtPct(result.missProbability)} of the time.
            </div>
          </CardContent>
        </Card>

        <div className="mb-4 grid grid-cols-1 gap-3 md:grid-cols-3">
          <BatchPolicyCard
            style={OPTIMAL_STYLE}
            label="Optimal batch"
            tooltip="The physical replay batch size that minimizes the prevalence cap after a clean audit."
            policy={result.prevalenceOptimal}
            result={result}
            outputBytes={outputBytes}
          />
          <BatchPolicyCard
            style={MANUAL_STYLE}
            label="Your batch"
            tooltip="The batch size set in the Physical batching controls, for comparison against the optimum."
            policy={result.manual}
            result={result}
            outputBytes={outputBytes}
          />
          <BatchPolicyCard
            style={FULL_STYLE}
            label="Full batch"
            tooltip="Running full physical batches maximizes utilization, but tier-violating requests can hide under the shared b·C cap."
            policy={result.fullBatch}
            result={result}
            outputBytes={outputBytes}
          />
        </div>

        <TierBoundsCard
          rows={tierBounds}
          selectedMultiplier={result.costMultiplier}
          requestCount={result.requestCount}
          outputBytes={outputBytes}
        />

        <Card className="mb-4">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-semibold">Prevalence cap across replay batch sizes</CardTitle>
          </CardHeader>
          <CardContent>
            <PrevalenceChart
              evaluations={result.evaluations}
              optimalBatch={result.prevalenceOptimal.batchSize}
              manualBatch={result.manual.batchSize}
            />
            <p className="mt-2 text-xs text-muted-foreground">
              Lower is stronger. <span className="font-medium text-emerald-600 dark:text-emerald-400">Green</span> marks the
              optimum (b={result.prevalenceOptimal.batchSize}); <span className="font-medium text-amber-600 dark:text-amber-400">amber</span> marks
              your batch (b={result.manual.batchSize}). Here b is the isolated replay/accounting group. Small batches catch tail concentration;
              large batches share slack across sampled prompts. Vertical scale is logarithmic.
            </p>
          </CardContent>
        </Card>

        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <Card>
            <CardHeader className="pb-3"><CardTitle className="text-sm font-semibold">Population &amp; audit</CardTitle></CardHeader>
            <CardContent className="space-y-5">
              <LogSlider
                label="Committed requests"
                tooltip="Population size n. Every request and output is committed before the auditor samples."
                valueExp={state.requestCountExp}
                onChange={value => setField("requestCountExp", value)}
                min={6} max={14} step={0.05}
                formatValue={value => fmtCount(Math.round(value), 2)}
              />
              <LogSlider
                label="Replay compute share"
                tooltip="ρ: the audit budget as a share of the total nC claimed compute. Sets how many requests can be replayed."
                valueExp={state.replayShareExp}
                onChange={value => setField("replayShareExp", value)}
                min={-10} max={-0.3} step={0.05}
                formatValue={fmtPct}
              />
              <LogSlider
                label="Selected compute tier"
                tooltip="a: this page bounds how common requests are that need at least aC compute to replay."
                valueExp={state.costMultiplierExp}
                onChange={value => setField("costMultiplierExp", value)}
                min={Math.log10(1.01)} max={3} step={0.01}
                formatValue={fmtMultiplier}
              />
              <LogSlider
                label="Task output size limit"
                tooltip="Maximum task output payload. This translates the prevalence cap into an output-byte budget."
                valueExp={state.outputSizeExp}
                onChange={value => setField("outputSizeExp", value)}
                min={0} max={12} step={0.05}
                formatValue={fmtBytes}
              />
              <LogSlider
                label="Confidence"
                tooltip="1−δ. A clean audit rejects any prevalence at or above the displayed cap at this confidence."
                valueExp={state.detectionMissExp}
                onChange={value => setField("detectionMissExp", value)}
                min={-8} max={Math.log10(0.5)} step={0.01}
                formatValue={miss => `${fmtPct(1 - miss)} confidence`}
              />
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3"><CardTitle className="text-sm font-semibold">Physical batching</CardTitle></CardHeader>
            <CardContent className="space-y-5">
              <LinearSlider
                label="Maximum physical batch"
                tooltip="B: the physical batch size at which the server reaches full throughput."
                value={state.maxBatchSize}
                onChange={value => setField("maxBatchSize", Math.round(value))}
                min={1} max={512} step={1}
                formatValue={value => `${Math.round(value)} requests`}
              />
              <LinearSlider
                label="Throughput at half batch"
                tooltip="Sets the utilization curve u(b) = (b/B)^γ. 50% means linear utilization; lower values penalize small batches more."
                value={state.halfBatchUtilization}
                onChange={value => setField("halfBatchUtilization", value)}
                min={0.05} max={1} step={0.01}
                formatValue={fmtPct}
              />
              <LinearSlider
                label="Your batch size"
                tooltip="The physical replay batch you choose to compare against the optimum. Its b sampled requests share one aggregate budget of b·C."
                value={Math.min(state.manualBatchSize, state.maxBatchSize)}
                onChange={value => setField("manualBatchSize", Math.round(value))}
                min={1} max={state.maxBatchSize} step={1}
                formatValue={value => `${Math.round(value)} requests`}
              />
              <div className="rounded-lg border bg-muted/30 px-4 py-4 text-xs leading-relaxed text-muted-foreground">
                <div className="font-mono text-sm text-foreground">u(b) = (b / {result.maxBatchSize})^{result.utilizationGamma.toFixed(3)}</div>
                <div className="mt-2">
                  Your batch b={result.manual.batchSize} runs at {fmtPct(result.manual.utilization)} throughput, audits {fmtCount(result.manual.auditedRequests, 2)} requests
                  in {fmtCount(result.manual.batchCount, 2)} independent batches, and fails only if {result.manual.failuresNeeded} sampled tier-violating requests land in one batch.
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        <p className="mt-5 max-w-3xl text-xs leading-relaxed text-muted-foreground">
          Audit capacity is K = ρ·n·u(b) requests, split into M = ⌊K/b⌋ independently randomized replay batches. A batch shares one
          aggregate cap of b·C, so it fails only when at least ⌊b/a⌋+1 sampled requests land in the selected compute tier. The prevalence
          cap q is the largest tier frequency for which a fully clean audit still occurs with probability δ. Interpreted as isolated replay,
          this assumes outputs are committed before sampling, challenges are sampled against a time-ordered ledger, sampling is uniform, and
          no replay batch can borrow compute from another batch.
        </p>
      </div>
    </div>
  )
}
