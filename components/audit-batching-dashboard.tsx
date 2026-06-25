"use client"

import { useMemo, useState, type ReactNode } from "react"
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
  detectionMissExp: number
  targetBadShareExp: number
  maxBatchSize: number
  halfBatchUtilization: number
  manualBatchSize: number
}

const DEFAULT_STATE: DashboardState = {
  requestCountExp: 11,
  replayShareExp: -4,
  costMultiplierExp: 1,
  detectionMissExp: -2,
  targetBadShareExp: -3,
  maxBatchSize: 200,
  halfBatchUtilization: 0.5,
  manualBatchSize: 9,
}

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

function fmtProbability(p: number): string {
  if (p >= 0.999999) return ">99.9999%"
  return fmtPct(p)
}

function fmtMultiplier(a: number): string {
  if (a >= 100) return `${a.toFixed(0)}× C`
  if (a >= 10) return `${a.toFixed(1)}× C`
  return `${a.toFixed(2)}× C`
}

function fmtOneIn(q: number): string {
  return q > 0 ? `1 in ${fmtCount(1 / q, 2)}` : "--"
}

function InfoTip({ text }: { text: string }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button type="button" className="ml-1 inline-flex h-3.5 w-3.5 items-center justify-center rounded-full bg-muted text-[9px] font-medium text-muted-foreground hover:bg-muted-foreground/20">
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
    <button type="button" onClick={onClick} title={title} aria-label={title} className="inline-flex h-8 w-8 items-center justify-center rounded-md text-muted-foreground hover:bg-muted hover:text-foreground">
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
    const number = (candidate: unknown, fallback: number) => typeof candidate === "number" && Number.isFinite(candidate) ? candidate : fallback
    const maxBatchSize = clamp(Math.round(number(value.maxBatchSize, 200)), 1, 512)
    return {
      requestCountExp: clamp(number(value.requestCountExp, 11), 6, 14),
      replayShareExp: clamp(number(value.replayShareExp, -4), -10, -0.3),
      costMultiplierExp: clamp(number(value.costMultiplierExp, 1), Math.log10(1.01), 3),
      detectionMissExp: clamp(number(value.detectionMissExp, -2), -8, Math.log10(0.5)),
      targetBadShareExp: clamp(number(value.targetBadShareExp, -3), -10, Math.log10(0.5)),
      maxBatchSize,
      halfBatchUtilization: clamp(number(value.halfBatchUtilization, 0.5), 0.05, 1),
      manualBatchSize: clamp(Math.round(number(value.manualBatchSize, 9)), 1, maxBatchSize),
    }
  } catch { return null }
}

function PolicyCard({ title, description, policy, result, emphasized = false }: {
  title: string
  description: string
  policy: BatchEvaluation
  result: AuditResult
  emphasized?: boolean
}) {
  return (
    <div className={`rounded-lg border px-4 py-4 ${emphasized ? "border-foreground/30 bg-muted/35" : "border-border bg-background"}`}>
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-sm font-semibold">{title}</div>
          <div className="mt-1 text-xs leading-relaxed text-muted-foreground">{description}</div>
        </div>
        <span className="rounded-md border bg-background px-2 py-1 font-mono text-sm font-semibold">b={policy.batchSize}</span>
      </div>
      <div className="mt-4 grid grid-cols-2 gap-3 text-xs">
        <Stat label="Clean upper bound" value={fmtPct(policy.cleanUpperBound)} />
        <Stat label="Target detection" value={fmtProbability(policy.targetDetection)} />
        <Stat label="Utilization" value={fmtPct(policy.utilization)} />
        <Stat label="Audited requests" value={fmtCount(policy.auditedRequests, 2)} />
        <Stat label="Independent batches" value={fmtCount(policy.batchCount, 2)} />
        <Stat label="Bad jobs to fail" value={policy.failuresNeeded.toString()} />
      </div>
      <div className="mt-3 rounded-md bg-muted/50 px-3 py-2 text-xs text-muted-foreground">
        At {fmtPct(result.targetBadShare)} prevalence, expect {fmtCount(policy.targetExpectedBadRequests, 2)} audited {fmtMultiplier(result.costMultiplier)} requests.
      </div>
    </div>
  )
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-muted-foreground">{label}</div>
      <div className="mt-1 font-mono text-base font-semibold">{value}</div>
    </div>
  )
}

function LineChart({
  evaluations,
  value,
  logScale,
  optimalBatch,
  manualBatch,
  yLabel,
}: {
  evaluations: BatchEvaluation[]
  value: (evaluation: BatchEvaluation) => number
  logScale: boolean
  optimalBatch: number
  manualBatch: number
  yLabel: string
}) {
  const width = 820
  const height = 270
  const left = 76
  const right = 24
  const top = 18
  const bottom = 46
  const plotWidth = width - left - right
  const plotHeight = height - top - bottom
  const maxBatch = evaluations.length
  const transform = (v: number) => logScale ? Math.log10(Math.max(v, 1e-14)) : clamp(v, 0, 1)
  const transformed = evaluations.map(item => transform(value(item)))
  const rawMin = Math.min(...transformed)
  const rawMax = Math.max(...transformed)
  const yMin = logScale ? Math.floor(rawMin) : 0
  const yMax = logScale ? Math.max(0, Math.ceil(rawMax)) : 1
  const span = Math.max(1e-9, yMax - yMin)
  const x = (batch: number) => left + ((batch - 1) / Math.max(1, maxBatch - 1)) * plotWidth
  const y = (v: number) => top + ((yMax - transform(v)) / span) * plotHeight
  const path = evaluations.map((item, index) => `${index === 0 ? "M" : "L"}${x(item.batchSize).toFixed(2)},${y(value(item)).toFixed(2)}`).join(" ")
  const yTicks = logScale
    ? Array.from({ length: yMax - yMin + 1 }, (_, index) => yMin + index)
    : [0, 0.25, 0.5, 0.75, 1]

  return (
    <div className="overflow-x-auto">
      <svg viewBox={`0 0 ${width} ${height}`} className="min-w-[700px]" role="img" aria-label={yLabel}>
        {yTicks.map(tick => {
          const raw = logScale ? 10 ** tick : tick
          const position = top + ((yMax - tick) / span) * plotHeight
          return (
            <g key={tick}>
              <line x1={left} x2={width - right} y1={position} y2={position} className="stroke-border" />
              <text x={left - 10} y={position + 4} textAnchor="end" className="fill-muted-foreground font-mono text-[10px]">{fmtPct(raw)}</text>
            </g>
          )
        })}
        <line x1={x(manualBatch)} x2={x(manualBatch)} y1={top} y2={height - bottom} className="stroke-amber-500/60" strokeDasharray="5 4" />
        <line x1={x(optimalBatch)} x2={x(optimalBatch)} y1={top} y2={height - bottom} className="stroke-emerald-500/70" strokeDasharray="5 4" />
        <path d={path} fill="none" className="stroke-foreground" strokeWidth="2" />
        <circle cx={x(optimalBatch)} cy={y(value(evaluations[optimalBatch - 1]))} r="4" className="fill-emerald-500" />
        <circle cx={x(manualBatch)} cy={y(value(evaluations[manualBatch - 1]))} r="4" className="fill-amber-500" />
        {[1, Math.round(maxBatch / 4), Math.round(maxBatch / 2), Math.round(3 * maxBatch / 4), maxBatch]
          .filter((batch, index, all) => batch >= 1 && all.indexOf(batch) === index)
          .map(batch => (
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

function Equation({ title, formula, children }: { title: string; formula: string; children: ReactNode }) {
  return (
    <div className="rounded-lg border bg-background px-4 py-4">
      <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">{title}</div>
      <div className="mt-2 font-mono text-sm">{formula}</div>
      <p className="mt-2 text-xs leading-relaxed text-muted-foreground">{children}</p>
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
    targetBadShare: 10 ** state.targetBadShareExp,
    maxBatchSize: state.maxBatchSize,
    halfBatchUtilization: state.halfBatchUtilization,
    manualBatchSize: state.manualBatchSize,
  }), [state])

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

  const best = result.certificateOptimal
  const upperCount = best.cleanUpperBound * result.requestCount
  const oneHitLimit = Math.max(0, Math.ceil(result.costMultiplier) - 1)

  return (
    <div className="min-h-screen bg-background">
      <div className="mx-auto max-w-6xl px-4 py-6">
        <div className="mb-6 flex items-start justify-between gap-4">
          <div>
            <div className="mb-1.5 text-[11px] font-mono uppercase tracking-[0.22em] text-muted-foreground">Interactive · Physical batching only</div>
            <h1 className="text-2xl font-bold tracking-tight">Replay audit batching certificate</h1>
            <p className="mt-1 max-w-3xl text-sm leading-relaxed text-muted-foreground">
              Optimize random replay auditing when the physical server batch is also the accounting group and requests may share its aggregate compute cap.
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
            <div className="text-sm font-medium">Certificate after zero failed batches</div>
            <div className="mt-2 font-mono text-5xl font-bold tracking-tight md:text-6xl">{fmtPct(best.cleanUpperBound)}</div>
            <p className="mt-3 max-w-4xl text-sm leading-relaxed text-muted-foreground">
              At {fmtPct(result.confidence)} confidence, the best physical-batch policy rules out populations in which this fraction or more of requests cost at least {fmtMultiplier(result.costMultiplier)}. That is about {fmtOneIn(best.cleanUpperBound)}, corresponding to {fmtCount(upperCount, 2)} requests across the selected population.
            </p>
            <div className="mt-5 grid grid-cols-2 gap-3 lg:grid-cols-5">
              <StatBox label="Optimal batch" value={`b=${best.batchSize}`} detail={`${best.failuresNeeded} bad job${best.failuresNeeded === 1 ? "" : "s"} needed to fail`} />
              <StatBox label="Utilization" value={fmtPct(best.utilization)} detail={`${fmtCount(best.auditedRequests, 2)} requests audited`} />
              <StatBox label="Independent batches" value={fmtCount(best.batchCount, 2)} detail={`Each capped at ${best.batchSize}C`} />
              <StatBox label="Target detection" value={fmtProbability(best.targetDetection)} detail={`At ${fmtPct(result.targetBadShare)} prevalence`} />
              <StatBox label="Full-capacity requests" value={fmtCount(result.fullUtilizationCapacity, 2)} detail={`${fmtPct(result.replayShare)} of nC`} />
            </div>
            <div className="mt-5 rounded-lg border border-amber-500/30 bg-amber-500/10 px-4 py-3 text-sm leading-relaxed text-amber-900 dark:text-amber-100">
              This is a prevalence certificate, not a proof that every request costs at most C. A population at the displayed bound would still produce a clean audit with probability {fmtPct(result.missProbability)}.
            </div>
          </CardContent>
        </Card>

        <div className="mb-4 grid grid-cols-1 gap-4 xl:grid-cols-3">
          <PolicyCard title="Best clean certificate" description="Minimizes the upper confidence bound after a clean audit." policy={result.certificateOptimal} result={result} emphasized />
          <PolicyCard title="Manual policy" description="Uses the batch size selected in the server controls." policy={result.manual} result={result} />
          <PolicyCard title="Full physical batch" description="Maximizes occupancy but permits the most cross-subsidization." policy={result.fullBatch} result={result} />
        </div>

        <div className="mb-4 grid grid-cols-1 gap-4 lg:grid-cols-2">
          <Card>
            <CardHeader className="pb-3"><CardTitle className="text-sm font-semibold">Clean upper bound by batch size</CardTitle></CardHeader>
            <CardContent>
              <LineChart evaluations={result.evaluations} value={item => item.cleanUpperBound} logScale optimalBatch={result.certificateOptimal.batchSize} manualBatch={result.manual.batchSize} yLabel="Clean-audit prevalence upper bound" />
              <p className="mt-2 text-xs text-muted-foreground">Lower is stronger. Green marks the optimum; amber marks the manual policy. Vertical scale is logarithmic.</p>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-3"><CardTitle className="text-sm font-semibold">Detection power at target prevalence</CardTitle></CardHeader>
            <CardContent>
              <LineChart evaluations={result.evaluations} value={item => item.targetDetection} logScale={false} optimalBatch={result.targetOptimal.batchSize} manualBatch={result.manual.batchSize} yLabel="Detection probability at target prevalence" />
              <p className="mt-2 text-xs text-muted-foreground">Green marks the policy with maximum power against q={fmtPct(result.targetBadShare)}.</p>
            </CardContent>
          </Card>
        </div>

        <div className="mb-4 grid grid-cols-1 gap-4 md:grid-cols-2">
          <Card>
            <CardHeader className="pb-3"><CardTitle className="text-sm font-semibold">Population and statistical target</CardTitle></CardHeader>
            <CardContent className="space-y-5">
              <LogSlider label="Committed requests" tooltip="Population size n. Requests and outputs must be committed before sampling." valueExp={state.requestCountExp} onChange={value => setField("requestCountExp", value)} min={6} max={14} step={0.05} formatValue={value => fmtCount(Math.round(value), 2)} />
              <LogSlider label="Replay compute share" tooltip="ρ: audit capacity as a share of total claimed nC compute." valueExp={state.replayShareExp} onChange={value => setField("replayShareExp", value)} min={-10} max={-0.3} step={0.05} formatValue={fmtPct} />
              <LogSlider label="Bad-request cost threshold" tooltip="a: bound the prevalence of requests costing at least aC." valueExp={state.costMultiplierExp} onChange={value => setField("costMultiplierExp", value)} min={Math.log10(1.01)} max={3} step={0.01} formatValue={fmtMultiplier} />
              <LogSlider label="Detection confidence" tooltip="1−δ. The clean outcome has probability at most δ at the reported upper bound." valueExp={state.detectionMissExp} onChange={value => setField("detectionMissExp", value)} min={-8} max={Math.log10(0.5)} step={0.01} formatValue={miss => `${fmtPct(1 - miss)} confidence`} />
              <LogSlider label="Target bad-request prevalence" tooltip="q used only for the detection-power chart and comparisons." valueExp={state.targetBadShareExp} onChange={value => setField("targetBadShareExp", value)} min={-10} max={Math.log10(0.5)} step={0.05} formatValue={fmtPct} />
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3"><CardTitle className="text-sm font-semibold">Physical server and utilization</CardTitle></CardHeader>
            <CardContent className="space-y-5">
              <LinearSlider label="Maximum physical batch" tooltip="B: full relative throughput is reached at this batch size." value={state.maxBatchSize} onChange={value => setField("maxBatchSize", Math.round(value))} min={1} max={512} step={1} formatValue={value => `${Math.round(value)} requests`} />
              <LinearSlider label="Throughput at half batch" tooltip="Defines u(b)=(b/B)^γ. The default 50% gives linear utilization." value={state.halfBatchUtilization} onChange={value => setField("halfBatchUtilization", value)} min={0.05} max={1} step={0.01} formatValue={fmtPct} />
              <LinearSlider label="Manual physical batch" tooltip="Used for the manual policy card and amber chart marker." value={Math.min(state.manualBatchSize, state.maxBatchSize)} onChange={value => setField("manualBatchSize", Math.round(value))} min={1} max={state.maxBatchSize} step={1} formatValue={value => `${Math.round(value)} requests`} />
              <div className="rounded-lg border bg-muted/30 px-4 py-4 text-xs leading-relaxed text-muted-foreground">
                <div className="font-medium text-foreground">Current utilization model</div>
                <div className="mt-2 font-mono text-sm text-foreground">u(b) = (b / {result.maxBatchSize})^{result.utilizationGamma.toFixed(3)}</div>
                <div className="mt-2">Manual b={result.manual.batchSize} runs at {fmtPct(result.manual.utilization)} throughput and audits {fmtCount(result.manual.auditedRequests, 2)} complete-batch requests.</div>
              </div>
              <div className="rounded-lg border bg-muted/30 px-4 py-4 text-xs leading-relaxed text-muted-foreground">
                <div className="font-medium text-foreground">One-hit region</div>
                <div className="mt-2">One {fmtMultiplier(result.costMultiplier)} request necessarily breaks any batch with b≤{oneHitLimit}. Crossing the next integer boundary usually raises the required collision count from one to two.</div>
              </div>
            </CardContent>
          </Card>
        </div>

        <Card className="mb-4">
          <CardHeader className="pb-3"><CardTitle className="text-sm font-semibold">Model and equations</CardTitle></CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
              <Equation title="Adversarial alternative" formula="H₁(a,q): at least qn requests cost ≥ aC">The least detectable construction sets those requests to exactly aC and every other request to zero.</Equation>
              <Equation title="Audit capacity" formula="K_b=floor(ρnu(b)); M_b=floor(K_b/b)">K_b is capacity after utilization loss; M_b is the number of complete independently randomized physical batches.</Equation>
              <Equation title="Failure threshold" formula="r_b=floor(b/a)+1">A batch fails only when at least r_b bad requests collide. Equality with the bC cap passes.</Equation>
              <Equation title="Batch pass probability" formula="s_b(q)=P[Binomial(b,q)<r_b]">For an enormous committed population and a tiny sample share, binomial sampling closely approximates sampling without replacement.</Equation>
              <Equation title="Detection power" formula="D_b(q)=1−s_b(q)^(M_b)">This is the probability that at least one physical batch exceeds its aggregate compute cap.</Equation>
              <Equation title="Clean certificate" formula="s_b(q_U)^(M_b)=δ">After zero failures, q≥q_U is rejected at significance δ; the dashboard chooses b minimizing q_U.</Equation>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3"><CardTitle className="text-sm font-semibold">Plain-language interpretation</CardTitle></CardHeader>
          <CardContent className="space-y-3 text-sm leading-relaxed text-muted-foreground">
            <p>The optimal physical batch is b={best.batchSize}. It audits {fmtCount(best.auditedRequests, 2)} requests in {fmtCount(best.batchCount, 2)} independent batches, and requires {best.failuresNeeded} sampled {fmtMultiplier(result.costMultiplier)} request{best.failuresNeeded === 1 ? "" : "s"} in one batch to reject.</p>
            <p>If all batches finish within their aggregate caps, a population with prevalence q={fmtPct(best.cleanUpperBound)} would pass only {fmtPct(result.missProbability)} of repeated audits. The resulting one-sided {fmtPct(result.confidence)} confidence statement is q&lt;{fmtPct(best.cleanUpperBound)} under the stated model.</p>
            <p>The least-detectable two-point construction has average replay cost qaC. It exceeds the claimed average C once q&gt;{fmtPct(1 / result.costMultiplier)}, but small physical batches can detect tail concentration well below that average-cost break-even point.</p>
            <p className="text-xs">Assumptions: precommitment before sampling; uniform random sampling; hard per-physical-batch cap bC; no compute borrowing across batches; observable failures; and zero failed batches in the reported audit outcome.</p>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

function StatBox({ label, value, detail }: { label: string; value: string; detail: string }) {
  return (
    <div className="rounded-lg border bg-background px-3 py-3">
      <div className="text-xs font-medium text-muted-foreground">{label}</div>
      <div className="mt-2 font-mono text-2xl font-semibold leading-none">{value}</div>
      <div className="mt-2 text-xs text-muted-foreground">{detail}</div>
    </div>
  )
}
