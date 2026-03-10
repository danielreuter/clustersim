"use client"

import { useState, useMemo } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import {
  simulate,
  logRange,
  HARDWARE,
  WORKLOADS_V2,
  MODEL_MAP,
  GPU_MAP,
  honestLoadFromFractions,
  type Scenario,
  type GammaResult,
  type SweepPoint,
  type CovertWorkloadInference,
  type CovertWorkloadTraining,
  type TrainingSyncPolicy,
  type SimulationSnapshot,
} from "@/lib/sim"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function fmt(n: number, decimals = 1): string {
  if (!Number.isFinite(n)) return "∞"
  if (n >= 1e15) return `${(n / 1e15).toFixed(decimals)} PF/s`
  if (n >= 1e12) return `${(n / 1e12).toFixed(decimals)} TF/s`
  if (n >= 1e9) return `${(n / 1e9).toFixed(decimals)} G`
  if (n >= 1e6) return `${(n / 1e6).toFixed(decimals)} M`
  if (n >= 1e3) return `${(n / 1e3).toFixed(decimals)} K`
  return n.toFixed(decimals)
}

function fmtBytes(n: number): string {
  if (n >= 1e12) return `${(n / 1e12).toFixed(1)} TB`
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)} GB`
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)} MB`
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)} KB`
  return `${n.toFixed(0)} B`
}

function fmtBw(bps: number): string {
  if (bps >= 1e9) return `${(bps / 1e9).toFixed(1)} GB/s`
  if (bps >= 1e6) return `${(bps / 1e6).toFixed(1)} MB/s`
  if (bps >= 1e3) return `${(bps / 1e3).toFixed(1)} KB/s`
  return `${bps.toFixed(0)} B/s`
}

function fmtGamma(g: number): string {
  if (!Number.isFinite(g)) return "∞"
  if (g >= 1000) return `${(g / 1000).toFixed(1)}K×`
  return `${g.toFixed(1)}×`
}

function fmtTime(s: number): string {
  if (!Number.isFinite(s)) return "∞"
  if (s >= 86400) return `${(s / 86400).toFixed(1)} days`
  if (s >= 3600) return `${(s / 3600).toFixed(1)} hrs`
  if (s >= 60) return `${(s / 60).toFixed(1)} min`
  return `${s.toFixed(2)}s`
}

const DOMINANT_LABELS: Record<string, string> = {
  "memory-fit": "Memory (doesn't fit)",
  duty: "Sanitization (reload time)",
  compute: "Compute",
  ingress: "Ingress bandwidth",
  egress: "Egress bandwidth",
}

const DOMINANT_COLORS: Record<string, string> = {
  "memory-fit": "text-red-600",
  duty: "text-purple-600",
  compute: "text-blue-600",
  ingress: "text-amber-600",
  egress: "text-orange-600",
}

const REGIME_LABELS: Record<string, string> = {
  compute: "Compute-bound",
  "memory-bandwidth": "Memory BW-bound",
  "under-batched": "Under-batched",
  memory: "Memory-limited",
  latency: "Latency-limited",
  comm: "Comm-limited",
}

const REGIME_COLORS: Record<string, string> = {
  compute: "bg-blue-100 text-blue-800",
  "memory-bandwidth": "bg-amber-100 text-amber-800",
  "under-batched": "bg-red-100 text-red-800",
}

// ---------------------------------------------------------------------------
// Operational overhead bar — red if it's the bottleneck
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Inline legend dot
// ---------------------------------------------------------------------------

function LegendDot({ color, label }: { color: string; label: string }) {
  return (
    <span className="inline-flex items-center gap-1 text-[11px] text-muted-foreground">
      <span className={`inline-block w-2.5 h-2.5 rounded-sm ${color}`} />
      {label}
    </span>
  )
}

function OpBar({ label, value, max, isBottleneck, disabled }: { label: string; value: number; max: number; isBottleneck: boolean; disabled?: boolean }) {
  const width = Number.isFinite(value) ? Math.min(100, (Math.log10(Math.max(1, value)) / Math.log10(Math.max(10, max))) * 100) : 100
  const isInf = !Number.isFinite(value)
  if (disabled) {
    return (
      <div className="flex items-center gap-2 text-sm opacity-30">
        <span className="w-20 text-right text-muted-foreground shrink-0">{label}</span>
        <div className="flex-1 h-5 bg-muted rounded overflow-hidden" />
        <span className="w-16 text-right font-mono shrink-0">--</span>
      </div>
    )
  }
  return (
    <div className="flex items-center gap-2 text-sm">
      <span className={`w-20 text-right shrink-0 ${isBottleneck ? "text-red-600 font-medium" : "text-muted-foreground"}`}>{label}</span>
      <div className="flex-1 h-5 bg-muted rounded overflow-hidden">
        <div
          className={`h-full rounded transition-all ${isInf ? "bg-red-500" : isBottleneck ? "bg-red-500" : "bg-foreground/25"}`}
          style={{ width: `${width}%` }}
        />
      </div>
      <span className={`w-16 text-right font-mono shrink-0 ${isBottleneck ? "text-red-600" : ""}`}>{fmtGamma(value)}</span>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Epoch timeline — three-segment bar: download | operational | sanitization
// ---------------------------------------------------------------------------

function EpochTimeline({ result, epochS, downtimeS, disabled }: { result: GammaResult; epochS: number; downtimeS: number; disabled?: boolean }) {
  if (!Number.isFinite(epochS) || epochS <= 0) return null

  if (disabled) {
    return (
      <div className="flex items-center gap-2 text-sm opacity-30">
        <span className="w-20 text-right text-muted-foreground shrink-0">--</span>
        <div className="flex-1 h-5 bg-muted rounded overflow-hidden" />
        <span className="w-16 text-right text-muted-foreground shrink-0">--</span>
      </div>
    )
  }

  const tReload = result.tReload
  const tCovert = Math.max(0, result.tCovert)
  const tSanitize = downtimeS

  // Fractions of the full epoch
  const downloadFrac = Math.min(tReload / epochS, 1)
  const operationalFrac = Math.min(tCovert / epochS, 1 - downloadFrac)
  const sanitizeFrac = Math.min(tSanitize / epochS, 1 - downloadFrac - operationalFrac)

  return (
    <div className="flex items-center gap-2 text-sm">
      <span className="w-20 text-right text-muted-foreground shrink-0">{fmtGamma(result.gammaDuty)}</span>
      <div className="flex-1 h-5 bg-muted rounded overflow-hidden flex">
        <div className="h-full bg-amber-400 transition-all" style={{ width: `${downloadFrac * 100}%` }} />
        <div className="h-full bg-blue-300 transition-all" style={{ width: `${operationalFrac * 100}%` }} />
        <div className="h-full bg-foreground/15 transition-all" style={{ width: `${sanitizeFrac * 100}%` }} />
      </div>
      <span className="w-16 text-right text-muted-foreground shrink-0">{operationalFrac > 0 ? `${(operationalFrac * 100).toFixed(0)}%` : "0%"}</span>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Memory fit bar — covert + honest vs total HBM
// ---------------------------------------------------------------------------

function MemoryFitBar({ result, totalHbm, honestMem, covertState }: { result: GammaResult; totalHbm: number; honestMem: number; covertState: number }) {
  const honestFrac = Math.min(honestMem / totalHbm, 1)
  const covertFrac = Math.min(covertState / totalHbm, 1 - honestFrac)
  const overflows = result.fitMarginBytes < 0

  return (
    <div className="flex items-center gap-2 text-sm">
      <span className={`w-20 text-right shrink-0 ${overflows ? "text-red-600" : "text-muted-foreground"}`}>{overflows ? "No fit" : "Fits"}</span>
      <div className="flex-1 h-5 bg-muted rounded overflow-hidden">
        <div className="h-full flex">
          <div
            className="h-full bg-blue-300 transition-all"
            style={{ width: `${honestFrac * 100}%` }}
            title={`Honest: ${fmtBytes(honestMem)}`}
          />
          <div
            className={`h-full transition-all ${overflows ? "bg-red-400" : "bg-amber-400"}`}
            style={{ width: `${covertFrac * 100}%` }}
            title={`Covert: ${fmtBytes(covertState)}`}
          />
        </div>
      </div>
      <span className={`w-16 text-right shrink-0 ${overflows ? "text-red-600" : "text-muted-foreground"}`}>{fmtBytes(totalHbm)}</span>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Sparkline SVG for sweep
// ---------------------------------------------------------------------------

function SweepChart({ points, paramLabel, xFormatter }: { points: SweepPoint[]; paramLabel: string; xFormatter?: (v: number) => string }) {
  const W = 400
  const H = 120
  const PAD = { top: 10, right: 10, bottom: 24, left: 50 }
  const w = W - PAD.left - PAD.right
  const h = H - PAD.top - PAD.bottom

  const finitePoints = points.filter((p) => Number.isFinite(p.result.gamma))
  if (finitePoints.length < 2) {
    return <div className="text-sm text-muted-foreground italic">All values are ∞ (sanitization dominates)</div>
  }

  const xMin = Math.log10(finitePoints[0].value)
  const xMax = Math.log10(finitePoints[finitePoints.length - 1].value)
  const yMax = Math.log10(Math.max(...finitePoints.map((p) => p.result.gamma), 10))
  const yMin = 0

  const toX = (v: number) => PAD.left + ((Math.log10(v) - xMin) / (xMax - xMin)) * w
  const toY = (g: number) => PAD.top + h - ((Math.log10(Math.max(1, g)) - yMin) / (yMax - yMin)) * h

  const path = finitePoints.map((p, i) => `${i === 0 ? "M" : "L"}${toX(p.value).toFixed(1)},${toY(p.result.gamma).toFixed(1)}`).join(" ")

  const fmtX = xFormatter ?? fmtBw

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full max-w-md">
      {/* axes */}
      <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={PAD.top + h} stroke="currentColor" strokeOpacity={0.2} />
      <line x1={PAD.left} y1={PAD.top + h} x2={PAD.left + w} y2={PAD.top + h} stroke="currentColor" strokeOpacity={0.2} />
      {/* 1× line */}
      <line x1={PAD.left} y1={toY(1)} x2={PAD.left + w} y2={toY(1)} stroke="currentColor" strokeOpacity={0.1} strokeDasharray="4 2" />
      <text x={PAD.left - 4} y={toY(1) + 3} textAnchor="end" fontSize={9} fill="currentColor" opacity={0.4}>1×</text>
      {/* top label */}
      <text x={PAD.left - 4} y={PAD.top + 8} textAnchor="end" fontSize={9} fill="currentColor" opacity={0.4}>{fmtGamma(10 ** yMax)}</text>
      {/* x labels */}
      <text x={PAD.left} y={H - 2} fontSize={9} fill="currentColor" opacity={0.4}>{fmtX(finitePoints[0].value)}</text>
      <text x={PAD.left + w} y={H - 2} textAnchor="end" fontSize={9} fill="currentColor" opacity={0.4}>{fmtX(finitePoints[finitePoints.length - 1].value)}</text>
      <text x={PAD.left + w / 2} y={H - 2} textAnchor="middle" fontSize={9} fill="currentColor" opacity={0.5}>{paramLabel}</text>
      {/* line */}
      <path d={path} fill="none" stroke="var(--primary)" strokeWidth={2} />
      {/* dots for dominant transitions */}
      {finitePoints.map((p, i) => {
        const prev = i > 0 ? finitePoints[i - 1] : null
        if (prev && prev.result.dominant !== p.result.dominant) {
          return <circle key={i} cx={toX(p.value)} cy={toY(p.result.gamma)} r={3} fill="var(--destructive)" />
        }
        return null
      })}
    </svg>
  )
}

// ---------------------------------------------------------------------------
// Context length sweep for v2
// ---------------------------------------------------------------------------

function useContextSweep(scenario: Scenario) {
  return useMemo(() => {
    const wl = scenario.covert
    if (!("backend" in wl) || wl.kind !== "inference") return []

    const ctxValues = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
    return ctxValues.map((ctx) => {
      const modified: Scenario = {
        ...scenario,
        covert: { ...wl, contextLength: ctx } as CovertWorkloadInference,
      }
      return { value: ctx, result: simulate(modified) }
    })
  }, [scenario])
}

// ---------------------------------------------------------------------------
// Main Dashboard
// ---------------------------------------------------------------------------

export function GammaDashboard() {
  const [hwKey, setHwKey] = useState("gb200-nvl72")
  const [computeFrac, setComputeFrac] = useState(50)
  const [memoryFrac, setMemoryFrac] = useState(50)
  const [alpha, setAlpha] = useState(100)
  const [bOutExp, setBOutExp] = useState(Math.log10(20e3))
  const [bInExp, setBInExp] = useState(Math.log10(100e3))
  const [sanitization, setSanitization] = useState(true)
  const [epochSExp, setEpochSExp] = useState(Math.log10(5)) // log10(seconds)
  const epochS = 10 ** epochSExp
  const [downtimeS, setDowntimeS] = useState(0.25)
  const [survivingGB, setSurvivingGB] = useState(17)

  // v2-specific state
  const [v2WlKey, setV2WlKey] = useState(Object.keys(WORKLOADS_V2)[0])
  const [v2ModelKey, setV2ModelKey] = useState("Llama 3 70B")
  const [v2GpuKey, setV2GpuKey] = useState("H100")
  const [v2NGpu, setV2NGpu] = useState(8)
  const [v2CtxExp, setV2CtxExp] = useState(11) // 2^11 = 2048
  const [v2WorkloadKind, setV2WorkloadKind] = useState<"inference" | "training">("inference")
  const [v2SyncMode, setV2SyncMode] = useState<"none" | "checkpoint" | "periodic-updates">("none")
  const [v2UsePreset, setV2UsePreset] = useState(true)

  const v2ContextLength = Math.round(2 ** v2CtxExp)

  const scenario: Scenario = useMemo(() => {
    const hw = HARDWARE[hwKey]
    const verifier = {
      alpha: alpha / 100,
      covertIngressBps: 10 ** bInExp,
      covertEgressBps: 10 ** bOutExp,
      survivingStateBytes: survivingGB * 1e9,
      epochSeconds: epochS,
      downtimeSeconds: downtimeS,
      sanitizationEnabled: sanitization,
    }

    let covert: CovertWorkloadInference | CovertWorkloadTraining
    if (v2UsePreset) {
      covert = WORKLOADS_V2[v2WlKey]
    } else if (v2WorkloadKind === "inference") {
      covert = {
        label: `${v2ModelKey} on ${v2NGpu}×${v2GpuKey}`,
        kind: "inference",
        unit: "token",
        backend: "roofline-lite",
        modelKey: v2ModelKey,
        gpuKey: v2GpuKey,
        nGpu: v2NGpu,
        contextLength: v2ContextLength,
      }
    } else {
      const syncPolicy: TrainingSyncPolicy =
        v2SyncMode === "none"
          ? { mode: "none" }
          : v2SyncMode === "checkpoint"
            ? { mode: "checkpoint", bytesOutPerSync: MODEL_MAP[v2ModelKey].totalParams * 2, tokensPerSync: 1e6 }
            : { mode: "periodic-updates", bytesInPerSync: MODEL_MAP[v2ModelKey].totalParams * 2, bytesOutPerSync: MODEL_MAP[v2ModelKey].totalParams * 2, tokensPerSync: 1e6 }

      covert = {
        label: `Train ${v2ModelKey} on ${v2NGpu}×${v2GpuKey}`,
        kind: "training",
        unit: "train-token",
        backend: "roofline-lite",
        modelKey: v2ModelKey,
        gpuKey: v2GpuKey,
        nGpu: v2NGpu,
        syncPolicy,
      }
    }

    return {
      hardware: hw,
      honest: honestLoadFromFractions(hw, computeFrac / 100, memoryFrac / 100),
      verifier,
      covert,
    }
  }, [hwKey, computeFrac, memoryFrac, alpha, bOutExp, bInExp, sanitization, epochSExp, downtimeS, survivingGB, v2WlKey, v2ModelKey, v2GpuKey, v2NGpu, v2ContextLength, v2WorkloadKind, v2SyncMode, v2UsePreset])

  const result = useMemo(() => simulate(scenario), [scenario])

  const snapshot: SimulationSnapshot = useMemo(() => ({ input: scenario, output: result }), [scenario, result])
  const [copied, setCopied] = useState(false)
  const handleCopy = () => {
    navigator.clipboard.writeText(JSON.stringify(snapshot, null, 2))
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  // Sweep: b_out
  const bOutSweep = useMemo(
    () => logRange(1, 10, 60).map((v) => ({
      value: v,
      result: simulate({ ...scenario, verifier: { ...scenario.verifier, covertEgressBps: v } }),
    })),
    [scenario],
  )

  // Sweep: proven compute share (log-spaced from 1% to 99%)
  const computeSweep = useMemo(
    () => logRange(-2, Math.log10(0.99), 50).map((v) => ({
      value: v,
      result: simulate({
        ...scenario,
        honest: { ...scenario.honest, claimedComputeFlops: v * scenario.hardware.computeFlops },
        verifier: { ...scenario.verifier, alpha: 1 },
      }),
    })),
    [scenario],
  )

  // Context sweep (v2 inference only)
  const ctxSweep = useContextSweep(scenario)

  const opMax = Math.max(result.gammaCompute, result.gammaIngress, result.gammaEgress, 10)
  const opBottleneck: "compute" | "ingress" | "egress" = result.gammaCompute >= result.gammaIngress && result.gammaCompute >= result.gammaEgress
    ? "compute"
    : result.gammaIngress >= result.gammaEgress
      ? "ingress"
      : "egress"

  const GPU_COUNTS = [1, 2, 4, 8, 16, 32, 64, 72]

  return (
    <div className="min-h-screen bg-background">
      <div className="mx-auto max-w-5xl px-4 py-6">
        {/* Header */}
        <div className="mb-6">
          <h1 className="text-2xl font-bold tracking-tight text-foreground">Covert Overhead Simulator</h1>
          <p className="text-sm text-muted-foreground mt-1">
            How much slower does a covert workload run under verification?
          </p>
        </div>

        {/* === Result Card === */}
        <Card className="mb-6">
          <CardContent className="pt-6">
            {/* Header: Γ value + copy button */}
            <div className="flex items-baseline gap-4 mb-5 h-14">
              <span className="text-5xl font-bold font-mono tracking-tight leading-none">
                {fmtGamma(result.gamma)}
              </span>
              <span className="text-sm text-muted-foreground">overhead</span>
              {result.v2 && (
                <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${REGIME_COLORS[result.v2.regime] ?? "bg-gray-100 text-gray-800"}`}>
                  {REGIME_LABELS[result.v2.regime] ?? result.v2.regime}
                </span>
              )}
              <div className="ml-auto" />
              <button
                onClick={handleCopy}
                className="p-1.5 rounded-md text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
                title="Copy config + result as JSON"
              >
                {copied ? (
                  <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M20 6 9 17l-5-5"/></svg>
                ) : (
                  <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect width="14" height="14" x="8" y="8" rx="2" ry="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/></svg>
                )}
              </button>
            </div>

            {result.reason && (
              <p className="text-sm text-muted-foreground mb-4 italic">{result.reason}</p>
            )}

            {/* --- Layer 1: Operational overhead --- */}
            <div className="mb-5">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">Operational overhead</span>
                <span className="flex items-center gap-3">
                  <LegendDot color="bg-foreground/25" label="Non-bottleneck" />
                  <LegendDot color="bg-red-500" label="Bottleneck" />
                </span>
              </div>
              <div className="space-y-1.5">
                <OpBar label="Compute" value={result.gammaCompute} max={opMax} isBottleneck={opBottleneck === "compute"} disabled={result.fitMarginBytes < 0} />
                <OpBar label="Ingress" value={result.gammaIngress} max={opMax} isBottleneck={opBottleneck === "ingress"} disabled={result.fitMarginBytes < 0} />
                <OpBar label="Egress" value={result.gammaEgress} max={opMax} isBottleneck={opBottleneck === "egress"} disabled={result.fitMarginBytes < 0} />
              </div>
            </div>

            {/* --- Layer 2: Memory fit --- */}
            <div className="mb-5">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">Memory fit</span>
                <span className="flex items-center gap-3">
                  <LegendDot color="bg-blue-300" label="Honest" />
                  <LegendDot color="bg-amber-400" label="Covert" />
                </span>
              </div>
              <MemoryFitBar
                result={result}
                totalHbm={scenario.hardware.hbmBytes}
                honestMem={scenario.honest.claimedMemoryBytes}
                covertState={result.v2 ? result.v2.nPersistBytes + result.v2.workspaceBytes : ("stateBytes" in scenario.covert ? (scenario.covert as { stateBytes: number }).stateBytes : 0)}
              />
            </div>

            {/* --- Layer 3: Sanitization cycle --- */}
            {sanitization && (
              <div className="mb-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">Sanitization cycle</span>
                  <span className="flex items-center gap-3">
                    <LegendDot color="bg-amber-400" label="Covert download" />
                    <LegendDot color="bg-blue-300" label="Operational" />
                    <LegendDot color="bg-foreground/15" label="Sanitization" />
                  </span>
                </div>
                <EpochTimeline result={result} epochS={epochS} downtimeS={downtimeS} disabled={result.fitMarginBytes < 0} />
              </div>
            )}

            {/* Key numbers */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 pt-4 border-t text-sm">
              <div>
                <div className="text-muted-foreground">Dedicated θ₀</div>
                <div className="font-mono">{fmt(result.theta0)} {scenario.covert.unit}/s</div>
              </div>
              <div>
                <div className="text-muted-foreground">Verified θ</div>
                <div className="font-mono">{fmt(result.thetaVerified)} {scenario.covert.unit}/s</div>
              </div>
              {result.v2 && (
                <>
                  <div>
                    <div className="text-muted-foreground">Batch size</div>
                    <div className="font-mono">{result.v2.optimalBatchSize}</div>
                  </div>
                  <div>
                    <div className="text-muted-foreground">Persistent state</div>
                    <div className="font-mono">{fmtBytes(result.v2.nPersistBytes)}</div>
                  </div>
                </>
              )}
            </div>
          </CardContent>
        </Card>

        {/* === Controls === */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
          {/* Presets */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-semibold">Configuration</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <Label className="text-xs text-muted-foreground">Hardware</Label>
                <Select value={hwKey} onValueChange={setHwKey}>
                  <SelectTrigger className="mt-1"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {Object.entries(HARDWARE).map(([k, v]) => (
                      <SelectItem key={k} value={k}>{v.name}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {/* Preset vs custom toggle */}
              <div className="flex items-center gap-3">
                <button
                  onClick={() => setV2UsePreset(!v2UsePreset)}
                  className={`relative inline-flex h-5 w-9 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors ${v2UsePreset ? "bg-primary" : "bg-muted"}`}
                >
                  <span className={`pointer-events-none block h-4 w-4 rounded-full bg-background shadow-lg transition-transform ${v2UsePreset ? "translate-x-4" : "translate-x-0"}`} />
                </button>
                <Label className="text-xs text-muted-foreground">Use preset</Label>
              </div>

                  {v2UsePreset ? (
                    <div>
                      <Label className="text-xs text-muted-foreground">V2 Workload Preset</Label>
                      <Select value={v2WlKey} onValueChange={setV2WlKey}>
                        <SelectTrigger className="mt-1"><SelectValue /></SelectTrigger>
                        <SelectContent>
                          {Object.entries(WORKLOADS_V2).map(([k, v]) => (
                            <SelectItem key={k} value={k}>{v.label}</SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  ) : (
                    <>
                      {/* Workload kind */}
                      <div className="flex gap-2">
                        <button
                          onClick={() => setV2WorkloadKind("inference")}
                          className={`px-2 py-1 text-xs rounded ${v2WorkloadKind === "inference" ? "bg-primary text-primary-foreground" : "bg-muted text-muted-foreground"}`}
                        >
                          Inference
                        </button>
                        <button
                          onClick={() => setV2WorkloadKind("training")}
                          className={`px-2 py-1 text-xs rounded ${v2WorkloadKind === "training" ? "bg-primary text-primary-foreground" : "bg-muted text-muted-foreground"}`}
                        >
                          Training
                        </button>
                      </div>

                      {/* Model */}
                      <div>
                        <Label className="text-xs text-muted-foreground">Model</Label>
                        <Select value={v2ModelKey} onValueChange={setV2ModelKey}>
                          <SelectTrigger className="mt-1"><SelectValue /></SelectTrigger>
                          <SelectContent>
                            {Object.keys(MODEL_MAP).map((k) => (
                              <SelectItem key={k} value={k}>{k}</SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>

                      {/* GPU */}
                      <div>
                        <Label className="text-xs text-muted-foreground">GPU</Label>
                        <Select value={v2GpuKey} onValueChange={setV2GpuKey}>
                          <SelectTrigger className="mt-1"><SelectValue /></SelectTrigger>
                          <SelectContent>
                            {Object.keys(GPU_MAP).map((k) => (
                              <SelectItem key={k} value={k}>{k}</SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>

                      {/* GPU count */}
                      <div>
                        <Label className="text-xs text-muted-foreground">GPU count</Label>
                        <Select value={String(v2NGpu)} onValueChange={(v) => setV2NGpu(Number(v))}>
                          <SelectTrigger className="mt-1"><SelectValue /></SelectTrigger>
                          <SelectContent>
                            {GPU_COUNTS.map((n) => (
                              <SelectItem key={n} value={String(n)}>{n}</SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>

                      {/* Context length (inference) */}
                      {v2WorkloadKind === "inference" && (
                        <div>
                          <div className="flex justify-between">
                            <Label className="text-xs text-muted-foreground">Context length</Label>
                            <span className="text-xs font-mono">{v2ContextLength.toLocaleString()}</span>
                          </div>
                          <Slider
                            value={[v2CtxExp]}
                            onValueChange={([v]) => setV2CtxExp(v)}
                            min={8} max={17} step={0.5}
                            className="mt-1"
                          />
                        </div>
                      )}

                      {/* Sync policy (training) */}
                      {v2WorkloadKind === "training" && (
                        <div>
                          <Label className="text-xs text-muted-foreground">Sync policy</Label>
                          <Select value={v2SyncMode} onValueChange={(v) => setV2SyncMode(v as typeof v2SyncMode)}>
                            <SelectTrigger className="mt-1"><SelectValue /></SelectTrigger>
                            <SelectContent>
                              <SelectItem value="none">No sync</SelectItem>
                              <SelectItem value="checkpoint">Checkpoint every 1M tokens</SelectItem>
                              <SelectItem value="periodic-updates">Periodic gradient sync</SelectItem>
                            </SelectContent>
                          </Select>
                        </div>
                      )}
                    </>
                  )}

              <div>
                <div className="flex justify-between">
                  <Label className="text-xs text-muted-foreground">Honest compute (f*/Ĝ)</Label>
                  <span className="text-xs font-mono">{computeFrac}%</span>
                </div>
                <Slider
                  value={[computeFrac]}
                  onValueChange={([v]) => setComputeFrac(v)}
                  min={0} max={100} step={1}
                  className="mt-1"
                />
              </div>
              <div>
                <div className="flex justify-between">
                  <Label className="text-xs text-muted-foreground">Honest memory (m*/M̂)</Label>
                  <span className="text-xs font-mono">{memoryFrac}%</span>
                </div>
                <Slider
                  value={[memoryFrac]}
                  onValueChange={([v]) => setMemoryFrac(v)}
                  min={0} max={100} step={1}
                  className="mt-1"
                />
              </div>
              <div className="text-xs text-muted-foreground pt-1 border-t">
                {"Proven compute: α \u00D7 f*/\u011C = "}{((alpha / 100) * computeFrac).toFixed(1)}{"% of \u011C"}
              </div>
            </CardContent>
          </Card>

          {/* Verification */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-semibold">Verification Parameters</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <div className="flex justify-between">
                  <Label className="text-xs text-muted-foreground">Covert egress (b_out)</Label>
                  <span className="text-xs font-mono">{fmtBw(10 ** bOutExp)}</span>
                </div>
                <Slider
                  value={[bOutExp]}
                  onValueChange={([v]) => setBOutExp(v)}
                  min={1} max={10} step={0.1}
                  className="mt-1"
                />
              </div>
              <div>
                <div className="flex justify-between">
                  <Label className="text-xs text-muted-foreground">Covert ingress (b_in)</Label>
                  <span className="text-xs font-mono">{fmtBw(10 ** bInExp)}</span>
                </div>
                <Slider
                  value={[bInExp]}
                  onValueChange={([v]) => setBInExp(v)}
                  min={1} max={10} step={0.1}
                  className="mt-1"
                />
              </div>
              <div>
                <div className="flex justify-between">
                  <Label className="text-xs text-muted-foreground">Matmul transparency (α)</Label>
                  <span className="text-xs font-mono">{alpha}%</span>
                </div>
                <Slider
                  value={[alpha]}
                  onValueChange={([v]) => setAlpha(v)}
                  min={0} max={100} step={1}
                  className="mt-1"
                />
              </div>
              <div className="flex items-center gap-3">
                <button
                  onClick={() => setSanitization(!sanitization)}
                  className={`relative inline-flex h-5 w-9 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors ${sanitization ? "bg-primary" : "bg-muted"}`}
                >
                  <span className={`pointer-events-none block h-4 w-4 rounded-full bg-background shadow-lg transition-transform ${sanitization ? "translate-x-4" : "translate-x-0"}`} />
                </button>
                <Label className="text-xs text-muted-foreground">Memory sanitization</Label>
              </div>
              {sanitization && (
                <>
                  <div>
                    <div className="flex justify-between">
                      <Label className="text-xs text-muted-foreground">Epoch length (τ)</Label>
                      <span className="text-xs font-mono">{fmtTime(epochS)}</span>
                    </div>
                    <Slider
                      value={[epochSExp]}
                      onValueChange={([v]) => setEpochSExp(v)}
                      min={-1} max={6} step={0.05}
                      className="mt-1"
                    />
                  </div>
                  <div>
                    <div className="flex justify-between">
                      <Label className="text-xs text-muted-foreground">Downtime per epoch (T)</Label>
                      <span className="text-xs font-mono">{downtimeS}s</span>
                    </div>
                    <Slider
                      value={[downtimeS]}
                      onValueChange={([v]) => setDowntimeS(v)}
                      min={0.01} max={10} step={0.01}
                      className="mt-1"
                    />
                  </div>
                  <div>
                    <div className="flex justify-between">
                      <Label className="text-xs text-muted-foreground">Covert persistence capacity (C)</Label>
                      <span className="text-xs font-mono">{survivingGB} GB</span>
                    </div>
                    <Slider
                      value={[survivingGB]}
                      onValueChange={([v]) => setSurvivingGB(v)}
                      min={0} max={200} step={1}
                      className="mt-1"
                    />
                  </div>
                </>
              )}
            </CardContent>
          </Card>
        </div>

        {/* === Sweeps === */}
        <Tabs defaultValue="egress">
          <TabsList className={`grid w-full ${ctxSweep.length > 0 ? "grid-cols-3" : "grid-cols-2"}`}>
            <TabsTrigger value="egress">Egress Sweep</TabsTrigger>
            <TabsTrigger value="compute">Compute Sweep</TabsTrigger>
            {ctxSweep.length > 0 && (
              <TabsTrigger value="context">Context Length</TabsTrigger>
            )}
          </TabsList>
          <TabsContent value="egress" className="mt-4">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-semibold">Γ vs Covert Egress Bandwidth</CardTitle>
              </CardHeader>
              <CardContent>
                <SweepChart points={bOutSweep} paramLabel="b_out" />
              </CardContent>
            </Card>
          </TabsContent>
          <TabsContent value="compute" className="mt-4">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-semibold">Γ vs Honest Compute (f*/Ĝ, with α=1)</CardTitle>
              </CardHeader>
              <CardContent>
                <SweepChart
                  points={computeSweep}
                  paramLabel="f*/Ĝ"
                  xFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                />
              </CardContent>
            </Card>
          </TabsContent>
          {ctxSweep.length > 0 && (
            <TabsContent value="context" className="mt-4">
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-semibold">Γ vs Context Length</CardTitle>
                </CardHeader>
                <CardContent>
                  <SweepChart
                    points={ctxSweep}
                    paramLabel="Context Length"
                    xFormatter={(v) => v >= 1024 ? `${(v / 1024).toFixed(0)}K` : String(v)}
                  />
                </CardContent>
              </Card>
            </TabsContent>
          )}
        </Tabs>
      </div>
    </div>
  )
}
