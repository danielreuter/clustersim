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
  sweep,
  logRange,
  linRange,
  HARDWARE,
  WORKLOADS,
  VERIFIER_FULL,
  honestLoadFromFractions,
  type Scenario,
  type GammaResult,
  type SweepPoint,
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

// ---------------------------------------------------------------------------
// Mini bar chart for Γ factors
// ---------------------------------------------------------------------------

function GammaBar({ label, value, max }: { label: string; value: number; max: number }) {
  const width = Number.isFinite(value) ? Math.min(100, (Math.log10(Math.max(1, value)) / Math.log10(Math.max(10, max))) * 100) : 100
  const isInf = !Number.isFinite(value)
  return (
    <div className="flex items-center gap-2 text-sm">
      <span className="w-20 text-right text-muted-foreground shrink-0">{label}</span>
      <div className="flex-1 h-5 bg-muted rounded overflow-hidden">
        <div
          className={`h-full rounded transition-all ${isInf ? "bg-red-500" : "bg-primary"}`}
          style={{ width: `${width}%` }}
        />
      </div>
      <span className="w-16 text-right font-mono shrink-0">{fmtGamma(value)}</span>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Sparkline SVG for sweep
// ---------------------------------------------------------------------------

function SweepChart({ points, paramLabel }: { points: SweepPoint[]; paramLabel: string }) {
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
      <text x={PAD.left} y={H - 2} fontSize={9} fill="currentColor" opacity={0.4}>{fmtBw(finitePoints[0].value)}</text>
      <text x={PAD.left + w} y={H - 2} textAnchor="end" fontSize={9} fill="currentColor" opacity={0.4}>{fmtBw(finitePoints[finitePoints.length - 1].value)}</text>
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
// Main Dashboard
// ---------------------------------------------------------------------------

export function GammaDashboard() {
  const [hwKey, setHwKey] = useState("gb200-nvl72")
  const [wlKey, setWlKey] = useState("inf-1t")
  const [computeFrac, setComputeFrac] = useState(50)
  const [memoryFrac, setMemoryFrac] = useState(50)
  const [alpha, setAlpha] = useState(100)
  const [bOutExp, setBOutExp] = useState(Math.log10(20e3)) // log10(bytes/s)
  const [bInExp, setBInExp] = useState(Math.log10(100e3))
  const [sanitization, setSanitization] = useState(true)
  const [epochS, setEpochS] = useState(5)

  const scenario: Scenario = useMemo(() => {
    const hw = HARDWARE[hwKey]
    return {
      hardware: hw,
      honest: honestLoadFromFractions(hw, computeFrac / 100, memoryFrac / 100),
      verifier: {
        alpha: alpha / 100,
        covertIngressBps: 10 ** bInExp,
        covertEgressBps: 10 ** bOutExp,
        survivingStateBytes: VERIFIER_FULL.survivingStateBytes,
        epochSeconds: epochS,
        downtimeSeconds: VERIFIER_FULL.downtimeSeconds,
        sanitizationEnabled: sanitization,
      },
      covert: WORKLOADS[wlKey],
    }
  }, [hwKey, wlKey, computeFrac, memoryFrac, alpha, bOutExp, bInExp, sanitization, epochS])

  const result = useMemo(() => simulate(scenario), [scenario])

  // Sweep: b_out
  const bOutSweep = useMemo(
    () =>
      sweep(
        scenario,
        (s, v) => ({ ...s, verifier: { ...s.verifier, covertEgressBps: v } }),
        logRange(1, 7, 60), // 10 B/s to 10 MB/s
      ),
    [scenario],
  )

  // Sweep: proven compute share (αf*/Ĝ)
  const computeSweep = useMemo(
    () =>
      sweep(
        scenario,
        (s, v) => ({
          ...s,
          honest: { ...s.honest, claimedComputeFlops: v * s.hardware.computeFlops },
          verifier: { ...s.verifier, alpha: 1 },
        }),
        linRange(0.01, 0.99, 50),
      ),
    [scenario],
  )

  const gammaMax = Math.max(result.gammaDuty, result.gammaCompute, result.gammaIngress, result.gammaEgress, 10)

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
            <div className="flex items-baseline gap-4 mb-4">
              <span className="text-5xl font-bold font-mono tracking-tight">
                {fmtGamma(result.gamma)}
              </span>
              <span className="text-sm text-muted-foreground">slowdown</span>
              <span className={`text-sm font-medium ml-auto ${DOMINANT_COLORS[result.dominant]}`}>
                Bottleneck: {DOMINANT_LABELS[result.dominant]}
              </span>
            </div>

            {result.reason && (
              <p className="text-sm text-muted-foreground mb-4 italic">{result.reason}</p>
            )}

            {/* Γ factor bars */}
            <div className="space-y-1.5 mb-4">
              <GammaBar label="Duty" value={result.gammaDuty} max={gammaMax} />
              <GammaBar label="Compute" value={result.gammaCompute} max={gammaMax} />
              <GammaBar label="Ingress" value={result.gammaIngress} max={gammaMax} />
              <GammaBar label="Egress" value={result.gammaEgress} max={gammaMax} />
            </div>

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
              <div>
                <div className="text-muted-foreground">Reload time</div>
                <div className="font-mono">{fmtTime(result.tReload)}</div>
              </div>
              <div>
                <div className="text-muted-foreground">Memory margin</div>
                <div className={`font-mono ${result.fitMarginBytes < 0 ? "text-red-600" : ""}`}>
                  {fmtBytes(result.fitMarginBytes)}
                </div>
              </div>
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
              <div>
                <Label className="text-xs text-muted-foreground">Covert Workload</Label>
                <Select value={wlKey} onValueChange={setWlKey}>
                  <SelectTrigger className="mt-1"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {Object.entries(WORKLOADS).map(([k, v]) => (
                      <SelectItem key={k} value={k}>{v.label}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div>
                <div className="flex justify-between">
                  <Label className="text-xs text-muted-foreground">Proven honest compute (αf*/Ĝ)</Label>
                  <span className="text-xs font-mono">{computeFrac}%</span>
                </div>
                <Slider
                  value={[computeFrac]}
                  onValueChange={([v]) => setComputeFrac(v)}
                  min={0} max={99} step={1}
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
                  min={0} max={99} step={1}
                  className="mt-1"
                />
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
                  min={1} max={7} step={0.1}
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
                  min={1} max={7} step={0.1}
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
                {sanitization && (
                  <span className="text-xs font-mono text-muted-foreground ml-auto">τ = {epochS}s</span>
                )}
              </div>
              {sanitization && (
                <div>
                  <div className="flex justify-between">
                    <Label className="text-xs text-muted-foreground">Epoch length (τ)</Label>
                    <span className="text-xs font-mono">{epochS}s</span>
                  </div>
                  <Slider
                    value={[epochS]}
                    onValueChange={([v]) => setEpochS(v)}
                    min={1} max={600} step={1}
                    className="mt-1"
                  />
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* === Sweeps === */}
        <Tabs defaultValue="egress">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="egress">Egress Sweep</TabsTrigger>
            <TabsTrigger value="compute">Compute Sweep</TabsTrigger>
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
                <CardTitle className="text-sm font-semibold">Γ vs Proven Honest Compute Share</CardTitle>
              </CardHeader>
              <CardContent>
                <SweepChart
                  points={computeSweep.map((p) => ({ ...p, value: p.value * 1e6 }))}
                  paramLabel="αf*/Ĝ"
                />
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}
