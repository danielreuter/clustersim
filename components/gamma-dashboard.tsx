"use client"

import { useState, useMemo, useEffect, useRef } from "react"
import { ThemeToggle } from "@/components/theme-toggle"
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
  simulateDirect,
  sweepDirect,
  logRange,
  computeHardwarePreset,
  computeCovertPreset,
  HARDWARE_PRESETS,
  COVERT_PRESETS,
  VERIFIER_FULL,
  type DirectScenario,
  type CovertWorkloadV1,
  type GammaResult,
  type SweepPoint,
} from "@/lib/sim"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function fmt(n: number, decimals = 1): string {
  if (!Number.isFinite(n)) return "—"
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

function fmtGamma(g: number, large?: boolean): string {
  if (!Number.isFinite(g)) return large ? "Infeasible" : "—"
  if (g >= 1000) return `${(g / 1000).toFixed(1)}K×`
  return `${g.toFixed(1)}×`
}

function fmtTime(s: number): string {
  if (!Number.isFinite(s)) return "—"
  if (s >= 86400) return `${(s / 86400).toFixed(1)} days`
  if (s >= 3600) return `${(s / 3600).toFixed(1)} hrs`
  if (s >= 60) return `${(s / 60).toFixed(1)} min`
  return `${s.toFixed(2)}s`
}

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
// Epoch timeline
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
  const tTotal = tSanitize + tReload + tCovert
  const overflows = tTotal > epochS

  if (overflows) {
    // Nothing fits — show grayed-out bar with infinity
    const sanitizePct = (tSanitize / tTotal) * 100
    const downloadPct = (tReload / tTotal) * 100
    const operationalPct = (tCovert / tTotal) * 100
    return (
      <div className="flex items-center gap-2 text-sm">
        <span className="w-20 text-right text-red-600 shrink-0">—</span>
        <div className="flex-1 h-5 bg-muted rounded overflow-hidden flex opacity-40">
          <div className="h-full bg-emerald-400 transition-all" style={{ width: `${sanitizePct}%` }} />
          <div className="h-full bg-amber-400 transition-all" style={{ width: `${downloadPct}%` }} />
          <div className="h-full bg-blue-300 transition-all" style={{ width: `${operationalPct}%` }} />
        </div>
        <span className="w-16 text-right text-red-600 shrink-0">No fit</span>
      </div>
    )
  }

  const sanitizeFrac = tSanitize / epochS
  const downloadFrac = tReload / epochS
  const operationalFrac = tCovert / epochS

  return (
    <div className="flex items-center gap-2 text-sm">
      <span className="w-20 text-right text-muted-foreground shrink-0">{fmtGamma(result.gammaDuty)}</span>
      <div className="flex-1 h-5 bg-muted rounded overflow-hidden flex">
        <div className="h-full bg-emerald-400 transition-all" style={{ width: `${sanitizeFrac * 100}%` }} />
        <div className="h-full bg-amber-400 transition-all" style={{ width: `${downloadFrac * 100}%` }} />
        <div className="h-full bg-blue-300 transition-all" style={{ width: `${operationalFrac * 100}%` }} />
      </div>
      <span className="w-16 text-right text-muted-foreground shrink-0">{operationalFrac > 0 ? `${(operationalFrac * 100).toFixed(0)}%` : "0%"}</span>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Memory fit bar
// ---------------------------------------------------------------------------

function MemoryFitBar({ result, totalHbm, honestMem, covertState }: { result: GammaResult; totalHbm: number; honestMem: number; covertState: number }) {
  const covertFrac = Math.min(covertState / totalHbm, 1)
  const honestFrac = Math.min(honestMem / totalHbm, 1 - covertFrac)
  const overflows = result.fitMarginBytes < 0

  return (
    <div className="flex items-center gap-2 text-sm">
      <span className={`w-20 text-right shrink-0 ${overflows ? "text-red-600" : "text-muted-foreground"}`}>{overflows ? "No fit" : "Fits"}</span>
      <div className="flex-1 h-5 bg-muted rounded overflow-hidden">
        <div className="h-full flex">
          <div
            className={`h-full transition-all ${overflows ? "bg-red-400" : "bg-amber-400"}`}
            style={{ width: `${covertFrac * 100}%` }}
            title={`Covert: ${fmtBytes(covertState)}`}
          />
          <div
            className="h-full bg-blue-300 transition-all"
            style={{ width: `${honestFrac * 100}%` }}
            title={`Honest: ${fmtBytes(honestMem)}`}
          />
        </div>
      </div>
      <span className={`w-16 text-right shrink-0 ${overflows ? "text-red-600" : "text-muted-foreground"}`}>{fmtBytes(totalHbm)}</span>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Sweep chart
// ---------------------------------------------------------------------------

function SweepChart({ points, paramLabel, xFormatter }: { points: SweepPoint[]; paramLabel: string; xFormatter?: (v: number) => string }) {
  const W = 400
  const H = 120
  const PAD = { top: 10, right: 10, bottom: 24, left: 50 }
  const w = W - PAD.left - PAD.right
  const h = H - PAD.top - PAD.bottom

  const finitePoints = points.filter((p) => Number.isFinite(p.result.gamma))
  if (finitePoints.length < 2) {
    return <div className="text-sm text-muted-foreground italic">All values infeasible</div>
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
      <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={PAD.top + h} stroke="currentColor" strokeOpacity={0.2} />
      <line x1={PAD.left} y1={PAD.top + h} x2={PAD.left + w} y2={PAD.top + h} stroke="currentColor" strokeOpacity={0.2} />
      <line x1={PAD.left} y1={toY(1)} x2={PAD.left + w} y2={toY(1)} stroke="currentColor" strokeOpacity={0.1} strokeDasharray="4 2" />
      <text x={PAD.left - 4} y={toY(1) + 3} textAnchor="end" fontSize={9} fill="currentColor" opacity={0.4}>1×</text>
      <text x={PAD.left - 4} y={PAD.top + 8} textAnchor="end" fontSize={9} fill="currentColor" opacity={0.4}>{fmtGamma(10 ** yMax)}</text>
      <text x={PAD.left} y={H - 2} fontSize={9} fill="currentColor" opacity={0.4}>{fmtX(finitePoints[0].value)}</text>
      <text x={PAD.left + w} y={H - 2} textAnchor="end" fontSize={9} fill="currentColor" opacity={0.4}>{fmtX(finitePoints[finitePoints.length - 1].value)}</text>
      <text x={PAD.left + w / 2} y={H - 2} textAnchor="middle" fontSize={9} fill="currentColor" opacity={0.5}>{paramLabel}</text>
      <path d={path} fill="none" stroke="var(--primary)" strokeWidth={2} />
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
// Log-scale slider helper
// ---------------------------------------------------------------------------

function LogSlider({ label, valueExp, onValueExpChange, min, max, step, formatValue }: {
  label: string
  valueExp: number
  onValueExpChange: (v: number) => void
  min: number
  max: number
  step: number
  formatValue: (v: number) => string
}) {
  return (
    <div>
      <div className="flex justify-between">
        <Label className="text-xs text-muted-foreground">{label}</Label>
        <span className="text-xs font-mono">{formatValue(10 ** valueExp)}</span>
      </div>
      <Slider
        value={[valueExp]}
        onValueChange={([v]) => onValueExpChange(v)}
        min={min} max={max} step={step}
        className="mt-1"
      />
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main Dashboard
// ---------------------------------------------------------------------------

// Default hardware: 8xH100
const DEFAULT_HW = computeHardwarePreset("H100", 8)
// Default covert: Llama 70B inference
const DEFAULT_COVERT = computeCovertPreset(DEFAULT_HW, COVERT_PRESETS["inf-llama70b"])

// ---------------------------------------------------------------------------
// URL hash state serialization (write on copy-link only, read on init)
// ---------------------------------------------------------------------------

type DashState = {
  cfe: number; hbe: number  // hardware: computeFlopsExp, hbmBytesExp
  cf: number; mf: number; a: number  // honest load
  bo: number; bi: number; sn: boolean; ep: number; dt: number; sg: number  // verifier
  ne: number; ge: number; die: number; dien: boolean; doe: number; doen: boolean  // covert
}

function encodeState(s: DashState): string {
  try { return btoa(JSON.stringify(s)) } catch { return "" }
}

function decodeState(hash: string): DashState | null {
  try {
    const raw = hash.startsWith("#") ? hash.slice(1) : hash
    if (!raw) return null
    const parsed = JSON.parse(atob(raw))
    // Current format has `cfe` — return directly
    if ("cfe" in parsed) return parsed as DashState
    // Legacy format: has `gk`/`ng`/`wl` keys — migrate
    if ("gk" in parsed && "ng" in parsed) {
      const hw = computeHardwarePreset(parsed.gk, parsed.ng)
      const wlKey = parsed.wl as string | undefined
      const covertConfig = wlKey && COVERT_PRESETS[wlKey] ? COVERT_PRESETS[wlKey] : null
      const covert = covertConfig ? computeCovertPreset(hw, covertConfig) : null
      return {
        cfe: Math.log10(hw.computeFlops),
        hbe: Math.log10(hw.hbmBytes),
        cf: parsed.cf ?? 50,
        mf: parsed.mf ?? 50,
        a: parsed.a ?? 100,
        bo: parsed.bo ?? Math.log10(20e3),
        bi: parsed.bi ?? Math.log10(100e3),
        sn: parsed.sn ?? true,
        ep: parsed.ep ?? Math.log10(5),
        dt: parsed.dt ?? 0.25,
        sg: parsed.sg ?? 17,
        ne: covert ? Math.log10(covert.stateBytes) : Math.log10(DEFAULT_COVERT.stateBytes),
        ge: covert ? Math.log10(covert.flopPerUnit) : Math.log10(DEFAULT_COVERT.flopPerUnit),
        die: covert && covert.ingressBytesPerUnit > 0 ? Math.log10(covert.ingressBytesPerUnit) : 0,
        dien: covert ? covert.ingressBytesPerUnit > 0 : DEFAULT_COVERT.ingressBytesPerUnit > 0,
        doe: covert && covert.egressBytesPerUnit > 0 ? Math.log10(covert.egressBytesPerUnit) : 0,
        doen: covert ? covert.egressBytesPerUnit > 0 : DEFAULT_COVERT.egressBytesPerUnit > 0,
      }
    }
    return null
  } catch { return null }
}

export function GammaDashboard() {
  // Read initial state from URL hash (once on mount)
  const initial = typeof window !== "undefined" ? decodeState(window.location.hash) : null

  // Hardware: raw log-scale sliders
  const [computeFlopsExp, setComputeFlopsExp] = useState(initial?.cfe ?? Math.log10(DEFAULT_HW.computeFlops))
  const [hbmBytesExp, setHbmBytesExp] = useState(initial?.hbe ?? Math.log10(DEFAULT_HW.hbmBytes))

  // Honest load
  const [computeFrac, setComputeFrac] = useState(initial?.cf ?? 50)
  const [memoryFrac, setMemoryFrac] = useState(initial?.mf ?? 50)

  // Verifier
  const [alpha, setAlpha] = useState(initial?.a ?? 100)
  const [bOutExp, setBOutExp] = useState(initial?.bo ?? Math.log10(20e3))
  const [bInExp, setBInExp] = useState(initial?.bi ?? Math.log10(100e3))
  const [sanitization, setSanitization] = useState(initial?.sn ?? true)
  const [epochSExp, setEpochSExp] = useState(initial?.ep ?? Math.log10(5))
  const epochS = 10 ** epochSExp
  const [downtimeSExp, setDowntimeSExp] = useState(initial?.dt ?? Math.log10(0.25))
  const downtimeS = 10 ** downtimeSExp
  const [survivingGB, setSurvivingGB] = useState(initial?.sg ?? 17)

  // Covert workload: raw log-scale sliders
  const [nBytesExp, setNBytesExp] = useState(initial?.ne ?? Math.log10(DEFAULT_COVERT.stateBytes))
  const [gFlopExp, setGFlopExp] = useState(initial?.ge ?? Math.log10(DEFAULT_COVERT.flopPerUnit))
  const [dInExp, setDInExp] = useState(initial?.die ?? (DEFAULT_COVERT.ingressBytesPerUnit > 0 ? Math.log10(DEFAULT_COVERT.ingressBytesPerUnit) : 0))
  const [dInEnabled, setDInEnabled] = useState(initial?.dien ?? DEFAULT_COVERT.ingressBytesPerUnit > 0)
  const [dOutExp, setDOutExp] = useState(initial?.doe ?? (DEFAULT_COVERT.egressBytesPerUnit > 0 ? Math.log10(DEFAULT_COVERT.egressBytesPerUnit) : 0))
  const [dOutEnabled, setDOutEnabled] = useState(initial?.doen ?? DEFAULT_COVERT.egressBytesPerUnit > 0)

  // Re-apply URL hash state after mount (handles cases where hash isn't available during initial render)
  const didApplyHash = useRef(initial !== null)
  useEffect(() => {
    if (didApplyHash.current) return
    const state = decodeState(window.location.hash)
    if (!state) return
    didApplyHash.current = true
    setComputeFlopsExp(state.cfe)
    setHbmBytesExp(state.hbe)
    setComputeFrac(state.cf)
    setMemoryFrac(state.mf)
    setAlpha(state.a)
    setBOutExp(state.bo)
    setBInExp(state.bi)
    setSanitization(state.sn)
    setEpochSExp(state.ep)
    setDowntimeSExp(state.dt)
    setSurvivingGB(state.sg)
    setNBytesExp(state.ne)
    setGFlopExp(state.ge)
    setDInExp(state.die)
    setDInEnabled(state.dien)
    setDOutExp(state.doe)
    setDOutEnabled(state.doen)
  }, [])

  const computeFlops = 10 ** computeFlopsExp
  const hbmBytes = 10 ** hbmBytesExp

  // Snap hardware preset
  const snapHardware = (key: string) => {
    const p = HARDWARE_PRESETS[key]
    if (!p) return
    const hw = computeHardwarePreset(p.gpuKey, p.nGpu)
    setComputeFlopsExp(Math.log10(hw.computeFlops))
    setHbmBytesExp(Math.log10(hw.hbmBytes))
  }

  // Snap covert preset
  const snapCovert = (key: string) => {
    const config = COVERT_PRESETS[key]
    if (!config) return
    const hw = { computeFlops, hbmBytes }
    const wl = computeCovertPreset(hw, config)
    setNBytesExp(Math.log10(Math.max(1, wl.stateBytes)))
    setGFlopExp(Math.log10(Math.max(1, wl.flopPerUnit)))
    if (wl.ingressBytesPerUnit > 0) {
      setDInExp(Math.log10(wl.ingressBytesPerUnit))
      setDInEnabled(true)
    } else {
      setDInEnabled(false)
    }
    if (wl.egressBytesPerUnit > 0) {
      setDOutExp(Math.log10(wl.egressBytesPerUnit))
      setDOutEnabled(true)
    } else {
      setDOutEnabled(false)
    }
  }

  const covert: CovertWorkloadV1 = useMemo(() => ({
    label: "Custom",
    kind: "inference",
    unit: "unit",
    stateBytes: 10 ** nBytesExp,
    flopPerUnit: 10 ** gFlopExp,
    ingressBytesPerUnit: dInEnabled ? 10 ** dInExp : 0,
    egressBytesPerUnit: dOutEnabled ? 10 ** dOutExp : 0,
  }), [nBytesExp, gFlopExp, dInExp, dInEnabled, dOutExp, dOutEnabled])

  const scenario: DirectScenario = useMemo(() => ({
    hardware: { computeFlops, hbmBytes },
    honest: {
      claimedComputeFlops: (computeFrac / 100) * computeFlops,
      claimedMemoryBytes: (memoryFrac / 100) * hbmBytes,
    },
    verifier: {
      alpha: alpha / 100,
      covertIngressBps: 10 ** bInExp,
      covertEgressBps: 10 ** bOutExp,
      survivingStateBytes: survivingGB * 1e9,
      epochSeconds: epochS,
      downtimeSeconds: downtimeS,
      sanitizationEnabled: sanitization,
    },
    covert,
  }), [computeFlops, hbmBytes, computeFrac, memoryFrac, alpha, bOutExp, bInExp, sanitization, epochS, downtimeSExp, survivingGB, covert])

  const result = useMemo(() => simulateDirect(scenario), [scenario])

  const [copied, setCopied] = useState(false)
  const handleCopy = () => {
    navigator.clipboard.writeText(JSON.stringify({ input: scenario, output: result }, null, 2))
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  // Copy link: encode current state into URL hash on click (not on every change)
  const [linkCopied, setLinkCopied] = useState(false)
  const handleCopyLink = () => {
    const state: DashState = {
      cfe: computeFlopsExp, hbe: hbmBytesExp,
      cf: computeFrac, mf: memoryFrac, a: alpha,
      bo: bOutExp, bi: bInExp, sn: sanitization, ep: epochSExp, dt: downtimeSExp, sg: survivingGB,
      ne: nBytesExp, ge: gFlopExp, die: dInExp, dien: dInEnabled, doe: dOutExp, doen: dOutEnabled,
    }
    const encoded = encodeState(state)
    const url = `${window.location.origin}${window.location.pathname}#${encoded}`
    navigator.clipboard.writeText(url)
    setLinkCopied(true)
    setTimeout(() => setLinkCopied(false), 2000)
  }

  // Sweep: b_out
  const bOutSweep = useMemo(
    () => logRange(1, 10, 60).map((v) => ({
      value: v,
      result: simulateDirect({ ...scenario, verifier: { ...scenario.verifier, covertEgressBps: v } }),
    })),
    [scenario],
  )

  // Sweep: proven compute share
  const computeSweep = useMemo(
    () => logRange(-2, Math.log10(0.99), 50).map((v) => ({
      value: v,
      result: simulateDirect({
        ...scenario,
        honest: { ...scenario.honest, claimedComputeFlops: v * computeFlops },
        verifier: { ...scenario.verifier, alpha: 1 },
      }),
    })),
    [scenario, computeFlops],
  )

  // Sweep: covert state (n)
  const nSweep = useMemo(
    () => logRange(6, 13, 60).map((v) => ({
      value: v,
      result: simulateDirect({ ...scenario, covert: { ...scenario.covert, stateBytes: v } }),
    })),
    [scenario],
  )

  const opMax = Math.max(result.gammaCompute, result.gammaIngress, result.gammaEgress, 10)
  const opBottleneck: "compute" | "ingress" | "egress" = result.gammaCompute >= result.gammaIngress && result.gammaCompute >= result.gammaEgress
    ? "compute"
    : result.gammaIngress >= result.gammaEgress
      ? "ingress"
      : "egress"

  return (
    <div className="min-h-screen bg-background">
      <div className="mx-auto max-w-5xl px-4 py-6">
        {/* Header */}
        <div className="mb-6 flex items-start justify-between">
          <div>
          <h1 className="text-2xl font-bold tracking-tight text-foreground">Covert Overhead Simulator</h1>
          <p className="text-sm text-muted-foreground mt-1">
            How much slower does a covert workload run under verification?
          </p>
          </div>
          <ThemeToggle />
        </div>

        {/* === Result Card === */}
        <Card className="mb-6">
          <CardContent className="pt-6">
            {/* Header: Γ value + copy button */}
            <div className="flex items-baseline gap-4 mb-5 h-14">
              <span className="text-5xl font-bold font-mono tracking-tight leading-none">
                {fmtGamma(result.gamma, true)}
              </span>
              {result.finite && <span className="text-sm text-muted-foreground">overhead</span>}
              <div className="ml-auto" />
              <button
                onClick={handleCopyLink}
                className="p-1.5 rounded-md text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
                title="Copy shareable link"
              >
                {linkCopied ? (
                  <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M20 6 9 17l-5-5"/></svg>
                ) : (
                  <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/></svg>
                )}
              </button>
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
                  <LegendDot color="bg-amber-400" label="Covert" />
                  <LegendDot color="bg-blue-300" label="Honest" />
                </span>
              </div>
              <MemoryFitBar
                result={result}
                totalHbm={hbmBytes}
                honestMem={scenario.honest.claimedMemoryBytes}
                covertState={covert.stateBytes}
              />
            </div>

            {/* --- Layer 3: Sanitization cycle --- */}
            {sanitization && (
              <div className="mb-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">Sanitization cycle</span>
                  <span className="flex items-center gap-3">
                    <LegendDot color="bg-emerald-400" label="Sanitization" />
                    <LegendDot color="bg-amber-400" label="Covert download" />
                    <LegendDot color="bg-blue-300" label="Operational" />
                  </span>
                </div>
                <EpochTimeline result={result} epochS={epochS} downtimeS={downtimeS} disabled={result.fitMarginBytes < 0} />
              </div>
            )}

            {/* Key numbers */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 pt-4 border-t text-sm">
              <div>
                <div className="text-muted-foreground">Dedicated θ₀</div>
                <div className="font-mono">{fmt(result.theta0)} {covert.unit}/s</div>
              </div>
              <div>
                <div className="text-muted-foreground">Verified θ</div>
                <div className="font-mono">{fmt(result.thetaVerified)} {covert.unit}/s</div>
              </div>
              <div>
                <div className="text-muted-foreground">Covert state</div>
                <div className="font-mono">{fmtBytes(covert.stateBytes)}</div>
              </div>
              <div>
                <div className="text-muted-foreground">FLOP/unit</div>
                <div className="font-mono">{fmt(covert.flopPerUnit)}</div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* === Controls === */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
          {/* Hardware + Covert workload */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-semibold">Configuration</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Hardware section */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">Hardware</span>
                  <Select value="" onValueChange={snapHardware}>
                    <SelectTrigger className="w-40 h-7 text-xs">
                      <SelectValue placeholder="Load preset..." />
                    </SelectTrigger>
                    <SelectContent>
                      {Object.entries(HARDWARE_PRESETS).map(([k, v]) => (
                        <SelectItem key={k} value={k}>{v.nGpu}× {v.gpuKey}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-3">
                  <LogSlider
                    label="Compute (FLOP/s)"
                    valueExp={computeFlopsExp}
                    onValueExpChange={setComputeFlopsExp}
                    min={12} max={18} step={0.05}
                    formatValue={fmt}
                  />
                  <LogSlider
                    label="HBM (bytes)"
                    valueExp={hbmBytesExp}
                    onValueExpChange={setHbmBytesExp}
                    min={9} max={14} step={0.05}
                    formatValue={fmtBytes}
                  />
                </div>
              </div>

              {/* Covert workload section */}
              <div className="pt-3 border-t">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">Covert workload</span>
                  <Select value="" onValueChange={snapCovert}>
                    <SelectTrigger className="w-48 h-7 text-xs">
                      <SelectValue placeholder="Load preset..." />
                    </SelectTrigger>
                    <SelectContent>
                      {Object.entries(COVERT_PRESETS).map(([k, v]) => (
                        <SelectItem key={k} value={k}>
                          {v.kind === "inference" ? v.modelKey : `Train ${v.modelKey}`}
                          {v.syncPolicy && v.syncPolicy.mode !== "none" ? ` (${v.syncPolicy.mode})` : ""}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-3">
                  <LogSlider
                    label="Covert state n (bytes)"
                    valueExp={nBytesExp}
                    onValueExpChange={setNBytesExp}
                    min={6} max={13} step={0.05}
                    formatValue={fmtBytes}
                  />
                  <LogSlider
                    label="FLOP per unit (g)"
                    valueExp={gFlopExp}
                    onValueExpChange={setGFlopExp}
                    min={6} max={15} step={0.05}
                    formatValue={fmt}
                  />
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => setDInEnabled(!dInEnabled)}
                      className={`relative inline-flex h-4 w-7 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors ${dInEnabled ? "bg-primary" : "bg-muted"}`}
                    >
                      <span className={`pointer-events-none block h-3 w-3 rounded-full bg-background shadow-lg transition-transform ${dInEnabled ? "translate-x-3" : "translate-x-0"}`} />
                    </button>
                    <div className="flex-1">
                      <LogSlider
                        label="Ingress per unit d_in (bytes)"
                        valueExp={dInExp}
                        onValueExpChange={setDInExp}
                        min={0} max={6} step={0.05}
                        formatValue={(v) => dInEnabled ? fmtBytes(v) : "0"}
                      />
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => setDOutEnabled(!dOutEnabled)}
                      className={`relative inline-flex h-4 w-7 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors ${dOutEnabled ? "bg-primary" : "bg-muted"}`}
                    >
                      <span className={`pointer-events-none block h-3 w-3 rounded-full bg-background shadow-lg transition-transform ${dOutEnabled ? "translate-x-3" : "translate-x-0"}`} />
                    </button>
                    <div className="flex-1">
                      <LogSlider
                        label="Egress per unit d_out (bytes)"
                        valueExp={dOutExp}
                        onValueExpChange={setDOutExp}
                        min={0} max={6} step={0.05}
                        formatValue={(v) => dOutEnabled ? fmtBytes(v) : "0"}
                      />
                    </div>
                  </div>
                </div>
              </div>

              {/* Honest load */}
              <div className="pt-3 border-t">
                <span className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">Honest load</span>
                <div className="space-y-3 mt-2">
                  <div>
                    <div className="flex justify-between">
                      <Label className="text-xs text-muted-foreground">Compute utilization (f*/Ĝ)</Label>
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
                      <Label className="text-xs text-muted-foreground">Memory utilization (m*/M̂)</Label>
                      <span className="text-xs font-mono">{memoryFrac}%</span>
                    </div>
                    <Slider
                      value={[memoryFrac]}
                      onValueChange={([v]) => setMemoryFrac(v)}
                      min={0} max={100} step={1}
                      className="mt-1"
                    />
                  </div>
                </div>
                <div className="text-xs text-muted-foreground pt-1">
                  {"Proven compute: α \u00D7 f*/\u011C = "}{((alpha / 100) * computeFrac).toFixed(1)}{"% of \u011C"}
                </div>
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
                      <span className="text-xs font-mono">{fmtTime(downtimeS)}</span>
                    </div>
                    <Slider
                      value={[downtimeSExp]}
                      onValueChange={([v]) => setDowntimeSExp(v)}
                      min={-2} max={3.76} step={0.05}
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
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="egress">Egress Sweep</TabsTrigger>
            <TabsTrigger value="compute">Compute Sweep</TabsTrigger>
            <TabsTrigger value="state">Covert State Sweep</TabsTrigger>
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
          <TabsContent value="state" className="mt-4">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-semibold">Γ vs Covert State (n)</CardTitle>
              </CardHeader>
              <CardContent>
                <SweepChart
                  points={nSweep}
                  paramLabel="Covert state (n)"
                  xFormatter={fmtBytes}
                />
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}
