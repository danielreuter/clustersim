"use client"

import { useState, useMemo, useEffect, useRef, useCallback } from "react"
import { ThemeToggle } from "@/components/theme-toggle"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import {
  simulateDirect,
  computeHardwarePreset,
  computeCovertPreset,
  COVERT_PRESETS,
  type DirectScenario,
  type CovertWorkloadV1,
  type GammaResult,
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
// Info icon for tooltips
// ---------------------------------------------------------------------------

function InfoTip({ text }: { text: string }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          className="inline-flex items-center justify-center w-3.5 h-3.5 rounded-full bg-muted text-muted-foreground text-[9px] font-medium leading-none hover:bg-muted-foreground/20 transition-colors ml-1"
        >
          i
        </button>
      </TooltipTrigger>
      <TooltipContent side="top" sideOffset={4}>
        {text}
      </TooltipContent>
    </Tooltip>
  )
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

function OverflowStripes() {
  return (
    <div
      className="absolute inset-0 pointer-events-none rounded"
      style={{
        backgroundImage: "repeating-linear-gradient(135deg, transparent, transparent 2px, rgba(0,0,0,0.25) 2px, rgba(0,0,0,0.25) 5px)",
      }}
    />
  )
}

function OpBar({ label, value, max, isBottleneck, disabled }: { label: string; value: number; max: number; isBottleneck: boolean; disabled?: boolean }) {
  const width = Number.isFinite(value) ? Math.min(100, (Math.log10(Math.max(1, value)) / Math.log10(Math.max(10, max))) * 100) : 100
  if (disabled) {
    return (
      <div className="flex items-center gap-2 text-sm">
        <span className="w-20 text-right text-muted-foreground shrink-0">{label}</span>
        <div className="flex-1 h-5 bg-muted rounded overflow-hidden opacity-50" />
        <span className="w-16 text-right font-mono text-muted-foreground shrink-0">--</span>
      </div>
    )
  }
  return (
    <div className="flex items-center gap-2 text-sm">
      <span className={`w-20 text-right shrink-0 ${isBottleneck ? "font-medium" : "text-muted-foreground"}`}>{label}</span>
      <div className="flex-1 h-5 bg-muted rounded overflow-hidden">
        <div
          className={`h-full rounded transition-all ${isBottleneck ? "bg-foreground/80" : "bg-foreground/35"}`}
          style={{ width: `${width}%` }}
        />
      </div>
      <span className={`w-16 text-right font-mono shrink-0 ${isBottleneck ? "font-medium" : ""}`}>{fmtGamma(value)}</span>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Epoch timeline
// ---------------------------------------------------------------------------

function EpochTimeline({ result, epochS, downtimeS, disabled, overflows }: { result: GammaResult; epochS: number; downtimeS: number; disabled?: boolean; overflows?: boolean }) {
  if (!Number.isFinite(epochS) || epochS <= 0) return null

  if (disabled) {
    return (
      <div className="flex items-center gap-2 text-sm">
        <span className="w-20 text-right text-muted-foreground shrink-0">Time</span>
        <div className="flex-1 h-5 bg-muted rounded overflow-hidden opacity-50" />
        <span className="w-16 text-right font-mono text-muted-foreground shrink-0">--</span>
      </div>
    )
  }

  const tReload = result.tReload
  const tCovert = Math.max(0, result.tCovert)
  const tSanitize = downtimeS
  const tTotal = tSanitize + tReload + tCovert

  // When overflows, show proportions of total time needed (exceeds epoch)
  const denom = overflows ? tTotal : epochS
  const sanitizeFrac = denom > 0 ? tSanitize / denom : 0
  const downloadFrac = denom > 0 ? tReload / denom : 0
  const operationalFrac = denom > 0 ? tCovert / denom : 0

  return (
    <div className="flex items-center gap-2 text-sm">
      <span className="w-20 text-right text-muted-foreground shrink-0">Time</span>
      <div className="relative flex-1 h-5 bg-muted rounded overflow-hidden flex">
        <div className="h-full bg-emerald-400 transition-all flex items-center justify-center overflow-hidden" style={{ width: `${sanitizeFrac * 100}%` }}>
          {sanitizeFrac >= 0.15 && <span className="text-[10px] font-mono text-white truncate px-1 drop-shadow-sm">{fmtTime(tSanitize)}</span>}
        </div>
        <div className="h-full bg-pink-300 transition-all flex items-center justify-center overflow-hidden" style={{ width: `${downloadFrac * 100}%` }}>
          {downloadFrac >= 0.15 && <span className="text-[10px] font-mono text-white truncate px-1 drop-shadow-sm">{fmtTime(tReload)}</span>}
        </div>
        <div className="h-full bg-red-400 transition-all flex items-center justify-center overflow-hidden" style={{ width: `${operationalFrac * 100}%` }}>
          {operationalFrac >= 0.15 && <span className="text-[10px] font-mono text-white truncate px-1 drop-shadow-sm">{fmtTime(tCovert)}</span>}
        </div>
        {overflows && <OverflowStripes />}
      </div>
      <span className="w-16 text-right font-mono shrink-0">{fmtGamma(result.gammaDuty)}</span>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Memory fit bar
// ---------------------------------------------------------------------------

function MemoryFitBar({ result, totalHbm, honestMem, covertState, disabled }: { result: GammaResult; totalHbm: number; honestMem: number; covertState: number; disabled?: boolean }) {
  const covertFrac = Math.min(covertState / totalHbm, 1)
  const honestFrac = Math.min(honestMem / totalHbm, 1 - covertFrac)
  const overflows = result.fitMarginBytes < 0

  if (disabled) {
    return (
      <div className="flex items-center gap-2 text-sm">
        <span className="w-20 text-right text-muted-foreground shrink-0">Memory</span>
        <div className="flex-1 h-5 bg-muted rounded overflow-hidden opacity-50" />
        <span className="w-16 text-right shrink-0 text-muted-foreground">{fmtBytes(totalHbm)}</span>
      </div>
    )
  }

  return (
    <div className="flex items-center gap-2 text-sm">
      <span className="w-20 text-right text-muted-foreground shrink-0">Memory</span>
      <div className="relative flex-1 h-5 bg-muted rounded overflow-hidden">
        <div className="h-full flex">
          <div
            className="h-full bg-red-400 transition-all flex items-center justify-center overflow-hidden"
            style={{ width: `${covertFrac * 100}%` }}
            title={`Covert: ${fmtBytes(covertState)}`}
          >
            {covertFrac >= 0.15 && <span className="text-[10px] font-mono text-white truncate px-1 drop-shadow-sm">{fmtBytes(covertState)}</span>}
          </div>
          <div
            className="h-full bg-blue-300 transition-all flex items-center justify-center overflow-hidden"
            style={{ width: `${honestFrac * 100}%` }}
            title={`Honest: ${fmtBytes(honestMem)}`}
          >
            {honestFrac >= 0.15 && <span className="text-[10px] font-mono text-white truncate px-1 drop-shadow-sm">{fmtBytes(honestMem)}</span>}
          </div>
        </div>
        {overflows && <OverflowStripes />}
      </div>
      <span className="w-16 text-right shrink-0 text-muted-foreground">{fmtBytes(totalHbm)}</span>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Log-scale slider helper
// ---------------------------------------------------------------------------

function LogSlider({ label, tooltip, valueExp, onValueExpChange, min, max, step, formatValue }: {
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
      <div className="flex justify-between">
        <Label className="text-xs text-muted-foreground">
          {label}
          {tooltip && <InfoTip text={tooltip} />}
        </Label>
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
// Linear slider helper
// ---------------------------------------------------------------------------

function LinearSlider({ label, tooltip, value, onValueChange, min, max, step, formatValue, clampMin }: {
  label: string
  tooltip?: string
  value: number
  onValueChange: (v: number) => void
  min: number
  max: number
  step: number
  formatValue: (v: number) => string
  clampMin?: number
}) {
  const effectiveMin = clampMin != null ? Math.max(min, clampMin) : min
  const atFloor = clampMin != null && clampMin > min && value <= clampMin
  return (
    <div>
      <div className="flex justify-between">
        <Label className={`text-xs ${atFloor ? "text-red-500" : "text-muted-foreground"}`}>
          {label}
          {tooltip && <InfoTip text={tooltip} />}
          {atFloor && <span className="ml-1 text-[10px] text-red-500/70">(at proof-of-work floor)</span>}
        </Label>
        <span className={`text-xs font-mono ${atFloor ? "text-red-500" : ""}`}>{formatValue(value)}</span>
      </div>
      <div className="relative mt-1">
        {clampMin != null && clampMin > min && (
          <div
            className="absolute top-1/2 -translate-y-1/2 h-1.5 rounded-full bg-red-500/30"
            style={{ width: `${((clampMin - min) / (max - min)) * 100}%` }}
          />
        )}
        <Slider
          value={[value]}
          onValueChange={([v]) => onValueChange(Math.max(v, effectiveMin))}
          min={min} max={max} step={step}
          className={atFloor ? "[&_[data-slot=slider-range]]:bg-red-500 [&_[data-slot=slider-thumb]]:border-red-500" : ""}
        />
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Phase map axis definitions
// ---------------------------------------------------------------------------

type AxisDef = {
  key: string
  label: string
  min: number
  max: number
  isLog: boolean
  format: (v: number) => string
  applyToScenario: (s: DirectScenario, v: number) => DirectScenario
}

const PHASE_AXES: AxisDef[] = [
  {
    key: "stateBytes", label: "Covert state size", min: 6, max: 13, isLog: true,
    format: (v) => fmtBytes(10 ** v),
    applyToScenario: (s, v) => ({ ...s, covert: { ...s.covert, stateBytes: 10 ** v } }),
  },
  {
    key: "flopPerUnit", label: "Compute per output", min: 3, max: 15, isLog: true,
    format: (v) => fmt(10 ** v),
    applyToScenario: (s, v) => ({ ...s, covert: { ...s.covert, flopPerUnit: 10 ** v } }),
  },
  {
    key: "dOut", label: "Egress per output", min: -1, max: 6, isLog: true,
    format: (v) => v <= -1 ? "0" : fmtBytes(10 ** v),
    applyToScenario: (s, v) => ({ ...s, covert: { ...s.covert, egressBytesPerUnit: v <= -1 ? 0 : 10 ** v } }),
  },
  {
    key: "dIn", label: "Ingress per input", min: -1, max: 6, isLog: true,
    format: (v) => v <= -1 ? "0" : fmtBytes(10 ** v),
    applyToScenario: (s, v) => ({ ...s, covert: { ...s.covert, ingressBytesPerUnit: v <= -1 ? 0 : 10 ** v } }),
  },
  {
    key: "bOut", label: "Covert egress bandwidth", min: 1, max: 12, isLog: true,
    format: (v) => fmtBw(10 ** v),
    applyToScenario: (s, v) => ({ ...s, verifier: { ...s.verifier, covertEgressBps: 10 ** v } }),
  },
  {
    key: "bIn", label: "Covert ingress bandwidth", min: 1, max: 12, isLog: true,
    format: (v) => fmtBw(10 ** v),
    applyToScenario: (s, v) => ({ ...s, verifier: { ...s.verifier, covertIngressBps: 10 ** v } }),
  },
  {
    key: "computeFlops", label: "Compute capacity", min: 12, max: 18, isLog: true,
    format: (v) => fmt(10 ** v),
    applyToScenario: (s, v) => ({
      ...s,
      hardware: { ...s.hardware, computeFlops: 10 ** v },
      honest: { ...s.honest, claimedComputeFlops: s.honest.claimedComputeFlops * (10 ** v) / s.hardware.computeFlops },
    }),
  },
  {
    key: "hbmBytes", label: "Memory capacity", min: 9, max: 14, isLog: true,
    format: (v) => fmtBytes(10 ** v),
    applyToScenario: (s, v) => ({
      ...s,
      hardware: { ...s.hardware, hbmBytes: 10 ** v },
      honest: { ...s.honest, claimedMemoryBytes: s.honest.claimedMemoryBytes * (10 ** v) / s.hardware.hbmBytes },
    }),
  },
  {
    key: "alpha", label: "Proved compute utilization", min: 0, max: 100, isLog: false,
    format: (v) => `${v.toFixed(0)}%`,
    applyToScenario: (s, v) => ({ ...s, verifier: { ...s.verifier, alpha: v / 100 } }),
  },
  {
    key: "alphaMemory", label: "Proved memory utilization", min: 0, max: 100, isLog: false,
    format: (v) => `${v.toFixed(0)}%`,
    applyToScenario: (s, v) => ({ ...s, verifier: { ...s.verifier, alphaMemory: v / 100 } }),
  },
  {
    key: "computeFrac", label: "Claimed compute util", min: 0, max: 100, isLog: false,
    format: (v) => `${v.toFixed(0)}%`,
    applyToScenario: (s, v) => ({
      ...s,
      honest: { ...s.honest, claimedComputeFlops: (v / 100) * s.hardware.computeFlops },
    }),
  },
  {
    key: "memoryFrac", label: "Claimed memory util", min: 0, max: 100, isLog: false,
    format: (v) => `${v.toFixed(0)}%`,
    applyToScenario: (s, v) => ({
      ...s,
      honest: { ...s.honest, claimedMemoryBytes: (v / 100) * s.hardware.hbmBytes },
    }),
  },
  {
    key: "epochS", label: "Epoch length", min: -1, max: 6, isLog: true,
    format: (v) => fmtTime(10 ** v),
    applyToScenario: (s, v) => ({ ...s, verifier: { ...s.verifier, epochSeconds: 10 ** v } }),
  },
  {
    key: "downtimeS", label: "Downtime per epoch", min: -2, max: 2, isLog: true,
    format: (v) => fmtTime(10 ** v),
    applyToScenario: (s, v) => ({ ...s, verifier: { ...s.verifier, downtimeSeconds: 10 ** v } }),
  },
  {
    key: "survivingGB", label: "Covert persistence", min: 0, max: 200, isLog: false,
    format: (v) => `${v.toFixed(0)} GB`,
    applyToScenario: (s, v) => ({ ...s, verifier: { ...s.verifier, survivingStateBytes: v * 1e9 } }),
  },
]

// Get current axis value from dashboard state
function getAxisValue(key: string, state: {
  computeFlopsExp: number; hbmBytesExp: number; computeFrac: number; memoryFrac: number
  alpha: number; alphaMemory: number; bOutExp: number; bInExp: number; epochSExp: number; downtimeSExp: number
  survivingGB: number; nBytesExp: number; gFlopExp: number; dInExp: number; dOutExp: number
}): number {
  switch (key) {
    case "stateBytes": return state.nBytesExp
    case "flopPerUnit": return state.gFlopExp
    case "dOut": return state.dOutExp
    case "dIn": return state.dInExp
    case "bOut": return state.bOutExp
    case "bIn": return state.bInExp
    case "computeFlops": return state.computeFlopsExp
    case "hbmBytes": return state.hbmBytesExp
    case "alpha": return state.alpha
    case "alphaMemory": return state.alphaMemory
    case "computeFrac": return state.computeFrac
    case "memoryFrac": return state.memoryFrac
    case "epochS": return state.epochSExp
    case "downtimeS": return state.downtimeSExp
    case "survivingGB": return state.survivingGB
    default: return 0
  }
}

// ---------------------------------------------------------------------------
// Gamma → color mapping
// ---------------------------------------------------------------------------

function gammaToColor(gamma: number, finite: boolean): string {
  if (!finite) return "#374151" // gray-700 for infeasible
  const t = Math.log10(Math.max(1, gamma))
  // green(1x) → yellow(~3x) → orange(~30x) → red(~1000x)
  if (t <= 0) return "#22c55e"
  if (t <= 0.5) { // 1x - ~3x: green → yellow
    const f = t / 0.5
    return lerpColor("#22c55e", "#eab308", f)
  }
  if (t <= 1.5) { // ~3x - ~30x: yellow → orange
    const f = (t - 0.5) / 1.0
    return lerpColor("#eab308", "#f97316", f)
  }
  if (t <= 3) { // ~30x - 1000x: orange → red
    const f = (t - 1.5) / 1.5
    return lerpColor("#f97316", "#ef4444", f)
  }
  return "#ef4444" // 1000x+: solid red
}

function lerpColor(a: string, b: string, t: number): string {
  const pa = [parseInt(a.slice(1, 3), 16), parseInt(a.slice(3, 5), 16), parseInt(a.slice(5, 7), 16)]
  const pb = [parseInt(b.slice(1, 3), 16), parseInt(b.slice(3, 5), 16), parseInt(b.slice(5, 7), 16)]
  const r = Math.round(pa[0] + (pb[0] - pa[0]) * t)
  const g = Math.round(pa[1] + (pb[1] - pa[1]) * t)
  const bl = Math.round(pa[2] + (pb[2] - pa[2]) * t)
  return `#${r.toString(16).padStart(2, "0")}${g.toString(16).padStart(2, "0")}${bl.toString(16).padStart(2, "0")}`
}

const DOMINANT_COLORS: Record<string, string> = {
  compute: "#3b82f6",    // blue
  egress: "#f97316",     // orange
  ingress: "#a855f7",    // purple
  duty: "#6b7280",       // gray
  "memory-fit": "#ef4444", // red
}

// ---------------------------------------------------------------------------
// PhaseMap component
// ---------------------------------------------------------------------------

const GRID_SIZE = 30

function PhaseMap({ scenario, dashState }: {
  scenario: DirectScenario
  dashState: {
    computeFlopsExp: number; hbmBytesExp: number; computeFrac: number; memoryFrac: number
    alpha: number; alphaMemory: number; bOutExp: number; bInExp: number; epochSExp: number; downtimeSExp: number
    survivingGB: number; nBytesExp: number; gFlopExp: number; dInExp: number; dOutExp: number
  }
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [xAxisKey, setXAxisKey] = useState("stateBytes")
  const [yAxisKey, setYAxisKey] = useState("bIn")
  const [colorMode, setColorMode] = useState<"gamma" | "regime">("gamma")

  const xAxis = PHASE_AXES.find(a => a.key === xAxisKey)!
  const yAxis = PHASE_AXES.find(a => a.key === yAxisKey)!

  // Compute grid of gamma values
  const grid = useMemo(() => {
    const results: GammaResult[][] = []
    for (let yi = 0; yi < GRID_SIZE; yi++) {
      const row: GammaResult[] = []
      const yVal = yAxis.min + (yAxis.max - yAxis.min) * yi / (GRID_SIZE - 1)
      for (let xi = 0; xi < GRID_SIZE; xi++) {
        const xVal = xAxis.min + (xAxis.max - xAxis.min) * xi / (GRID_SIZE - 1)
        const s1 = xAxis.applyToScenario(scenario, xVal)
        const s2 = yAxis.applyToScenario(s1, yVal)
        row.push(simulateDirect(s2))
      }
      results.push(row)
    }
    return results
  }, [scenario, xAxis, yAxis])

  // Crosshair position
  const xCurrent = getAxisValue(xAxisKey, dashState)
  const yCurrent = getAxisValue(yAxisKey, dashState)
  const xFrac = (xCurrent - xAxis.min) / (xAxis.max - xAxis.min)
  const yFrac = (yCurrent - yAxis.min) / (yAxis.max - yAxis.min)

  // Paint canvas
  const paint = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext("2d")
    if (!ctx) return

    const W = canvas.width
    const H = canvas.height
    const cellW = W / GRID_SIZE
    const cellH = H / GRID_SIZE

    // Draw heatmap cells
    for (let yi = 0; yi < GRID_SIZE; yi++) {
      for (let xi = 0; xi < GRID_SIZE; xi++) {
        const r = grid[yi][xi]
        const color = colorMode === "gamma"
          ? gammaToColor(r.gamma, r.finite)
          : (DOMINANT_COLORS[r.dominant] ?? "#6b7280")
        ctx.fillStyle = color
        // Y axis is inverted (low values at bottom)
        ctx.fillRect(xi * cellW, (GRID_SIZE - 1 - yi) * cellH, cellW + 0.5, cellH + 0.5)
      }
    }

    // Contour lines (gamma mode only)
    if (colorMode === "gamma") {
      const contours = [1.5, 2, 3, 5, 10]
      ctx.strokeStyle = "rgba(255,255,255,0.5)"
      ctx.lineWidth = 1
      for (const threshold of contours) {
        ctx.beginPath()
        const logT = Math.log10(threshold)
        // Simple marching squares
        for (let yi = 0; yi < GRID_SIZE - 1; yi++) {
          for (let xi = 0; xi < GRID_SIZE - 1; xi++) {
            const vals = [
              grid[yi][xi].finite ? Math.log10(Math.max(1, grid[yi][xi].gamma)) : 999,
              grid[yi][xi + 1].finite ? Math.log10(Math.max(1, grid[yi][xi + 1].gamma)) : 999,
              grid[yi + 1][xi + 1].finite ? Math.log10(Math.max(1, grid[yi + 1][xi + 1].gamma)) : 999,
              grid[yi + 1][xi].finite ? Math.log10(Math.max(1, grid[yi + 1][xi].gamma)) : 999,
            ]
            const bits = vals.map(v => v >= logT ? 1 : 0)
            const code = bits[0] | (bits[1] << 1) | (bits[2] << 2) | (bits[3] << 3)
            if (code === 0 || code === 15) continue

            // Interpolate edge crossings
            const cx = (xi + 0.5) * cellW
            const cy = (GRID_SIZE - 1 - yi - 0.5) * cellH

            // Simplified: just draw a small segment at the cell center for non-uniform cells
            const edges: [number, number][] = []
            // Top edge (0→1)
            if ((bits[0] !== bits[1])) {
              const t = (logT - vals[0]) / (vals[1] - vals[0])
              edges.push([(xi + t) * cellW, (GRID_SIZE - 1 - yi) * cellH])
            }
            // Right edge (1→2)
            if ((bits[1] !== bits[2])) {
              const t = (logT - vals[1]) / (vals[2] - vals[1])
              edges.push([(xi + 1) * cellW, (GRID_SIZE - 1 - yi - t) * cellH])
            }
            // Bottom edge (3→2)
            if ((bits[3] !== bits[2])) {
              const t = (logT - vals[3]) / (vals[2] - vals[3])
              edges.push([(xi + t) * cellW, (GRID_SIZE - 1 - yi - 1) * cellH])
            }
            // Left edge (0→3)
            if ((bits[0] !== bits[3])) {
              const t = (logT - vals[0]) / (vals[3] - vals[0])
              edges.push([xi * cellW, (GRID_SIZE - 1 - yi - t) * cellH])
            }

            if (edges.length >= 2) {
              ctx.moveTo(edges[0][0], edges[0][1])
              ctx.lineTo(edges[1][0], edges[1][1])
            }
          }
        }
        ctx.stroke()

        // Label the contour
        // Find a cell near the middle where this contour passes
        const midY = Math.floor(GRID_SIZE / 2)
        for (let xi = 0; xi < GRID_SIZE - 1; xi++) {
          const v0 = grid[midY][xi].finite ? Math.log10(Math.max(1, grid[midY][xi].gamma)) : 999
          const v1 = grid[midY][xi + 1].finite ? Math.log10(Math.max(1, grid[midY][xi + 1].gamma)) : 999
          if ((v0 < logT) !== (v1 < logT)) {
            const t = (logT - v0) / (v1 - v0)
            const lx = (xi + t) * cellW
            const ly = (GRID_SIZE - 1 - midY) * cellH
            ctx.fillStyle = "rgba(255,255,255,0.8)"
            ctx.font = "10px monospace"
            ctx.fillText(`${threshold}×`, lx + 2, ly - 2)
            break
          }
        }
      }
    }

    // Crosshair
    const crossX = Math.max(0, Math.min(1, xFrac)) * W
    const crossY = (1 - Math.max(0, Math.min(1, yFrac))) * H
    ctx.strokeStyle = "rgba(255,255,255,0.8)"
    ctx.lineWidth = 1
    ctx.setLineDash([4, 3])
    ctx.beginPath()
    ctx.moveTo(crossX, 0); ctx.lineTo(crossX, H)
    ctx.moveTo(0, crossY); ctx.lineTo(W, crossY)
    ctx.stroke()
    ctx.setLineDash([])
    // Dot
    ctx.beginPath()
    ctx.arc(crossX, crossY, 4, 0, Math.PI * 2)
    ctx.fillStyle = "white"
    ctx.fill()
    ctx.strokeStyle = "rgba(0,0,0,0.5)"
    ctx.lineWidth = 1
    ctx.stroke()
  }, [grid, colorMode, xFrac, yFrac, xAxis, yAxis])

  useEffect(() => { paint() }, [paint])

  // Generate tick labels
  const xTicks = Array.from({ length: 5 }, (_, i) => {
    const v = xAxis.min + (xAxis.max - xAxis.min) * i / 4
    return { frac: i / 4, label: xAxis.format(v) }
  })
  const yTicks = Array.from({ length: 5 }, (_, i) => {
    const v = yAxis.min + (yAxis.max - yAxis.min) * i / 4
    return { frac: 1 - i / 4, label: yAxis.format(v) }
  })

  return (
    <div className="space-y-3">
      {/* Controls row */}
      <div className="flex items-center gap-3 flex-wrap">
        <div className="flex items-center gap-1.5">
          <span className="text-xs text-muted-foreground">X:</span>
          <Select value={xAxisKey} onValueChange={setXAxisKey}>
            <SelectTrigger className="h-7 text-xs w-44"><SelectValue /></SelectTrigger>
            <SelectContent>
              {PHASE_AXES.filter(a => a.key !== yAxisKey).map(a => (
                <SelectItem key={a.key} value={a.key}>{a.label}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="text-xs text-muted-foreground">Y:</span>
          <Select value={yAxisKey} onValueChange={setYAxisKey}>
            <SelectTrigger className="h-7 text-xs w-44"><SelectValue /></SelectTrigger>
            <SelectContent>
              {PHASE_AXES.filter(a => a.key !== xAxisKey).map(a => (
                <SelectItem key={a.key} value={a.key}>{a.label}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="ml-auto flex items-center gap-1 bg-muted rounded-md p-0.5">
          <button
            onClick={() => setColorMode("gamma")}
            className={`px-2 py-0.5 rounded text-xs transition-colors ${colorMode === "gamma" ? "bg-background shadow-sm font-medium" : "text-muted-foreground"}`}
          >Gamma</button>
          <button
            onClick={() => setColorMode("regime")}
            className={`px-2 py-0.5 rounded text-xs transition-colors ${colorMode === "regime" ? "bg-background shadow-sm font-medium" : "text-muted-foreground"}`}
          >Regime</button>
        </div>
      </div>

      {/* Canvas with axis labels */}
      <div className="relative" ref={containerRef}>
        {/* Y-axis tick labels */}
        <div className="absolute left-0 top-0 bottom-6 w-14 flex flex-col justify-between items-end pr-1.5">
          {yTicks.map((t, i) => (
            <span key={i} className="text-[10px] font-mono text-muted-foreground leading-none">{t.label}</span>
          ))}
        </div>

        {/* Canvas */}
        <div className="ml-14 mb-6">
          <canvas
            ref={canvasRef}
            width={600}
            height={380}
            className="w-full rounded border border-border"
            style={{ height: "380px", imageRendering: "pixelated" }}
          />
        </div>

        {/* X-axis tick labels */}
        <div className="ml-14 flex justify-between">
          {xTicks.map((t, i) => (
            <span key={i} className="text-[10px] font-mono text-muted-foreground leading-none">{t.label}</span>
          ))}
        </div>
      </div>

      {/* Color legend */}
      {colorMode === "gamma" ? (
        <div className="flex items-center gap-2 ml-14">
          <div className="flex-1 h-3 rounded-sm overflow-hidden flex">
            {Array.from({ length: 60 }, (_, i) => {
              const t = i / 59
              // Map to gamma range: 1x → 10000x
              const gamma = 10 ** (t * 4)
              return <div key={i} className="flex-1 h-full" style={{ backgroundColor: gammaToColor(gamma, true) }} />
            })}
            <div className="w-6 h-full" style={{ backgroundColor: "#374151" }} />
          </div>
          <div className="flex gap-3 text-[10px] text-muted-foreground font-mono shrink-0">
            <span>1×</span><span>10×</span><span>1K×</span><span>∞</span>
          </div>
        </div>
      ) : (
        <div className="flex items-center gap-3 ml-14 text-[10px]">
          {Object.entries(DOMINANT_COLORS).map(([key, color]) => (
            <span key={key} className="flex items-center gap-1">
              <span className="inline-block w-2.5 h-2.5 rounded-sm" style={{ backgroundColor: color }} />
              <span className="text-muted-foreground capitalize">{key === "memory-fit" ? "Memory" : key}</span>
            </span>
          ))}
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main Dashboard
// ---------------------------------------------------------------------------

// GPU options for hardware preset
const GPU_KEYS = ["H100", "H200", "A100", "H20", "B200", "Rubin"]
const GPU_COUNTS = [1, 2, 4, 8, 16, 32, 64, 72, 128, 256, 512, 1024]

// Default hardware: 8xH100
const DEFAULT_HW = computeHardwarePreset("H100", 8)
// Default covert: Llama 70B inference
const DEFAULT_COVERT = computeCovertPreset(DEFAULT_HW, COVERT_PRESETS["inf-llama70b"])

// ---------------------------------------------------------------------------
// URL hash state serialization (write on copy-link only, read on init)
// ---------------------------------------------------------------------------

type DashState = {
  cfe: number; hbe: number  // hardware: computeFlopsExp, hbmBytesExp
  cf: number; mf: number; a: number; am?: number  // honest load + verification
  bo: number; bi: number; sn: boolean; ep: number; dt: number; sg: number  // verifier
  ne: number; ge: number; die: number; doe: number  // covert
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
        cf: parsed.cf ?? 96,
        mf: parsed.mf ?? 83,
        a: parsed.a ?? 83,
        bo: parsed.bo ?? Math.log10(1e6),
        bi: parsed.bi ?? Math.log10(100e6),
        sn: parsed.sn ?? true,
        ep: parsed.ep ?? Math.log10(3981),
        dt: parsed.dt ?? Math.log10(10),
        sg: parsed.sg ?? 100,
        ne: covert ? Math.log10(covert.stateBytes) : Math.log10(DEFAULT_COVERT.stateBytes),
        ge: covert ? Math.log10(covert.flopPerUnit) : Math.log10(DEFAULT_COVERT.flopPerUnit),
        die: covert && covert.ingressBytesPerUnit > 0 ? Math.log10(covert.ingressBytesPerUnit) : -1,
        doe: covert && covert.egressBytesPerUnit > 0 ? Math.log10(covert.egressBytesPerUnit) : -1,
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

  // Hardware preset selectors
  const [gpuKey, setGpuKey] = useState("H100")
  const [gpuCount, setGpuCount] = useState(8)

  // Honest load
  const [computeFrac, setComputeFrac] = useState(initial?.cf ?? 96)
  const [memoryFrac, setMemoryFrac] = useState(initial?.mf ?? 83)

  // Verifier
  const [alpha, setAlpha] = useState(initial?.a ?? 83)
  const [alphaMemory, setAlphaMemory] = useState(initial?.am ?? 80)
  const [bOutExp, setBOutExp] = useState(initial?.bo ?? Math.log10(1e6))
  const [bInExp, setBInExp] = useState(initial?.bi ?? Math.log10(100e6))
  const [sanitization, setSanitization] = useState(initial?.sn ?? true)
  const [epochSExp, setEpochSExp] = useState(initial?.ep ?? Math.log10(3981))
  const epochS = 10 ** epochSExp
  const [downtimeSExp, setDowntimeSExp] = useState(initial?.dt ?? Math.log10(10))
  const downtimeS = 10 ** downtimeSExp
  const [survivingGB, setSurvivingGB] = useState(initial?.sg ?? 100)


  // Covert workload: raw log-scale sliders
  const [nBytesExp, setNBytesExp] = useState(initial?.ne ?? Math.log10(DEFAULT_COVERT.stateBytes))
  const [gFlopExp, setGFlopExp] = useState(initial?.ge ?? Math.log10(DEFAULT_COVERT.flopPerUnit))
  // d_in/d_out: use -1 as sentinel for "off" (0 bytes). Slider min is -1.
  const DIO_OFF = -1
  const [dInExp, setDInExp] = useState(initial?.die ?? (DEFAULT_COVERT.ingressBytesPerUnit > 0 ? Math.log10(DEFAULT_COVERT.ingressBytesPerUnit) : DIO_OFF))
  const [dOutExp, setDOutExp] = useState(initial?.doe ?? (DEFAULT_COVERT.egressBytesPerUnit > 0 ? Math.log10(DEFAULT_COVERT.egressBytesPerUnit) : DIO_OFF))

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
    if (state.am != null) setAlphaMemory(state.am)
    setBOutExp(state.bo)
    setBInExp(state.bi)
    setSanitization(state.sn)
    setEpochSExp(state.ep)
    setDowntimeSExp(state.dt)
    setSurvivingGB(state.sg)
    setNBytesExp(state.ne)
    setGFlopExp(state.ge)
    setDInExp(state.die)
    setDOutExp(state.doe)
  }, [])

  const computeFlops = 10 ** computeFlopsExp
  const hbmBytes = 10 ** hbmBytesExp

  // Snap hardware preset
  const snapHardware = (gpu: string, count: number) => {
    const hw = computeHardwarePreset(gpu, count)
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
    setDInExp(wl.ingressBytesPerUnit > 0 ? Math.log10(wl.ingressBytesPerUnit) : DIO_OFF)
    setDOutExp(wl.egressBytesPerUnit > 0 ? Math.log10(wl.egressBytesPerUnit) : DIO_OFF)
  }

  const covert: CovertWorkloadV1 = useMemo(() => ({
    label: "Custom",
    kind: "inference",
    unit: "unit",
    stateBytes: 10 ** nBytesExp,
    flopPerUnit: 10 ** gFlopExp,
    ingressBytesPerUnit: dInExp <= DIO_OFF ? 0 : 10 ** dInExp,
    egressBytesPerUnit: dOutExp <= DIO_OFF ? 0 : 10 ** dOutExp,
  }), [nBytesExp, gFlopExp, dInExp, dOutExp])

  const scenario: DirectScenario = useMemo(() => ({
    hardware: { computeFlops, hbmBytes },
    honest: {
      claimedComputeFlops: (computeFrac / 100) * computeFlops,
      claimedMemoryBytes: (memoryFrac / 100) * hbmBytes,
    },
    verifier: {
      alpha: alpha / 100,
      alphaMemory: alphaMemory / 100,
      covertIngressBps: 10 ** bInExp,
      covertEgressBps: 10 ** bOutExp,
      survivingStateBytes: survivingGB * 1e9,
      epochSeconds: epochS,
      downtimeSeconds: downtimeS,
      sanitizationEnabled: sanitization,
    },
    covert,
  }), [computeFlops, hbmBytes, computeFrac, memoryFrac, alpha, alphaMemory, bOutExp, bInExp, sanitization, epochS, downtimeSExp, survivingGB, covert])

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
      cf: computeFrac, mf: memoryFrac, a: alpha, am: alphaMemory,
      bo: bOutExp, bi: bInExp, sn: sanitization, ep: epochSExp, dt: downtimeSExp, sg: survivingGB,
      ne: nBytesExp, ge: gFlopExp, die: dInExp, doe: dOutExp,
    }
    const encoded = encodeState(state)
    const url = `${window.location.origin}${window.location.pathname}#${encoded}`
    navigator.clipboard.writeText(url)
    setLinkCopied(true)
    setTimeout(() => setLinkCopied(false), 2000)
  }

  const memoryOverflow = result.fitMarginBytes < 0
  const dutyOverflow = sanitization && !Number.isFinite(result.gammaDuty)

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
        <div className="pb-4">
          <Card>
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

              <Tabs defaultValue="bottlenecks">
                <TabsList className="mb-4">
                  <TabsTrigger value="bottlenecks" className="text-xs">Bottlenecks</TabsTrigger>
                  <TabsTrigger value="phases" className="text-xs">Phases</TabsTrigger>
                </TabsList>

                <TabsContent value="bottlenecks" className="mt-0">
                  {/* --- Layer 1: Operational overhead --- */}
                  <div className="mb-5">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">Operational overhead</span>
                      <span className="flex items-center gap-3">
                        <LegendDot color="bg-foreground/35" label="Non-bottleneck" />
                        <LegendDot color="bg-foreground/80" label="Bottleneck" />
                      </span>
                    </div>
                    <div className="space-y-1.5">
                      <OpBar label="Compute" value={result.gammaCompute} max={opMax} isBottleneck={opBottleneck === "compute"} disabled={memoryOverflow || dutyOverflow} />
                      <OpBar label="Ingress" value={result.gammaIngress} max={opMax} isBottleneck={opBottleneck === "ingress"} disabled={memoryOverflow || dutyOverflow} />
                      <OpBar label="Egress" value={result.gammaEgress} max={opMax} isBottleneck={opBottleneck === "egress"} disabled={memoryOverflow || dutyOverflow} />
                    </div>
                  </div>

                  {/* --- Layer 2: Memory fit --- */}
                  <div className="mb-5">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">Memory utilization</span>
                      <span className="flex items-center gap-3">
                        <LegendDot color="bg-red-400" label="Covert state" />
                        <LegendDot color="bg-blue-300" label="Honest state" />
                      </span>
                    </div>
                    <MemoryFitBar
                      result={result}
                      totalHbm={hbmBytes}
                      honestMem={scenario.honest.claimedMemoryBytes}
                      covertState={covert.stateBytes}
                      disabled={dutyOverflow}
                    />
                  </div>

                  {/* --- Layer 3: Sanitization epoch --- */}
                  {sanitization && (
                    <div className="mb-4">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">Sanitization epoch</span>
                        <span className="flex items-center gap-3">
                          <LegendDot color="bg-emerald-400" label="Sanitization" />
                          <LegendDot color="bg-pink-300" label="Covert download" />
                          <LegendDot color="bg-red-400" label="Covert work" />
                        </span>
                      </div>
                      <EpochTimeline result={result} epochS={epochS} downtimeS={downtimeS} disabled={memoryOverflow} overflows={!!dutyOverflow} />
                    </div>
                  )}
                </TabsContent>

                <TabsContent value="phases" className="mt-0">
                  <PhaseMap
                    scenario={scenario}
                    dashState={{
                      computeFlopsExp, hbmBytesExp, computeFrac, memoryFrac,
                      alpha, alphaMemory, bOutExp, bInExp, epochSExp, downtimeSExp,
                      survivingGB, nBytesExp, gFlopExp, dInExp, dOutExp,
                    }}
                  />
                </TabsContent>
              </Tabs>

          </CardContent>
        </Card>
        </div>

        {/* === Controls === */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Prover controls */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-semibold">Prover controls</CardTitle>
            </CardHeader>
            <CardContent>
              <Tabs defaultValue="hardware">
                <TabsList className="grid w-full grid-cols-3 mb-4">
                  <TabsTrigger value="hardware" className="text-xs">Hardware</TabsTrigger>
                  <TabsTrigger value="honest" className="text-xs">Honest</TabsTrigger>
                  <TabsTrigger value="covert" className="text-xs">Covert</TabsTrigger>
                </TabsList>

                {/* Hardware configuration */}
                <TabsContent value="hardware" className="space-y-3 mt-0 min-h-[320px]">
                  <LogSlider
                    label="Compute capacity"
                    tooltip="Total system compute in FLOP/s"
                    valueExp={computeFlopsExp}
                    onValueExpChange={setComputeFlopsExp}
                    min={12} max={18} step={0.05}
                    formatValue={fmt}
                  />
                  <LogSlider
                    label="Memory capacity"
                    tooltip="Total HBM across all GPUs"
                    valueExp={hbmBytesExp}
                    onValueExpChange={setHbmBytesExp}
                    min={9} max={14} step={0.05}
                    formatValue={fmtBytes}
                  />
                  <div className="pt-2 border-t">
                    <Label className="text-xs text-muted-foreground mb-2 block">Load from preset</Label>
                    <div className="flex items-center gap-2">
                      <Select value={gpuKey} onValueChange={setGpuKey}>
                        <SelectTrigger className="flex-1 h-7 text-xs">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {GPU_KEYS.map((k) => (
                            <SelectItem key={k} value={k}>{k}</SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                      <span className="text-xs text-muted-foreground">&times;</span>
                      <Select value={String(gpuCount)} onValueChange={(v) => setGpuCount(Number(v))}>
                        <SelectTrigger className="w-20 h-7 text-xs">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {GPU_COUNTS.map((n) => (
                            <SelectItem key={n} value={String(n)}>{n}</SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                      <button
                        onClick={() => snapHardware(gpuKey, gpuCount)}
                        className="h-7 px-3 text-xs rounded-md bg-muted hover:bg-muted-foreground/20 text-foreground transition-colors"
                      >
                        Apply
                      </button>
                    </div>
                  </div>
                </TabsContent>

                {/* Honest workload */}
                <TabsContent value="honest" className="space-y-3 mt-0 min-h-[320px]">
                  <LinearSlider
                    label="Claimed compute utilization"
                    tooltip="Fraction of compute the prover claims the honest workload uses"
                    value={computeFrac}
                    onValueChange={setComputeFrac}
                    min={0} max={100} step={1}
                    formatValue={(v) => `${v}%`}
                  />
                  <LinearSlider
                    label="Claimed memory utilization"
                    tooltip="Fraction of HBM the prover claims the honest workload occupies"
                    value={memoryFrac}
                    onValueChange={setMemoryFrac}
                    min={0} max={100} step={1}
                    formatValue={(v) => `${v}%`}
                  />
                </TabsContent>

                {/* Covert workload */}
                <TabsContent value="covert" className="space-y-3 mt-0 min-h-[320px]">
                  <LogSlider
                    label="Covert state size"
                    tooltip="Bytes of covert state that must reside in HBM. For inference, this is model weights (e.g. 140 GB for Llama 70B in FP16). For training, weights + optimizer state (~1.3 TB for Llama 70B)."
                    valueExp={nBytesExp}
                    onValueExpChange={setNBytesExp}
                    min={6} max={13} step={0.05}
                    formatValue={fmtBytes}
                  />
                  <LogSlider
                    label="Compute per output"
                    tooltip="FLOP per unit of covert output. For inference, a 'unit' is one output token (e.g. ~140 GFLOP for Llama 70B). For training, a 'unit' is one training token (e.g. ~420 GFLOP for Llama 70B)."
                    valueExp={gFlopExp}
                    onValueExpChange={setGFlopExp}
                    min={6} max={15} step={0.05}
                    formatValue={fmt}
                  />
                  <LogSlider
                    label="Ingress per input"
                    tooltip="Covert bytes received per unit. For inference, ~4 B/token (prompt bytes amortized). For training without external data, set to 0."
                    valueExp={dInExp}
                    onValueExpChange={setDInExp}
                    min={-1} max={6} step={0.05}
                    formatValue={(v) => v <= 10 ** DIO_OFF ? "0" : fmtBytes(v)}
                  />
                  <LogSlider
                    label="Egress per output"
                    tooltip="Covert bytes sent per unit. For inference, ~4 B/token. For training with periodic 140 GB checkpoints every 1B tokens, ~140 B/token. Set to 0 if no sync needed."
                    valueExp={dOutExp}
                    onValueExpChange={setDOutExp}
                    min={-1} max={6} step={0.05}
                    formatValue={(v) => v <= 10 ** DIO_OFF ? "0" : fmtBytes(v)}
                  />
                  <div className="pt-2 border-t">
                    <Label className="text-xs text-muted-foreground mb-2 block">Load from preset</Label>
                    <Select value="" onValueChange={snapCovert}>
                      <SelectTrigger className="w-full h-7 text-xs">
                        <SelectValue placeholder="Select workload..." />
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
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>

          {/* Verifier controls */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-semibold">Verifier controls</CardTitle>
            </CardHeader>
            <CardContent>
              <Tabs defaultValue="matmul">
                <TabsList className="grid w-full grid-cols-3 mb-4">
                  <TabsTrigger value="matmul" className="text-xs">Compute</TabsTrigger>
                  <TabsTrigger value="network" className="text-xs">Network</TabsTrigger>
                  <TabsTrigger value="memory" className="text-xs">Memory</TabsTrigger>
                </TabsList>

                {/* Compute transparency */}
                <TabsContent value="matmul" className="space-y-3 mt-0 min-h-[240px]">
                  <LinearSlider
                    label="Proved compute utilization (share of claimed)"
                    tooltip="Share of claimed compute that the verifier can prove was actually performed (e.g. via matmul transparency)"
                    value={alpha}
                    onValueChange={setAlpha}
                    min={0} max={100} step={1}
                    formatValue={(v) => `${v}%`}
                  />
                  <LinearSlider
                    label="Proved memory utilization (share of claimed)"
                    tooltip="Share of claimed memory that the verifier can prove is actually occupied by the honest workload"
                    value={alphaMemory}
                    onValueChange={setAlphaMemory}
                    min={0} max={100} step={1}
                    formatValue={(v) => `${v}%`}
                  />
                </TabsContent>

                {/* Network transparency */}
                <TabsContent value="network" className="space-y-3 mt-0 min-h-[240px]">
                  <LogSlider
                    label="Covert egress bandwidth"
                    tooltip="Maximum covert data the prover can send out per second"
                    valueExp={bOutExp}
                    onValueExpChange={setBOutExp}
                    min={1} max={10} step={0.1}
                    formatValue={fmtBw}
                  />
                  <LogSlider
                    label="Covert ingress bandwidth"
                    tooltip="Maximum covert data the prover can receive per second"
                    valueExp={bInExp}
                    onValueExpChange={setBInExp}
                    min={1} max={10} step={0.1}
                    formatValue={fmtBw}
                  />
                </TabsContent>

                {/* Memory transparency */}
                <TabsContent value="memory" className="space-y-3 mt-0 min-h-[240px]">
                  <div className="flex items-center gap-3">
                    <button
                      onClick={() => setSanitization(!sanitization)}
                      className={`relative inline-flex h-5 w-9 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors ${sanitization ? "bg-primary" : "bg-muted"}`}
                    >
                      <span className={`pointer-events-none block h-4 w-4 rounded-full bg-background shadow-lg transition-transform ${sanitization ? "translate-x-4" : "translate-x-0"}`} />
                    </button>
                    <Label className="text-xs text-muted-foreground">Enable sanitization</Label>
                  </div>
                  {sanitization && (
                    <>
                      <LogSlider
                        label="Epoch length"
                        tooltip="Time between sanitization events"
                        valueExp={epochSExp}
                        onValueExpChange={setEpochSExp}
                        min={-1} max={6} step={0.05}
                        formatValue={(v) => fmtTime(v)}
                      />
                      <LogSlider
                        label="Downtime per epoch"
                        tooltip="Duration of each sanitization event"
                        valueExp={downtimeSExp}
                        onValueExpChange={setDowntimeSExp}
                        min={-2} max={2} step={0.05}
                        formatValue={(v) => fmtTime(v)}
                      />
                      <LinearSlider
                        label="Covert persistence capacity"
                        tooltip="Covert bytes that persist through a sanitization boundary"
                        value={survivingGB}
                        onValueChange={setSurvivingGB}
                        min={0} max={200} step={1}
                        formatValue={(v) => `${v} GB`}
                      />
                    </>
                  )}
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
