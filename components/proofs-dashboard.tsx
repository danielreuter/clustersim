"use client"

import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { ThemeToggle } from "@/components/theme-toggle"
import { Card, CardContent } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"

// ---------------------------------------------------------------------------
// Math
// ---------------------------------------------------------------------------

function proofs(alpha: number, rho: number): number {
  return Math.ceil(Math.log(1 / alpha) / -Math.log(1 - rho))
}

function shareOf(s: number, gamma: number, M: number): number {
  const sg = s * gamma
  return sg / (M + sg)
}

// ---------------------------------------------------------------------------
// Formatting
// ---------------------------------------------------------------------------

const SUP: Record<string, string> = {
  "-": "⁻", "0": "⁰", "1": "¹", "2": "²", "3": "³",
  "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹",
}

function sup(n: number | string): string {
  return String(n).split("").map(c => SUP[c] ?? c).join("")
}

function pow10Label(e: number): string {
  return "10" + sup(e)
}

function sci(v: number): string {
  if (!isFinite(v)) return "—"
  if (v === 0) return "0"
  const e = Math.floor(Math.log10(Math.abs(v)))
  let m = v / Math.pow(10, e)
  m = Math.round(m * 10) / 10
  let ex = e
  if (m >= 10) { m /= 10; ex++ }
  if (ex >= 0 && ex < 5) return Math.round(v).toLocaleString("en-US")
  return `${m.toFixed(1)} × 10${sup(ex)}`
}

function intStr(v: number): string {
  return v >= 1e6 ? sci(v) : Math.round(v).toLocaleString("en-US")
}

function pct(frac: number): string {
  const p = frac * 100
  if (p >= 10) return `${p.toFixed(1)}%`
  if (p >= 1) return `${p.toFixed(2)}%`
  if (p >= 0.0001) return `${p.toPrecision(2)}%`
  if (p === 0) return "0%"
  const e = Math.floor(Math.log10(p))
  let m = p / Math.pow(10, e)
  m = Math.round(m * 10) / 10
  let ex = e
  if (m >= 10) { m /= 10; ex++ }
  return `${m.toFixed(1)} × 10${sup(ex)}%`
}

function pctTick(e: number): string {
  const pe = e + 2
  const map: Record<number, string> = { 2: "100%", 1: "10%", 0: "1%", [-1]: "0.1%", [-2]: "0.01%", [-3]: "0.001%" }
  if (map[pe] !== undefined) return map[pe]
  return "10" + sup(pe) + "%"
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

type Rho = { rho: number; label: string; color: string }

const RHOS: Rho[] = [
  { rho: 0.01,    label: "1%",     color: "#b5862f" },
  { rho: 0.001,   label: "0.1%",   color: "#c2632b" },
  { rho: 0.0001,  label: "0.01%",  color: "#9c2b28" },
  { rho: 0.00001, label: "0.001%", color: "#5e2750" },
]

const CONFS = [
  { p: "90%",    alpha: 0.1 },
  { p: "99%",    alpha: 0.01 },
  { p: "99.9%",  alpha: 0.001 },
  { p: "99.99%", alpha: 0.0001 },
]

const GMIN = 2
const GMAX = 9

// ---------------------------------------------------------------------------
// Canvas plot
// ---------------------------------------------------------------------------

function ProofsPlot({
  alpha, mLog, gLog, target, setGLog,
}: {
  alpha: number
  mLog: number
  gLog: number
  target: number
  setGLog: (v: number) => void
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const containerRef = useRef<HTMLDivElement | null>(null)
  const [reveal, setReveal] = useState(0)
  const draggingRef = useRef(false)

  // Reveal animation on mount
  useEffect(() => {
    let raf = 0
    const tick = () => {
      setReveal(r => {
        const next = Math.min(1, r + 0.05)
        if (next < 1) raf = requestAnimationFrame(tick)
        return next
      })
    }
    const t = setTimeout(() => { raf = requestAnimationFrame(tick) }, 160)
    return () => { clearTimeout(t); cancelAnimationFrame(raf) }
  }, [])

  const M = Math.pow(10, mLog)
  const ss = useMemo(() => RHOS.map(r => proofs(alpha, r.rho)), [alpha])

  // Compute y-axis range
  const { elo, ehi } = useMemo(() => {
    let ymin = Infinity, ymax = -Infinity
    for (let i = 0; i < ss.length; i++) {
      const lo = shareOf(ss[i], Math.pow(10, GMIN), M)
      const hi = shareOf(ss[i], Math.pow(10, GMAX), M)
      if (lo < ymin) ymin = lo
      if (hi > ymax) ymax = hi
    }
    let elo = Math.floor(Math.log10(ymin))
    let ehi = Math.ceil(Math.log10(ymax))
    if (ehi > 0) ehi = 0
    if (ehi - elo < 3) elo = ehi - 3
    return { elo, ehi }
  }, [ss, M])

  const draw = useCallback(() => {
    const cv = canvasRef.current
    if (!cv) return
    const ctx = cv.getContext("2d")
    if (!ctx) return

    const cs = getComputedStyle(document.documentElement)
    const colInk = cs.getPropertyValue("--foreground").trim() || "#211d17"
    const colSoft = cs.getPropertyValue("--muted-foreground").trim() || "#5c5447"
    const colBorder = cs.getPropertyValue("--border").trim() || "#d9d1bf"
    const colBg = cs.getPropertyValue("--card").trim() || "#fbf9f3"
    const wrap = (c: string) => c.startsWith("oklch") || c.startsWith("#") || c.startsWith("rgb") || c.startsWith("hsl") ? c : `oklch(${c})`
    const ink = wrap(colInk)
    const soft = wrap(colSoft)
    const grid = wrap(colBorder)
    const bg = wrap(colBg)

    const dpr = window.devicePixelRatio || 1
    const rect = cv.getBoundingClientRect()
    const W = rect.width, H = rect.height
    cv.width = W * dpr; cv.height = H * dpr
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, W, H)

    const mL = 70, mR = 22, mT = 22, mB = 46
    const pL = mL, pR = W - mR, pT = mT, pB = H - mB
    const pW = pR - pL, pH = pB - pT

    const xPx = (ge: number) => pL + (ge - GMIN) / (GMAX - GMIN) * pW
    const yPx = (v: number) => {
      const ly = Math.log10(Math.max(v, 1e-300))
      const t = (ly - elo) / (ehi - elo)
      return pB - Math.max(-0.04, Math.min(1.04, t)) * pH
    }

    ctx.font = '10px ui-monospace, "SF Mono", Menlo, monospace'
    ctx.textBaseline = "middle"

    // Grid
    ctx.strokeStyle = grid
    ctx.lineWidth = 1
    for (let ge = GMIN; ge <= GMAX; ge++) {
      const x = Math.round(xPx(ge)) + 0.5
      ctx.beginPath(); ctx.moveTo(x, pT); ctx.lineTo(x, pB); ctx.stroke()
    }
    for (let ye = elo; ye <= ehi; ye++) {
      const y = Math.round(yPx(Math.pow(10, ye))) + 0.5
      ctx.beginPath(); ctx.moveTo(pL, y); ctx.lineTo(pR, y); ctx.stroke()
    }

    // Reference line at 1% of all compute
    if (0.01 >= Math.pow(10, elo) && 0.01 <= Math.pow(10, ehi)) {
      const ry = Math.round(yPx(0.01)) + 0.5
      ctx.save()
      ctx.strokeStyle = soft; ctx.lineWidth = 1; ctx.setLineDash([2, 3])
      ctx.beginPath(); ctx.moveTo(pL, ry); ctx.lineTo(pR, ry); ctx.stroke()
      ctx.setLineDash([])
      ctx.fillStyle = soft; ctx.textAlign = "left"
      ctx.fillText("1% of all compute", pL + 6, ry - 7)
      ctx.restore()
    }

    // Axes
    ctx.strokeStyle = ink; ctx.lineWidth = 1.4
    ctx.beginPath()
    ctx.moveTo(pL, pT); ctx.lineTo(pL, pB); ctx.lineTo(pR, pB); ctx.stroke()

    // Ticks
    ctx.fillStyle = soft; ctx.lineWidth = 1; ctx.strokeStyle = ink
    ctx.textAlign = "center"
    for (let ge = GMIN; ge <= GMAX; ge++) {
      const tx = xPx(ge)
      ctx.beginPath(); ctx.moveTo(tx, pB); ctx.lineTo(tx, pB + 4); ctx.stroke()
      ctx.fillText(pow10Label(ge), tx, pB + 15)
    }
    ctx.textAlign = "right"
    for (let ye = elo; ye <= ehi; ye++) {
      const ty = yPx(Math.pow(10, ye))
      ctx.beginPath(); ctx.moveTo(pL - 4, ty); ctx.lineTo(pL, ty); ctx.stroke()
      ctx.fillText(pctTick(ye), pL - 8, ty)
    }

    // Axis titles
    ctx.fillStyle = soft; ctx.textAlign = "center"
    ctx.fillText("γ  —  PER-PROOF OVERHEAD  (× ONE INFERENCE)", (pL + pR) / 2, H - 7)
    ctx.save()
    ctx.translate(15, (pT + pB) / 2); ctx.rotate(-Math.PI / 2)
    ctx.fillText("TOTAL COMPUTE COST OF PROOFS  ·  SHARE OF ALL COMPUTE", 0, 0)
    ctx.restore()

    // Curves
    const N = 160
    const lim = Math.floor(N * reveal)
    for (let i = 0; i < RHOS.length; i++) {
      const s = ss[i]; const isT = i === target
      ctx.save()
      ctx.strokeStyle = RHOS[i].color
      ctx.globalAlpha = isT ? 1 : 0.5
      ctx.lineWidth = isT ? 3 : 1.7
      ctx.lineJoin = "round"; ctx.lineCap = "round"
      ctx.beginPath()
      for (let k = 0; k <= lim; k++) {
        const ge2 = GMIN + (GMAX - GMIN) * k / N
        const px = xPx(ge2)
        const py = yPx(shareOf(s, Math.pow(10, ge2), M))
        if (k === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py)
      }
      ctx.stroke()
      ctx.restore()
    }

    // Marker
    if (reveal >= 1) {
      const gx = xPx(gLog)
      const gamma = Math.pow(10, gLog)
      ctx.save()
      ctx.strokeStyle = ink; ctx.lineWidth = 1; ctx.setLineDash([3, 3])
      ctx.beginPath(); ctx.moveTo(gx, pT); ctx.lineTo(gx, pB); ctx.stroke()
      ctx.setLineDash([])
      for (let i = 0; i < RHOS.length; i++) {
        const cy = yPx(shareOf(ss[i], gamma, M))
        const isT = i === target
        ctx.beginPath(); ctx.arc(gx, cy, isT ? 5 : 3.6, 0, 7)
        ctx.fillStyle = bg; ctx.fill()
        ctx.lineWidth = isT ? 2.4 : 1.8; ctx.strokeStyle = RHOS[i].color; ctx.stroke()
      }
      ctx.restore()
    }
  }, [alpha, mLog, gLog, target, reveal, ss, elo, ehi, M])

  useEffect(() => { draw() }, [draw])

  // Redraw on resize
  useEffect(() => {
    const onResize = () => draw()
    window.addEventListener("resize", onResize)
    return () => window.removeEventListener("resize", onResize)
  }, [draw])

  // Redraw on theme change (class changes on <html>)
  useEffect(() => {
    const obs = new MutationObserver(() => draw())
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] })
    return () => obs.disconnect()
  }, [draw])

  const setGFromEvent = useCallback((clientX: number) => {
    const cv = canvasRef.current
    if (!cv) return
    const rect = cv.getBoundingClientRect()
    const mL = 70, mR = 22, pW = rect.width - mL - mR
    const t = (clientX - rect.left - mL) / pW
    const g = Math.max(GMIN, Math.min(GMAX, GMIN + t * (GMAX - GMIN)))
    setGLog(Math.round(g / 0.02) * 0.02)
  }, [setGLog])

  useEffect(() => {
    const onMove = (e: MouseEvent) => { if (draggingRef.current) setGFromEvent(e.clientX) }
    const onUp = () => { draggingRef.current = false }
    window.addEventListener("mousemove", onMove)
    window.addEventListener("mouseup", onUp)
    return () => { window.removeEventListener("mousemove", onMove); window.removeEventListener("mouseup", onUp) }
  }, [setGFromEvent])

  return (
    <div ref={containerRef} className="relative w-full" style={{ height: 466 }}>
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full cursor-crosshair rounded border border-border"
        onMouseDown={e => { draggingRef.current = true; setGFromEvent(e.clientX) }}
        onTouchStart={e => { draggingRef.current = true; setGFromEvent(e.touches[0].clientX); e.preventDefault() }}
        onTouchMove={e => { if (draggingRef.current) { setGFromEvent(e.touches[0].clientX); e.preventDefault() } }}
        onTouchEnd={() => { draggingRef.current = false }}
      />
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export function ProofsDashboard() {
  const [confIdx, setConfIdx] = useState(1)
  const [mLog, setMLog] = useState(12)
  const [gLog, setGLog] = useState(5)
  const [target, setTarget] = useState(1)

  const alpha = CONFS[confIdx].alpha
  const gamma = Math.pow(10, gLog)
  const M = Math.pow(10, mLog)

  const rows = useMemo(() => RHOS.map((r, i) => {
    const s = proofs(alpha, r.rho)
    return { i, r, s, share: shareOf(s, gamma, M) }
  }), [alpha, gamma, M])

  const t = rows[target]

  return (
    <div className="min-h-screen bg-background">
      <div className="mx-auto max-w-5xl px-4 py-6">
        {/* Header */}
        <div className="mb-6 flex items-start justify-between">
          <div>
            <div className="text-[11px] font-mono uppercase tracking-[0.22em] text-muted-foreground mb-1.5">
              Interactive · Appendix A, Figure 2
            </div>
            <h1 className="text-2xl font-bold tracking-tight text-foreground">
              The total compute cost of <span className="italic font-semibold">the proofs</span>
            </h1>
            <p className="text-sm text-muted-foreground mt-1 max-w-2xl">
              Each colored line is a ceiling on the share of outputs allowed to be bad. For a fixed
              confidence and a fixed total inference volume, it shows what fraction of <em>all</em>{" "}
              compute the verification proofs consume — as the per-proof overhead{" "}
              <em className="font-medium">γ</em> varies.
            </p>
          </div>
          <ThemeToggle />
        </div>

        {/* Main card */}
        <Card>
          <CardContent className="pt-6 space-y-4">
            {/* Topbar */}
            <div className="flex flex-wrap items-center justify-between gap-3">
              <span className="text-[11px] font-mono uppercase tracking-[0.16em] text-muted-foreground">
                Figure 2 — compute cost of proofs vs per-proof overhead
              </span>
              <div className="flex items-center gap-2">
                <span className="text-[10px] font-mono uppercase tracking-wider text-muted-foreground">
                  Confidence
                </span>
                <Tabs value={String(confIdx)} onValueChange={v => setConfIdx(parseInt(v, 10))}>
                  <TabsList className="h-8">
                    {CONFS.map((c, i) => (
                      <TabsTrigger key={i} value={String(i)} className="text-[11px] font-mono px-2.5">
                        {c.p}
                      </TabsTrigger>
                    ))}
                  </TabsList>
                </Tabs>
              </div>
            </div>

            {/* Legend */}
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-[9.5px] font-mono uppercase tracking-wider text-muted-foreground mr-1">
                Bad-output ceiling
              </span>
              {RHOS.map((r, i) => {
                const on = i === target
                return (
                  <button
                    key={i}
                    onClick={() => setTarget(i)}
                    className={`inline-flex items-center gap-2 text-[13px] px-2.5 py-1 rounded border transition-colors ${
                      on
                        ? "border-foreground bg-background text-foreground font-medium"
                        : "border-border bg-muted/50 text-muted-foreground hover:border-foreground/40"
                    }`}
                  >
                    <span
                      className="inline-block rounded-sm"
                      style={{
                        width: 18,
                        height: on ? 4 : 3,
                        background: r.color,
                        opacity: on ? 1 : 0.55,
                      }}
                    />
                    ρ ≤ {r.label}
                  </button>
                )
              })}
            </div>

            {/* Plot */}
            <ProofsPlot alpha={alpha} mLog={mLog} gLog={gLog} target={target} setGLog={setGLog} />

            {/* Sliders */}
            <div className="space-y-3 pt-2">
              <SliderRow
                tagLabel="Parameter"
                tagAccent
                label={<>total inference&nbsp;<span className="italic font-medium">M</span> =</>}
                value={`${sci(M)} pairs`}
                min={9}
                max={16}
                step={0.05}
                rawValue={mLog}
                onChange={setMLog}
              />
              <SliderRow
                tagLabel="Marker"
                label={<>overhead&nbsp;<span className="italic font-medium">γ</span> =</>}
                value={sci(gamma)}
                min={2}
                max={9}
                step={0.02}
                rawValue={gLog}
                onChange={setGLog}
              />
            </div>

            {/* Takeaway */}
            <div className="text-sm italic text-foreground bg-muted/40 border-l-2 border-foreground pl-3 pr-3 py-2.5 rounded-r">
              At γ = {sci(gamma)}, M = {sci(M)} and {CONFS[confIdx].p} confidence, holding bad outputs to{" "}
              <b className="not-italic font-semibold">≤ {t.r.label}</b> means the proofs consume{" "}
              <b className="not-italic font-semibold">{pct(t.share)}</b> of all compute — from{" "}
              {intStr(t.s)} spot-check proofs.
            </div>

            {/* Table */}
            <div>
              <div className="text-[10px] font-mono uppercase tracking-wider text-muted-foreground mb-2">
                At the marker — click a row to highlight its line
              </div>
              <table className="w-full text-sm border-collapse">
                <thead>
                  <tr>
                    <th className="text-left text-[9.5px] font-mono font-normal uppercase tracking-wider text-muted-foreground py-1.5 px-2 border-b border-border">
                      Bad-output ceiling ρ
                    </th>
                    <th className="text-right text-[9.5px] font-mono font-normal uppercase tracking-wider text-muted-foreground py-1.5 px-2 border-b border-border">
                      Proofs required s
                    </th>
                    <th className="text-right text-[9.5px] font-mono font-normal uppercase tracking-wider text-muted-foreground py-1.5 px-2 border-b border-border">
                      Compute cost of proofs (% of all compute)
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map(row => {
                    const on = row.i === target
                    return (
                      <tr
                        key={row.i}
                        onClick={() => setTarget(row.i)}
                        className={`cursor-pointer transition-colors ${on ? "bg-muted/50" : "hover:bg-muted/30"}`}
                        style={on ? { boxShadow: `inset 3px 0 0 ${row.r.color}` } : undefined}
                      >
                        <td className="py-2 px-2 border-b border-border/60 font-medium">
                          <span
                            className="inline-block w-2.5 h-2.5 rounded-sm mr-2 align-[-1px]"
                            style={{ background: row.r.color }}
                          />
                          ≤ {row.r.label}
                        </td>
                        <td className="py-2 px-2 border-b border-border/60 text-right font-mono">
                          {intStr(row.s)}
                        </td>
                        <td className="py-2 px-2 border-b border-border/60 text-right font-mono font-medium">
                          {pct(row.share)}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>

            {/* Formula */}
            <div className="bg-muted/40 border border-border rounded p-3 flex flex-wrap gap-x-6 gap-y-1">
              <span className="font-mono text-xs text-muted-foreground">
                <span className="text-foreground font-medium">s</span> = ⌈ ln(1/α) / ln(1/(1−ρ)) ⌉
              </span>
              <span className="font-mono text-xs text-muted-foreground">
                cost <span className="text-foreground font-medium">share</span> = sγ / (M + sγ)
              </span>
            </div>
          </CardContent>
        </Card>

        {/* Footer */}
        <p className="mt-6 text-xs text-muted-foreground max-w-2xl">
          The y-axis is{" "}
          <code className="font-mono bg-muted px-1 py-0.5 rounded text-[11px]">
            proof compute / (inference compute + proof compute)
          </code>{" "}
          — proofs as a share of <em>all</em> compute. The per-inference cost{" "}
          <code className="font-mono bg-muted px-1 py-0.5 rounded text-[11px]">c</code> cancels, and
          since all outputs are the same size, &ldquo;share of outputs&rdquo; equals &ldquo;share of
          output volume.&rdquo; Each line rises with slope 1 while proofs are cheap, then bends toward
          100% once proving rivals the inference itself. Sample size{" "}
          <code className="font-mono bg-muted px-1 py-0.5 rounded text-[11px]">s</code> depends only
          on ρ and α, so the curves shift bodily as{" "}
          <code className="font-mono bg-muted px-1 py-0.5 rounded text-[11px]">M</code> changes — that
          is where total volume re-enters.
        </p>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Slider row
// ---------------------------------------------------------------------------

function SliderRow({
  tagLabel, tagAccent, label, value, min, max, step, rawValue, onChange,
}: {
  tagLabel: string
  tagAccent?: boolean
  label: React.ReactNode
  value: string
  min: number
  max: number
  step: number
  rawValue: number
  onChange: (v: number) => void
}) {
  return (
    <div className="flex items-center gap-3 pt-2.5 border-t border-border/60">
      <span
        className={`text-[8.5px] font-mono uppercase tracking-wider text-background px-1.5 py-0.5 rounded shrink-0 ${
          tagAccent ? "bg-foreground" : "bg-muted-foreground"
        }`}
      >
        {tagLabel}
      </span>
      <Label className="text-sm whitespace-nowrap shrink-0">{label}</Label>
      <span className="text-xs font-mono bg-muted/60 border border-border px-2 py-0.5 rounded whitespace-nowrap shrink-0">
        {value}
      </span>
      <Slider
        value={[rawValue]}
        onValueChange={([v]) => onChange(v)}
        min={min}
        max={max}
        step={step}
        className="flex-1 min-w-[90px]"
      />
    </div>
  )
}
