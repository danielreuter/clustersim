"use client"

import { useMemo } from "react"

interface SensitivityChartProps {
  data: { resourceValue: number; slowdown: number }[]
  currentValue: number
  color: string
  isActive: boolean
}

export function SensitivityChart({ data, currentValue, color, isActive }: SensitivityChartProps) {
  const { path, currentX, currentY, threshold100Y } = useMemo(() => {
    if (!data || data.length === 0) return { path: "", currentX: 0, currentY: 0, threshold100Y: 0 }

    // Chart dimensions
    const width = 200
    const height = 120
    const padding = { top: 10, right: 10, bottom: 20, left: 35 }
    const chartWidth = width - padding.left - padding.right
    const chartHeight = height - padding.top - padding.bottom

    // Log scale helpers
    const xMin = Math.log10(0.1)
    const xMax = Math.log10(100)
    const yMin = Math.log10(1)
    const yMax = Math.log10(10000)

    const scaleX = (val: number) => {
      const log = Math.log10(Math.max(0.1, Math.min(100, val)))
      return padding.left + ((log - xMin) / (xMax - xMin)) * chartWidth
    }

    const scaleY = (val: number) => {
      const log = Math.log10(Math.max(1, Math.min(10000, val)))
      return padding.top + chartHeight - ((log - yMin) / (yMax - yMin)) * chartHeight
    }

    // Build path
    const points = data.map((d) => ({ x: scaleX(d.resourceValue), y: scaleY(d.slowdown) }))
    const pathD = points.map((p, i) => `${i === 0 ? "M" : "L"} ${p.x} ${p.y}`).join(" ")

    // Current position
    const currentPoint = data.find((d) => Math.abs(d.resourceValue - currentValue) < currentValue * 0.15)
    const cx = scaleX(currentValue)
    const cy = currentPoint ? scaleY(currentPoint.slowdown) : scaleY(1)

    return {
      path: pathD,
      currentX: cx,
      currentY: cy,
      threshold100Y: scaleY(100),
      width,
      height,
      padding,
      scaleX,
      scaleY,
    }
  }, [data, currentValue])

  const currentSlowdown = data.find((d) => Math.abs(d.resourceValue - currentValue) < currentValue * 0.15)?.slowdown

  return (
    <div className={`transition-opacity ${!isActive ? "opacity-40" : ""}`}>
      <svg viewBox="0 0 200 120" className="w-full h-auto">
        {/* Grid lines */}
        <line x1="35" y1="10" x2="35" y2="100" stroke="currentColor" className="text-border" strokeWidth="1" />
        <line x1="35" y1="100" x2="190" y2="100" stroke="currentColor" className="text-border" strokeWidth="1" />

        {/* Y-axis labels */}
        <text x="32" y="100" textAnchor="end" className="fill-muted-foreground text-[8px]">
          1x
        </text>
        <text x="32" y="70" textAnchor="end" className="fill-muted-foreground text-[8px]">
          10x
        </text>
        <text x="32" y="40" textAnchor="end" className="fill-muted-foreground text-[8px]">
          100x
        </text>
        <text x="32" y="13" textAnchor="end" className="fill-muted-foreground text-[8px]">
          10kx
        </text>

        {/* X-axis labels */}
        <text x="35" y="115" textAnchor="start" className="fill-muted-foreground text-[8px]">
          0.1%
        </text>
        <text x="112" y="115" textAnchor="middle" className="fill-muted-foreground text-[8px]">
          10%
        </text>
        <text x="190" y="115" textAnchor="end" className="fill-muted-foreground text-[8px]">
          100%
        </text>

        {/* 100x threshold line */}
        <line
          x1="35"
          y1={threshold100Y}
          x2="190"
          y2={threshold100Y}
          stroke="var(--destructive)"
          strokeWidth="1"
          strokeDasharray="4 2"
        />

        {/* Data line */}
        <path d={path} fill="none" stroke={color} strokeWidth="2" />

        {/* Current position marker */}
        <line x1={currentX} y1="10" x2={currentX} y2="100" stroke={color} strokeWidth="1.5" strokeDasharray="3 2" />
        <circle cx={currentX} cy={currentY} r="4" fill={color} />
      </svg>

      {/* Current value display */}
      <div className="flex items-center justify-between text-xs mt-1 px-1">
        <span className="text-muted-foreground">{currentValue.toFixed(1)}%</span>
        {currentSlowdown && (
          <span className="font-mono font-semibold text-foreground">{currentSlowdown.toFixed(1)}x</span>
        )}
      </div>
    </div>
  )
}
