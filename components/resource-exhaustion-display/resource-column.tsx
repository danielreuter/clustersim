"use client"

import { Slider } from "@/components/ui/slider"
import { SensitivityChart } from "./sensitivity-chart"
import type { SensitivityPoint } from "@/lib/training-model"

interface ResourceColumnProps {
  label: string
  symbol: string
  value: number
  onChange: (value: number) => void
  sensitivityData: SensitivityPoint[]
  color: string
  isActive: boolean
}

export function ResourceColumn({
  label,
  symbol,
  value,
  onChange,
  sensitivityData,
  color,
  isActive,
}: ResourceColumnProps) {
  // Convert value to log scale position (0-100)
  const logMin = Math.log10(0.01)
  const logMax = Math.log10(100)
  const logValue = Math.log10(value)
  const sliderPosition = ((logValue - logMin) / (logMax - logMin)) * 100

  const handleChange = (newPosition: number[]) => {
    const pos = newPosition[0]
    const logVal = logMin + (pos / 100) * (logMax - logMin)
    const newValue = Math.pow(10, logVal)
    onChange(Math.round(newValue * 100) / 100)
  }

  return (
    <div className="flex flex-col gap-3">
      {/* Header */}
      <div className="text-center">
        <div className="text-sm font-medium text-foreground">{label}</div>
        <div className="text-xs text-muted-foreground font-mono">{symbol}</div>
      </div>

      {/* Chart */}
      <SensitivityChart data={sensitivityData} currentValue={value} color={color} isActive={isActive} />

      {/* Slider */}
      <div className="px-2">
        <Slider value={[sliderPosition]} onValueChange={handleChange} max={100} min={0} step={0.1} className="w-full" />
        <div className="flex justify-between text-[10px] text-muted-foreground font-mono mt-1">
          <span>0.01%</span>
          <span>100%</span>
        </div>
      </div>
    </div>
  )
}
