"use client"

import { Slider } from "@/components/ui/slider"
import { Label } from "@/components/ui/label"

interface LinearSliderProps {
  label: string
  value: number
  onChange: (value: number) => void
  min: number
  max: number
  step?: number
  unit?: string
  formatValue?: (val: number) => string
}

export function LinearSlider({
  label,
  value,
  onChange,
  min,
  max,
  step = 1,
  unit = "",
  formatValue,
}: LinearSliderProps) {
  const displayValue = formatValue ? formatValue(value) : `${value}${unit}`

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <Label className="text-sm font-medium text-foreground">{label}</Label>
        <span className="font-mono text-sm font-semibold text-foreground">{displayValue}</span>
      </div>
      <Slider
        value={[value]}
        onValueChange={(v) => onChange(v[0])}
        min={min}
        max={max}
        step={step}
        className="w-full"
      />
    </div>
  )
}
