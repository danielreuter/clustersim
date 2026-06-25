"use client"

import * as React from "react"

import { cn } from "@/lib/utils"

type SliderProps = Omit<
  React.InputHTMLAttributes<HTMLInputElement>,
  "defaultValue" | "max" | "min" | "onChange" | "step" | "type" | "value"
> & {
  defaultValue?: number[]
  max?: number
  min?: number
  onValueChange?: (value: number[]) => void
  onValueCommit?: (value: number[]) => void
  orientation?: "horizontal" | "vertical"
  step?: number
  value?: number[]
}

function clamp(n: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, n))
}

function firstValue(value: number[] | undefined, fallback: number): number {
  const next = Array.isArray(value) && Number.isFinite(value[0]) ? value[0] : fallback
  return next
}

function Slider({
  className,
  defaultValue,
  disabled,
  max = 100,
  min = 0,
  onValueChange,
  onValueCommit,
  orientation = "horizontal",
  step = 1,
  value,
  ...props
}: SliderProps) {
  const fallbackValue = firstValue(defaultValue, min)
  const [internalValue, setInternalValue] = React.useState(fallbackValue)
  const isControlled = Array.isArray(value)
  const rawValue = isControlled ? firstValue(value, fallbackValue) : internalValue
  const currentValue = clamp(rawValue, min, max)
  const range = max - min
  const percentage = range > 0 ? ((currentValue - min) / range) * 100 : 0

  const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const next = Number(event.currentTarget.value)
    if (!isControlled) setInternalValue(next)
    onValueChange?.([next])
  }

  const handleCommit = (event: React.SyntheticEvent<HTMLInputElement>) => {
    onValueCommit?.([Number(event.currentTarget.value)])
  }

  return (
    <span
      data-disabled={disabled ? "" : undefined}
      data-orientation={orientation}
      data-slot="slider"
      className={cn(
        "relative flex h-4 w-full touch-pan-y items-center select-none data-[disabled]:opacity-50",
        className
      )}
    >
      <input
        {...props}
        data-slot="slider-input"
        disabled={disabled}
        max={max}
        min={min}
        onBlur={handleCommit}
        onChange={handleChange}
        onKeyUp={handleCommit}
        onMouseUp={handleCommit}
        onTouchEnd={handleCommit}
        step={step}
        type="range"
        value={currentValue}
        className="peer absolute inset-0 z-10 m-0 h-full w-full cursor-pointer opacity-0 disabled:cursor-not-allowed"
      />
      <span
        data-slot="slider-track"
        className="relative h-1.5 w-full grow overflow-hidden rounded-full bg-muted"
      >
        <span
          data-slot="slider-range"
          className="absolute h-full bg-primary"
          style={{ left: 0, width: `${percentage}%` }}
        />
      </span>
      <span
        data-slot="slider-thumb"
        className="pointer-events-none absolute top-1/2 block size-4 shrink-0 rounded-full border border-primary bg-white shadow-sm ring-ring/50 transition-[color,box-shadow] hover:ring-4 peer-focus-visible:ring-4 peer-focus-visible:outline-hidden disabled:pointer-events-none disabled:opacity-50"
        style={{ left: `${percentage}%`, transform: "translate(-50%, -50%)" }}
      />
    </span>
  )
}

export { Slider }
