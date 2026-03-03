"use client"

import type { TimeBreakdown, DerivedQuantities } from "@/lib/training-model"

interface OverheadSummaryProps {
  time: TimeBreakdown
  derived: DerivedQuantities
}

export function OverheadSummary({ time, derived }: OverheadSummaryProps) {
  const slowdown = time.totalSlowdown
  const isAboveThreshold = slowdown >= 100

  return (
    <div className="flex items-center gap-8">
      {/* Main slowdown number */}
      <div className="text-center">
        <div className={`text-5xl font-bold font-mono ${isAboveThreshold ? "text-destructive" : "text-foreground"}`}>
          {slowdown >= 1000 ? `${(slowdown / 1000).toFixed(1)}k` : slowdown.toFixed(1)}x
        </div>
        <div className="text-sm text-muted-foreground mt-1">Total Slowdown</div>
      </div>

      {/* Key metrics */}
      <div className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm">
        <div>
          <span className="text-muted-foreground">Bottleneck:</span>
          <span className="ml-2 font-medium text-foreground">{time.bottleneck}</span>
        </div>
        <div>
          <span className="text-muted-foreground">Min GPUs:</span>
          <span className="ml-2 font-mono font-medium text-foreground">{derived.minGpus}</span>
        </div>
        <div>
          <span className="text-muted-foreground">Pipeline Efficiency:</span>
          <span className="ml-2 font-mono font-medium text-foreground">
            {(derived.pipelineEfficiency * 100).toFixed(0)}%
          </span>
        </div>
        <div>
          <span className="text-muted-foreground">Comm Regime:</span>
          <span className="ml-2 font-medium text-foreground">{derived.communicationRegime}</span>
        </div>
      </div>

      {/* Threshold indicator */}
      {isAboveThreshold && (
        <div className="px-3 py-1.5 bg-destructive/10 border border-destructive/30 rounded-md">
          <span className="text-destructive text-sm font-medium">Above 100x threshold</span>
        </div>
      )}
    </div>
  )
}
