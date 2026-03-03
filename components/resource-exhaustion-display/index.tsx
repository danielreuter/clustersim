"use client"

import { useState, useMemo } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ResourceColumn } from "./resource-column"
import { LinearSlider } from "./linear-slider"
import { HardwareSelector } from "./hardware-selector"
import { OverheadSummary } from "./overhead-summary"
import {
  HARDWARE_PROFILES,
  calculateDerivedQuantities,
  calculateTimeBreakdown,
  calculateSensitivityCurve,
  type ResourceLevels,
  type ModelConfig,
  type AdversaryConfig,
  type HardwareProfile,
} from "@/lib/training-model"

export function ResourceExhaustionDisplay() {
  // Resource levels
  const [resources, setResources] = useState<ResourceLevels>({
    compute: 10,
    hbmCapacity: 10,
    nvlinkBandwidth: 50,
    infinibandBandwidth: 50,
  })

  // Model configuration
  const [model, setModel] = useState<ModelConfig>({
    totalParams: 70,
    hiddenDim: 8192,
    numLayers: 80,
    attentionHeads: 64,
    sequenceLength: 2048,
    microbatchSize: 2048,
  })

  // Adversary configuration
  const [adversary, setAdversary] = useState<AdversaryConfig>({
    maxPipelineStages: 8,
    overlapFactor: 0.5,
    bytesPerParam: 16,
  })

  // Hardware selection
  const [hardwareKey, setHardwareKey] = useState("h100")
  const [customHardware, setCustomHardware] = useState<HardwareProfile>(HARDWARE_PROFILES.h100)
  const hardware = hardwareKey === "custom" ? customHardware : HARDWARE_PROFILES[hardwareKey]

  // Calculations
  const derived = useMemo(
    () => calculateDerivedQuantities(resources, model, adversary, hardware),
    [resources, model, adversary, hardware],
  )

  const time = useMemo(
    () => calculateTimeBreakdown(resources, model, adversary, hardware, derived),
    [resources, model, adversary, hardware, derived],
  )

  // Sensitivity curves
  const sensitivityData = useMemo(
    () => ({
      compute: calculateSensitivityCurve(resources, model, adversary, hardware, "compute"),
      hbmCapacity: calculateSensitivityCurve(resources, model, adversary, hardware, "hbmCapacity"),
      nvlinkBandwidth: calculateSensitivityCurve(resources, model, adversary, hardware, "nvlinkBandwidth"),
      infinibandBandwidth: calculateSensitivityCurve(resources, model, adversary, hardware, "infinibandBandwidth"),
    }),
    [resources, model, adversary, hardware],
  )

  const updateResource = (key: keyof ResourceLevels, value: number) => {
    setResources((prev) => ({ ...prev, [key]: value }))
  }

  const updateModel = (key: keyof ModelConfig, value: number) => {
    setModel((prev) => ({ ...prev, [key]: value }))
  }

  const updateAdversary = (key: keyof AdversaryConfig, value: number) => {
    setAdversary((prev) => ({ ...prev, [key]: value }))
  }

  // Determine which resource is active (bottleneck)
  const getIsActive = (resourceKey: string) => {
    if (time.bottleneck === "compute" && resourceKey === "compute") return true
    if (time.bottleneck === "pipeline bubbles (HBM capacity)" && resourceKey === "hbmCapacity") return true
    if (time.bottleneck === "network (NVLink)" && resourceKey === "nvlinkBandwidth") return true
    if (time.bottleneck === "network (InfiniBand)" && resourceKey === "infinibandBandwidth") return true
    return false
  }

  const chartColors = {
    compute: "hsl(220, 70%, 50%)",
    hbmCapacity: "hsl(150, 60%, 45%)",
    nvlinkBandwidth: "hsl(35, 90%, 50%)",
    infinibandBandwidth: "hsl(340, 70%, 50%)",
  }

  return (
    <div className="min-h-screen bg-background">
      <div className="mx-auto max-w-6xl px-4 py-6">
        {/* Header */}
        <div className="mb-6">
          <h1 className="text-2xl font-bold tracking-tight text-foreground">Training Overhead Analysis</h1>
          <p className="text-sm text-muted-foreground mt-1">
            Model how datacenter resource exhaustion affects covert AI training throughput
          </p>
        </div>

        {/* Top Section: Overhead Output */}
        <Card className="mb-6 bg-card border-border">
          <CardContent className="pt-6">
            {/* Summary */}
            <OverheadSummary time={time} derived={derived} />

            {/* 4-column resource grid: chart + slider per column */}
            <div className="grid grid-cols-4 gap-4 mt-6 pt-6 border-t border-border">
              <ResourceColumn
                label="Compute"
                symbol="f_C"
                value={resources.compute}
                onChange={(v) => updateResource("compute", v)}
                sensitivityData={sensitivityData.compute}
                color={chartColors.compute}
                isActive={getIsActive("compute")}
              />
              <ResourceColumn
                label="HBM Capacity"
                symbol="f_HBM"
                value={resources.hbmCapacity}
                onChange={(v) => updateResource("hbmCapacity", v)}
                sensitivityData={sensitivityData.hbmCapacity}
                color={chartColors.hbmCapacity}
                isActive={getIsActive("hbmCapacity")}
              />
              <ResourceColumn
                label="NVLink BW"
                symbol="f_NV"
                value={resources.nvlinkBandwidth}
                onChange={(v) => updateResource("nvlinkBandwidth", v)}
                sensitivityData={sensitivityData.nvlinkBandwidth}
                color={chartColors.nvlinkBandwidth}
                isActive={getIsActive("nvlinkBandwidth")}
              />
              <ResourceColumn
                label="InfiniBand BW"
                symbol="f_IB"
                value={resources.infinibandBandwidth}
                onChange={(v) => updateResource("infinibandBandwidth", v)}
                sensitivityData={sensitivityData.infinibandBandwidth}
                color={chartColors.infinibandBandwidth}
                isActive={getIsActive("infinibandBandwidth")}
              />
            </div>
          </CardContent>
        </Card>

        {/* Bottom Section: Configuration Tabs */}
        <Tabs defaultValue="model" className="w-full">
          <TabsList className="grid w-full grid-cols-3 bg-muted">
            <TabsTrigger value="model">Model</TabsTrigger>
            <TabsTrigger value="adversary">Adversary</TabsTrigger>
            <TabsTrigger value="hardware">Hardware</TabsTrigger>
          </TabsList>

          <TabsContent value="model" className="mt-4">
            <Card className="bg-card border-border">
              <CardHeader className="pb-4">
                <CardTitle className="text-base font-semibold text-foreground">Model Configuration</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-6">
                  <LinearSlider
                    label="Total Parameters"
                    value={model.totalParams}
                    onChange={(v) => updateModel("totalParams", v)}
                    min={1}
                    max={1000}
                    step={1}
                    formatValue={(v) => `${v}B`}
                  />
                  <LinearSlider
                    label="Hidden Dimension"
                    value={model.hiddenDim}
                    onChange={(v) => updateModel("hiddenDim", v)}
                    min={1024}
                    max={32768}
                    step={256}
                  />
                  <LinearSlider
                    label="Number of Layers"
                    value={model.numLayers}
                    onChange={(v) => updateModel("numLayers", v)}
                    min={8}
                    max={256}
                    step={1}
                  />
                  <LinearSlider
                    label="Sequence Length"
                    value={model.sequenceLength}
                    onChange={(v) => updateModel("sequenceLength", v)}
                    min={512}
                    max={131072}
                    step={512}
                  />
                  <LinearSlider
                    label="Microbatch Size"
                    value={model.microbatchSize}
                    onChange={(v) => updateModel("microbatchSize", v)}
                    min={256}
                    max={65536}
                    step={256}
                    formatValue={(v) => `${v} tokens`}
                  />
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="adversary" className="mt-4">
            <Card className="bg-card border-border">
              <CardHeader className="pb-4">
                <CardTitle className="text-base font-semibold text-foreground">Adversary Capability</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-6">
                  <LinearSlider
                    label="Max Pipeline Stages"
                    value={adversary.maxPipelineStages}
                    onChange={(v) => updateAdversary("maxPipelineStages", v)}
                    min={1}
                    max={64}
                    step={1}
                  />
                  <LinearSlider
                    label="Comm/Compute Overlap"
                    value={adversary.overlapFactor}
                    onChange={(v) => updateAdversary("overlapFactor", v)}
                    min={0}
                    max={1}
                    step={0.05}
                    formatValue={(v) => `${(v * 100).toFixed(0)}%`}
                  />
                  <LinearSlider
                    label="Bytes per Parameter"
                    value={adversary.bytesPerParam}
                    onChange={(v) => updateAdversary("bytesPerParam", v)}
                    min={12}
                    max={16}
                    step={4}
                    formatValue={(v) => `${v}B`}
                  />
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="hardware" className="mt-4">
            <Card className="bg-card border-border">
              <CardHeader className="pb-4">
                <CardTitle className="text-base font-semibold text-foreground">Hardware Profile</CardTitle>
              </CardHeader>
              <CardContent>
                <HardwareSelector
                  value={hardwareKey}
                  onChange={setHardwareKey}
                  customHardware={customHardware}
                  onCustomChange={setCustomHardware}
                />
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}
