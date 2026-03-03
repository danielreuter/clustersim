"use client"

import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Label } from "@/components/ui/label"
import { HARDWARE_PROFILES, type HardwareProfile } from "@/lib/training-model"

interface HardwareSelectorProps {
  value: string
  onChange: (value: string) => void
  customHardware: HardwareProfile
  onCustomChange: (hardware: HardwareProfile) => void
}

export function HardwareSelector({ value, onChange }: HardwareSelectorProps) {
  return (
    <div className="space-y-2">
      <Label className="text-sm font-medium text-foreground">Hardware Profile</Label>
      <Select value={value} onValueChange={onChange}>
        <SelectTrigger className="w-full bg-background border-input">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {Object.entries(HARDWARE_PROFILES).map(([key, profile]) => (
            <SelectItem key={key} value={key}>
              {profile.name}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      {value !== "custom" && (
        <div className="mt-3 grid grid-cols-2 gap-2 text-xs text-muted-foreground">
          <div>HBM: {HARDWARE_PROFILES[value]?.hbmPerGpu} GB</div>
          <div>FLOPS: {HARDWARE_PROFILES[value]?.peakFlops} TF</div>
          <div>NVLink: {HARDWARE_PROFILES[value]?.nvlinkBandwidth} GB/s</div>
          <div>IB: {HARDWARE_PROFILES[value]?.infinibandBandwidth} GB/s</div>
        </div>
      )}
    </div>
  )
}
