# Cluster Simulation (Charlotte)

GPU cluster inference throughput simulator. Models covert overhead (Gamma) under different transparency regimes (matmul, network, memory sanitization) based on Erdil 2025.

## Quick start

```bash
npm run dev    # Next.js dev server on localhost:3000
npm run build  # production build check
```

## Running the simulator from the CLI

You can run the simulator directly without the browser:

```bash
npx tsx scripts/sim.ts                          # defaults: 8xH100, Llama 70B-ish covert workload
npx tsx scripts/sim.ts '{"gpu":"B200","count":72,"alpha":0.8}'
npx tsx scripts/sim.ts '{"gpu":"Rubin","count":1024,"sanitization":false,"stateBytes":1e12}'
```

All fields are optional (sensible defaults apply):

| Field | Type | Default | Description |
|---|---|---|---|
| `gpu` | string | `"H100"` | GPU type: H100, H200, A100, H20, B200, Rubin |
| `count` | number | `8` | Number of GPUs |
| `computeUtil` | 0-1 | `0.96` | Honest compute utilization |
| `memoryUtil` | 0-1 | `0.83` | Honest memory utilization |
| `alpha` | 0-1 | `0.83` | Proved compute fraction (matmul transparency) |
| `bOut` | bytes/s | `1e6` | Covert egress bandwidth |
| `bIn` | bytes/s | `100e6` | Covert ingress bandwidth |
| `sanitization` | bool | `true` | Enable memory sanitization |
| `epochS` | seconds | `3981` | Sanitization epoch length |
| `downtimeS` | seconds | `10` | Downtime per sanitization event |
| `survivingBytes` | bytes | `100e9` | Covert state surviving sanitization |
| `stateBytes` | bytes | `140e9` | Total covert state size (e.g. model weights) |
| `flopPerUnit` | FLOP | `580e6` | Compute per unit of covert output |
| `ingressBytesPerUnit` | bytes | `0` | Covert ingress per input |
| `egressBytesPerUnit` | bytes | `200` | Covert egress per output |

Output is JSON with `gamma`, `dominant` bottleneck, `finite` flag, per-channel gammas, and throughput numbers.

### Interesting things to explore

- How does Gamma change across GPU generations for the same workload? (H100 vs B200 vs Rubin)
- At what cluster size does covert inference become infeasible under full transparency?
- What's the minimum alpha (proved compute fraction) needed to make covert training impractical?
- How sensitive is Gamma to egress bandwidth? (sweep `bOut` from 1e3 to 1e8)
- What happens when you disable sanitization but crank alpha to 1.0?
- Compare small covert models (8B, `stateBytes:16e9, flopPerUnit:16e9`) vs large ones (405B, `stateBytes:810e9, flopPerUnit:8e12`)
- When does memory-fit become the binding constraint vs compute vs egress?

## Project structure

- `components/gamma-dashboard.tsx` — main UI, all controls and result visualization
- `lib/sim/gamma.ts` — core Gamma computation (`simulateDirect`, `composeGamma`)
- `lib/sim/types.ts` — type definitions (DirectScenario, GammaResult, etc.)
- `lib/sim/presets.ts` — hardware/covert workload presets, `computeHardwarePreset()`
- `lib/erdil/gpus.ts` — GPU definitions (H100, H200, A100, H20, B200, Rubin)
- `lib/erdil/models.ts` — model definitions for roofline analysis
- `scripts/sim.ts` — CLI wrapper for running simulations programmatically

## Key concepts

- **Gamma (Γ)** = dedicated throughput / verified throughput. Higher = more overhead for the covert workload.
- **Dominant bottleneck**: compute, egress, ingress, memory-fit, or duty (sanitization timing).
- **Matmul transparency (alpha)**: proves a fraction of claimed compute was real, setting a floor on honest utilization.
- **Memory sanitization**: periodic wipes force the covert workload to reload state, creating duty-cycle overhead.
- Gamma = infinity (finite: false) means the covert workload is completely infeasible.
