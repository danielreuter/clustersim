#!/bin/bash
# Dump key source files into a single text file for external review.
# Usage: ./scripts/dump-for-review.sh [output_file]

OUT="${1:-/tmp/clustersim-review.txt}"

FILES=(
  "lib/sim/types.ts"
  "lib/sim/gamma.ts"
  "lib/sim/roofline.ts"
  "lib/sim/presets.ts"
  "lib/sim/index.ts"
  "components/gamma-dashboard.tsx"
)

cd "$(dirname "$0")/.." || exit 1

> "$OUT"
for f in "${FILES[@]}"; do
  echo "================================================================================" >> "$OUT"
  echo "FILE: $f" >> "$OUT"
  echo "================================================================================" >> "$OUT"
  cat "$f" >> "$OUT"
  echo "" >> "$OUT"
  echo "" >> "$OUT"
done

echo "Wrote $(wc -l < "$OUT" | tr -d ' ') lines to $OUT"
