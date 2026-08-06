#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$script_dir/.." && pwd)

usage() {
    cat <<'EOF'
Generate a coherent, high-quality evolutionary Clifford-attractor PNG sequence.

Usage:
  scripts/generate-evolution.sh [OUTPUT_DIR] [-- additional Go flags]

Default output:
  output/evolution-240-2048-YYYYMMDD-HHMMSS

Environment overrides:
  FRAMES=240
  WIDTH=2048
  HEIGHT=2048
  ITERATIONS=100000000
  SCREEN_ITERATIONS=200000
  SCREEN_SIZE=256
  SAMPLES=256
  EVOLVE_OFFSPRING=48
  EVOLVE_MUTATION=0.035
  EVOLVE_MIN_SCORE=0.65
  GAMMA=1.8
  EXPOSURE=0.9
  WHITE_PERCENTILE=99.7
  SUPERSAMPLE=3
  GLOW_STRENGTH=0.32
  GLOW_RADIUS=14
  GLOW_THRESHOLD=0.65
  SOFTNESS=2
  WORKERS=<half the available CPUs>
  SEED=0

Examples:
  scripts/generate-evolution.sh
  SEED=42 FRAMES=120 scripts/generate-evolution.sh output/my-lineage
  FRAMES=24 ITERATIONS=2000000 WIDTH=512 HEIGHT=512 \
    scripts/generate-evolution.sh output/evolution-preview

Flags after "--" are passed last and override script defaults.
EOF
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
    usage
    exit 0
fi

cpu_count=$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '2')
default_workers=$((cpu_count / 2))
if ((default_workers < 1)); then default_workers=1; fi

timestamp=$(date -u +%Y%m%d-%H%M%S)
frames=${FRAMES:-240}
width=${WIDTH:-2048}
height=${HEIGHT:-2048}
output_dir=${1:-"output/evolution-${frames}-${width}-${timestamp}"}
if (($# > 0)); then shift; fi
if [[ ${1:-} == "--" ]]; then shift; fi

iterations=${ITERATIONS:-100000000}
screen_iterations=${SCREEN_ITERATIONS:-200000}
screen_size=${SCREEN_SIZE:-256}
samples=${SAMPLES:-256}
offspring=${EVOLVE_OFFSPRING:-48}
mutation=${EVOLVE_MUTATION:-0.035}
min_score=${EVOLVE_MIN_SCORE:-0.65}
gamma=${GAMMA:-1.8}
exposure=${EXPOSURE:-0.9}
white_percentile=${WHITE_PERCENTILE:-99.7}
supersample=${SUPERSAMPLE:-3}
glow_strength=${GLOW_STRENGTH:-0.32}
glow_radius=${GLOW_RADIUS:-14}
glow_threshold=${GLOW_THRESHOLD:-0.65}
softness=${SOFTNESS:-2}
workers=${WORKERS:-$default_workers}
seed=${SEED:-0}

if [[ -e $output_dir ]] && find "$output_dir" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null | grep -q .; then
    printf 'error: output directory is not empty: %s\n' "$output_dir" >&2
    exit 1
fi
mkdir -p -- "$output_dir"

printf 'Generating %s-frame %sx%s attractor evolution in %s\n' "$frames" "$width" "$height" "$output_dir"
printf 'Samples/frame=%s offspring=%s mutation=%s minimum-score=%s workers=%s seed=%s\n' \
    "$iterations" "$offspring" "$mutation" "$min_score" "$workers" "$seed"

cd -- "$repo_dir"
go run . \
    -out "$output_dir" \
    -evolve-frames "$frames" \
    -evolve-offspring "$offspring" \
    -evolve-mutation "$mutation" \
    -evolve-min-score "$min_score" \
    -samples "$samples" \
    -workers "$workers" \
    -seed "$seed" \
    -width "$width" \
    -height "$height" \
    -iterations "$iterations" \
    -screen-iterations "$screen_iterations" \
    -screen-size "$screen_size" \
    -gamma "$gamma" \
    -exposure "$exposure" \
    -white-percentile "$white_percentile" \
    -supersample "$supersample" \
    -glow-strength "$glow_strength" \
    -glow-radius "$glow_radius" \
    -glow-threshold "$glow_threshold" \
    -softness "$softness" \
    "$@"

printf 'Complete: %s\n' "$output_dir"
