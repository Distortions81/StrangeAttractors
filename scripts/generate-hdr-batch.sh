#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$script_dir/.." && pwd)

usage() {
    cat <<'EOF'
Generate a ranked batch of high-quality Clifford attractors.

Usage:
  scripts/generate-hdr-batch.sh [OUTPUT_DIR] [-- additional Go flags]

Default output:
  output/candidates-1000-2048-hdr-YYYYMMDD-HHMMSS

Environment overrides:
  COUNT=1000
  WIDTH=2048
  HEIGHT=2048
  ITERATIONS=200000000
  SCREEN_ITERATIONS=200000
  SCREEN_SIZE=256
  MIN_SCORE=0.80
  MAX_SCORE=1.0
  GAMMA=1.8
  EXPOSURE=0.9
  WHITE_PERCENTILE=99.7
  SUPERSAMPLE=3
  GLOW_STRENGTH=0.32
  GLOW_RADIUS=14
  GLOW_THRESHOLD=0.65
  SOFTNESS=2
  WORKERS=<half the available CPUs>
  SEED=0                       # zero generates and records a time-based seed

Examples:
  scripts/generate-hdr-batch.sh
  SEED=42 scripts/generate-hdr-batch.sh output/reproducible-hdr
  COUNT=20 ITERATIONS=5000000 scripts/generate-hdr-batch.sh output/preview
  scripts/generate-hdr-batch.sh output/custom -- -gamma 1.9 -exposure 3.0

Flags after "--" are passed last and therefore override script defaults.
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
output_dir=${1:-"output/candidates-1000-2048-hdr-$timestamp"}
if (($# > 0)); then shift; fi
if [[ ${1:-} == "--" ]]; then shift; fi

count=${COUNT:-1000}
width=${WIDTH:-2048}
height=${HEIGHT:-2048}
iterations=${ITERATIONS:-200000000}
screen_iterations=${SCREEN_ITERATIONS:-200000}
screen_size=${SCREEN_SIZE:-256}
min_score=${MIN_SCORE:-0.80}
max_score=${MAX_SCORE:-1.0}
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

cleanup_staging() {
    local stage
    while IFS= read -r -d '' stage; do
        rm -rf -- "$stage"
    done < <(find "$output_dir" -mindepth 1 -maxdepth 1 -type d -name '.rendering-*' -print0 2>/dev/null)
}
trap cleanup_staging EXIT INT TERM

printf 'Generating %s ranked %sx%s HDR attractors in %s\n' "$count" "$width" "$height" "$output_dir"
printf 'Samples/image=%s score=[%s,%s] workers=%s seed=%s\n' "$iterations" "$min_score" "$max_score" "$workers" "$seed"

cd -- "$repo_dir"
go run . \
    -out "$output_dir" \
    -count "$count" \
    -workers "$workers" \
    -seed "$seed" \
    -width "$width" \
    -height "$height" \
    -iterations "$iterations" \
    -screen-iterations "$screen_iterations" \
    -screen-size "$screen_size" \
    -min-score "$min_score" \
    -max-score "$max_score" \
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
