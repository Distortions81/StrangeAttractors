# Strange Attractors

A small, CPU-only Go program that searches random Clifford-map coefficients,
rejects collapsed or short-looping orbits, scores the survivors, and renders the
best one as a density PNG with JSON metadata.

The Clifford map is:

```text
x(n+1) = sin(a*y(n)) + c*cos(a*x(n))
y(n+1) = sin(b*x(n)) + d*cos(b*y(n))
```

## Run

Go 1.22 or newer is sufficient; there are no third-party dependencies.

```sh
go run . -out output/attractor -seed 42
```

This writes `output/attractor.png` and `output/attractor.json`. Leave `-seed`
at zero to choose a time-based seed; the chosen seed is always saved so a run
can be reproduced.

To generate a ranked batch:

```sh
go run . -out output/candidates -count 1000 \
  -width 1024 -height 1024 -seed 20260805
```

Batch output is named `0001_score-0.812345_id-000123.png` (with a matching
JSON file), so normal ascending filename order puts the highest-ranked image
first. Ranking and score filtering use the fixed-size screening histogram, so
scores remain comparable when render resolution or quality changes.
`batch.json` provides a machine-readable index.
Rendering runs concurrently; use `-workers` to control CPU and memory use.

For higher-quality 16-bit output restricted to a preferred score band:

```sh
go run . -out output/high-quality -count 1000 -width 1024 -height 1024 \
  -iterations 20000000 -min-score 0.3 -max-score 0.7 -gamma 2.2
```

Final images use 32-bit density buffers with subpixel splatting, then a gamma
lookup table maps density into 16-bit-per-channel PNG. PNG does not support
32-bit integer channels; the 32-bit values are retained as the accumulation
precision before tone mapping.

For a quick preview:

```sh
go run . -out output/preview -seed 42 \
  -samples 16 -screen-iterations 20000 \
  -iterations 250000 -width 800 -height 800
```

Useful flags:

| Flag | Default | Meaning |
| --- | ---: | --- |
| `-samples` | 64 | random coefficient sets tested |
| `-range` | 3 | coefficients are sampled uniformly from this signed range |
| `-screen-iterations` | 100000 | points used to score each candidate |
| `-iterations` | 2000000 | points used in the final render |
| `-burn-in` | 1000 | transient points discarded from each orbit |
| `-width`, `-height` | 1600 | final image dimensions |
| `-count` | 1 | accepted attractors to render; values above 1 enable ranked batch mode |
| `-workers` | half available CPUs | concurrent batch renderers |
| `-min-score`, `-max-score` | 0, 1 | accepted batch screening-score interval |
| `-gamma` | 2.2 | density tone-map gamma; larger values reveal faint details |

## Selection heuristic

Each candidate is accumulated into a small histogram. Candidates that revisit a
quantized point within 64 steps or occupy fewer than 64 histogram cells are
rejected. The remaining candidates are ranked using:

```text
score = 0.35*sqrt(occupancy) + 0.50*entropy + 0.15*symmetry
```

- **Occupancy** is the fraction of histogram cells touched by the orbit.
- **Entropy** is Shannon entropy normalized by the number of occupied cells.
- **Symmetry** is the better of horizontal and vertical reflection similarity.

The square root on occupancy keeps fine, filamentary attractors competitive
with dense blobs. The final JSON records all component metrics, coefficients,
bounds, settings, rejection counts, and seed rather than only the aggregate
score.

## Test

```sh
go test ./...
```
