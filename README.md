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

Final images use 32-bit density buffers with subpixel splatting. A robust
nonzero-density percentile establishes the white point, an ACES-style filmic
curve compresses highlights, and a gamma LUT feeds a 16-bit aurora palette from
indigo through cyan to warm highlights. PNG does not support 32-bit integer
channels; the 32-bit values are retained as accumulation precision before tone
mapping.

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
| `-exposure` | 2.5 | exposure applied before the filmic curve |
| `-white-percentile` | 99.5 | nonzero density percentile used as the HDR white point |

## Selection heuristic

Each candidate is accumulated into a small histogram. Candidates that revisit a
quantized point within 64 steps or occupy fewer than 64 histogram cells are
rejected. The visual-interest score combines:

```text
score = 0.30*coverage_preference
      + 0.40*dimension_preference
      + 0.25*global_entropy_preference
      + 0.05*chaos_preference
```

- **Coverage preference** favors moderate occupancy and penalizes both tiny
  loops and screen-filling clouds.
- **Dimension preference** uses multiscale box counting to favor richly layered
  structures around projected dimension 1.72 over simple curves or solid noise.
- **Global entropy preference** measures entropy against the entire histogram
  and penalizes both near-empty and maximally noisy images.
- **Chaos preference** uses the largest Lyapunov exponent from the analytic
  Clifford-map Jacobian only as a small tie-breaker; image-space complexity is
  intentionally more important than abstract dynamical chaos.

Horizontal and vertical symmetry are recorded as diagnostic metadata but do not
affect ranking; irregular chaotic structures are intentionally not penalized.
Coverage also gates the combined score, preventing an almost empty loop or a
nearly solid cloud from compensating for poor framing with entropy or chaos.

Framing uses the 0.1% and 99.9% coordinate quantiles, preventing rare outlier
points from shrinking the dominant structure. The final JSON records every
component metric, coefficients, bounds, settings, rejection counts, and seed.

## Test

```sh
go test ./...
```
