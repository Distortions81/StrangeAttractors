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

For the tuned 2048×2048 HDR workflow, use the bundled script:

```sh
scripts/generate-hdr-batch.sh
```

It defaults to 1,000 images, 200 million samples per image, score `0.80–1.00`,
and a timestamped directory under `output/`. Its settings can be overridden by
environment variables or trailing Go flags:

```sh
SEED=42 WORKERS=12 scripts/generate-hdr-batch.sh output/my-hdr-batch
COUNT=20 scripts/generate-hdr-batch.sh output/test -- -exposure 3.0
```

The script refuses to write into a nonempty directory. Ranked PNG/JSON pairs are
written directly into the requested folder as workers finish, so progress is
visible and completed output remains available if the run is interrupted.

## Evolution sequences

Generate a coherent lineage of nearby attractors as numbered PNG frames:

```sh
scripts/generate-evolution.sh
```

Each frame is selected from viable mutations of the previous frame. Selection
balances visual-interest score with novelty relative to the recent lineage,
while coefficient momentum and a temporally smoothed camera keep the motion
continuous. Evolution screening waits through a full extra screening window,
preventing long transients that later collapse into a periodic cycle from
entering the lineage. The default run produces 240 high-quality 2048x2048
frames. Use a smaller preview before committing to a full render:

```sh
FRAMES=24 WIDTH=512 HEIGHT=512 ITERATIONS=2000000 \
  scripts/generate-evolution.sh output/evolution-preview
```

Frames use Resolve-friendly names such as `frame_000000.png`; scores stay in
matching JSON sidecars instead of the image filename. The sidecars capture the
parent, mutation distance, raw and smoothed framing, coefficients, score, and
render settings. `evolution.json` stores the artwork title and indexes the
complete lineage. Set `TITLE="Electric Wings"` to give a render a display name.
To make a 10-bit HEVC movie at 30 fps from inside its output directory:

```sh
ffmpeg -framerate 30 -i 'frame_%06d.png' \
  -c:v libx265 -pix_fmt yuv420p10le evolution.mp4
```

Final images use 64-bit density buffers with 16-bit bilinear weights on a 3×
supersampled grid, then downsample in linear density space. A robust nonzero
density percentile establishes the white point without clipping values above
it. Two gentle binomial reconstruction passes soften pixel-scale texture before
bright knots seed a broad linear-light glow and an ACES-style filmic curve
and gamma LUT feed a cinematic 16-bit palette with indigo shadows, cyan/teal
midtones, and warm highlights. The wide accumulator
and unclipped tone curve preserve very intense regions instead of flattening
them into one color.

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
| `-gamma` | 1.8 | density tone-map gamma; larger values reveal faint details |
| `-exposure` | 0.9 | exposure applied before the filmic curve |
| `-white-percentile` | 99.7 | nonzero density percentile used as the HDR white point |
| `-supersample` | 3 | linear-density supersampling factor |
| `-glow-strength` | 0.32 | intensity of highlight-driven glow |
| `-glow-radius` | 14 | glow radius in output pixels |
| `-glow-threshold` | 0.65 | glow onset relative to the density white point |
| `-softness` | 2 | linear-density reconstruction smoothing passes |
| `-evolve-frames` | 0 | numbered lineage frames; zero disables evolution mode |
| `-evolve-name` | empty | artwork title stored in `evolution.json` |
| `-evolve-offspring` | 48 | viable mutations considered for each next frame |
| `-evolve-mutation` | 0.035 | coefficient mutation scale per frame |
| `-evolve-min-score` | 0.65 | minimum visual score retained throughout the lineage |

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
