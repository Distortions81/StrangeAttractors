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
