package main

import (
	"math"
	"math/rand"
	"testing"
)

func TestCliffordStep(t *testing.T) {
	p := coefficients{A: -1.4, B: 1.6, C: 1.0, D: 0.7}
	x, y := clifford(p, 0.1, 0.1)
	wantX := math.Sin(p.A*0.1) + p.C*math.Cos(p.A*0.1)
	wantY := math.Sin(p.B*0.1) + p.D*math.Cos(p.B*0.1)
	if math.Abs(x-wantX) > 1e-15 || math.Abs(y-wantY) > 1e-15 {
		t.Fatalf("clifford() = (%g, %g), want (%g, %g)", x, y, wantX, wantY)
	}
}

func TestScoreHistogram(t *testing.T) {
	uniform := []uint64{1, 1, 1, 1}
	m := scoreHistogram(uniform, 2, 2, 0.1)
	if m.Occupancy != 1 || math.Abs(m.Entropy-1) > 1e-12 || math.Abs(m.GlobalEntropy-1) > 1e-12 || math.Abs(m.Symmetry-1) > 1e-12 {
		t.Fatalf("unexpected uniform metrics: %+v", m)
	}

	concentrated := []uint64{100, 0, 0, 0}
	c := scoreHistogram(concentrated, 2, 2, 0)
	if c.Occupancy != 0.25 || c.Entropy != 0 || c.Symmetry != 0 {
		t.Fatalf("unexpected concentrated metrics: %+v", c)
	}
}

func TestVisualScorePenalizesEmptyAndFullHistograms(t *testing.T) {
	const size = 64
	full := make([]uint64, size*size)
	for i := range full {
		full[i] = 1
	}
	sparse := make([]uint64, size*size)
	for i := 0; i < 8; i++ {
		sparse[i*size+i] = 1
	}
	structured := make([]uint64, size*size)
	for y := 8; y < 56; y++ {
		x := 16 + (y*y/19)%32
		structured[y*size+x] = 1
		structured[y*size+x+1] = 1
	}
	fullScore := scoreHistogram(full, size, size, 0.1).Score
	sparseScore := scoreHistogram(sparse, size, size, 0.1).Score
	structuredScore := scoreHistogram(structured, size, size, 0.1).Score
	if structuredScore <= sparseScore || structuredScore <= fullScore {
		t.Fatalf("structured score %.3f should exceed sparse %.3f and full %.3f", structuredScore, sparseScore, fullScore)
	}
}

func TestLyapunovKnownCliffordMap(t *testing.T) {
	chaotic := coefficients{A: -1.4, B: 1.6, C: 1.0, D: 0.7}
	exponent := estimateLyapunov(chaotic, 1000, 20_000)
	if math.IsNaN(exponent) || math.IsInf(exponent, 0) {
		t.Fatalf("invalid exponent: %g", exponent)
	}
}

func TestHistogramQuantilesIgnoreOutliers(t *testing.T) {
	counts := []uint64{1, 0, 1000, 1000, 0, 1}
	low, high := histogramQuantiles(counts, 0, 6, 0.01, 0.99)
	if low != 2 || high != 4 {
		t.Fatalf("quantiles = (%g, %g), want (2, 4)", low, high)
	}
}

func TestSearchIsDeterministic(t *testing.T) {
	cfg := config{output: "unused", width: 64, height: 64, iterations: 1000, burnIn: 100, samples: 8, screenIters: 2000, screenSize: 32, coefficientRange: 3, seed: 42}
	a, _, err := search(rand.New(rand.NewSource(cfg.seed)), cfg)
	if err != nil {
		t.Fatal(err)
	}
	b, _, err := search(rand.New(rand.NewSource(cfg.seed)), cfg)
	if err != nil {
		t.Fatal(err)
	}
	if a.Coefficients != b.Coefficients || a.Metrics.Score != b.Metrics.Score {
		t.Fatalf("same seed produced different candidates: %+v and %+v", a, b)
	}
}

func TestValidateConfig(t *testing.T) {
	valid := config{output: "x", width: 2, height: 2, iterations: 100, samples: 1, screenIters: 100, screenSize: 8, coefficientRange: 1, count: 1, workers: 1, maxScore: 1, gamma: 1.8, exposure: 0.9, whitePercentile: 99.7, supersample: 3, glowStrength: 0.32, glowRadius: 14, glowThreshold: 0.65, softness: 2}
	if err := validateConfig(valid); err != nil {
		t.Fatalf("valid config rejected: %v", err)
	}
	valid.samples = 0
	if err := validateConfig(valid); err == nil {
		t.Fatal("zero samples accepted")
	}
}

func TestCollectCandidatesReturnsRequestedCount(t *testing.T) {
	cfg := config{count: 3, burnIn: 50, screenIters: 1000, screenSize: 32, coefficientRange: 3, maxScore: 1}
	got, _, attempts, err := collectCandidates(rand.New(rand.NewSource(42)), cfg)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != cfg.count {
		t.Fatalf("got %d candidates, want %d", len(got), cfg.count)
	}
	if attempts < cfg.count {
		t.Fatalf("attempts = %d, want at least %d", attempts, cfg.count)
	}
	for i, c := range got {
		if c.Index != i+1 {
			t.Fatalf("candidate %d has index %d", i, c.Index)
		}
	}
}

func TestDensitySplat(t *testing.T) {
	density := make([]uint64, 4)
	addDensity(density, 2, 2, 0, 0, 10)
	addDensity(density, 2, 2, 1, 1, 20)
	addDensity(density, 2, 2, -1, 0, 30)
	if density[0] != 10 || density[3] != 20 || density[1] != 0 || density[2] != 0 {
		t.Fatalf("unexpected density: %v", density)
	}
}

func TestDensityPercentileIgnoresHotPixel(t *testing.T) {
	density := []float64{0, 10, 10, 10, 10, 10000}
	white := densityPercentile(density, 10000, 0.8)
	if white < 9 || white > 11 {
		t.Fatalf("white point = %g, want approximately 10", white)
	}
}

func TestDownsampleDensity(t *testing.T) {
	source := []uint64{4, 4, 8, 8, 4, 4, 8, 8, 12, 12, 16, 16, 12, 12, 16, 16}
	got, width, height := downsampleDensity(source, 4, 4, 2)
	want := []float64{4, 8, 12, 16}
	if width != 2 || height != 2 {
		t.Fatalf("size = %dx%d", width, height)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("density[%d] = %g, want %g", i, got[i], want[i])
		}
	}
}

func TestGlowSpreadsHighlight(t *testing.T) {
	source := make([]float32, 25)
	source[12] = 1
	got := gaussianApproximation(source, 5, 5, 1)
	if got[12] <= 0 || got[11] <= 0 || got[7] <= 0 {
		t.Fatalf("glow did not spread: %v", got)
	}
}

func TestSoftnessPreservesMass(t *testing.T) {
	source := make([]float64, 25)
	source[12] = 16
	got := softenDensity(source, 5, 5, 2)
	var sum float64
	for _, n := range got {
		sum += n
	}
	if math.Abs(sum-16) > 1e-12 {
		t.Fatalf("softening changed mass to %g", sum)
	}
	if got[12] >= 16 || got[11] <= 0 {
		t.Fatalf("softening did not spread center: %v", got)
	}
}

func TestACESToneMapIsMonotonic(t *testing.T) {
	previous := 0.0
	for _, input := range []float64{0, 0.01, 0.1, 0.5, 1, 2, 10} {
		got := acesToneMap(input)
		if got < previous || got < 0 || got > 1 {
			t.Fatalf("tone map(%g) = %g after %g", input, got, previous)
		}
		previous = got
	}
}
