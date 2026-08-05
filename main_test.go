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
	m := scoreHistogram(uniform, 2, 2)
	if m.Occupancy != 1 || math.Abs(m.Entropy-1) > 1e-12 || math.Abs(m.Symmetry-1) > 1e-12 {
		t.Fatalf("unexpected uniform metrics: %+v", m)
	}

	concentrated := []uint64{100, 0, 0, 0}
	c := scoreHistogram(concentrated, 2, 2)
	if c.Occupancy != 0.25 || c.Entropy != 0 || c.Symmetry != 0 {
		t.Fatalf("unexpected concentrated metrics: %+v", c)
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
	valid := config{output: "x", width: 2, height: 2, iterations: 100, samples: 1, screenIters: 100, screenSize: 8, coefficientRange: 1}
	if err := validateConfig(valid); err != nil {
		t.Fatalf("valid config rejected: %v", err)
	}
	valid.samples = 0
	if err := validateConfig(valid); err == nil {
		t.Fatal("zero samples accepted")
	}
}
