package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"time"
)

const programVersion = "0.1.0"

type coefficients struct {
	A float64 `json:"a"`
	B float64 `json:"b"`
	C float64 `json:"c"`
	D float64 `json:"d"`
}

type bounds struct {
	MinX float64 `json:"min_x"`
	MaxX float64 `json:"max_x"`
	MinY float64 `json:"min_y"`
	MaxY float64 `json:"max_y"`
}

type metrics struct {
	OccupiedBins       int     `json:"occupied_bins"`
	HistogramBins      int     `json:"histogram_bins"`
	Occupancy          float64 `json:"occupancy"`
	Entropy            float64 `json:"entropy"`
	VerticalSymmetry   float64 `json:"vertical_symmetry"`
	HorizontalSymmetry float64 `json:"horizontal_symmetry"`
	Symmetry           float64 `json:"symmetry"`
	Score              float64 `json:"score"`
}

type candidate struct {
	Coefficients coefficients
	Bounds       bounds
	Metrics      metrics
}

type rejectionCounts struct {
	Diverged  int `json:"diverged"`
	ShortLoop int `json:"short_loop"`
	Sparse    int `json:"sparse"`
}

type metadata struct {
	Version          string          `json:"version"`
	GeneratedAt      string          `json:"generated_at"`
	Seed             int64           `json:"seed"`
	Samples          int             `json:"samples"`
	Iterations       int             `json:"iterations"`
	BurnIn           int             `json:"burn_in"`
	Width            int             `json:"width"`
	Height           int             `json:"height"`
	CoefficientRange float64         `json:"coefficient_range"`
	Coefficients     coefficients    `json:"coefficients"`
	Bounds           bounds          `json:"bounds"`
	Metrics          metrics         `json:"metrics"`
	Rejections       rejectionCounts `json:"rejections"`
	PNG              string          `json:"png"`
}

type config struct {
	output           string
	width            int
	height           int
	iterations       int
	burnIn           int
	samples          int
	screenIters      int
	screenSize       int
	coefficientRange float64
	seed             int64
}

func main() {
	cfg := parseFlags()
	if err := run(cfg); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func parseFlags() config {
	var cfg config
	flag.StringVar(&cfg.output, "out", "attractor", "output path prefix")
	flag.IntVar(&cfg.width, "width", 1600, "PNG width in pixels")
	flag.IntVar(&cfg.height, "height", 1600, "PNG height in pixels")
	flag.IntVar(&cfg.iterations, "iterations", 2_000_000, "points used for the final render")
	flag.IntVar(&cfg.burnIn, "burn-in", 1_000, "initial iterations to discard")
	flag.IntVar(&cfg.samples, "samples", 64, "random coefficient sets to score")
	flag.IntVar(&cfg.screenIters, "screen-iterations", 100_000, "points used to screen each candidate")
	flag.IntVar(&cfg.screenSize, "screen-size", 256, "screening histogram width and height")
	flag.Float64Var(&cfg.coefficientRange, "range", 3.0, "sample coefficients uniformly from [-range, range]")
	flag.Int64Var(&cfg.seed, "seed", 0, "random seed; zero chooses the current time")
	flag.Parse()
	return cfg
}

func run(cfg config) error {
	if err := validateConfig(cfg); err != nil {
		return err
	}
	if cfg.seed == 0 {
		cfg.seed = time.Now().UnixNano()
	}
	rng := rand.New(rand.NewSource(cfg.seed))

	best, rejects, err := search(rng, cfg)
	if err != nil {
		return err
	}
	counts, renderBounds, err := buildHistogram(best.Coefficients, cfg.burnIn, cfg.iterations, cfg.width, cfg.height)
	if err != nil {
		return fmt.Errorf("final render: %w", err)
	}
	best.Bounds = renderBounds
	best.Metrics = scoreHistogram(counts, cfg.width, cfg.height)

	pngPath := cfg.output + ".png"
	jsonPath := cfg.output + ".json"
	if err := ensureParent(pngPath); err != nil {
		return err
	}
	if err := writePNG(pngPath, counts, cfg.width, cfg.height); err != nil {
		return err
	}
	meta := metadata{
		Version: programVersion, GeneratedAt: time.Now().UTC().Format(time.RFC3339),
		Seed: cfg.seed, Samples: cfg.samples, Iterations: cfg.iterations, BurnIn: cfg.burnIn,
		Width: cfg.width, Height: cfg.height, CoefficientRange: cfg.coefficientRange,
		Coefficients: best.Coefficients, Bounds: best.Bounds, Metrics: best.Metrics,
		Rejections: rejects, PNG: filepath.Base(pngPath),
	}
	if err := writeJSON(jsonPath, meta); err != nil {
		return err
	}
	fmt.Printf("wrote %s and %s\n", pngPath, jsonPath)
	fmt.Printf("coefficients: a=%.8f b=%.8f c=%.8f d=%.8f\n", best.Coefficients.A, best.Coefficients.B, best.Coefficients.C, best.Coefficients.D)
	fmt.Printf("score=%.4f occupancy=%.4f entropy=%.4f symmetry=%.4f seed=%d\n", best.Metrics.Score, best.Metrics.Occupancy, best.Metrics.Entropy, best.Metrics.Symmetry, cfg.seed)
	return nil
}

func validateConfig(cfg config) error {
	switch {
	case cfg.output == "":
		return errors.New("out must not be empty")
	case cfg.width < 2 || cfg.height < 2:
		return errors.New("width and height must be at least 2")
	case cfg.iterations < 100:
		return errors.New("iterations must be at least 100")
	case cfg.burnIn < 0:
		return errors.New("burn-in must not be negative")
	case cfg.samples < 1:
		return errors.New("samples must be at least 1")
	case cfg.screenIters < 100:
		return errors.New("screen-iterations must be at least 100")
	case cfg.screenSize < 8:
		return errors.New("screen-size must be at least 8")
	case cfg.coefficientRange <= 0:
		return errors.New("range must be positive")
	}
	return nil
}

func search(rng *rand.Rand, cfg config) (candidate, rejectionCounts, error) {
	var best candidate
	var rejects rejectionCounts
	found := false
	for i := 0; i < cfg.samples; i++ {
		p := coefficients{
			A: uniform(rng, cfg.coefficientRange), B: uniform(rng, cfg.coefficientRange),
			C: uniform(rng, cfg.coefficientRange), D: uniform(rng, cfg.coefficientRange),
		}
		if reason := inspectOrbit(p, cfg.burnIn, cfg.screenIters); reason != "" {
			if reason == "diverged" {
				rejects.Diverged++
			} else {
				rejects.ShortLoop++
			}
			continue
		}
		counts, b, err := buildHistogram(p, cfg.burnIn, cfg.screenIters, cfg.screenSize, cfg.screenSize)
		if err != nil {
			rejects.Diverged++
			continue
		}
		m := scoreHistogram(counts, cfg.screenSize, cfg.screenSize)
		// Very sparse plots are usually fixed points, thin cycles, or uninteresting dust.
		if m.Occupancy < 0.005 || m.OccupiedBins < 64 {
			rejects.Sparse++
			continue
		}
		if !found || m.Score > best.Metrics.Score {
			best = candidate{Coefficients: p, Bounds: b, Metrics: m}
			found = true
		}
	}
	if !found {
		return candidate{}, rejects, errors.New("no usable attractor found; increase -samples or try another -seed")
	}
	return best, rejects, nil
}

func uniform(rng *rand.Rand, span float64) float64 { return (rng.Float64()*2 - 1) * span }

func clifford(p coefficients, x, y float64) (float64, float64) {
	return math.Sin(p.A*y) + p.C*math.Cos(p.A*x),
		math.Sin(p.B*x) + p.D*math.Cos(p.B*y)
}

func inspectOrbit(p coefficients, burnIn, iterations int) string {
	x, y := 0.1, 0.1
	seen := make(map[[2]int64]int, 4096)
	checkUntil := burnIn + min(iterations, 20_000)
	for i := 0; i < checkUntil; i++ {
		x, y = clifford(p, x, y)
		if math.IsNaN(x) || math.IsNaN(y) || math.IsInf(x, 0) || math.IsInf(y, 0) || math.Abs(x) > 1e6 || math.Abs(y) > 1e6 {
			return "diverged"
		}
		if i >= burnIn {
			key := [2]int64{int64(math.Round(x * 1e9)), int64(math.Round(y * 1e9))}
			if previous, ok := seen[key]; ok && i-previous <= 64 {
				return "short_loop"
			}
			seen[key] = i
		}
	}
	return ""
}

func orbitBounds(p coefficients, burnIn, iterations int) (bounds, error) {
	x, y := 0.1, 0.1
	b := bounds{MinX: math.Inf(1), MaxX: math.Inf(-1), MinY: math.Inf(1), MaxY: math.Inf(-1)}
	for i := 0; i < burnIn+iterations; i++ {
		x, y = clifford(p, x, y)
		if math.IsNaN(x) || math.IsNaN(y) || math.IsInf(x, 0) || math.IsInf(y, 0) {
			return bounds{}, errors.New("orbit diverged")
		}
		if i >= burnIn {
			b.MinX = math.Min(b.MinX, x)
			b.MaxX = math.Max(b.MaxX, x)
			b.MinY = math.Min(b.MinY, y)
			b.MaxY = math.Max(b.MaxY, y)
		}
	}
	if b.MaxX-b.MinX < 1e-12 || b.MaxY-b.MinY < 1e-12 {
		return bounds{}, errors.New("orbit collapsed to a point or line")
	}
	// A small border keeps the plotted orbit off the image edges.
	padX, padY := (b.MaxX-b.MinX)*0.015, (b.MaxY-b.MinY)*0.015
	b.MinX -= padX
	b.MaxX += padX
	b.MinY -= padY
	b.MaxY += padY
	return b, nil
}

func buildHistogram(p coefficients, burnIn, iterations, width, height int) ([]uint64, bounds, error) {
	b, err := orbitBounds(p, burnIn, iterations)
	if err != nil {
		return nil, bounds{}, err
	}
	counts := make([]uint64, width*height)
	x, y := 0.1, 0.1
	for i := 0; i < burnIn+iterations; i++ {
		x, y = clifford(p, x, y)
		if i < burnIn {
			continue
		}
		px := int((x-b.MinX)/(b.MaxX-b.MinX)*float64(width-1) + 0.5)
		py := int((b.MaxY-y)/(b.MaxY-b.MinY)*float64(height-1) + 0.5)
		if px >= 0 && px < width && py >= 0 && py < height {
			counts[py*width+px]++
		}
	}
	return counts, b, nil
}

func scoreHistogram(counts []uint64, width, height int) metrics {
	var total, occupied uint64
	for _, n := range counts {
		total += n
		if n > 0 {
			occupied++
		}
	}
	m := metrics{OccupiedBins: int(occupied), HistogramBins: len(counts)}
	if total == 0 || occupied == 0 {
		return m
	}
	m.Occupancy = float64(occupied) / float64(len(counts))
	var entropy float64
	for _, n := range counts {
		if n == 0 {
			continue
		}
		p := float64(n) / float64(total)
		entropy -= p * math.Log(p)
	}
	if occupied > 1 {
		m.Entropy = entropy / math.Log(float64(occupied))
	}
	m.VerticalSymmetry = reflectionSimilarity(counts, width, height, true)
	m.HorizontalSymmetry = reflectionSimilarity(counts, width, height, false)
	m.Symmetry = math.Max(m.VerticalSymmetry, m.HorizontalSymmetry)
	// Occupancy is square-rooted so delicate structures can compete with solid blobs.
	m.Score = 0.35*math.Sqrt(m.Occupancy) + 0.50*m.Entropy + 0.15*m.Symmetry
	return m
}

func reflectionSimilarity(counts []uint64, width, height int, vertical bool) float64 {
	var difference, mass uint64
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			mx, my := x, y
			if vertical {
				mx = width - 1 - x
			} else {
				my = height - 1 - y
			}
			a, b := counts[y*width+x], counts[my*width+mx]
			if a > b {
				difference += a - b
			} else {
				difference += b - a
			}
			mass += a + b
		}
	}
	if mass == 0 {
		return 0
	}
	return 1 - float64(difference)/float64(mass)
}

func writePNG(path string, counts []uint64, width, height int) error {
	var maxCount uint64
	for _, n := range counts {
		if n > maxCount {
			maxCount = n
		}
	}
	if maxCount == 0 {
		return errors.New("cannot render empty histogram")
	}
	img := image.NewRGBA(image.Rect(0, 0, width, height))
	denominator := math.Log1p(float64(maxCount))
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			v := math.Log1p(float64(counts[y*width+x])) / denominator
			img.SetRGBA(x, y, palette(v))
		}
	}
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create PNG: %w", err)
	}
	if err := png.Encode(f, img); err != nil {
		f.Close()
		return fmt.Errorf("encode PNG: %w", err)
	}
	if err := f.Close(); err != nil {
		return fmt.Errorf("close PNG: %w", err)
	}
	return nil
}

func palette(v float64) color.RGBA {
	if v <= 0 {
		return color.RGBA{R: 3, G: 5, B: 12, A: 255}
	}
	// A compact black-blue-cyan-gold density ramp.
	r := uint8(255 * math.Pow(v, 1.6))
	g := uint8(255 * math.Pow(v, 0.85))
	b := uint8(255 * math.Min(1, 1.8*math.Pow(v, 0.45)))
	return color.RGBA{R: r, G: g, B: b, A: 255}
}

func ensureParent(path string) error {
	dir := filepath.Dir(path)
	if dir == "." {
		return nil
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("create output directory: %w", err)
	}
	return nil
}

func writeJSON(path string, value any) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create metadata: %w", err)
	}
	encoder := json.NewEncoder(f)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(value); err != nil {
		f.Close()
		return fmt.Errorf("encode metadata: %w", err)
	}
	if err := f.Close(); err != nil {
		return fmt.Errorf("close metadata: %w", err)
	}
	return nil
}
