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
	"runtime"
	"sort"
	"sync"
	"sync/atomic"
	"time"
)

const programVersion = "0.3.0"

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
	Coefficients  coefficients
	Bounds        bounds
	Metrics       metrics
	ScreenMetrics metrics
	Index         int
	TempPNG       string
}

type rejectionCounts struct {
	Diverged         int `json:"diverged"`
	ShortLoop        int `json:"short_loop"`
	Sparse           int `json:"sparse"`
	OutsideScoreBand int `json:"outside_score_band"`
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
	Rank             int             `json:"rank,omitempty"`
	CandidateIndex   int             `json:"candidate_index,omitempty"`
	ScreenMetrics    *metrics        `json:"screen_metrics,omitempty"`
	Gamma            float64         `json:"gamma"`
	BitDepth         int             `json:"bit_depth"`
	DensityBits      int             `json:"density_bits"`
}

type batchEntry struct {
	Rank           int          `json:"rank"`
	CandidateIndex int          `json:"candidate_index"`
	Score          float64      `json:"score"`
	Coefficients   coefficients `json:"coefficients"`
	PNG            string       `json:"png"`
	Metadata       string       `json:"metadata"`
}

type batchMetadata struct {
	Version          string          `json:"version"`
	GeneratedAt      string          `json:"generated_at"`
	Seed             int64           `json:"seed"`
	Count            int             `json:"count"`
	Attempts         int             `json:"attempts"`
	Iterations       int             `json:"iterations"`
	BurnIn           int             `json:"burn_in"`
	Width            int             `json:"width"`
	Height           int             `json:"height"`
	CoefficientRange float64         `json:"coefficient_range"`
	Rejections       rejectionCounts `json:"rejections"`
	Entries          []batchEntry    `json:"entries"`
	MinScore         float64         `json:"min_score"`
	MaxScore         float64         `json:"max_score"`
	ScreenIterations int             `json:"screen_iterations"`
	ScreenSize       int             `json:"screen_size"`
	Gamma            float64         `json:"gamma"`
	BitDepth         int             `json:"bit_depth"`
	DensityBits      int             `json:"density_bits"`
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
	count            int
	workers          int
	minScore         float64
	maxScore         float64
	gamma            float64
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
	flag.IntVar(&cfg.count, "count", 1, "number of accepted attractors to render; values above 1 use batch mode")
	flag.IntVar(&cfg.workers, "workers", max(1, runtime.NumCPU()/2), "parallel render workers in batch mode")
	flag.Float64Var(&cfg.minScore, "min-score", 0, "minimum stable screening score accepted in batch mode")
	flag.Float64Var(&cfg.maxScore, "max-score", 1, "maximum stable screening score accepted in batch mode")
	flag.Float64Var(&cfg.gamma, "gamma", 2.2, "density-to-image gamma; larger values reveal fainter structure")
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
	if cfg.count > 1 {
		return runBatch(rng, cfg)
	}

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
	density, _, err := buildDensity32(best.Coefficients, cfg.burnIn, cfg.iterations, cfg.width, cfg.height)
	if err != nil {
		return fmt.Errorf("final density: %w", err)
	}
	if err := writePNG16Gamma(pngPath, density, cfg.width, cfg.height, cfg.gamma); err != nil {
		return err
	}
	meta := metadata{
		Version: programVersion, GeneratedAt: time.Now().UTC().Format(time.RFC3339),
		Seed: cfg.seed, Samples: cfg.samples, Iterations: cfg.iterations, BurnIn: cfg.burnIn,
		Width: cfg.width, Height: cfg.height, CoefficientRange: cfg.coefficientRange,
		Coefficients: best.Coefficients, Bounds: best.Bounds, Metrics: best.Metrics,
		Rejections: rejects, PNG: filepath.Base(pngPath),
		Gamma: cfg.gamma, BitDepth: 16, DensityBits: 32,
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
	case cfg.count < 1:
		return errors.New("count must be at least 1")
	case cfg.workers < 1:
		return errors.New("workers must be at least 1")
	case cfg.minScore < 0 || cfg.maxScore > 1 || cfg.minScore > cfg.maxScore:
		return errors.New("score range must satisfy 0 <= min-score <= max-score <= 1")
	case cfg.gamma <= 0:
		return errors.New("gamma must be positive")
	}
	return nil
}

func runBatch(rng *rand.Rand, cfg config) error {
	if err := os.MkdirAll(cfg.output, 0o755); err != nil {
		return fmt.Errorf("create batch directory: %w", err)
	}
	stageDir, err := os.MkdirTemp(cfg.output, ".rendering-")
	if err != nil {
		return fmt.Errorf("create staging directory: %w", err)
	}

	candidates, rejects, attempts, err := collectCandidates(rng, cfg)
	if err != nil {
		return err
	}
	fmt.Printf("accepted %d candidates after %d attempts; rendering with %d workers\n", len(candidates), attempts, min(cfg.workers, cfg.count))

	jobs := make(chan int)
	errCh := make(chan error, 1)
	var wg sync.WaitGroup
	var completed atomic.Int64
	workerCount := min(cfg.workers, cfg.count)
	for worker := 0; worker < workerCount; worker++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range jobs {
				c := &candidates[i]
				density, renderBounds, renderErr := buildDensity32(c.Coefficients, cfg.burnIn, cfg.iterations, cfg.width, cfg.height)
				if renderErr == nil {
					c.Bounds = renderBounds
					c.TempPNG = filepath.Join(stageDir, fmt.Sprintf("candidate-%06d.png", c.Index))
					renderErr = writePNG16Gamma(c.TempPNG, density, cfg.width, cfg.height, cfg.gamma)
				}
				if renderErr != nil {
					select {
					case errCh <- fmt.Errorf("candidate %d: %w", c.Index, renderErr):
					default:
					}
					continue
				}
				done := completed.Add(1)
				if done%25 == 0 || int(done) == cfg.count {
					fmt.Printf("rendered %d/%d\n", done, cfg.count)
				}
			}
		}()
	}
	for i := range candidates {
		jobs <- i
	}
	close(jobs)
	wg.Wait()
	select {
	case renderErr := <-errCh:
		return renderErr
	default:
	}

	sort.SliceStable(candidates, func(i, j int) bool {
		return candidates[i].Metrics.Score > candidates[j].Metrics.Score
	})
	entries := make([]batchEntry, 0, len(candidates))
	generatedAt := time.Now().UTC().Format(time.RFC3339)
	for i := range candidates {
		c := &candidates[i]
		rank := i + 1
		stem := fmt.Sprintf("%04d_score-%.6f_id-%06d", rank, c.Metrics.Score, c.Index)
		pngPath, jsonPath := filepath.Join(cfg.output, stem+".png"), filepath.Join(cfg.output, stem+".json")
		if fileExists(pngPath) || fileExists(jsonPath) {
			return fmt.Errorf("refusing to overwrite existing result %s", stem)
		}
		if err := os.Rename(c.TempPNG, pngPath); err != nil {
			return fmt.Errorf("publish candidate %d: %w", c.Index, err)
		}
		meta := metadata{
			Version: programVersion, GeneratedAt: generatedAt, Seed: cfg.seed, Samples: 1,
			Iterations: cfg.iterations, BurnIn: cfg.burnIn, Width: cfg.width, Height: cfg.height,
			CoefficientRange: cfg.coefficientRange, Coefficients: c.Coefficients, Bounds: c.Bounds,
			Metrics: c.Metrics, PNG: filepath.Base(pngPath), Rank: rank, CandidateIndex: c.Index,
			ScreenMetrics: &c.ScreenMetrics, Gamma: cfg.gamma, BitDepth: 16, DensityBits: 32,
		}
		if err := writeJSON(jsonPath, meta); err != nil {
			return err
		}
		entries = append(entries, batchEntry{
			Rank: rank, CandidateIndex: c.Index, Score: c.Metrics.Score, Coefficients: c.Coefficients,
			PNG: filepath.Base(pngPath), Metadata: filepath.Base(jsonPath),
		})
	}
	if err := os.Remove(stageDir); err != nil {
		return fmt.Errorf("remove empty staging directory: %w", err)
	}
	summary := batchMetadata{
		Version: programVersion, GeneratedAt: generatedAt, Seed: cfg.seed, Count: cfg.count,
		Attempts: attempts, Iterations: cfg.iterations, BurnIn: cfg.burnIn, Width: cfg.width,
		Height: cfg.height, CoefficientRange: cfg.coefficientRange, Rejections: rejects, Entries: entries,
		MinScore: cfg.minScore, MaxScore: cfg.maxScore, ScreenIterations: cfg.screenIters,
		ScreenSize: cfg.screenSize, Gamma: cfg.gamma, BitDepth: 16, DensityBits: 32,
	}
	if err := writeJSON(filepath.Join(cfg.output, "batch.json"), summary); err != nil {
		return err
	}
	fmt.Printf("wrote %d ranked attractors to %s (best score %.6f)\n", cfg.count, cfg.output, candidates[0].Metrics.Score)
	return nil
}

func collectCandidates(rng *rand.Rand, cfg config) ([]candidate, rejectionCounts, int, error) {
	candidates := make([]candidate, 0, cfg.count)
	var rejects rejectionCounts
	maxAttempts := cfg.count * 100
	for attempts := 1; attempts <= maxAttempts; attempts++ {
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
		counts, b, histogramErr := buildHistogram(p, cfg.burnIn, cfg.screenIters, cfg.screenSize, cfg.screenSize)
		if histogramErr != nil {
			rejects.Diverged++
			continue
		}
		m := scoreHistogram(counts, cfg.screenSize, cfg.screenSize)
		if m.Occupancy < 0.005 || m.OccupiedBins < 64 {
			rejects.Sparse++
			continue
		}
		if m.Score < cfg.minScore || m.Score > cfg.maxScore {
			rejects.OutsideScoreBand++
			continue
		}
		candidates = append(candidates, candidate{
			Coefficients: p, Bounds: b, Metrics: m, ScreenMetrics: m, Index: len(candidates) + 1,
		})
		if len(candidates) == cfg.count {
			return candidates, rejects, attempts, nil
		}
	}
	return nil, rejects, maxAttempts, fmt.Errorf("found only %d usable attractors in %d attempts", len(candidates), maxAttempts)
}

func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
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

// buildDensity32 accumulates a high-quality density field. Each orbit point is
// distributed across four neighboring pixels with six bits of subpixel weight.
// With the validated interestingness band this leaves ample uint32 headroom
// while reducing the hard, grainy edges caused by nearest-pixel binning.
func buildDensity32(p coefficients, burnIn, iterations, width, height int) ([]uint32, bounds, error) {
	b, err := orbitBounds(p, burnIn, iterations)
	if err != nil {
		return nil, bounds{}, err
	}
	density := make([]uint32, width*height)
	x, y := 0.1, 0.1
	for i := 0; i < burnIn+iterations; i++ {
		x, y = clifford(p, x, y)
		if i < burnIn {
			continue
		}
		fx := (x - b.MinX) / (b.MaxX - b.MinX) * float64(width-1)
		fy := (b.MaxY - y) / (b.MaxY - b.MinY) * float64(height-1)
		x0, y0 := int(math.Floor(fx)), int(math.Floor(fy))
		if x0 < 0 || x0 >= width || y0 < 0 || y0 >= height {
			continue
		}
		dx, dy := fx-float64(x0), fy-float64(y0)
		addDensity(density, width, height, x0, y0, uint32(math.Round(64*(1-dx)*(1-dy))))
		addDensity(density, width, height, x0+1, y0, uint32(math.Round(64*dx*(1-dy))))
		addDensity(density, width, height, x0, y0+1, uint32(math.Round(64*(1-dx)*dy)))
		addDensity(density, width, height, x0+1, y0+1, uint32(math.Round(64*dx*dy)))
	}
	return density, b, nil
}

func addDensity(density []uint32, width, height, x, y int, weight uint32) {
	if weight == 0 || x < 0 || x >= width || y < 0 || y >= height {
		return
	}
	i := y*width + x
	if ^uint32(0)-density[i] < weight {
		density[i] = ^uint32(0)
	} else {
		density[i] += weight
	}
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

func writePNG16Gamma(path string, density []uint32, width, height int, gamma float64) error {
	var maxDensity uint32
	for _, n := range density {
		if n > maxDensity {
			maxDensity = n
		}
	}
	if maxDensity == 0 {
		return errors.New("cannot render empty density")
	}
	const lutSize = 65536
	lut := make([]uint16, lutSize)
	invGamma := 1 / gamma
	for i := 1; i < lutSize; i++ {
		linear := float64(i) / float64(lutSize-1)
		lut[i] = uint16(math.Round(65535 * math.Pow(linear, invGamma)))
	}
	img := image.NewRGBA64(image.Rect(0, 0, width, height))
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			n := density[y*width+x]
			index := uint32(uint64(n) * (lutSize - 1) / uint64(maxDensity))
			v := float64(lut[index]) / 65535
			img.SetRGBA64(x, y, palette16(v))
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

func palette16(v float64) color.RGBA64 {
	if v <= 0 {
		return color.RGBA64{R: 771, G: 1285, B: 3084, A: 65535}
	}
	r := uint16(65535 * math.Pow(v, 1.6))
	g := uint16(65535 * math.Pow(v, 0.85))
	b := uint16(65535 * math.Min(1, 1.8*math.Pow(v, 0.45)))
	return color.RGBA64{R: r, G: g, B: b, A: 65535}
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
