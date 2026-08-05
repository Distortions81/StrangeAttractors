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
	GlobalEntropy      float64 `json:"global_entropy"`
	BoxDimension       float64 `json:"box_dimension"`
	VerticalSymmetry   float64 `json:"vertical_symmetry"`
	HorizontalSymmetry float64 `json:"horizontal_symmetry"`
	Symmetry           float64 `json:"symmetry"`
	CoverageScore      float64 `json:"coverage_score"`
	EntropyScore       float64 `json:"entropy_score"`
	DimensionScore     float64 `json:"dimension_score"`
	LyapunovExponent   float64 `json:"lyapunov_exponent"`
	ChaosScore         float64 `json:"chaos_score"`
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
	ToneMap          string          `json:"tone_map"`
	Palette          string          `json:"palette"`
	Exposure         float64         `json:"exposure"`
	WhitePercentile  float64         `json:"white_percentile"`
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
	ToneMap          string          `json:"tone_map"`
	Palette          string          `json:"palette"`
	Exposure         float64         `json:"exposure"`
	WhitePercentile  float64         `json:"white_percentile"`
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
	exposure         float64
	whitePercentile  float64
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
	flag.Float64Var(&cfg.exposure, "exposure", 2.5, "HDR exposure before filmic tone mapping")
	flag.Float64Var(&cfg.whitePercentile, "white-percentile", 99.5, "nonzero density percentile mapped near white")
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
	best.Metrics = scoreHistogram(counts, cfg.width, cfg.height, best.Metrics.LyapunovExponent)

	pngPath := cfg.output + ".png"
	jsonPath := cfg.output + ".json"
	if err := ensureParent(pngPath); err != nil {
		return err
	}
	density, _, err := buildDensity32(best.Coefficients, cfg.burnIn, cfg.iterations, cfg.width, cfg.height)
	if err != nil {
		return fmt.Errorf("final density: %w", err)
	}
	if err := writePNG16HDR(pngPath, density, cfg.width, cfg.height, cfg.gamma, cfg.exposure, cfg.whitePercentile); err != nil {
		return err
	}
	meta := metadata{
		Version: programVersion, GeneratedAt: time.Now().UTC().Format(time.RFC3339),
		Seed: cfg.seed, Samples: cfg.samples, Iterations: cfg.iterations, BurnIn: cfg.burnIn,
		Width: cfg.width, Height: cfg.height, CoefficientRange: cfg.coefficientRange,
		Coefficients: best.Coefficients, Bounds: best.Bounds, Metrics: best.Metrics,
		Rejections: rejects, PNG: filepath.Base(pngPath),
		Gamma: cfg.gamma, BitDepth: 16, DensityBits: 32, ToneMap: "aces-fitted",
		Palette: "aurora", Exposure: cfg.exposure, WhitePercentile: cfg.whitePercentile,
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
	case cfg.exposure <= 0:
		return errors.New("exposure must be positive")
	case cfg.whitePercentile <= 0 || cfg.whitePercentile > 100:
		return errors.New("white-percentile must be in (0, 100]")
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
					renderErr = writePNG16HDR(c.TempPNG, density, cfg.width, cfg.height, cfg.gamma, cfg.exposure, cfg.whitePercentile)
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
			ToneMap: "aces-fitted", Palette: "aurora", Exposure: cfg.exposure, WhitePercentile: cfg.whitePercentile,
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
		ToneMap: "aces-fitted", Palette: "aurora", Exposure: cfg.exposure, WhitePercentile: cfg.whitePercentile,
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
		lyapunov := estimateLyapunov(p, cfg.burnIn, min(cfg.screenIters, 50_000))
		m := scoreHistogram(counts, cfg.screenSize, cfg.screenSize, lyapunov)
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
		lyapunov := estimateLyapunov(p, cfg.burnIn, min(cfg.screenIters, 50_000))
		m := scoreHistogram(counts, cfg.screenSize, cfg.screenSize, lyapunov)
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
	const quantileBins = 4096
	x, y := 0.1, 0.1
	limitX, limitY := 1+math.Abs(p.C), 1+math.Abs(p.D)
	var xCounts, yCounts [quantileBins]uint64
	for i := 0; i < burnIn+iterations; i++ {
		x, y = clifford(p, x, y)
		if math.IsNaN(x) || math.IsNaN(y) || math.IsInf(x, 0) || math.IsInf(y, 0) {
			return bounds{}, errors.New("orbit diverged")
		}
		if i >= burnIn {
			xCounts[boundedBin(x, -limitX, limitX, quantileBins)]++
			yCounts[boundedBin(y, -limitY, limitY, quantileBins)]++
		}
	}
	minX, maxX := histogramQuantiles(xCounts[:], -limitX, limitX, 0.001, 0.999)
	minY, maxY := histogramQuantiles(yCounts[:], -limitY, limitY, 0.001, 0.999)
	b := bounds{MinX: minX, MaxX: maxX, MinY: minY, MaxY: maxY}
	if b.MaxX-b.MinX < 1e-12 || b.MaxY-b.MinY < 1e-12 {
		return bounds{}, errors.New("orbit collapsed to a point or line")
	}
	// Quantile framing ignores rare outliers that would otherwise shrink the
	// visually dominant structure. A small border keeps it off the image edges.
	padX, padY := (b.MaxX-b.MinX)*0.015, (b.MaxY-b.MinY)*0.015
	b.MinX -= padX
	b.MaxX += padX
	b.MinY -= padY
	b.MaxY += padY
	return b, nil
}

func boundedBin(value, low, high float64, bins int) int {
	i := int((value - low) / (high - low) * float64(bins))
	if i < 0 {
		return 0
	}
	if i >= bins {
		return bins - 1
	}
	return i
}

func histogramQuantiles(counts []uint64, low, high, lowerQ, upperQ float64) (float64, float64) {
	var total uint64
	for _, n := range counts {
		total += n
	}
	if total == 0 {
		return low, high
	}
	lowerTarget := uint64(math.Ceil(float64(total) * lowerQ))
	upperTarget := uint64(math.Ceil(float64(total) * upperQ))
	lowerIndex, upperIndex := 0, len(counts)-1
	var cumulative uint64
	foundLower := false
	for i, n := range counts {
		cumulative += n
		if !foundLower && cumulative >= lowerTarget {
			lowerIndex, foundLower = i, true
		}
		if cumulative >= upperTarget {
			upperIndex = i
			break
		}
	}
	step := (high - low) / float64(len(counts))
	return low + float64(lowerIndex)*step, low + float64(upperIndex+1)*step
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

func scoreHistogram(counts []uint64, width, height int, lyapunov float64) metrics {
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
	if len(counts) > 1 {
		m.GlobalEntropy = entropy / math.Log(float64(len(counts)))
	}
	m.BoxDimension = boxCountingDimension(counts, width, height)
	m.VerticalSymmetry = reflectionSimilarity(counts, width, height, true)
	m.HorizontalSymmetry = reflectionSimilarity(counts, width, height, false)
	m.Symmetry = math.Max(m.VerticalSymmetry, m.HorizontalSymmetry)

	// Visual interest is non-monotonic: both near-empty loops and screen-filling
	// noise are less useful than moderately occupied, filamentary structures.
	m.CoverageScore = bandPreference(m.Occupancy, 0.01, 0.04, 0.22, 0.48)
	m.EntropyScore = bandPreference(m.GlobalEntropy, 0.20, 0.45, 0.82, 0.96)
	m.DimensionScore = math.Exp(-0.5 * math.Pow((m.BoxDimension-1.72)/0.10, 2))
	m.LyapunovExponent = lyapunov
	m.ChaosScore = smoothStep((lyapunov - 0.05) / (1.0 - 0.05))
	baseScore := 0.30*m.CoverageScore + 0.40*m.DimensionScore + 0.25*m.EntropyScore + 0.05*m.ChaosScore
	// Coverage is also a viability gate: an almost empty loop or a nearly solid
	// cloud cannot offset poor framing merely by scoring well on another axis.
	m.Score = baseScore * (0.10 + 0.90*math.Sqrt(m.CoverageScore))
	return m
}

// estimateLyapunov follows a tangent vector through the analytic Jacobian of
// the Clifford map. Positive values indicate sensitive dependence on initial
// conditions; larger positive values receive the maximum chaos preference.
func estimateLyapunov(p coefficients, burnIn, iterations int) float64 {
	x, y := 0.1, 0.1
	vx, vy := 1.0, 0.0
	var sum float64
	for i := 0; i < burnIn+iterations; i++ {
		j00 := -p.A * p.C * math.Sin(p.A*x)
		j01 := p.A * math.Cos(p.A*y)
		j10 := p.B * math.Cos(p.B*x)
		j11 := -p.B * p.D * math.Sin(p.B*y)
		nx, ny := j00*vx+j01*vy, j10*vx+j11*vy
		norm := math.Hypot(nx, ny)
		if norm < 1e-300 || math.IsNaN(norm) || math.IsInf(norm, 0) {
			return math.Inf(-1)
		}
		vx, vy = nx/norm, ny/norm
		x, y = clifford(p, x, y)
		if i >= burnIn {
			sum += math.Log(norm)
		}
	}
	return sum / float64(iterations)
}

func bandPreference(value, low, idealLow, idealHigh, high float64) float64 {
	switch {
	case value <= low || value >= high:
		return 0
	case value < idealLow:
		return smoothStep((value - low) / (idealLow - low))
	case value <= idealHigh:
		return 1
	default:
		return smoothStep((high - value) / (high - idealHigh))
	}
}

func smoothStep(x float64) float64 {
	x = math.Max(0, math.Min(1, x))
	return x * x * (3 - 2*x)
}

func boxCountingDimension(counts []uint64, width, height int) float64 {
	maxScale := min(width, height) / 4
	var xs, ys []float64
	for boxSize := 2; boxSize <= 32 && boxSize <= maxScale; boxSize *= 2 {
		boxesX := (width + boxSize - 1) / boxSize
		boxesY := (height + boxSize - 1) / boxSize
		occupied := make([]bool, boxesX*boxesY)
		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				if counts[y*width+x] > 0 {
					occupied[(y/boxSize)*boxesX+x/boxSize] = true
				}
			}
		}
		var n int
		for _, hit := range occupied {
			if hit {
				n++
			}
		}
		if n > 1 {
			xs = append(xs, math.Log(float64(min(width, height))/float64(boxSize)))
			ys = append(ys, math.Log(float64(n)))
		}
	}
	if len(xs) < 2 {
		return 0
	}
	var meanX, meanY float64
	for i := range xs {
		meanX += xs[i]
		meanY += ys[i]
	}
	meanX /= float64(len(xs))
	meanY /= float64(len(ys))
	var covariance, variance float64
	for i := range xs {
		covariance += (xs[i] - meanX) * (ys[i] - meanY)
		variance += (xs[i] - meanX) * (xs[i] - meanX)
	}
	if variance == 0 {
		return 0
	}
	return covariance / variance
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

func writePNG16HDR(path string, density []uint32, width, height int, gamma, exposure, whitePercentile float64) error {
	var maxDensity uint32
	for _, n := range density {
		if n > maxDensity {
			maxDensity = n
		}
	}
	if maxDensity == 0 {
		return errors.New("cannot render empty density")
	}
	whitePoint := densityPercentile(density, maxDensity, whitePercentile/100)
	const lutSize = 65536
	lut := make([]uint16, lutSize)
	invGamma := 1 / gamma
	for i := 1; i < lutSize; i++ {
		linear := float64(i) / float64(lutSize-1) * exposure
		mapped := acesToneMap(linear)
		lut[i] = uint16(math.Round(65535 * math.Pow(mapped, invGamma)))
	}
	img := image.NewRGBA64(image.Rect(0, 0, width, height))
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			n := density[y*width+x]
			index := uint32(min(uint64(lutSize-1), uint64(n)*(lutSize-1)/uint64(whitePoint)))
			v := float64(lut[index]) / 65535
			img.SetRGBA64(x, y, auroraPalette(v))
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

func densityPercentile(density []uint32, maxDensity uint32, percentile float64) uint32 {
	const bins = 65536
	histogram := make([]uint64, bins)
	var nonzero uint64
	for _, n := range density {
		if n == 0 {
			continue
		}
		index := uint32(uint64(n) * (bins - 1) / uint64(maxDensity))
		histogram[index]++
		nonzero++
	}
	target := uint64(math.Ceil(float64(nonzero) * percentile))
	var cumulative uint64
	for i, n := range histogram {
		cumulative += n
		if cumulative >= target {
			return max(1, uint32(uint64(i)*uint64(maxDensity)/(bins-1)))
		}
	}
	return maxDensity
}

func acesToneMap(x float64) float64 {
	value := x * (2.51*x + 0.03) / (x*(2.43*x+0.59) + 0.14)
	return math.Max(0, math.Min(1, value))
}

type colorStop struct{ Position, R, G, B float64 }

func auroraPalette(v float64) color.RGBA64 {
	stops := [...]colorStop{
		{0.00, 0.012, 0.020, 0.047},
		{0.08, 0.025, 0.035, 0.160},
		{0.24, 0.180, 0.075, 0.520},
		{0.42, 0.035, 0.360, 1.000},
		{0.62, 0.020, 0.920, 0.940},
		{0.78, 0.480, 1.000, 0.760},
		{0.91, 1.000, 0.620, 0.160},
		{1.00, 1.000, 0.985, 0.900},
	}
	if v <= 0 {
		return rgba64(stops[0].R, stops[0].G, stops[0].B)
	}
	for i := 1; i < len(stops); i++ {
		if v <= stops[i].Position {
			a, b := stops[i-1], stops[i]
			t := smoothStep((v - a.Position) / (b.Position - a.Position))
			return rgba64(a.R+(b.R-a.R)*t, a.G+(b.G-a.G)*t, a.B+(b.B-a.B)*t)
		}
	}
	last := stops[len(stops)-1]
	return rgba64(last.R, last.G, last.B)
}

func rgba64(r, g, b float64) color.RGBA64 {
	return color.RGBA64{R: uint16(math.Round(65535 * r)), G: uint16(math.Round(65535 * g)), B: uint16(math.Round(65535 * b)), A: 65535}
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
