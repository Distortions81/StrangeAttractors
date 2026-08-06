package main

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"
)

type evolutionFrame struct {
	Candidate        candidate
	RawBounds        bounds
	MutationDistance float64
	PNGPath          string
	JSONPath         string
}

type evolutionFrameMetadata struct {
	metadata
	Frame            int     `json:"frame"`
	ParentFrame      int     `json:"parent_frame"`
	RawBounds        bounds  `json:"raw_bounds"`
	MutationDistance float64 `json:"mutation_distance"`
}

type evolutionEntry struct {
	Frame            int          `json:"frame"`
	ParentFrame      int          `json:"parent_frame"`
	Score            float64      `json:"score"`
	MutationDistance float64      `json:"mutation_distance"`
	Coefficients     coefficients `json:"coefficients"`
	PNG              string       `json:"png"`
	Metadata         string       `json:"metadata"`
}

type evolutionMetadata struct {
	Version          string           `json:"version"`
	GeneratedAt      string           `json:"generated_at"`
	Seed             int64            `json:"seed"`
	Frames           int              `json:"frames"`
	Width            int              `json:"width"`
	Height           int              `json:"height"`
	Iterations       int              `json:"iterations_per_frame"`
	Offspring        int              `json:"offspring_per_frame"`
	Mutation         float64          `json:"mutation"`
	MinimumScore     float64          `json:"minimum_score"`
	CoefficientRange float64          `json:"coefficient_range"`
	Entries          []evolutionEntry `json:"entries"`
}

func runEvolution(rng *rand.Rand, cfg config) error {
	if err := os.MkdirAll(cfg.output, 0o755); err != nil {
		return fmt.Errorf("create evolution directory: %w", err)
	}
	evaluationCfg := cfg
	evaluationCfg.minScore = cfg.evolveMinScore
	evaluationCfg.maxScore = 1
	// Evolution tends to walk near stability boundaries. Score a later window
	// so long transients that eventually collapse into a cycle cannot masquerade
	// as richly chaotic descendants.
	evaluationCfg.burnIn = max(cfg.burnIn, cfg.screenIters)

	first, err := findEvolutionStart(rng, evaluationCfg)
	if err != nil {
		return err
	}
	frames := make([]evolutionFrame, 0, cfg.evolveFrames)
	first.Index = 1
	frames = append(frames, evolutionFrame{Candidate: first, RawBounds: first.Bounds})
	velocity := coefficients{}
	for frame := 1; frame < cfg.evolveFrames; frame++ {
		historyStart := max(0, len(frames)-8)
		history := frames[historyStart:]
		next, delta, nextErr := evolveDescendant(rng, frames[len(frames)-1].Candidate, history, velocity, evaluationCfg)
		if nextErr != nil {
			return fmt.Errorf("evolve frame %d: %w", frame, nextErr)
		}
		next.Index = frame + 1
		frames = append(frames, evolutionFrame{
			Candidate: next, RawBounds: next.Bounds, MutationDistance: coefficientDistance(frames[len(frames)-1].Candidate.Coefficients, next.Coefficients),
		})
		velocity = delta
		if (frame+1)%10 == 0 || frame+1 == cfg.evolveFrames {
			fmt.Printf("evolved %d/%d frames\n", frame+1, cfg.evolveFrames)
		}
	}

	rawBounds := make([]bounds, len(frames))
	for i := range frames {
		rawBounds[i] = frames[i].RawBounds
	}
	smoothed := smoothEvolutionBounds(rawBounds, min(6, max(1, len(frames)/12)))
	for i := range frames {
		frames[i].Candidate.Bounds = smoothed[i]
		stem := fmt.Sprintf("frame-%06d_score-%.6f", i, frames[i].Candidate.Metrics.Score)
		frames[i].PNGPath = filepath.Join(cfg.output, stem+".png")
		frames[i].JSONPath = filepath.Join(cfg.output, stem+".json")
		if fileExists(frames[i].PNGPath) || fileExists(frames[i].JSONPath) {
			return fmt.Errorf("refusing to overwrite evolution frame %d", i)
		}
	}

	generatedAt := time.Now().UTC().Format(time.RFC3339)
	jobs := make(chan int)
	errCh := make(chan error, 1)
	var completed atomic.Int64
	var wg sync.WaitGroup
	workerCount := min(cfg.workers, len(frames))
	for worker := 0; worker < workerCount; worker++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range jobs {
				frame := &frames[i]
				densityWidth, densityHeight := cfg.width*cfg.supersample, cfg.height*cfg.supersample
				density, renderErr := buildDensity64InBounds(frame.Candidate.Coefficients, frame.Candidate.Bounds, cfg.burnIn, cfg.iterations, densityWidth, densityHeight)
				if renderErr == nil {
					renderErr = writePNG16GlowHDR(frame.PNGPath, density, densityWidth, densityHeight, cfg.supersample, cfg.softness, cfg.gamma, cfg.exposure, cfg.whitePercentile, cfg.glowStrength, cfg.glowRadius, cfg.glowThreshold)
				}
				if renderErr == nil {
					parent := -1
					if i > 0 {
						parent = i - 1
					}
					meta := evolutionFrameMetadata{
						metadata: metadata{
							Version: programVersion, GeneratedAt: generatedAt, Seed: cfg.seed, Iterations: cfg.iterations,
							BurnIn: cfg.burnIn, Width: cfg.width, Height: cfg.height, CoefficientRange: cfg.coefficientRange,
							Coefficients: frame.Candidate.Coefficients, Bounds: frame.Candidate.Bounds, Metrics: frame.Candidate.Metrics,
							PNG: filepath.Base(frame.PNGPath), Gamma: cfg.gamma, BitDepth: 16, DensityBits: 64,
							ToneMap: "aces-glow-unclipped", Palette: "deep-space", Exposure: cfg.exposure,
							WhitePercentile: cfg.whitePercentile, Supersample: cfg.supersample, Softness: cfg.softness,
							GlowStrength: cfg.glowStrength, GlowRadius: cfg.glowRadius, GlowThreshold: cfg.glowThreshold,
						},
						Frame: i, ParentFrame: parent, RawBounds: frame.RawBounds, MutationDistance: frame.MutationDistance,
					}
					renderErr = writeJSON(frame.JSONPath, meta)
				}
				if renderErr != nil {
					select {
					case errCh <- fmt.Errorf("frame %d: %w", i, renderErr):
					default:
					}
					continue
				}
				done := completed.Add(1)
				if done%10 == 0 || int(done) == len(frames) {
					fmt.Printf("rendered %d/%d frames\n", done, len(frames))
				}
			}
		}()
	}
	for i := range frames {
		jobs <- i
	}
	close(jobs)
	wg.Wait()
	select {
	case renderErr := <-errCh:
		return renderErr
	default:
	}

	entries := make([]evolutionEntry, len(frames))
	for i, frame := range frames {
		parent := -1
		if i > 0 {
			parent = i - 1
		}
		entries[i] = evolutionEntry{
			Frame: i, ParentFrame: parent, Score: frame.Candidate.Metrics.Score,
			MutationDistance: frame.MutationDistance, Coefficients: frame.Candidate.Coefficients,
			PNG: filepath.Base(frame.PNGPath), Metadata: filepath.Base(frame.JSONPath),
		}
	}
	manifest := evolutionMetadata{
		Version: programVersion, GeneratedAt: generatedAt, Seed: cfg.seed, Frames: len(frames),
		Width: cfg.width, Height: cfg.height, Iterations: cfg.iterations, Offspring: cfg.evolveOffspring,
		Mutation: cfg.evolveMutation, MinimumScore: cfg.evolveMinScore, CoefficientRange: cfg.coefficientRange,
		Entries: entries,
	}
	if err := writeJSON(filepath.Join(cfg.output, "evolution.json"), manifest); err != nil {
		return err
	}
	fmt.Printf("wrote %d-frame evolutionary sequence to %s\n", len(frames), cfg.output)
	return nil
}

func findEvolutionStart(rng *rand.Rand, cfg config) (candidate, error) {
	trials := max(cfg.samples, cfg.evolveOffspring*4)
	parameters := make([]coefficients, trials)
	for i := range parameters {
		parameters[i] = randomCoefficients(rng, cfg.coefficientRange)
	}
	results := evaluateCandidateBatch(parameters, cfg)
	var best candidate
	found := false
	for _, result := range results {
		if result.reason == "" && (!found || result.candidate.Metrics.Score > best.Metrics.Score) {
			best, found = result.candidate, true
		}
	}
	if !found {
		return candidate{}, errors.New("no viable evolution seed found; lower -evolve-min-score or increase -samples")
	}
	return best, nil
}

func evolveDescendant(rng *rand.Rand, parent candidate, history []evolutionFrame, velocity coefficients, cfg config) (candidate, coefficients, error) {
	for retry := 0; retry < 4; retry++ {
		scale := cfg.evolveMutation * math.Pow(0.6, float64(retry))
		parameters := make([]coefficients, cfg.evolveOffspring)
		// Retain the parent as an elite candidate. This guarantees that a lineage
		// can cross a locally fragile region without accepting a collapsed orbit;
		// any viable, interesting mutation still wins on novelty.
		parameters[0] = parent.Coefficients
		for i := 1; i < len(parameters); i++ {
			parameters[i] = mutateCoefficients(rng, parent.Coefficients, velocity, scale, cfg.coefficientRange)
		}
		results := evaluateCandidateBatch(parameters, cfg)
		bestUtility := math.Inf(-1)
		var best candidate
		found := false
		for _, result := range results {
			if result.reason != "" {
				continue
			}
			novelty := recentNovelty(result.candidate.Metrics, history)
			utility := 0.65*result.candidate.Metrics.Score + 0.35*novelty
			if utility > bestUtility {
				bestUtility, best, found = utility, result.candidate, true
			}
		}
		if found {
			return best, subtractCoefficients(best.Coefficients, parent.Coefficients), nil
		}
	}
	return candidate{}, coefficients{}, errors.New("no viable descendant found after reducing mutation scale")
}

func randomCoefficients(rng *rand.Rand, span float64) coefficients {
	return coefficients{A: uniform(rng, span), B: uniform(rng, span), C: uniform(rng, span), D: uniform(rng, span)}
}

func mutateCoefficients(rng *rand.Rand, parent, velocity coefficients, scale, limit float64) coefficients {
	const momentum = 0.45
	return coefficients{
		A: clampCoefficient(parent.A+momentum*velocity.A+rng.NormFloat64()*scale, limit),
		B: clampCoefficient(parent.B+momentum*velocity.B+rng.NormFloat64()*scale, limit),
		C: clampCoefficient(parent.C+momentum*velocity.C+rng.NormFloat64()*scale, limit),
		D: clampCoefficient(parent.D+momentum*velocity.D+rng.NormFloat64()*scale, limit),
	}
}

func clampCoefficient(value, limit float64) float64 { return math.Max(-limit, math.Min(limit, value)) }

func subtractCoefficients(a, b coefficients) coefficients {
	return coefficients{A: a.A - b.A, B: a.B - b.B, C: a.C - b.C, D: a.D - b.D}
}

func coefficientDistance(a, b coefficients) float64 {
	return math.Sqrt(math.Pow(a.A-b.A, 2) + math.Pow(a.B-b.B, 2) + math.Pow(a.C-b.C, 2) + math.Pow(a.D-b.D, 2))
}

func recentNovelty(m metrics, history []evolutionFrame) float64 {
	if len(history) == 0 {
		return 1
	}
	minimum := math.Inf(1)
	for _, previous := range history {
		d := descriptorDistance(m, previous.Candidate.Metrics)
		minimum = math.Min(minimum, d)
	}
	return math.Min(1, minimum)
}

func descriptorDistance(a, b metrics) float64 {
	terms := []float64{
		(a.Occupancy - b.Occupancy) / 0.20,
		(a.GlobalEntropy - b.GlobalEntropy) / 0.25,
		(a.BoxDimension - b.BoxDimension) / 0.25,
		(a.LyapunovExponent - b.LyapunovExponent) / 1.0,
	}
	var sum float64
	for _, term := range terms {
		sum += term * term
	}
	return math.Sqrt(sum / float64(len(terms)))
}

func smoothEvolutionBounds(input []bounds, radius int) []bounds {
	output := make([]bounds, len(input))
	for i := range input {
		var centerX, centerY, logWidth, logHeight, weightSum float64
		for j := max(0, i-radius); j <= min(len(input)-1, i+radius); j++ {
			weight := float64(radius + 1 - absInt(i-j))
			b := input[j]
			centerX += (b.MinX + b.MaxX) / 2 * weight
			centerY += (b.MinY + b.MaxY) / 2 * weight
			logWidth += math.Log(b.MaxX-b.MinX) * weight
			logHeight += math.Log(b.MaxY-b.MinY) * weight
			weightSum += weight
		}
		centerX, centerY = centerX/weightSum, centerY/weightSum
		width, height := math.Exp(logWidth/weightSum), math.Exp(logHeight/weightSum)
		output[i] = bounds{MinX: centerX - width/2, MaxX: centerX + width/2, MinY: centerY - height/2, MaxY: centerY + height/2}
	}
	return output
}

func absInt(value int) int {
	if value < 0 {
		return -value
	}
	return value
}
