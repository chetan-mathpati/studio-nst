# Studio NST

**Neural Style Transfer & Visual Experimentation**

Studio NST is a research-oriented implementation and evaluation project for Neural Style Transfer (NST). It brings three major approaches into one reproducible codebase:

1. **Optimization-based NST** — inspired by Gatys et al.
2. **Feed-forward NST** — inspired by Johnson et al.
3. **Arbitrary Style Transfer** — inspired by AdaIN (Huang & Belongie)

The goal is not to reproduce a course project. The goal is to build an independently structured implementation, document the underlying ideas, and measure practical trade-offs on real hardware.

## Research questions

- How does optimization-based NST change with image resolution and optimization steps?
- What practical trade-offs exist between iterative optimization and feed-forward inference?
- How does arbitrary style transfer change the style-selection workflow?
- What performance constraints appear on a 4 GB consumer GPU?

## Project principles

- Reproducible experiments
- No fabricated benchmark numbers
- Clear separation between implementation and experimental claims
- Research notes linked to engineering decisions
- Hardware-aware evaluation
- Original UI/UX and project architecture

## Current status

The repository foundation is intentionally being built before the visual frontend. The first milestone is a reproducible research core.

## Environment

The project is developed around:

- Python 3.14
- PyTorch
- TorchVision
- Pillow
- NumPy
- Matplotlib
- tqdm

CUDA support is used when available.

## Quick start

```powershell
.\.venv\Scripts\Activate.ps1
python scripts\check_environment.py
python scripts/run_gatys.py --content assets/content/example.jpg --style assets/styles/example.jpg --steps 300 --output outputs/gatys
```

Replace the example images with your own assets.

## Research documentation

See `docs/research/` for the research narrative and `docs/benchmarks/` for experiment methodology.

## Reproducibility rule

Every result reported in the final report must come from an actual run and include the relevant configuration and hardware context.
