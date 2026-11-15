# Halt-and-Fold

Halt-and-Fold is an inference method for token-adaptive recurrent vision
transformers. Tokens that stop updating remain available as context through
fixed-size sufficient statistics for normalized linear attention. The method
avoids recomputing keys and values for halted tokens while preserving dense
output slots.

## Repository layout

- `rdvb/model.py`: recurrent backbone, router, and execution modes
- `rdvb/blocks.py`: softmax, normalized linear, cached, and folded attention
- `rdvb/sparse.py`: packed gather/scatter operations
- `rdvb/finetune_halt_fold.py`: classification adaptation
- `rdvb/segmentation.py`: ADE20K transfer training
- `rdvb/benchmark_folded.py`: compiled full-versus-folded benchmark
- `rdvb/evaluate_*.py`: accuracy, precision, routing, and segmentation analysis
- `tests/`: model and baseline correctness tests
- `docs/REPRODUCIBILITY.md`: evaluation commands and experiment settings

Datasets, checkpoints, generated results, and cluster-specific launch scripts
are intentionally excluded.

## Installation

Python 3.10 or newer and a recent PyTorch installation are recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

## Data layout

Classification commands accept an ImageFolder-style directory:

```text
data/imagenette2-320/
├── train/
│   └── <class>/
└── val/
    └── <class>/
```

Segmentation commands expect the standard ADE20K structure:

```text
data/ADEChallengeData2016/
├── images/{training,validation}/
└── annotations/{training,validation}/
```

## Tests

```bash
pytest -q
```

The DeiT conversion tests are optional. Set `DEIT_PATH` to a compatible
checkpoint to enable them:

```bash
DEIT_PATH=/path/to/deit_small_patch16_224.pth pytest -q
```

## Core workflows

Adapt a recurrent softmax checkpoint to four-step normalized linear attention:

```bash
python -m rdvb.finetune_halt_fold \
  --checkpoint /path/to/source.pt \
  --data data/imagenette2-320 \
  --out outputs/halt_fold_seed0 \
  --schedule 1,.5,.25,.125 \
  --epochs 20 --seed 0
```

Evaluate explicit full statistics and folded execution:

```bash
python -m rdvb.evaluate_halt_fold \
  --checkpoint outputs/halt_fold_seed0/best.pt \
  --data data/imagenette2-320 \
  --out outputs/halt_fold_seed0/eval.json
```

Benchmark compiled execution:

```bash
python -m rdvb.benchmark_folded \
  --checkpoint /path/to/checkpoint.pt \
  --out outputs/profile.json \
  --sizes 1024,1344 \
  --schedule 1,.5,.25,.125,.0625,.03125,.015625,.0078125 \
  --warmup 10 --iters 100 --trials 10
```

Train the ADE20K decoder:

```bash
python -m rdvb.segmentation \
  --checkpoint /path/to/depth8.pt \
  --data data/ADEChallengeData2016 \
  --out outputs/ade_seed0 \
  --schedule 1,.75,.5,.25,.125,.0625,.03125,.015625 \
  --epochs 30 --seed 0
```

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the remaining
evaluation entry points.

## Notes

Halt-and-Fold requires monotone halting, shared recurrent projections,
token-local normalization, and final-state write-back. Folded sufficient
statistics accumulate in FP32 under mixed-precision execution.

## License

This project is available under the [MIT License](LICENSE).

