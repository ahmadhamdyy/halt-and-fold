# Reproducibility map

## Environment

- GPU: NVIDIA A30
- PyTorch: 2.10.0+cu128
- Mixed precision: bfloat16; folded sufficient statistics accumulate in FP32
- Python: 3.10 or newer
- Install: `pip install -e ".[dev]"`

## Data

- ImageNette: `data/imagenette2-320`
- ADE20K: `data/ADEChallengeData2016`
- ImageNette validation: 3,925 images
- ADE20K validation: 2,000 images

## Primary checkpoints

- Four-step classification seeds: `runs/halt_fold_seed{0,1,2}/best.pt`
- Eight-step classification: `runs/halt_fold_depth8/best.pt`
- ADE20K seeds: `runs/halt_fold_ade_seed{0,1,2}/best.pt`

## Result provenance

- Classification Pareto quality: `runs/halt_fold_depth8/pareto_quality.json`
- Pareto latency: `runs/halt_fold_depth8/pareto_latency_{0..4}.json`
- Three-seed ADE20K: `runs/halt_fold_ade_seed{0,1,2}/aspect_preserving_eval.json`
- Routing ablation: `runs/halt_fold_depth8/routing_ablation.json`
- ADE20K routing ablation: `runs/halt_fold_ade_seed{0,1,2}/routing_ablation.json`
- Precision evaluation: `runs/halt_fold_depth8/precision_eval.json`
- Repeated latency/memory profiles: `runs/halt_fold_depth8/profile_{0..4}.json`
- Boundary-depth analysis: `runs/halt_fold_ade_seed0/boundary_depth.json`

## Evaluation entry points

```bash
python -m rdvb.evaluate_halt_fold --checkpoint CHECKPOINT --data DATA --out OUT
python -m rdvb.evaluate_segmentation_original --checkpoint CHECKPOINT --data DATA --out OUT
python -m rdvb.evaluate_routing_ablation --checkpoint CHECKPOINT --data DATA --out OUT
python -m rdvb.evaluate_precision --checkpoint CHECKPOINT --data DATA --out OUT
python -m rdvb.analyze_boundary_depth --checkpoint CHECKPOINT --data DATA --out OUT
python -m rdvb.benchmark_folded --checkpoint CHECKPOINT --out OUT --schedule SCHEDULE
```

Fixed-capacity schedules determine tensor shapes and mean updates; learned
router scores determine which spatial locations occupy those capacities.

## Verification

```bash
pytest -q
```

Set `DEIT_PATH` to run the optional checkpoint-conversion tests.
