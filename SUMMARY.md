# CoBench Summary

## Core Architecture
- **API**: `Model` → `Data` → `CoLearning` → `Strategy`
- **Models**: PReNet, DeepSAD, DevNet (ADBench wrappers)
- **Data**: 60/20/20 split, semi-supervised labels, pseudo-label store

## CoLearning Flow
```
Warmup (N epochs) → [Exchange pseudo-labels → Train chapter] × M → Evaluate
```
- `send_anomalies()`: scores > 0.95 → label=1
- `send_normals()`: scores < 0.05 → label=0
- Early stopping via `PlateauStrategy` on validation AUC

## Key Classes
| Class | Purpose |
|-------|---------|
| `SimpleCoLearner` | Multi-model pseudo-label exchange |
| `SingleModel` | Solo baseline (no collaboration) |
| `ClassicalADBenchData` | NPZ loader + stratified splits |
| `PseudoLabelStore` | Per-model label/confidence tracking |

## Status
- ✅ Tabular AD (ADBench)
- ⏳ Graph AD (GADBench/PyGOD)
- ⏳ RecurrentModel (ensemble on embeddings)

## Submodules
- ADBench: 57 datasets, 30 algorithms
- GADBench: graph AD benchmark
- PyGOD: PyTorch Geometric detectors