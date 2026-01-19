# MVP Notebook - CoLearner Comprehensive Tests

## Overview

The MVP notebook (`src/mvp.ipynb`) now contains a complete test suite for the CoLearner collaborative learning framework. All tests use real models (PReNet, XGBOD) and real data (ADBench Classical dataset).

## Notebook Structure

### Cells 1-9: Setup and Model Training
- **Cell 1**: Clear cached modules for fresh import
- **Cell 2**: Setup paths and imports
- **Cell 3**: Markdown with setup instructions
- **Cell 4**: Data mapping configuration
- **Cell 5**: Load ADBench Classical dataset (annthyroid)
- **Cell 6**: Train individual models (PReNet, DeepSAD, DevNet, XGBOD)
- **Cell 7**: Markdown header for collaborative learning tests
- **Cell 8**: Initial CoLearner test (fixed with correct imports)
- **Cell 9**: Markdown introduction to detailed tests

### Cells 10-14: CoLearner Tests
- **Cell 10 (TEST 1)**: Index Selection Functions
- **Cell 11 (TEST 2)**: Manual Exchange Round
- **Cell 12 (TEST 3)**: Full Collaborative Training Loop
- **Cell 13 (TEST 4)**: Pseudo-Label Propagation Analysis
- **Cell 14 (TEST 5)**: Single Model Baseline Comparison
- **Cell 15**: Summary markdown

## Test Details

### TEST 1: Index Selection Functions
**Purpose**: Verify that models correctly identify anomalies and normals

**Process**:
1. Generate dummy scores with clear separation (0-0.4 for normals, 0.6-1.0 for anomalies)
2. Create CoLearner instance with 0.5 threshold
3. Call `get_anomaly_indexes()` and `get_normal_indexes()`
4. Verify counts and score ranges

**Expected Results**:
- ~100 anomalies identified (scores > 0.5)
- ~100 normals identified (scores ≤ 0.5)
- Total indices equal input size

### TEST 2: Manual Exchange Round
**Purpose**: Test bi-directional pseudo-label exchange between two models

**Process**:
1. Get predictions from PReNet and XGBOD on training set
2. Create CoLearner with fresh data
3. PReNet identifies anomalies and normals, sends to XGBOD
4. XGBOD receives and stores pseudo-labels with confidence
5. XGBOD identifies anomalies and normals, sends to PReNet
6. Verify both models received correct labels and confidence scores

**Verifications**:
- XGBOD receives PReNet's anomalies (label=1) and normals (label=0)
- PReNet receives XGBOD's anomalies and normals
- Confidence scores properly scaled (0-1 range)
- Label distribution tracked in `data.pseudo_labels_by_model`

### TEST 3: Full Collaborative Training Loop
**Purpose**: Test the complete automated exchange and training cycle

**Process**:
1. Create fresh data and CoLearner
2. Run `cotrain()` with:
   - 1 warmup epoch
   - 3 collaborative chapters
   - Convergence strategy (patience=3)
3. Collect metrics per chapter

**Output**:
- Warmup phase metrics
- Per-chapter metrics for each model
- Final ensemble AUC

**Key Metrics**:
- `model_0` (PReNet): ROC-AUC
- `model_1` (XGBOD): ROC-AUC
- `ensemble`: Average of both models

### TEST 4: Pseudo-Label Propagation Analysis
**Purpose**: Analyze how labels flow through the system during collaborative training

**Inspections**:
- Pseudo-labels per model (count of anomalies vs normals)
- Confidence distribution (mean, std, min, max)
- Comparison with original training labels

**Expected Behavior**:
- Different models may generate different label distributions
- Confidence scores reflect label extremeness
- Original class imbalance compared with generated labels

### TEST 5: Single Model Baseline Comparison
**Purpose**: Measure improvement from collaborative learning

**Methodology**:
1. Train single PReNet model (standalone fit)
2. Get ROC-AUC on test set
3. Compare with collaborative PReNet AUC
4. Calculate absolute and relative improvement

**Metrics**:
- Single model baseline AUC
- Collaborative PReNet AUC
- Collaborative XGBOD AUC
- Ensemble AUC (average of both)
- Improvement: `collaborative - single`
- Relative improvement: `(improvement / single) * 100%`

## Key Fixes Applied

1. **Import Fix**: Changed from `SingleCoLearner` to `SingleModel`
2. **Strategy Import**: Added `SimpleStrategy` to imports
3. **All Test Cells**: Updated to match refactored function signatures:
   - `send_anomalies(model_idx, anomaly_indexes, scores)` ← added `anomaly_indexes`
   - `send_normals(model_idx, normal_indexes, scores)` ← added `normal_indexes`

## Data Flow Diagram

```
Models (trained)
    ↓ predict_scores()
    ↓
CoLearner.exchange()
    ├─ get_anomaly_indexes(scores) → array of indices
    ├─ get_normal_indexes(scores) → array of indices
    ├─ send_anomalies(model_idx, indices, scores)
    │   └─ Update data.pseudo_labels_by_model[model_name]
    └─ send_normals(model_idx, indices, scores)
        └─ Update data.pseudo_labels_by_model[model_name]
    ↓
Pseudo-Labels (stored in Data object)
    ├─ data.pseudo_labels_by_model[model_name] → label array
    └─ data.pseudo_label_confidence[model_name] → confidence array
    ↓
Next training epoch (models use pseudo-labels)
```

## Variables Created During Tests

### From TEST 1:
- `dummy_scores`: Synthetic score distribution
- `test_colearner`: CoLearner for testing index functions
- `anomaly_indexes`: Indices of detected anomalies
- `normal_indexes`: Indices of detected normals

### From TEST 2:
- `data_exchange`: Fresh data for manual exchange test
- `prenet_scores`, `xgbod_scores`: Model predictions
- `colearner`: CoLearner for manual exchange
- `prenet_pseudo`, `xgbod_pseudo`: Received pseudo-labels
- `prenet_conf`, `xgbod_conf`: Confidence scores

### From TEST 3:
- `data_collearn`: Fresh data for collaborative training
- `models_list`: List of models to collaborate
- `colearner_full`: Full CoLearner instance
- `history`: Training history with metrics per chapter

### From TEST 4:
- (Uses variables from TEST 3)
- Prints analysis of pseudo-label propagation

### From TEST 5:
- `data_single`: Fresh data for baseline training
- `single_learner`: Single model learner (no collaboration)
- `single_scores`, `single_auc`: Baseline results
- Comparison metrics printed

## Running the Tests

To run all tests:
1. Execute cells 1-9 (setup and training)
2. Execute cells 10-14 (tests) sequentially
3. Each test is independent and provides its own output

**Expected Runtime**: ~5-10 minutes depending on system performance

## Success Indicators

✓ **TEST 1**: "Index selection working correctly!" + equal counts
✓ **TEST 2**: "Manual exchange round completed successfully!" + label counts match
✓ **TEST 3**: History shows chapters with model metrics + final ensemble AUC
✓ **TEST 4**: Pseudo-label analysis shows distribution differences
✓ **TEST 5**: Improvement metrics show collaboration effect

## Troubleshooting

### Import Error: "cannot import name 'SimpleStrategy'"
→ Ensure base.py has SimpleStrategy defined (lines 407-450)

### Import Error: "cannot import name 'SingleModel'"
→ Check base.py for SingleModel class (line 839+)

### Model errors in initial training (Cell 6)
→ Ensure PyTorch, TensorFlow, scikit-learn dependencies installed
→ Run: `pip install scikit-learn==1.0.2 pyod==1.0.9 --no-cache-dir`

### Memory errors with large data
→ Reduce batch size in model configs
→ Reduce max_chapters in CoLearner initialization

## Next Steps

1. **Experiment with thresholds**: Test different `anomaly_threshold` values
2. **Add more models**: Include DeepSAD, DevNet in collaborative ensemble
3. **Different datasets**: Test with GADBench or PyGOD datasets
4. **Recurrent models**: Implement and test RecurrentModel on aggregated embeddings
5. **Advanced strategies**: Create custom Strategy implementations
6. **Performance analysis**: Track individual model AUC improvement over chapters

## Files Updated

- `/home/peppino58/CoBench/src/mvp.ipynb`: Complete test suite (15 cells)
- `/home/peppino58/CoBench/src/base.py`: Already contains all required classes
- `/home/peppino58/CoBench/test_colearning.py`: Original unit tests (still available)

All tests are now integrated into the MVP notebook for easy execution and demonstration!
