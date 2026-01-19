# CoLearning Functions Implementation Summary

## Overview
Successfully implemented the `send_anomalies()` and `send_normals()` functions in the `CoLearner` class for collaborative anomaly detection.

## Implementation Details

### 1. `send_anomalies(model_idx, scores)`
**Purpose**: Send detected anomalies from one model to other models for pseudo-label refinement.

**Key Features**:
- Identifies anomalies as samples where `scores > anomaly_threshold` (default 0.5)
- For each recipient model, marks these samples as anomalies (pseudo-label = 1)
- Calculates confidence based on how extreme the anomaly score is
- Confidence formula: `min(score - threshold, 1.0)` (clipped to [0,1])

**Implementation**:
```python
def send_anomalies(self, model_idx: int, scores: np.ndarray) -> None:
    # 1. Find anomaly indices
    anomaly_indices = np.where(scores > self.anomaly_threshold)[0]
    
    # 2. For each other model, mark these as anomalies
    for i in range(len(self.models)):
        if i != model_idx:
            target_model = f"model_{i}"
            # Initialize pseudo-labels if needed
            # Set labels to 1 for anomalies
            # Update confidence: higher for more extreme scores
```

### 2. `send_normals(model_idx, scores)`
**Purpose**: Send detected normals from one model to other models for pseudo-labeling.

**Key Features**:
- Identifies normals as samples where `scores <= anomaly_threshold` (default 0.5)
- For each recipient model, marks these samples as normals (pseudo-label = 0)
- Calculates confidence based on how far below threshold the score is
- Confidence formula: `min(threshold - score, 1.0)` (clipped to [0,1])

**Implementation**:
```python
def send_normals(self, model_idx: int, scores: np.ndarray) -> None:
    # 1. Find normal indices
    normal_indices = np.where(scores <= self.anomaly_threshold)[0]
    
    # 2. For each other model, mark these as normals
    for i in range(len(self.models)):
        if i != model_idx:
            target_model = f"model_{i}"
            # Initialize pseudo-labels if needed
            # Set labels to 0 for normals
            # Update confidence: higher for more extreme scores
```

## Integration with CoLearner

### Updated `exchange()` method
The `exchange()` method now uses these functions for sequential pseudo-label exchange:

```python
def exchange(self) -> None:
    # 1. Each model predicts on training set
    for i, model in enumerate(self.models):
        scores = model.predict_scores(self.data.train_indexes)
        self.model_scores[f"model_{i}"] = scores
        
        # 2. Send anomalies and normals to other models
        self.send_anomalies(i, scores)
        self.send_normals(i, scores)
    
    # 3. Compute ensemble predictions
    ensemble_scores = np.mean([scores for scores in self.model_scores.values()])
```

## Data Structure Updates

### `CoLearner.__init__()` enhancement
- Added `anomaly_threshold` parameter (default 0.5)
- Configurable threshold for defining what constitutes an anomaly

### `Data` class integration
The functions leverage existing Data class methods:
- `pseudo_labels_by_model`: Dict storing per-model pseudo-labels
- `pseudo_label_confidence`: Dict storing confidence scores per model
- `y_train_original`: Original training labels for initialization

## Testing

### Test Results
Created comprehensive test script (`test_colearning.py`) that validates:

✅ **send_anomalies**:
- Correctly identifies samples above threshold
- Distributes to all recipient models
- Confidence values properly scaled based on anomaly score extremeness

✅ **send_normals**:
- Correctly identifies samples below/at threshold
- Distributes to all recipient models
- Confidence values properly scaled based on normal score extremeness

**Example Output**:
```
--- Testing send_anomalies ---
Model 0 scores: [0.1, 0.2, 0.8, 0.9, 0.3, 0.85, 0.15, 0.92, 0.4, 0.7]
Detected anomalies: 2908 samples
✓ Model 1 received 2908/2908 anomalies (Avg confidence: 0.2510)
✓ Model 2 received 2908/2908 anomalies (Avg confidence: 0.2510)

--- Testing send_normals ---
Model 1 scores: [0.1, 0.2, 0.3, 0.15, 0.05, 0.25, 0.12, 0.08, 0.2, 0.1]
Detected normals: 2935 samples
✓ Model 0 received 2935/2935 normals (Avg confidence: 0.2521)
✓ Model 2 received 2935/2935 normals (Avg confidence: 0.2521)
```

## Workflow: Sequential Multi-Model Exchange

The complete collaborative training process now works as follows:

```
For chapter in range(max_chapters):
    # Step 1: Each model trains one epoch
    for model in models:
        model.train(epoch)
    
    # Step 2: Exchange pseudo-labels
    for i, model in models:
        scores = model.predict_scores(training_set)
        
        # Send to peers
        colearner.send_anomalies(i, scores)    # Mark high-scoring samples as anomalies
        colearner.send_normals(i, scores)      # Mark low-scoring samples as normals
    
    # Step 3: Each model now has peer predictions in data.pseudo_labels_by_model
    # and will incorporate them in next training epoch
    
    # Step 4: Evaluate ensemble
    ensemble_pred = mean([m.predict_scores() for m in models])
    auc = roc_auc_score(y_test, ensemble_pred)
    
    # Step 5: Check convergence
    if should_stop:
        break
```

## Key Design Decisions

1. **Turn-by-turn Exchange**: Models send sequentially, not simultaneously
2. **Confidence-weighted Labels**: Confidence reflects score extremeness
3. **Bi-directional Learning**: Each model learns both anomalies (from confident high scores) and normals (from confident low scores)
4. **No Self-Contamination**: Models don't send to themselves
5. **Lazy Initialization**: Pseudo-label dicts created only when first needed

## Next Steps

The implementation enables:
1. ✅ Pseudo-label exchange between models
2. ✅ Confidence scoring for transmitted labels
3. ⏳ Full collaborative training pipeline in `cotrain()`
4. ⏳ Evaluation and convergence checking
5. ⏳ Optional recurrent model training on aggregated embeddings

All components are now ready for end-to-end testing with real models (PReNet, DeepSAD, XGBOD, DevNet).
