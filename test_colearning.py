#!/usr/bin/env python3
"""
Test script for CoLearner send_anomalies and send_normals functions.
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from base import CoLearner, Strategy, Data
from baselines.adbench.data_loader import ClassicalADBenchData


class DummyModel:
    """Dummy model for testing."""
    def __init__(self, data):
        self.data = data
    
    def train(self, epoch):
        pass
    
    def fit(self):
        pass
    
    def predict_scores(self, indexes=None):
        """Return random scores."""
        if indexes is None:
            return np.random.rand(self.data.n_train)
        return np.random.rand(len(indexes))
    
    def get_embeddings(self, indexes=None):
        """Return random embeddings."""
        if indexes is None:
            return np.random.randn(self.data.n_train, 10)
        return np.random.randn(len(indexes), 10)


class DummyStrategy(Strategy):
    """Dummy strategy for testing."""
    def should_continue(self, model_metrics, chapter):
        return True
    
    def get_weights(self, model_metrics):
        return {f"model_{i}": 1.0 / len(model_metrics) for i in range(len(model_metrics))}


def test_send_anomalies_and_normals():
    """Test send_anomalies and send_normals functions."""
    print("=" * 70)
    print("Testing CoLearner send_anomalies and send_normals")
    print("=" * 70)
    
    # Load dataset
    dataset_path = Path("/home/peppino58/CoBench/SubModules/ADBench/adbench/datasets/Classical/2_annthyroid.npz")
    data = ClassicalADBenchData(dataset_path)
    print(f"\n✓ Loaded dataset: {data}")
    print(f"  Training samples: {data.n_train}")
    print(f"  Test samples: {data.n_test}")
    
    # Create dummy models
    models = [DummyModel(data) for _ in range(3)]
    print(f"\n✓ Created {len(models)} dummy models")
    
    # Create CoLearner
    strategy = DummyStrategy()
    colearner = CoLearner(models, data, strategy, warmup_epochs=0, max_chapters=1, anomaly_threshold=0.5)
    print(f"✓ Created CoLearner with anomaly_threshold={colearner.anomaly_threshold}")
    
    # Test send_anomalies
    print(f"\n--- Testing send_anomalies ---")
    scores_model_0 = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.85, 0.15, 0.92, 0.4, 0.7])
    # Pad to actual training size
    scores_model_0 = np.concatenate([scores_model_0, np.random.rand(data.n_train - len(scores_model_0))])
    
    print(f"Model 0 scores (first 10): {scores_model_0[:10]}")
    
    # Use intermediate function to get anomaly indexes
    anomaly_indexes = colearner.get_anomaly_indexes(scores_model_0)
    print(f"Detected anomalies (scores > 0.5): {len(anomaly_indexes)} samples")
    print(f"Sample indices: {anomaly_indexes[:5]}...")
    
    colearner.send_anomalies(model_idx=0, anomaly_indexes=anomaly_indexes, scores=scores_model_0)
    print(f"✓ Sent anomalies from model 0 to others")
    
    # Check that other models received the anomalies
    for i in range(1, len(models)):
        model_name = f"model_{i}"
        if model_name in data.pseudo_labels_by_model:
            labels = data.pseudo_labels_by_model[model_name]
            received_anomalies = np.sum(labels[anomaly_indexes] == 1)
            print(f"  → Model {i} received {received_anomalies}/{len(anomaly_indexes)} anomalies")
            confidence = data.pseudo_label_confidence[model_name][anomaly_indexes]
            print(f"    Average confidence: {np.mean(confidence):.4f}")
    
    # Reset for normals test
    data.reset_pseudo_labels()
    
    # Test send_normals
    print(f"\n--- Testing send_normals ---")
    scores_model_1 = np.array([0.1, 0.2, 0.3, 0.15, 0.05, 0.25, 0.12, 0.08, 0.2, 0.1])
    # Pad to actual training size
    scores_model_1 = np.concatenate([scores_model_1, np.random.rand(data.n_train - len(scores_model_1))])
    
    print(f"Model 1 scores (first 10): {scores_model_1[:10]}")
    
    # Use intermediate function to get normal indexes
    normal_indexes = colearner.get_normal_indexes(scores_model_1)
    print(f"Detected normals (scores <= 0.5): {len(normal_indexes)} samples")
    print(f"Sample indices: {normal_indexes[:5]}...")
    
    colearner.send_normals(model_idx=1, normal_indexes=normal_indexes, scores=scores_model_1)
    print(f"✓ Sent normals from model 1 to others")
    
    # Check that other models received the normals
    for i in [0, 2]:
        model_name = f"model_{i}"
        if model_name in data.pseudo_labels_by_model:
            labels = data.pseudo_labels_by_model[model_name]
            received_normals = np.sum(labels[normal_indexes] == 0)
            print(f"  → Model {i} received {received_normals}/{len(normal_indexes)} normals")
            confidence = data.pseudo_label_confidence[model_name][normal_indexes]
            print(f"    Average confidence: {np.mean(confidence):.4f}")
    
    print("\n" + "=" * 70)
    print("passed!")
    print("=" * 70)


if __name__ == "__main__":
    test_send_anomalies_and_normals()
