# -*- coding: utf-8 -*-
"""
TLH-AFP | Random Forest Feature Selection

Select informative complementary features from:
    - ChemBERTa
    - ProstT5
    - Handcrafted sequence features

The resulting selector and scaler are saved for use by the
training and evaluation pipelines.
"""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler


# ============================================================================
# Configuration
# ============================================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_PATH = ROOT / "feature_selector_rf.pkl"

SEED = 42

CHEMBERTA_DIM = 384
PROSTT5_DIM = 1024
GLOBAL_DIM = 488
TOTAL_FEATURES = CHEMBERTA_DIM + PROSTT5_DIM + GLOBAL_DIM

N_ESTIMATORS = 100
SELECTION_THRESHOLD = "mean"


# ============================================================================
# Data loading
# ============================================================================

def load_data() -> tuple[np.ndarray, np.ndarray, dict]:
    """Load complementary representations and training labels."""

    feature_paths = {
        "ChemBERTa": DATA_DIR / "train_chemberta.npy",
        "ProstT5": DATA_DIR / "train_prostt5.npy",
        "Handcrafted": DATA_DIR / "train_handcrafted_global.npy",
    }

    arrays = {}

    for name, path in feature_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Required file not found:\n{path}")
        arrays[name] = np.load(path).astype(np.float32)

    label_path = DATA_DIR / "train_smiles.csv"
    if not label_path.exists():
        raise FileNotFoundError(f"Required file not found:\n{label_path}")

    labels = pd.read_csv(label_path)["label"].astype(np.int64).to_numpy()

    if len(labels) != arrays["ChemBERTa"].shape[0]:
        raise ValueError("Number of labels does not match feature samples.")

    expected_dims = {
        "ChemBERTa": CHEMBERTA_DIM,
        "ProstT5": PROSTT5_DIM,
        "Handcrafted": GLOBAL_DIM,
    }

    for name, array in arrays.items():
        if array.shape[1] != expected_dims[name]:
            raise ValueError(
                f"{name} dimension mismatch: "
                f"{array.shape[1]} != {expected_dims[name]}"
            )

        if array.shape[0] != len(labels):
            raise ValueError(
                f"{name} sample count mismatch: "
                f"{array.shape[0]} != {len(labels)}"
            )

    X = np.concatenate(
        [
            arrays["ChemBERTa"],
            arrays["ProstT5"],
            arrays["Handcrafted"],
        ],
        axis=1,
    )

    if X.shape[1] != TOTAL_FEATURES:
        raise ValueError(
            f"Expected {TOTAL_FEATURES} complementary features, "
            f"got {X.shape[1]}"
        )

    feature_ranges = {
        "ChemBERTa": (0, CHEMBERTA_DIM),
        "ProstT5": (
            CHEMBERTA_DIM,
            CHEMBERTA_DIM + PROSTT5_DIM,
        ),
        "Handcrafted": (
            CHEMBERTA_DIM + PROSTT5_DIM,
            TOTAL_FEATURES,
        ),
    }

    return X, labels, feature_ranges


# ============================================================================
# Feature selection
# ============================================================================

def fit_feature_selector(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[SelectFromModel, StandardScaler, np.ndarray]:
    """Fit scaler, Random Forest, and mean-importance selector."""

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=SEED,
        n_jobs=-1,
        class_weight="balanced",
    )

    rf.fit(X_scaled, y)

    importances = rf.feature_importances_

    selector = SelectFromModel(
        rf,
        threshold=SELECTION_THRESHOLD,
        prefit=True,
    )

    selected_mask = selector.get_support()

    return selector, scaler, importances, selected_mask


# ============================================================================
# Save selector
# ============================================================================

def save_selector(
    selector: SelectFromModel,
    scaler: StandardScaler,
    importances: np.ndarray,
    selected_mask: np.ndarray,
    feature_ranges: dict,
) -> int:
    """Save the fitted preprocessing and feature-selection pipeline."""

    selected_features = int(selected_mask.sum())

    metadata = {
        "selector": selector,
        "scaler": scaler,
        "selected_features": selected_features,
        "total_features": TOTAL_FEATURES,
        "selected_mask": selected_mask,
        "feature_importances": importances,
        "threshold": SELECTION_THRESHOLD,
        "n_estimators": N_ESTIMATORS,
        "random_state": SEED,
        "feature_ranges": feature_ranges,
        "representations": [
            "ChemBERTa",
            "ProstT5",
            "Engineered Global Features",
        ],
        "pepbert_excluded": True,
        "model_name": "TLH-AFP",
        "selection_method": "Random Forest Feature Selection",
        "selection_threshold": "Mean Feature Importance",
    }

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(metadata, f)

    return selected_features


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print("=" * 70)
    print("TLH-AFP | RANDOM FOREST FEATURE SELECTION")
    print("=" * 70)

    print(f"Project root : {ROOT}")
    print(f"Data dir     : {DATA_DIR}")

    X, y, feature_ranges = load_data()

    print(f"\nSamples      : {len(y):,}")
    print(f"Features     : {X.shape[1]:,}")
    print(f"Positive     : {int(y.sum()):,}")
    print(f"Negative     : {len(y) - int(y.sum()):,}")

    selector, scaler, importances, selected_mask = fit_feature_selector(
        X, y
    )

    selected_features = save_selector(
        selector,
        scaler,
        importances,
        selected_mask,
        feature_ranges,
    )

    reduction = 100.0 * (
        1.0 - selected_features / TOTAL_FEATURES
    )

    print("\nFeature selection completed")
    print(f"Selected     : {selected_features:,} / {TOTAL_FEATURES:,}")
    print(f"Reduction    : {reduction:.2f}%")
    print(f"Saved to     : {OUTPUT_PATH}")

    print("\nSelected features by representation:")

    for name, (start, end) in feature_ranges.items():
        count = int(selected_mask[start:end].sum())
        total = end - start
        percentage = 100.0 * count / total

        print(
            f"  {name:12s}: "
            f"{count:4d}/{total:4d} "
            f"({percentage:.2f}%)"
        )

    print("\nTop 10 features:")

    top_indices = np.argsort(importances)[::-1][:10]

    for rank, index in enumerate(top_indices, start=1):
        feature_name = "Unknown"
        local_index = index

        for name, (start, end) in feature_ranges.items():
            if start <= index < end:
                feature_name = name
                local_index = index - start
                break

        print(
            f"  {rank:2d}. "
            f"{feature_name:12s} "
            f"index={local_index:4d} "
            f"importance={importances[index]:.8f}"
        )

    print("\n" + "=" * 70)
    print("FEATURE SELECTION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
