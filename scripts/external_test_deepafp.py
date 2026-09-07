# -*- coding: utf-8 -*-
"""
TLH-AFP | External Evaluation on DeepAFP-Main

Locked protocol:
- Five-fold ensemble (0.20 per fold)
- Decision threshold = 0.50
- No test-based model/weight/threshold optimization

DeepAFP-Main representations:
- PepBERT: 1024
- ChemBERTa: 768
- ProstT5: 1024
- Handcrafted features: 488

Raw total: 3304
Complementary: 2280
Selected complementary: 558
"""

from pathlib import Path
import gc
import pickle
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from torch.utils.data import DataLoader, Dataset


warnings.filterwarnings("ignore")


# =============================================================================
# Configuration
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

FEATURE_SELECTOR_PATH = ROOT / "feature_selector_rf.pkl"
CHECKPOINT_DIR = ROOT / "checkpoints"

RESULTS_DIR = (
    ROOT / "results" / "FINAL_DEEPAFP_EXTERNAL_TEST"
)
RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SEED = 42
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SELECTED_FOLDS = [0, 1, 2, 3, 4]

FOLD_WEIGHTS = np.full(
    len(SELECTED_FOLDS),
    0.20,
    dtype=np.float64,
)

THRESHOLD = 0.50
BATCH_SIZE = 32

# Feature dimensions
PEPBERT_DIM = 1024
CHEMBERTA_DIM = 768
PROSTT5_DIM = 1024
GLOBAL_DIM = 488

RAW_TOTAL_DIM = (
    PEPBERT_DIM
    + CHEMBERTA_DIM
    + PROSTT5_DIM
    + GLOBAL_DIM
)

RAW_COMPLEMENTARY_DIM = (
    CHEMBERTA_DIM
    + PROSTT5_DIM
    + GLOBAL_DIM
)

SELECTED_COMPLEMENTARY_DIM = 558
FEATURE_DIM_PER_BRANCH = 186

# Model dimensions
HIDDEN_DIM = 128
NUM_HEADS = 1
NUM_LAYERS = 2


# =============================================================================
# Reproducibility
# =============================================================================

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed()


# =============================================================================
# Dataset
# =============================================================================

class DeepAFPDataset(Dataset):
    """DeepAFP-Main test set with precomputed representations."""

    def __init__(
        self,
        data_dir: Path,
    ) -> None:

        self.pepbert = self._load(
            data_dir / "test_pepbert.npy"
        )

        self.chemberta = self._load(
            data_dir / "test_chem.npy"
        )

        self.prostt5 = self._load(
            data_dir / "test_prostt5.npy"
        )

        self.global_feat = self._load(
            data_dir / "test_handcrafted_global.npy"
        )

        label_path = data_dir / "DeepAFP-main-test.csv"

        if not label_path.exists():
            raise FileNotFoundError(
                f"Test labels not found:\n{label_path}"
            )

        df = pd.read_csv(label_path)

        if "label" not in df.columns:
            raise ValueError(
                f"'label' column not found in {label_path}"
            )

        self.labels = (
            df["label"]
            .astype(np.int64)
            .to_numpy()
        )

        self._validate()

    @staticmethod
    def _load(path: Path) -> np.ndarray:
        if not path.exists():
            raise FileNotFoundError(
                f"Required feature file not found:\n{path}"
            )

        return np.load(path).astype(np.float32)

    def _validate(self) -> None:
        n = len(self.labels)

        arrays = {
            "PepBERT": (
                self.pepbert,
                PEPBERT_DIM,
            ),
            "ChemBERTa": (
                self.chemberta,
                CHEMBERTA_DIM,
            ),
            "ProstT5": (
                self.prostt5,
                PROSTT5_DIM,
            ),
            "Handcrafted": (
                self.global_feat,
                GLOBAL_DIM,
            ),
        }

        for name, (array, expected_dim) in arrays.items():

            if array.shape[0] != n:
                raise ValueError(
                    f"{name} sample count mismatch: "
                    f"{array.shape[0]} != {n}"
                )

            if array.shape[1] != expected_dim:
                raise ValueError(
                    f"{name} dimension mismatch: "
                    f"{array.shape[1]} != {expected_dim}"
                )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict:
        return {
            "pepbert": torch.from_numpy(
                self.pepbert[index]
            ),
            "chemberta": torch.from_numpy(
                self.chemberta[index]
            ),
            "prostt5": torch.from_numpy(
                self.prostt5[index]
            ),
            "global": torch.from_numpy(
                self.global_feat[index]
            ),
            "label": torch.tensor(
                self.labels[index],
                dtype=torch.long,
            ),
        }


def collate_fn(batch: list[dict]) -> dict:
    return {
        key: torch.stack(
            [item[key] for item in batch]
        )
        for key in batch[0]
    }


# =============================================================================
# Model
# =============================================================================

class TLH_AFP_Model_Dual(nn.Module):
    """TLH-AFP architecture used for DeepAFP-Main evaluation."""

    def __init__(
        self,
        input_dim: int = SELECTED_COMPLEMENTARY_DIM,
        hidden_dim: int = HIDDEN_DIM,
        num_heads: int = NUM_HEADS,
        num_layers: int = NUM_LAYERS,
    ) -> None:
        super().__init__()

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=PEPBERT_DIM,
                nhead=num_heads,
                dim_feedforward=PEPBERT_DIM * 4,
                dropout=0.2,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=num_layers,
        )

        self.transformer_proj = nn.Linear(
            PEPBERT_DIM,
            hidden_dim,
        )

        self.transformer_norm = nn.LayerNorm(
            hidden_dim
        )

        self.complementary_proj = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
        )

        self.cross_attn_norm = nn.LayerNorm(
            hidden_dim
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
        )

    def forward(
        self,
        pepbert: torch.Tensor,
        complementary: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if pepbert.dim() == 2:
            pepbert = pepbert.unsqueeze(1)

        primary = self.transformer(
            pepbert
        )

        primary = self.transformer_proj(
            primary
        )

        primary = self.transformer_norm(
            primary
        )

        complementary = self.complementary_proj(
            complementary
        )

        complementary = complementary.unsqueeze(1)

        attended, _ = self.cross_attn(
            query=primary,
            key=complementary,
            value=complementary,
        )

        fused = self.cross_attn_norm(
            attended
        )

        pooled = fused.mean(dim=1)

        logits = self.classifier(
            pooled
        )

        z = self.projection(
            pooled
        )

        return logits, z


# =============================================================================
# Feature preprocessing
# =============================================================================

def prepare_features(
    pepbert: torch.Tensor,
    chemberta: torch.Tensor,
    prostt5: torch.Tensor,
    global_feat: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:

    arrays = [
        pepbert.detach().cpu().numpy(),
        chemberta.detach().cpu().numpy(),
        prostt5.detach().cpu().numpy(),
        global_feat.detach().cpu().numpy(),
    ]

    combined = np.concatenate(
        arrays,
        axis=1,
    )

    if combined.shape[1] != RAW_TOTAL_DIM:
        raise ValueError(
            f"Expected {RAW_TOTAL_DIM} raw features, "
            f"got {combined.shape[1]}"
        )

    scaled = SCALER.transform(
        combined
    )

    pepbert_scaled = scaled[
        :,
        :PEPBERT_DIM,
    ]

    complementary_scaled = scaled[
        :,
        PEPBERT_DIM:,
    ]

    if complementary_scaled.shape[1] != RAW_COMPLEMENTARY_DIM:
        raise ValueError(
            f"Expected {RAW_COMPLEMENTARY_DIM} complementary "
            f"features, got {complementary_scaled.shape[1]}"
        )

    selected = SELECTOR.transform(
        complementary_scaled
    )

    if selected.shape[1] != SELECTED_COMPLEMENTARY_DIM:
        raise ValueError(
            f"Expected {SELECTED_COMPLEMENTARY_DIM} selected "
            f"features, got {selected.shape[1]}"
        )

    chem_out = selected[
        :,
        :FEATURE_DIM_PER_BRANCH,
    ]

    prost_out = selected[
        :,
        FEATURE_DIM_PER_BRANCH:
        2 * FEATURE_DIM_PER_BRANCH,
    ]

    global_out = selected[
        :,
        2 * FEATURE_DIM_PER_BRANCH:
        3 * FEATURE_DIM_PER_BRANCH,
    ]

    return (
        torch.from_numpy(
            pepbert_scaled
        ).float(),
        torch.from_numpy(
            chem_out
        ).float(),
        torch.from_numpy(
            prost_out
        ).float(),
        torch.from_numpy(
            global_out
        ).float(),
    )


# =============================================================================
# Prediction
# =============================================================================

@torch.no_grad()
def predict_fold(
    model: nn.Module,
    loader: DataLoader,
) -> np.ndarray:

    model.eval()
    probabilities = []

    for batch in loader:

        (
            pepbert,
            chemberta,
            prostt5,
            global_feat,
        ) = prepare_features(
            batch["pepbert"],
            batch["chemberta"],
            batch["prostt5"],
            batch["global"],
        )

        logits, _ = model(
            pepbert.to(
                DEVICE,
                non_blocking=True,
            ),
            torch.cat(
                [
                    chemberta,
                    prostt5,
                    global_feat,
                ],
                dim=1,
            ).to(
                DEVICE,
                non_blocking=True,
            ),
        )

        probabilities.extend(
            torch.sigmoid(logits)
            .cpu()
            .numpy()
            .ravel()
        )

    return np.asarray(
        probabilities,
        dtype=np.float64,
    )


# =============================================================================
# Metrics
# =============================================================================

def calculate_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[dict, np.ndarray]:

    predictions = (
        probabilities >= THRESHOLD
    ).astype(np.int64)

    cm = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    sensitivity = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if tn + fp > 0
        else 0.0
    )

    metrics = {
        "AUROC": roc_auc_score(
            labels,
            probabilities,
        ),
        "PR_AUC": average_precision_score(
            labels,
            probabilities,
        ),
        "Accuracy": accuracy_score(
            labels,
            predictions,
        ),
        "Precision": precision_score(
            labels,
            predictions,
            zero_division=0,
        ),
        "Recall": recall_score(
            labels,
            predictions,
            zero_division=0,
        ),
        "F1_Score": f1_score(
            labels,
            predictions,
            zero_division=0,
        ),
        "MCC": matthews_corrcoef(
            labels,
            predictions,
        ),
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Balanced_Accuracy": (
            sensitivity + specificity
        ) / 2.0,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }

    return metrics, cm


# =============================================================================
# Load feature selector
# =============================================================================

if not FEATURE_SELECTOR_PATH.exists():
    raise FileNotFoundError(
        f"Feature selector not found:\n"
        f"{FEATURE_SELECTOR_PATH}"
    )

with open(
    FEATURE_SELECTOR_PATH,
    "rb",
) as f:
    feature_data = pickle.load(f)

SCALER = feature_data["scaler"]
SELECTOR = feature_data["selector"]

stored_total = int(
    feature_data.get(
        "total_features",
        RAW_COMPLEMENTARY_DIM,
    )
)

stored_selected = int(
    feature_data.get(
        "selected_features",
        SELECTED_COMPLEMENTARY_DIM,
    )
)

if stored_total != RAW_COMPLEMENTARY_DIM:
    raise ValueError(
        f"Selector expects {stored_total} complementary features; "
        f"expected {RAW_COMPLEMENTARY_DIM}."
    )

if stored_selected != SELECTED_COMPLEMENTARY_DIM:
    raise ValueError(
        f"Selector returns {stored_selected} features; "
        f"expected {SELECTED_COMPLEMENTARY_DIM}."
    )


# =============================================================================
# Load external dataset
# =============================================================================

dataset = DeepAFPDataset(
    DATA_DIR
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
    collate_fn=collate_fn,
)

test_labels = dataset.labels


# =============================================================================
# Validate ensemble configuration
# =============================================================================

if not np.isclose(
    FOLD_WEIGHTS.sum(),
    1.0,
):
    raise ValueError(
        "Fold weights must sum to 1."
    )

if len(SELECTED_FOLDS) != len(FOLD_WEIGHTS):
    raise ValueError(
        "Fold count and weight count do not match."
    )


# =============================================================================
# Locate checkpoints
# =============================================================================

checkpoint_paths = {}

for fold in SELECTED_FOLDS:

    candidates = [
        CHECKPOINT_DIR
        / f"best_tlh_afp_fold_{fold}.pt",
        CHECKPOINT_DIR
        / f"best_fold_{fold}.pt",
    ]

    matches = [
        path
        for path in candidates
        if path.exists()
    ]

    if len(matches) == 1:
        checkpoint_paths[fold] = matches[0]
        continue

    recursive_matches = list(
        ROOT.rglob(
            f"best_tlh_afp_fold_{fold}.pt"
        )
    )

    if len(recursive_matches) == 1:
        checkpoint_paths[fold] = recursive_matches[0]
    elif len(recursive_matches) == 0:
        raise FileNotFoundError(
            f"Checkpoint for Fold {fold} not found."
        )
    else:
        raise RuntimeError(
            f"Multiple checkpoints found for Fold {fold}:\n"
            + "\n".join(
                str(path)
                for path in recursive_matches
            )
        )


# =============================================================================
# Generate fold predictions
# =============================================================================

fold_probabilities = []
fold_results = []

for fold, weight in zip(
    SELECTED_FOLDS,
    FOLD_WEIGHTS,
):

    checkpoint = torch.load(
        checkpoint_paths[fold],
        map_location="cpu",
        weights_only=False,
    )

    state_dict = checkpoint.get(
        "model_state_dict",
        checkpoint,
    )

    model = TLH_AFP_Model_Dual(
        input_dim=SELECTED_COMPLEMENTARY_DIM,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
    ).to(DEVICE)

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False,
    )

    if missing:
        raise RuntimeError(
            f"Missing parameters in Fold {fold}:\n"
            + "\n".join(missing)
        )

    probabilities = predict_fold(
        model,
        loader,
    )

    if len(probabilities) != len(dataset):
        raise RuntimeError(
            f"Fold {fold}: prediction count does not "
            f"match the test-set size."
        )

    predictions = (
        probabilities >= THRESHOLD
    ).astype(np.int64)

    fold_metrics, _ = calculate_metrics(
        test_labels,
        probabilities,
    )

    fold_probabilities.append(
        probabilities
    )

    fold_results.append(
        {
            "Fold": fold,
            "Weight": weight,
            "AUROC": fold_metrics["AUROC"],
            "PR_AUC": fold_metrics["PR_AUC"],
            "Accuracy": fold_metrics["Accuracy"],
            "F1_Score": fold_metrics["F1_Score"],
            "MCC": fold_metrics["MCC"],
        }
    )

    print(
        f"Fold {fold} | "
        f"AUROC={fold_metrics['AUROC']:.4f} | "
        f"F1={fold_metrics['F1_Score']:.4f} | "
        f"MCC={fold_metrics['MCC']:.4f}"
    )

    del model
    del checkpoint

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# Five-fold ensemble
# =============================================================================

prediction_matrix = np.vstack(
    fold_probabilities
)

ensemble_probabilities = np.average(
    prediction_matrix,
    axis=0,
    weights=FOLD_WEIGHTS,
)

final_metrics, confusion = calculate_metrics(
    test_labels,
    ensemble_probabilities,
)


# =============================================================================
# Save numerical results
# =============================================================================

pd.DataFrame(
    fold_results
).to_csv(
    RESULTS_DIR / "DeepAFP_fold_results.csv",
    index=False,
)

pd.DataFrame(
    [
        {
            "Dataset": "DeepAFP-Main",
            "Folds": "0,1,2,3,4",
            "Weighting": "Equal",
            "Threshold": THRESHOLD,
            **final_metrics,
        }
    ]
).to_csv(
    RESULTS_DIR / "DeepAFP_FINAL_RESULTS.csv",
    index=False,
)


# =============================================================================
# Save predictions
# =============================================================================

prediction_data = {
    "label": test_labels,
    "ensemble_probability": ensemble_probabilities,
    "prediction": (
        ensemble_probabilities >= THRESHOLD
    ).astype(np.int64),
}

for index, fold in enumerate(SELECTED_FOLDS):
    prediction_data[
        f"fold_{fold}_probability"
    ] = prediction_matrix[index]

pd.DataFrame(
    prediction_data
).to_csv(
    RESULTS_DIR / "DeepAFP_FINAL_PREDICTIONS.csv",
    index=False,
)


# =============================================================================
# ROC / PR curves
# =============================================================================

fpr, tpr, _ = roc_curve(
    test_labels,
    ensemble_probabilities,
)

precision, recall, _ = precision_recall_curve(
    test_labels,
    ensemble_probabilities,
)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(13.5, 5.7),
)

axes[0].plot(
    fpr,
    tpr,
    linewidth=2.5,
    label=f"AUROC = {final_metrics['AUROC']:.4f}",
)

axes[0].plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1,
)

axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("(a) ROC Curve")
axes[0].legend(
    loc="lower right",
    frameon=False,
)
axes[0].grid(
    linestyle="--",
    linewidth=0.5,
    alpha=0.25,
)

axes[1].plot(
    recall,
    precision,
    linewidth=2.5,
    label=f"PR-AUC = {final_metrics['PR_AUC']:.4f}",
)

axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("(b) Precision–Recall Curve")
axes[1].legend(
    loc="lower left",
    frameon=False,
)
axes[1].grid(
    linestyle="--",
    linewidth=0.5,
    alpha=0.25,
)

fig.suptitle(
    "TLH-AFP on Independent DeepAFP-Main Test Set"
)

plt.tight_layout()

fig.savefig(
    RESULTS_DIR / "DeepAFP_ROC_PR.png",
    dpi=600,
    bbox_inches="tight",
)

plt.close(fig)


# =============================================================================
# Confusion matrix
# =============================================================================

fig, ax = plt.subplots(
    figsize=(6.5, 5.8)
)

image = ax.imshow(
    confusion,
    interpolation="nearest",
    cmap="Blues",
)

ax.set_xticks(
    [0, 1],
)
ax.set_yticks(
    [0, 1],
)

ax.set_xticklabels(
    ["non-AFP", "AFP"],
)

ax.set_yticklabels(
    ["non-AFP", "AFP"],
)

ax.set_xlabel("Predicted Class")
ax.set_ylabel("True Class")
ax.set_title(
    "TLH-AFP — DeepAFP-Main\nConfusion Matrix"
)

for i in range(2):
    for j in range(2):
        ax.text(
            j,
            i,
            str(confusion[i, j]),
            ha="center",
            va="center",
            fontsize=15,
        )

fig.colorbar(
    image,
    ax=ax,
    fraction=0.046,
    pad=0.04,
)

plt.tight_layout()

fig.savefig(
    RESULTS_DIR / "DeepAFP_Confusion_Matrix.png",
    dpi=600,
    bbox_inches="tight",
)

plt.close(fig)


# =============================================================================
# Protocol
# =============================================================================

protocol = f"""TLH-AFP FINAL EXTERNAL EVALUATION
=================================

Dataset: DeepAFP-Main

Folds used: 0, 1, 2, 3, 4
Fold weights: 0.20, 0.20, 0.20, 0.20, 0.20
Decision threshold: {THRESHOLD:.2f}

No test-based fold selection.
No test-based weight optimization.
No test-based threshold optimization.
No test-based model-combination search.

Feature dimensions:
PepBERT = {PEPBERT_DIM}
ChemBERTa = {CHEMBERTA_DIM}
ProstT5 = {PROSTT5_DIM}
Global = {GLOBAL_DIM}
Raw combined = {RAW_TOTAL_DIM}
Raw complementary = {RAW_COMPLEMENTARY_DIM}
Selected complementary = {SELECTED_COMPLEMENTARY_DIM}
"""

(
    RESULTS_DIR / "DeepAFP_FINAL_PROTOCOL.txt"
).write_text(
    protocol,
    encoding="utf-8",
)


# =============================================================================
# Final report
# =============================================================================

print("\n" + "=" * 72)
print("TLH-AFP | FINAL DeepAFP-Main EXTERNAL TEST")
print("=" * 72)

print(
    f"AUROC          : {final_metrics['AUROC']:.6f}"
)
print(
    f"PR-AUC         : {final_metrics['PR_AUC']:.6f}"
)
print(
    f"Accuracy       : {final_metrics['Accuracy']:.6f}"
)
print(
    f"Precision      : {final_metrics['Precision']:.6f}"
)
print(
    f"Recall         : {final_metrics['Recall']:.6f}"
)
print(
    f"F1-Score       : {final_metrics['F1_Score']:.6f}"
)
print(
    f"MCC            : {final_metrics['MCC']:.6f}"
)

print(
    f"\nTN = {final_metrics['TN']} | "
    f"FP = {final_metrics['FP']} | "
    f"FN = {final_metrics['FN']} | "
    f"TP = {final_metrics['TP']}"
)

print(
    f"\nResults saved to:\n{RESULTS_DIR}"
)

print("=" * 72)
print("EXTERNAL EVALUATION COMPLETED")
print("=" * 72)
