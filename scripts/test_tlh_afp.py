# -*- coding: utf-8 -*-
"""
TLH-AFP | Final Independent Test Evaluation

Locked evaluation protocol:
    - Five-fold development
    - Validation AUROC cutoff: 0.97
    - Selected folds: 1-4
    - Equal ensemble weights: 0.25
    - Decision threshold: 0.50
    - No test-based fold, weight, or threshold optimization
"""

from pathlib import Path
import pickle
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

from torch.utils.data import Dataset, DataLoader


# ============================================================================
# Configuration
# ============================================================================

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data"

CHECKPOINT_DIR = ROOT / "checkpoints" / "final_tlh_afp"
FEATURE_SELECTOR_PATH = ROOT / "feature_selector_rf.pkl"
RESULTS_DIR = ROOT / "results" / "FINAL_TLH_AFP_TEST"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

SELECTED_FOLDS = [1, 2, 3, 4]
EXCLUDED_FOLDS = [5]

VALIDATION_AUROC_CUTOFF = 0.97
THRESHOLD = 0.50

FOLD_WEIGHTS = {
    1: 0.25,
    2: 0.25,
    3: 0.25,
    4: 0.25,
}

BATCH_SIZE = 16

# Feature dimensions
PEPBERT_DIM = 1024
CHEMBERTA_DIM = 384
PROSTT5_DIM = 1024
GLOBAL_DIM = 488

COMPLEMENTARY_DIM = (
    CHEMBERTA_DIM
    + PROSTT5_DIM
    + GLOBAL_DIM
)

SELECTED_DIM = 547

# Model dimensions
HIDDEN_DIM = 128
NUM_HEADS = 1
NUM_LAYERS = 2

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================================
# Reproducibility
# ============================================================================

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed()


# ============================================================================
# Dataset
# ============================================================================

class TLHAFPDataset(Dataset):
    """Dataset wrapper for precomputed TLH-AFP representations."""

    def __init__(
        self,
        data_dir: Path,
        split: str,
    ) -> None:

        self.pepbert = self._load(
            data_dir / f"{split}_pepbert.npy"
        )

        self.chemberta = self._load(
            data_dir / f"{split}_chemberta.npy"
        )

        self.prostt5 = self._load(
            data_dir / f"{split}_prostt5.npy"
        )

        self.global_feat = self._load(
            data_dir / f"{split}_handcrafted_global.npy"
        )

        csv_path = data_dir / f"{split}_smiles.csv"

        if not csv_path.exists():
            raise FileNotFoundError(
                f"Missing dataset file:\n{csv_path}"
            )

        df = pd.read_csv(csv_path)

        if "label" not in df.columns:
            raise ValueError(
                f"'label' column not found in:\n{csv_path}"
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
                f"Missing feature file:\n{path}"
            )

        return np.load(path).astype(np.float32)

    def _validate(self) -> None:
        n = len(self.labels)

        arrays = {
            "PepBERT": (self.pepbert, PEPBERT_DIM),
            "ChemBERTa": (self.chemberta, CHEMBERTA_DIM),
            "ProstT5": (self.prostt5, PROSTT5_DIM),
            "Handcrafted": (self.global_feat, GLOBAL_DIM),
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


# ============================================================================
# Feature preprocessing
# ============================================================================

def prepare_complementary(
    chemberta: torch.Tensor,
    prostt5: torch.Tensor,
    global_feat: torch.Tensor,
) -> torch.Tensor:

    complementary = np.concatenate(
        [
            chemberta.detach().cpu().numpy(),
            prostt5.detach().cpu().numpy(),
            global_feat.detach().cpu().numpy(),
        ],
        axis=1,
    )

    if complementary.shape[1] != COMPLEMENTARY_DIM:
        raise ValueError(
            f"Expected {COMPLEMENTARY_DIM} complementary features, "
            f"got {complementary.shape[1]}"
        )

    scaled = SCALER.transform(complementary)
    selected = SELECTOR.transform(scaled)

    if selected.shape[1] != SELECTED_DIM:
        raise ValueError(
            f"Expected {SELECTED_DIM} selected features, "
            f"got {selected.shape[1]}"
        )

    return torch.from_numpy(
        selected
    ).float()


# ============================================================================
# Model
# ============================================================================

class TLH_AFP_Model_Dual(nn.Module):
    """Final TLH-AFP model used for independent evaluation."""

    def __init__(
        self,
        input_dim: int = SELECTED_DIM,
        hidden_dim: int = HIDDEN_DIM,
        num_heads: int = NUM_HEADS,
        num_layers: int = NUM_LAYERS,
    ) -> None:
        super().__init__()

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=PEPBERT_DIM,
                nhead=num_heads,
                dim_feedforward=4096,
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
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.3),
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
            primary + attended
        )

        pooled = fused.mean(dim=1)

        logits = self.classifier(
            pooled
        )

        z = F.normalize(
            self.projection(pooled),
            dim=-1,
        )

        return logits, z


# ============================================================================
# Prediction
# ============================================================================

@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
) -> tuple[np.ndarray, np.ndarray]:

    model.eval()

    probabilities = []
    labels = []

    for batch in loader:

        complementary = prepare_complementary(
            batch["chemberta"],
            batch["prostt5"],
            batch["global"],
        )

        pepbert = batch["pepbert"]

        logits, _ = model(
            pepbert.to(DEVICE, non_blocking=True),
            complementary.to(
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

        labels.extend(
            batch["label"]
            .cpu()
            .numpy()
            .ravel()
        )

    return (
        np.asarray(probabilities, dtype=np.float64),
        np.asarray(labels, dtype=np.int64),
    )


# ============================================================================
# Metrics
# ============================================================================

def calculate_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = THRESHOLD,
) -> dict:

    predictions = (
        probabilities >= threshold
    ).astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()

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

    return {
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
        "F1": f1_score(
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


# ============================================================================
# Feature selector
# ============================================================================

if not FEATURE_SELECTOR_PATH.exists():
    raise FileNotFoundError(
        f"Feature selector not found:\n{FEATURE_SELECTOR_PATH}"
    )

with open(
    FEATURE_SELECTOR_PATH,
    "rb",
) as f:
    fs_data = pickle.load(f)

SCALER = fs_data["scaler"]
SELECTOR = fs_data["selector"]

saved_total = int(
    fs_data.get(
        "total_features",
        COMPLEMENTARY_DIM,
    )
)

saved_selected = int(
    fs_data.get(
        "selected_features",
        SELECTED_DIM,
    )
)

if saved_total != COMPLEMENTARY_DIM:
    raise ValueError(
        f"Feature selector expects {saved_total} features; "
        f"current representation has {COMPLEMENTARY_DIM}."
    )

if saved_selected != SELECTED_DIM:
    raise ValueError(
        f"Feature selector returns {saved_selected} features; "
        f"expected {SELECTED_DIM}."
    )


# ============================================================================
# Locked protocol audit
# ============================================================================

validation_rows = []

for fold in range(1, 6):

    checkpoint_path = (
        CHECKPOINT_DIR
        / f"best_fold_{fold}.pt"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint:\n{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    best_auc = checkpoint.get(
        "best_auc"
    )

    if best_auc is None:
        raise ValueError(
            f"Fold {fold} checkpoint lacks 'best_auc'."
        )

    best_auc = float(best_auc)

    included = fold in SELECTED_FOLDS

    validation_rows.append(
        {
            "Fold": fold,
            "Validation_AUROC": best_auc,
            "Included": included,
            "AUROC_GE_0.97": (
                best_auc >= VALIDATION_AUROC_CUTOFF
            ),
            "Fixed_Weight": (
                FOLD_WEIGHTS[fold]
                if included
                else 0.0
            ),
        }
    )


validation_df = pd.DataFrame(
    validation_rows
)

# Strict audit of the predefined selection rule.
for fold in SELECTED_FOLDS:

    auc = float(
        validation_df.loc[
            validation_df["Fold"] == fold,
            "Validation_AUROC",
        ].iloc[0]
    )

    if auc < VALIDATION_AUROC_CUTOFF:
        raise RuntimeError(
            f"Selected Fold {fold} has "
            f"validation AUROC={auc:.6f}, below the "
            f"fixed cutoff of {VALIDATION_AUROC_CUTOFF:.2f}."
        )

excluded_auc = float(
    validation_df.loc[
        validation_df["Fold"] == 5,
        "Validation_AUROC",
    ].iloc[0]
)

if excluded_auc >= VALIDATION_AUROC_CUTOFF:
    raise RuntimeError(
        f"Fold 5 has validation AUROC={excluded_auc:.6f} "
        f"and therefore does not satisfy the predefined exclusion rule."
    )

if not np.isclose(
    sum(FOLD_WEIGHTS.values()),
    1.0,
):
    raise ValueError(
        "Selected fold weights must sum to 1.0."
    )


# ============================================================================
# Load independent test set
# ============================================================================

test_dataset = TLHAFPDataset(
    DATA_DIR,
    "test",
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
    collate_fn=collate_fn,
)

test_labels = test_dataset.labels


# ============================================================================
# Generate fold predictions
# ============================================================================

fold_probabilities = {}
individual_results = []

for fold in SELECTED_FOLDS:

    checkpoint_path = (
        CHECKPOINT_DIR
        / f"best_fold_{fold}.pt"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model = TLH_AFP_Model_Dual(
        input_dim=SELECTED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    probabilities, labels = predict(
        model,
        test_loader,
    )

    if not np.array_equal(
        labels,
        test_labels,
    ):
        raise RuntimeError(
            f"Fold {fold}: test-label ordering mismatch."
        )

    fold_probabilities[fold] = probabilities

    fold_metrics = calculate_metrics(
        test_labels,
        probabilities,
    )

    validation_auc = float(
        validation_df.loc[
            validation_df["Fold"] == fold,
            "Validation_AUROC",
        ].iloc[0]
    )

    individual_results.append(
        {
            "Fold": fold,
            "Validation_AUROC": validation_auc,
            "Weight": FOLD_WEIGHTS[fold],
            "Threshold": THRESHOLD,
            "Test_AUROC": fold_metrics["AUROC"],
            "Test_PR_AUC": fold_metrics["PR_AUC"],
            "Test_Accuracy": fold_metrics["Accuracy"],
            "Test_Precision": fold_metrics["Precision"],
            "Test_Recall": fold_metrics["Recall"],
            "Test_F1": fold_metrics["F1"],
            "Test_MCC": fold_metrics["MCC"],
        }
    )

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================================
# Four-fold equal-weight ensemble
# ============================================================================

probability_matrix = np.vstack(
    [
        fold_probabilities[fold]
        for fold in SELECTED_FOLDS
    ]
)

weight_array = np.asarray(
    [
        FOLD_WEIGHTS[fold]
        for fold in SELECTED_FOLDS
    ],
    dtype=np.float64,
)

final_probabilities = np.average(
    probability_matrix,
    axis=0,
    weights=weight_array,
)

final_metrics = calculate_metrics(
    test_labels,
    final_probabilities,
    threshold=THRESHOLD,
)


# ============================================================================
# Save numerical results
# ============================================================================

validation_df.to_csv(
    RESULTS_DIR / "validation_fold_audit.csv",
    index=False,
)

pd.DataFrame(
    individual_results
).to_csv(
    RESULTS_DIR / "individual_fold_test_results.csv",
    index=False,
)

pd.DataFrame(
    [
        {
            "Model": "TLH-AFP",
            "Selected_Folds": "1,2,3,4",
            "Excluded_Folds": "5",
            "Fold_Selection_Rule": (
                "Validation AUROC >= 0.97"
            ),
            "Ensemble_Method": (
                "Equal-weight four-fold ensemble"
            ),
            "Fold1_Weight": 0.25,
            "Fold2_Weight": 0.25,
            "Fold3_Weight": 0.25,
            "Fold4_Weight": 0.25,
            "Decision_Threshold": 0.50,
            "AUROC": final_metrics["AUROC"],
            "PR_AUC": final_metrics["PR_AUC"],
            "Accuracy": final_metrics["Accuracy"],
            "Precision": final_metrics["Precision"],
            "Recall": final_metrics["Recall"],
            "F1_Score": final_metrics["F1"],
            "MCC": final_metrics["MCC"],
            "Sensitivity": final_metrics["Sensitivity"],
            "Specificity": final_metrics["Specificity"],
            "Balanced_Accuracy": (
                final_metrics["Balanced_Accuracy"]
            ),
            "TN": final_metrics["TN"],
            "FP": final_metrics["FP"],
            "FN": final_metrics["FN"],
            "TP": final_metrics["TP"],
        }
    ]
).to_csv(
    RESULTS_DIR / "TLH_AFP_FINAL_RESULT.csv",
    index=False,
)


# ============================================================================
# Save predictions
# ============================================================================

prediction_data = {
    "label": test_labels,
    "ensemble_probability": final_probabilities,
    "ensemble_prediction": (
        final_probabilities >= THRESHOLD
    ).astype(np.int64),
}

for fold in SELECTED_FOLDS:
    prediction_data[
        f"Fold_{fold}_Probability"
    ] = fold_probabilities[fold]

pd.DataFrame(
    prediction_data
).to_csv(
    RESULTS_DIR / "TLH_AFP_FINAL_PREDICTIONS.csv",
    index=False,
)


# ============================================================================
# Confusion matrix
# ============================================================================

cm = confusion_matrix(
    test_labels,
    (
        final_probabilities >= THRESHOLD
    ).astype(np.int64),
    labels=[0, 1],
)

plt.figure(figsize=(6, 5))
plt.imshow(
    cm,
    interpolation="nearest",
)
plt.colorbar()

plt.xticks(
    [0, 1],
    ["Negative", "Positive"],
)

plt.yticks(
    [0, 1],
    ["Negative", "Positive"],
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("TLH-AFP — Final Four-Fold Ensemble")

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            fontsize=14,
        )

plt.tight_layout()

plt.savefig(
    RESULTS_DIR
    / "TLH_AFP_FINAL_Confusion_Matrix.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================================
# ROC and Precision-Recall curves
# ============================================================================

fpr, tpr, _ = roc_curve(
    test_labels,
    final_probabilities,
)

precision, recall, _ = precision_recall_curve(
    test_labels,
    final_probabilities,
)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(14, 6),
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
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(
    recall,
    precision,
    linewidth=2.5,
    label=f"PR-AUC = {final_metrics['PR_AUC']:.4f}",
)

axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("(b) Precision-Recall Curve")
axes[1].legend()
axes[1].grid(alpha=0.3)

fig.suptitle(
    "TLH-AFP — Final Independent Test"
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "TLH_AFP_FINAL_ROC_PR.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================================
# Prediction distribution
# ============================================================================

positive_probs = final_probabilities[test_labels == 1]
negative_probs = final_probabilities[test_labels == 0]

plt.figure(figsize=(9, 6))

try:
    from scipy.stats import gaussian_kde

    xmin = max(
        0.0,
        min(
            positive_probs.min(),
            negative_probs.min(),
        ) - 0.03,
    )

    xmax = min(
        1.0,
        max(
            positive_probs.max(),
            negative_probs.max(),
        ) + 0.03,
    )

    x_grid = np.linspace(
        xmin,
        xmax,
        500,
    )

    kde_negative = gaussian_kde(
        negative_probs
    )

    kde_positive = gaussian_kde(
        positive_probs
    )

    plt.plot(
        x_grid,
        kde_negative(x_grid),
        linewidth=2.5,
        label="Non-AFP",
    )

    plt.fill_between(
        x_grid,
        kde_negative(x_grid),
        alpha=0.25,
    )

    plt.plot(
        x_grid,
        kde_positive(x_grid),
        linewidth=2.5,
        label="AFP",
    )

    plt.fill_between(
        x_grid,
        kde_positive(x_grid),
        alpha=0.25,
    )

except Exception:
    plt.hist(
        negative_probs,
        bins=30,
        density=True,
        alpha=0.45,
        label="Non-AFP",
    )

    plt.hist(
        positive_probs,
        bins=30,
        density=True,
        alpha=0.45,
        label="AFP",
    )

plt.axvline(
    THRESHOLD,
    linestyle="--",
    linewidth=2,
    label=f"Threshold = {THRESHOLD:.2f}",
)

plt.xlabel("Predicted Probability")
plt.ylabel("Density")
plt.title(
    "TLH-AFP — Final Test Prediction Distribution"
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "TLH_AFP_FINAL_KDE.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================================
# Save locked protocol
# ============================================================================

protocol = """TLH-AFP FINAL TEST PROTOCOL
===========================

Selected folds: 1, 2, 3, 4
Excluded fold: 5
Fold inclusion criterion: Validation AUROC >= 0.97
Ensemble method: Equal-weight four-fold ensemble
Fold weights: 0.25, 0.25, 0.25, 0.25
Decision threshold: 0.50
Test-based threshold optimization: None
Test-based weight optimization: None
Test-based fold selection: None
"""

(RESULTS_DIR / "FINAL_PROTOCOL.txt").write_text(
    protocol,
    encoding="utf-8",
)


# ============================================================================
# Final report
# ============================================================================

print("=" * 72)
print("TLH-AFP | FINAL INDEPENDENT TEST")
print("=" * 72)

print(f"Selected folds : {SELECTED_FOLDS}")
print(f"Excluded fold : {EXCLUDED_FOLDS}")
print(f"Fold weights  : {weight_array}")
print(f"Threshold     : {THRESHOLD:.2f}")

print("\nTEST PERFORMANCE")
print(f"AUROC          : {final_metrics['AUROC']:.6f}")
print(f"PR-AUC         : {final_metrics['PR_AUC']:.6f}")
print(f"Accuracy       : {final_metrics['Accuracy']:.6f}")
print(f"Precision      : {final_metrics['Precision']:.6f}")
print(f"Recall         : {final_metrics['Recall']:.6f}")
print(f"F1-Score       : {final_metrics['F1']:.6f}")
print(f"MCC            : {final_metrics['MCC']:.6f}")
print(f"Sensitivity     : {final_metrics['Sensitivity']:.6f}")
print(f"Specificity     : {final_metrics['Specificity']:.6f}")
print(
    f"Balanced Acc.  : "
    f"{final_metrics['Balanced_Accuracy']:.6f}"
)

print(
    f"\nTN = {final_metrics['TN']} | "
    f"FP = {final_metrics['FP']} | "
    f"FN = {final_metrics['FN']} | "
    f"TP = {final_metrics['TP']}"
)

print("\nResults saved to:")
print(RESULTS_DIR)

print("=" * 72)
print("FINAL TEST EVALUATION COMPLETED")
print("=" * 72)
