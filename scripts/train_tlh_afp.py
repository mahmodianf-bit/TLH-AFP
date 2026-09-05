# -*- coding: utf-8 -*-
"""
TLH-AFP | Final Training
Multi-representation learning with feature selection,
Transformer encoding, cross-attention, focal learning,
and supervised contrastive learning.

This script:
    - trains five stratified folds
    - initializes from supervised contrastive pretraining
    - selects the best epoch by validation AUROC
    - supports resumable checkpoints
    - exports the best model of each fold

External test evaluation is handled separately.
"""

from pathlib import Path
import gc
import pickle
import random
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader, Subset

warnings.filterwarnings("ignore")


# ============================================================================
# CONFIGURATION
# ============================================================================

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"

FEATURE_SELECTOR_PATH = ROOT / "feature_selector_rf.pkl"
PRETRAINED_PATH = ROOT / "contrastive_pretrained_best.pt"
CHECKPOINT_DIR = ROOT / "checkpoints" / "final_tlh_afp"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

SELECTED_FEATURES = 547

# Model
HIDDEN_DIM = 128
NUM_HEADS = 1
NUM_LAYERS = 2

# Training
BATCH_SIZE = 8
MAX_EPOCHS = 200
PATIENCE = 30
LR = 1e-5
WEIGHT_DECAY = 0.05
GRAD_CLIP = 1.0

# Loss
FOCAL_ALPHA = 0.75
FOCAL_GAMMA = 2.5
LAMBDA_FOCAL = 1.2
LAMBDA_CONTRAST = 0.25
SUPCON_TEMP = 0.07


# ============================================================================
# REPRODUCIBILITY
# ============================================================================

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed()


# ============================================================================
# ENVIRONMENT
# ============================================================================

print("=" * 72)
print("TLH-AFP | FINAL MODEL TRAINING")
print("=" * 72)
print(f"Project root : {ROOT}")
print(f"Data dir     : {DATA_DIR}")
print(f"Checkpoint   : {CHECKPOINT_DIR}")
print(f"Device       : {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU          : {torch.cuda.get_device_name(0)}")


# ============================================================================
# FEATURE SELECTOR
# ============================================================================

if not FEATURE_SELECTOR_PATH.exists():
    raise FileNotFoundError(
        f"Feature selector not found:\n{FEATURE_SELECTOR_PATH}"
    )

with open(FEATURE_SELECTOR_PATH, "rb") as f:
    fs_data = pickle.load(f)

SELECTOR = fs_data["selector"]
SCALER = fs_data["scaler"]

SELECTOR_OUTPUT_DIM = int(fs_data["selected_features"])
SELECTOR_INPUT_DIM = int(fs_data["total_features"])

if SELECTOR_INPUT_DIM != COMPLEMENTARY_DIM:
    raise ValueError(
        f"Feature selector expects {SELECTOR_INPUT_DIM} features, "
        f"but current complementary features have {COMPLEMENTARY_DIM}."
    )

if SELECTOR_OUTPUT_DIM != SELECTED_FEATURES:
    raise ValueError(
        f"Expected {SELECTED_FEATURES} selected features, "
        f"but selector contains {SELECTOR_OUTPUT_DIM}."
    )

print("\nFeature selector:")
print(f"  Input  : {SELECTOR_INPUT_DIM}")
print(f"  Output : {SELECTOR_OUTPUT_DIM}")


# ============================================================================
# DATASET
# ============================================================================

class TLHAFPDataset(Dataset):
    """Dataset wrapper for precomputed TLH-AFP representations."""

    def __init__(self, data_dir: Path, split: str = "train") -> None:
        self.split = split

        self.pepbert = self._load_feature(
            data_dir / f"{split}_pepbert.npy"
        )
        self.chemberta = self._load_feature(
            data_dir / f"{split}_chemberta.npy"
        )
        self.prostt5 = self._load_feature(
            data_dir / f"{split}_prostt5.npy"
        )
        self.global_feat = self._load_feature(
            data_dir / f"{split}_handcrafted_global.npy"
        )

        csv_path = data_dir / f"{split}_smiles.csv"

        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset CSV not found:\n{csv_path}")

        df = pd.read_csv(csv_path)

        if "label" not in df.columns:
            raise ValueError(f"'label' column not found in {csv_path}")

        self.labels = df["label"].astype(np.int64).to_numpy()

        self._validate()
        self._summary()

    @staticmethod
    def _load_feature(path: Path) -> np.ndarray:
        if not path.exists():
            raise FileNotFoundError(f"Feature file not found:\n{path}")

        return np.load(path).astype(np.float32)

    def _validate(self) -> None:
        n = len(self.labels)

        arrays = {
            "PepBERT": (self.pepbert, PEPBERT_DIM),
            "ChemBERTa": (self.chemberta, CHEMBERTA_DIM),
            "ProstT5": (self.prostt5, PROSTT5_DIM),
            "Global": (self.global_feat, GLOBAL_DIM),
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

    def _summary(self) -> None:
        positive = int(self.labels.sum())
        negative = len(self.labels) - positive

        print(f"\n[{self.split}] Dataset")
        print(f"  Samples   : {len(self.labels)}")
        print(f"  PepBERT   : {self.pepbert.shape}")
        print(f"  ChemBERTa : {self.chemberta.shape}")
        print(f"  ProstT5   : {self.prostt5.shape}")
        print(f"  Global    : {self.global_feat.shape}")
        print(f"  Positive  : {positive}")
        print(f"  Negative  : {negative}")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict:
        return {
            "pepbert": torch.from_numpy(self.pepbert[index]),
            "chemberta": torch.from_numpy(self.chemberta[index]),
            "prostt5": torch.from_numpy(self.prostt5[index]),
            "global": torch.from_numpy(self.global_feat[index]),
            "label": torch.tensor(self.labels[index], dtype=torch.long),
        }


def collate_fn(batch: list[dict]) -> dict:
    return {
        key: torch.stack([item[key] for item in batch])
        for key in batch[0]
    }


# ============================================================================
# FEATURE PREPROCESSING
# ============================================================================

def prepare_batch(
    pepbert: torch.Tensor,
    chemberta: torch.Tensor,
    prostt5: torch.Tensor,
    global_feat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:

    complementary = np.concatenate(
        [
            chemberta.cpu().numpy(),
            prostt5.cpu().numpy(),
            global_feat.cpu().numpy(),
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

    if selected.shape[1] != SELECTOR_OUTPUT_DIM:
        raise ValueError(
            f"Expected {SELECTOR_OUTPUT_DIM} selected features, "
            f"got {selected.shape[1]}"
        )

    return pepbert.float(), torch.from_numpy(selected).float()


# ============================================================================
# MODEL
# ============================================================================

class TLH_AFP_Model_Dual(nn.Module):
    """Two-pathway TLH-AFP model."""

    def __init__(
        self,
        input_dim: int = SELECTED_FEATURES,
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
        self.transformer_norm = nn.LayerNorm(hidden_dim)

        self.complementary_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
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

        self.cross_attn_norm = nn.LayerNorm(hidden_dim)

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

        primary = self.transformer(pepbert)
        primary = self.transformer_proj(primary)
        primary = self.transformer_norm(primary)

        complementary = self.complementary_proj(complementary)
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

        logits = self.classifier(pooled)

        z = F.normalize(
            self.projection(pooled),
            dim=-1,
        )

        return logits, z


# ============================================================================
# LOSSES
# ============================================================================

class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha: float = FOCAL_ALPHA,
        gamma: float = FOCAL_GAMMA,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        bce = F.binary_cross_entropy_with_logits(
            inputs,
            targets,
            reduction="none",
        )

        pt = torch.exp(-bce)

        alpha_t = (
            self.alpha * targets
            + (1.0 - self.alpha) * (1.0 - targets)
        )

        return (
            alpha_t
            * (1.0 - pt).pow(self.gamma)
            * bce
        ).mean()


class SupConLoss(nn.Module):
    def __init__(
        self,
        temperature: float = SUPCON_TEMP,
    ) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:

        features = F.normalize(features, dim=-1)

        similarity = (
            torch.matmul(features, features.T)
            / self.temperature
        )

        labels = labels.contiguous()

        positive_mask = torch.eq(
            labels.unsqueeze(1),
            labels.unsqueeze(0),
        ).float()

        batch_size = features.size(0)
        diag = torch.eye(
            batch_size,
            device=features.device,
        )

        positive_mask *= 1.0 - diag
        logits_mask = 1.0 - diag

        exp_logits = (
            torch.exp(similarity)
            * logits_mask
        )

        log_prob = similarity - torch.log(
            exp_logits.sum(
                dim=1,
                keepdim=True,
            ) + 1e-8
        )

        positive_count = positive_mask.sum(dim=1)
        valid = positive_count > 0

        if not valid.any():
            return torch.zeros(
                (),
                device=features.device,
                requires_grad=True,
            )

        mean_log_prob = (
            (positive_mask * log_prob).sum(dim=1)
            / (positive_count + 1e-8)
        )

        return -mean_log_prob[valid].mean()


# ============================================================================
# CHECKPOINT HELPERS
# ============================================================================

def best_model_path(fold: int) -> Path:
    return CHECKPOINT_DIR / f"best_fold_{fold}.pt"


def resume_path(fold: int) -> Path:
    return CHECKPOINT_DIR / f"resume_fold_{fold}.pt"


def atomic_save(obj, path: Path) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, temp_path)
    temp_path.replace(path)


def save_resume_checkpoint(
    fold: int,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    best_auc: float,
    patience: int,
    best_state: dict | None,
) -> None:

    checkpoint = {
        "fold": fold,
        "epoch": epoch,
        "best_auc": best_auc,
        "patience": patience,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_state_dict": best_state,
        "random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        checkpoint["cuda_random_state"] = (
            torch.cuda.get_rng_state_all()
        )

    atomic_save(
        checkpoint,
        resume_path(fold),
    )


def load_resume_checkpoint(
    fold: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
):
    path = resume_path(fold)

    if not path.exists():
        return 0, -np.inf, 0, None

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    scheduler.load_state_dict(
        checkpoint["scheduler_state_dict"]
    )

    try:
        random.setstate(
            checkpoint["random_state"]
        )
        np.random.set_state(
            checkpoint["numpy_random_state"]
        )
        torch.set_rng_state(
            checkpoint["torch_random_state"]
        )

        if (
            torch.cuda.is_available()
            and "cuda_random_state" in checkpoint
        ):
            torch.cuda.set_rng_state_all(
                checkpoint["cuda_random_state"]
            )
    except Exception as exc:
        print(f"Warning: RNG restoration failed: {exc}")

    return (
        int(checkpoint["epoch"]),
        float(checkpoint["best_auc"]),
        int(checkpoint["patience"]),
        checkpoint.get("best_state_dict"),
    )


# ============================================================================
# PRETRAINED WEIGHTS
# ============================================================================

def load_pretrained(model: nn.Module) -> nn.Module:
    if not PRETRAINED_PATH.exists():
        print("Warning: pretrained contrastive checkpoint not found.")
        return model

    state = torch.load(
        PRETRAINED_PATH,
        map_location="cpu",
        weights_only=False,
    )

    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    model_state = model.state_dict()

    compatible = {
        key: value
        for key, value in state.items()
        if key in model_state
        and model_state[key].shape == value.shape
    }

    model.load_state_dict(
        compatible,
        strict=False,
    )

    print(
        f"Loaded {len(compatible)} compatible "
        f"pretrained parameter tensors."
    )

    return model


# ============================================================================
# VALIDATION
# ============================================================================

@torch.no_grad()
def evaluate_validation(
    model: nn.Module,
    loader: DataLoader,
) -> float:

    model.eval()

    probabilities = []
    labels = []

    for batch in loader:

        pepbert, complementary = prepare_batch(
            batch["pepbert"],
            batch["chemberta"],
            batch["prostt5"],
            batch["global"],
        )

        pepbert = pepbert.to(
            DEVICE,
            non_blocking=True,
        )

        complementary = complementary.to(
            DEVICE,
            non_blocking=True,
        )

        logits, _ = model(
            pepbert,
            complementary,
        )

        probabilities.extend(
            torch.sigmoid(logits)
            .cpu()
            .numpy()
            .ravel()
        )

        labels.extend(
            batch["label"].numpy()
        )

    return roc_auc_score(
        np.asarray(labels),
        np.asarray(probabilities),
    )


# ============================================================================
# DATA
# ============================================================================

dataset = TLHAFPDataset(
    DATA_DIR,
    "train",
)

labels = dataset.labels
indices = np.arange(len(dataset))

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED,
)

fold_splits = list(
    skf.split(indices, labels)
)


# ============================================================================
# FIVE-FOLD TRAINING
# ============================================================================

fold_results = []

for fold, (train_idx, val_idx) in enumerate(
    fold_splits,
    start=1,
):

    print("\n" + "=" * 72)
    print(f"FOLD {fold}/5")
    print("=" * 72)
    print(f"Train samples : {len(train_idx)}")
    print(f"Val samples   : {len(val_idx)}")

    best_path = best_model_path(fold)
    resume_file = resume_path(fold)

    # --------------------------------------------------------------
    # Skip completed fold
    # --------------------------------------------------------------
    if best_path.exists() and not resume_file.exists():

        saved = torch.load(
            best_path,
            map_location="cpu",
            weights_only=False,
        )

        best_auc = float(
            saved["best_auc"]
        )

        fold_results.append(best_auc)

        print(
            f"Completed fold found. "
            f"Best validation AUROC: {best_auc:.4f}"
        )

        continue

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )

    model = TLH_AFP_Model_Dual(
        input_dim=SELECTED_FEATURES,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR * 10,
        epochs=MAX_EPOCHS,
        steps_per_epoch=len(train_loader),
    )

    bce_loss = nn.BCEWithLogitsLoss()
    focal_loss = FocalLoss()
    supcon_loss = SupConLoss()

    start_epoch = 0
    best_auc = -np.inf
    patience = 0
    best_state = None

    # --------------------------------------------------------------
    # Resume or initialize from pretrained weights
    # --------------------------------------------------------------
    if resume_file.exists():

        (
            start_epoch,
            best_auc,
            patience,
            best_state,
        ) = load_resume_checkpoint(
            fold,
            model,
            optimizer,
            scheduler,
        )

        print(
            f"Resuming Fold {fold} from epoch "
            f"{start_epoch + 1}"
        )

    else:
        model = load_pretrained(model)

    # --------------------------------------------------------------
    # Training loop
    # --------------------------------------------------------------
    for epoch in range(
        start_epoch,
        MAX_EPOCHS,
    ):

        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()

        for batch in train_loader:

            pepbert, complementary = prepare_batch(
                batch["pepbert"],
                batch["chemberta"],
                batch["prostt5"],
                batch["global"],
            )

            pepbert = pepbert.to(
                DEVICE,
                non_blocking=True,
            )

            complementary = complementary.to(
                DEVICE,
                non_blocking=True,
            )

            labels_batch = batch["label"].to(
                DEVICE,
                non_blocking=True,
            )

            logits, z = model(
                pepbert,
                complementary,
            )

            targets = labels_batch.float().unsqueeze(1)

            loss_bce = bce_loss(
                logits,
                targets,
            )

            loss_focal = focal_loss(
                logits,
                targets,
            )

            loss_contrast = supcon_loss(
                z,
                labels_batch,
            )

            loss = (
                loss_bce
                + LAMBDA_FOCAL * loss_focal
                + LAMBDA_CONTRAST * loss_contrast
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP,
            )

            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

        current_auc = evaluate_validation(
            model,
            val_loader,
        )

        avg_loss = epoch_loss / max(
            len(train_loader),
            1,
        )

        if current_auc > best_auc:

            best_auc = current_auc
            patience = 0

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        else:
            patience += 1

        elapsed = time.time() - epoch_start

        print(
            f"Fold {fold} | "
            f"Epoch {epoch + 1:03d}/{MAX_EPOCHS} | "
            f"Loss {avg_loss:.4f} | "
            f"Val AUROC {current_auc:.4f} | "
            f"Best {best_auc:.4f} | "
            f"Patience {patience}/{PATIENCE} | "
            f"{elapsed:.1f}s"
        )

        save_resume_checkpoint(
            fold=fold,
            epoch=epoch + 1,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_auc=best_auc,
            patience=patience,
            best_state=best_state,
        )

        if patience >= PATIENCE:
            print(
                f"Early stopping at epoch {epoch + 1}."
            )
            break

    if best_state is None:
        raise RuntimeError(
            f"No best model obtained for Fold {fold}."
        )

    # --------------------------------------------------------------
    # Save best model
    # --------------------------------------------------------------
    atomic_save(
        {
            "model_state_dict": best_state,
            "fold": fold,
            "best_auc": float(best_auc),
            "selected_features": SELECTED_DIM,
            "feature_selector": "Random Forest",
            "selection_threshold": fs_data.get(
                "threshold",
                "mean",
            ),
            "seed": SEED,
        },
        best_path,
    )

    print(
        f"\nBest model for Fold {fold} saved."
    )
    print(
        f"Validation AUROC: {best_auc:.4f}"
    )

    if resume_file.exists():
        resume_file.unlink()

    fold_results.append(
        float(best_auc)
    )

    del model
    del optimizer
    del scheduler
    del train_loader
    del val_loader

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================================
# CROSS-VALIDATION SUMMARY
# ============================================================================

print("\n" + "=" * 72)
print("FIVE-FOLD CROSS-VALIDATION SUMMARY")
print("=" * 72)

for fold, auc in enumerate(
    fold_results,
    start=1,
):
    print(
        f"Fold {fold}: AUROC = {auc:.4f}"
    )

mean_auc = float(
    np.mean(fold_results)
)

std_auc = float(
    np.std(fold_results)
)

print(
    f"\nMean AUROC: {mean_auc:.4f} ± {std_auc:.4f}"
)


# ============================================================================
# SAVE SUMMARY
# ============================================================================

summary_path = CHECKPOINT_DIR / "cv_summary.csv"

pd.DataFrame(
    {
        "Fold": np.arange(
            1,
            len(fold_results) + 1,
        ),
        "Best_AUROC": fold_results,
    }
).to_csv(
    summary_path,
    index=False,
)

print(
    f"CV summary saved: {summary_path}"
)

print("\n" + "=" * 72)
print("TLH-AFP TRAINING COMPLETED")
print("=" * 72)
