# -*- coding: utf-8 -*-
"""
TLH-AFP | Ablation: Without PepBERT

PepBERT is removed while ChemBERTa, ProstT5, handcrafted features,
feature selection, cross-attention, and supervised contrastive learning
remain enabled.
"""

from pathlib import Path
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
from torch.utils.data import DataLoader, Dataset, Subset

from src.models.tlh_afp_model_no_pepbert import TLH_AFP_Model_Dual


warnings.filterwarnings("ignore")


# ============================================================================
# Configuration
# ============================================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

FEATURE_SELECTOR_PATH = (
    ROOT / "feature_selector_rf.pkl"
)

PRETRAINED_PATH = (
    ROOT
    / "checkpoints"
    / "pretrain_no_pepbert"
    / "contrastive_pretrained_no_pepbert_best.pt"
)

CHECKPOINT_DIR = (
    ROOT
    / "checkpoints"
    / "tlh_afp_no_pepbert"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SEED = 42
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Representation dimensions
CHEMBERTA_DIM = 384
PROSTT5_DIM = 1024
GLOBAL_DIM = 488

COMPLEMENTARY_DIM = (
    CHEMBERTA_DIM
    + PROSTT5_DIM
    + GLOBAL_DIM
)

# Training
BATCH_SIZE = 8
MAX_EPOCHS = 200
PATIENCE = 20

LR = 1e-5
WEIGHT_DECAY = 0.05
GRAD_CLIP = 1.0

# Model
HIDDEN_DIM = 128
NUM_HEADS = 1
NUM_LAYERS = 2

# Loss
FOCAL_ALPHA = 0.75
FOCAL_GAMMA = 2.5
LAMBDA_FOCAL = 1.2
LAMBDA_CONTRAST = 0.25
SUPCON_TEMP = 0.07


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
    """Training dataset for the No-PepBERT ablation."""

    def __init__(
        self,
        data_dir: Path,
        split: str = "train",
    ) -> None:

        self.chemberta = self._load(
            data_dir / f"{split}_chemberta.npy"
        )

        self.prostt5 = self._load(
            data_dir / f"{split}_prostt5.npy"
        )

        self.global_feat = self._load(
            data_dir / f"{split}_handcrafted_global.npy"
        )

        label_path = (
            data_dir / f"{split}_smiles.csv"
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
        self._summary(split)

    @staticmethod
    def _load(path: Path) -> np.ndarray:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

        return np.load(path).astype(
            np.float32
        )

    def _validate(self) -> None:
        n = len(self.labels)

        arrays = {
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

    def _summary(self, split: str) -> None:
        positive = int(self.labels.sum())
        negative = len(self.labels) - positive

        print(f"\n[{split}] Dataset")
        print(f"Samples     : {len(self.labels)}")
        print(f"ChemBERTa   : {self.chemberta.shape}")
        print(f"ProstT5     : {self.prostt5.shape}")
        print(f"Handcrafted : {self.global_feat.shape}")
        print(f"Positive    : {positive}")
        print(f"Negative    : {negative}")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict:
        return {
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

def prepare_batch(
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

    scaled = SCALER.transform(
        complementary
    )

    selected = SELECTOR.transform(
        scaled
    )

    if selected.shape[1] != SELECTED_DIM:
        raise ValueError(
            f"Expected {SELECTED_DIM} selected features, "
            f"got {selected.shape[1]}"
        )

    return torch.from_numpy(
        selected
    ).float()


# ============================================================================
# Losses
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

        features = F.normalize(
            features,
            dim=-1,
        )

        similarity = (
            torch.matmul(
                features,
                features.T,
            )
            / self.temperature
        )

        labels = labels.contiguous()

        mask = torch.eq(
            labels.unsqueeze(1),
            labels.unsqueeze(0),
        ).float()

        batch_size = features.size(0)

        diag = torch.eye(
            batch_size,
            device=features.device,
        )

        mask *= 1.0 - diag

        logits_mask = 1.0 - diag

        exp_logits = (
            torch.exp(similarity)
            * logits_mask
        )

        log_prob = (
            similarity
            - torch.log(
                exp_logits.sum(
                    dim=1,
                    keepdim=True,
                ) + 1e-8
            )
        )

        positive_count = mask.sum(
            dim=1
        )

        mean_log_prob_pos = (
            (mask * log_prob).sum(dim=1)
            / (positive_count + 1e-8)
        )

        valid = positive_count > 0

        if not valid.any():
            return torch.zeros(
                (),
                device=features.device,
                requires_grad=True,
            )

        return -mean_log_prob_pos[
            valid
        ].mean()


# ============================================================================
# Checkpoint helpers
# ============================================================================

def best_model_path(fold: int) -> Path:
    return (
        CHECKPOINT_DIR
        / f"best_no_pepbert_fold_{fold}.pt"
    )


def resume_path(fold: int) -> Path:
    return (
        CHECKPOINT_DIR
        / f"resume_no_pepbert_fold_{fold}.pt"
    )


def atomic_save(
    obj,
    path: Path,
) -> None:

    temp_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    torch.save(
        obj,
        temp_path,
    )

    temp_path.replace(path)


# ============================================================================
# Pretrained initialization
# ============================================================================

def load_pretrained(
    model: nn.Module,
) -> nn.Module:

    if not PRETRAINED_PATH.exists():
        raise FileNotFoundError(
            f"No-PepBERT pretrained checkpoint not found:\n"
            f"{PRETRAINED_PATH}"
        )

    checkpoint = torch.load(
        PRETRAINED_PATH,
        map_location="cpu",
        weights_only=False,
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state = checkpoint[
            "model_state_dict"
        ]
    else:
        state = checkpoint

    model_state = model.state_dict()

    compatible = {
        key: value
        for key, value in state.items()
        if (
            key in model_state
            and model_state[key].shape == value.shape
        )
    }

    model.load_state_dict(
        compatible,
        strict=False,
    )

    print(
        f"Loaded {len(compatible)} compatible "
        f"No-PepBERT pretrained tensors."
    )

    return model


# ============================================================================
# Validation
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

        complementary = prepare_batch(
            batch["chemberta"],
            batch["prostt5"],
            batch["global"],
        )

        logits, _ = model(
            complementary.to(
                DEVICE,
                non_blocking=True,
            )
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
# Feature selector
# ============================================================================

if not FEATURE_SELECTOR_PATH.exists():
    raise FileNotFoundError(
        f"Feature selector not found:\n"
        f"{FEATURE_SELECTOR_PATH}"
    )

with open(
    FEATURE_SELECTOR_PATH,
    "rb",
) as f:
    fs_data = pickle.load(f)

SELECTOR = fs_data["selector"]
SCALER = fs_data["scaler"]

SELECTED_DIM = int(
    fs_data["selected_features"]
)

TOTAL_FS_DIM = int(
    fs_data["total_features"]
)

if TOTAL_FS_DIM != COMPLEMENTARY_DIM:
    raise ValueError(
        f"Selector expects {TOTAL_FS_DIM} features, "
        f"but current data has {COMPLEMENTARY_DIM}."
    )


# ============================================================================
# Fixed five-fold cross-validation
# ============================================================================

dataset = TLHAFPDataset(
    DATA_DIR,
    "train",
)

labels = dataset.labels
indices = np.arange(
    len(dataset)
)

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED,
)

fold_splits = list(
    skf.split(
        indices,
        labels,
    )
)


# ============================================================================
# Training
# ============================================================================

fold_results = []

for fold, (train_idx, val_idx) in enumerate(
    fold_splits,
    start=1,
):

    print("\n" + "=" * 72)
    print(
        f"FOLD {fold}/5 | WITHOUT PepBERT"
    )
    print("=" * 72)

    best_path = best_model_path(fold)
    resume_file = resume_path(fold)

    # --------------------------------------------------------------
    # Existing completed fold
    # --------------------------------------------------------------
    if (
        best_path.exists()
        and not resume_file.exists()
    ):

        saved = torch.load(
            best_path,
            map_location="cpu",
            weights_only=False,
        )

        if saved.get(
            "pepbert",
            True,
        ) is not False:
            raise ValueError(
                "Existing checkpoint is not a No-PepBERT model."
            )

        best_auc = float(
            saved["best_auc"]
        )

        fold_results.append(
            best_auc
        )

        print(
            f"Completed fold found. "
            f"Best validation AUROC: {best_auc:.6f}"
        )

        continue

    train_loader = DataLoader(
        Subset(
            dataset,
            train_idx,
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        Subset(
            dataset,
            val_idx,
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )

    model = TLH_AFP_Model_Dual(
        input_dim=SELECTED_DIM,
        mode="finetune",
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        hidden_dim=HIDDEN_DIM,
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

        checkpoint = torch.load(
            resume_file,
            map_location="cpu",
            weights_only=False,
        )

        if checkpoint.get(
            "pepbert",
            True,
        ) is not False:
            raise ValueError(
                "Resume checkpoint is not a No-PepBERT model."
            )

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

        scheduler.load_state_dict(
            checkpoint[
                "scheduler_state_dict"
            ]
        )

        start_epoch = int(
            checkpoint["epoch"]
        )

        best_auc = float(
            checkpoint["best_auc"]
        )

        patience = int(
            checkpoint["patience"]
        )

        best_state = checkpoint.get(
            "best_state_dict"
        )

        print(
            f"Resuming Fold {fold} "
            f"from epoch {start_epoch + 1}"
        )

    else:
        model = load_pretrained(
            model
        )

    # --------------------------------------------------------------
    # Epoch loop
    # --------------------------------------------------------------
    for epoch in range(
        start_epoch,
        MAX_EPOCHS,
    ):

        model.train()

        total_loss = 0.0
        epoch_start = time.time()

        for batch in train_loader:

            complementary = prepare_batch(
                batch["chemberta"],
                batch["prostt5"],
                batch["global"],
            )

            complementary = complementary.to(
                DEVICE,
                non_blocking=True,
            )

            targets = batch["label"].to(
                DEVICE,
                non_blocking=True,
            )

            logits, z = model(
                complementary
            )

            targets_float = (
                targets.float()
                .unsqueeze(1)
            )

            loss_bce = bce_loss(
                logits,
                targets_float,
            )

            loss_focal = focal_loss(
                logits,
                targets_float,
            )

            loss_contrast = supcon_loss(
                z,
                targets,
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

            total_loss += loss.item()

        current_auc = evaluate_validation(
            model,
            val_loader,
        )

        avg_loss = total_loss / max(
            len(train_loader),
            1,
        )

        if current_auc > best_auc:

            best_auc = current_auc
            patience = 0

            best_state = {
                key: value.detach()
                .cpu()
                .clone()
                for key, value
                in model.state_dict().items()
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

        save_resume_checkpoint = {
            "fold": fold,
            "epoch": epoch + 1,
            "best_auc": float(best_auc),
            "patience": patience,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_state_dict": best_state,
            "selected_features": SELECTED_DIM,
            "pepbert": False,
            "feature_selection": True,
            "cross_attention": True,
            "contrastive": True,
            "seed": SEED,
        }

        atomic_save(
            save_resume_checkpoint,
            resume_file,
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

    atomic_save(
        {
            "model_state_dict": best_state,
            "fold": fold,
            "best_auc": float(best_auc),
            "selected_features": SELECTED_DIM,
            "pepbert": False,
            "feature_selector": "Random Forest",
            "feature_selection": True,
            "cross_attention": True,
            "contrastive": True,
            "loss": "BCE + 1.2*Focal + 0.25*SupCon",
            "seed": SEED,
        },
        best_path,
    )

    print(
        f"\nBest Fold {fold} model saved."
    )
    print(
        f"Validation AUROC: {best_auc:.6f}"
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

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================================
# Summary
# ============================================================================

mean_auc = float(
    np.mean(fold_results)
)

std_auc = float(
    np.std(fold_results)
)

print("\n" + "=" * 72)
print("FIVE-FOLD SUMMARY | WITHOUT PepBERT")
print("=" * 72)

for fold, auc in enumerate(
    fold_results,
    start=1,
):
    print(
        f"Fold {fold}: AUROC = {auc:.6f}"
    )

print(
    f"\nMean AUROC: {mean_auc:.6f} ± {std_auc:.6f}"
)

summary_path = (
    CHECKPOINT_DIR
    / "cv_summary_no_pepbert.csv"
)

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
    f"Summary saved: {summary_path}"
)

print("\n" + "=" * 72)
print("NO-PepBERT ABLATION COMPLETED")
print("=" * 72)
