# -*- coding: utf-8 -*-
"""
TLH-AFP | Supervised Contrastive Pre-training

This script performs supervised contrastive pre-training of the
TLH-AFP representation encoder using two augmented views of each
peptide representation.

The pretrained weights are used to initialize the final classifier.
"""

from pathlib import Path
import random
import time
import warnings

import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader


warnings.filterwarnings("ignore")


# ============================================================================
# Configuration
# ============================================================================

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data"

FEATURE_SELECTOR_PATH = ROOT / "feature_selector_rf.pkl"

CHECKPOINT_DIR = ROOT / "checkpoints" / "pretrain"
BEST_MODEL_PATH = ROOT / "contrastive_pretrained_best.pt"
FINAL_MODEL_PATH = ROOT / "contrastive_pretrained.pt"

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

EXPECTED_SELECTED_DIM = 547

# Training
BATCH_SIZE = 8
EPOCHS = 100
LR = 1e-5
WEIGHT_DECAY = 0.05
TEMPERATURE = 0.07
PATIENCE = 20
GRAD_CLIP = 1.0

# Model
HIDDEN_DIM = 128
NUM_HEADS = 1
NUM_TRANSFORMER_LAYERS = 2

# Augmentation
AUGMENT_DROP_PROB = 0.10
AUGMENT_NOISE_STD = 0.02


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
# Environment
# ============================================================================

print("=" * 72)
print("TLH-AFP | SUPERVISED CONTRASTIVE PRE-TRAINING")
print("=" * 72)
print(f"Project root : {ROOT}")
print(f"Data dir     : {DATA_DIR}")
print(f"Checkpoint   : {CHECKPOINT_DIR}")
print(f"Device       : {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU          : {torch.cuda.get_device_name(0)}")
    print(
        f"GPU memory   : "
        f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
    )


# ============================================================================
# Feature selector
# ============================================================================

if not FEATURE_SELECTOR_PATH.exists():
    raise FileNotFoundError(
        f"Feature selector not found:\n{FEATURE_SELECTOR_PATH}"
    )

with open(FEATURE_SELECTOR_PATH, "rb") as f:
    fs_data = pickle.load(f)

SELECTOR = fs_data["selector"]
SCALER = fs_data["scaler"]

SELECTED_DIM = int(fs_data["selected_features"])
SELECTOR_INPUT_DIM = int(fs_data["total_features"])

if SELECTOR_INPUT_DIM != COMPLEMENTARY_DIM:
    raise ValueError(
        f"Feature selector expects {SELECTOR_INPUT_DIM} features, "
        f"but current complementary features have {COMPLEMENTARY_DIM}."
    )

if SELECTED_DIM != EXPECTED_SELECTED_DIM:
    raise ValueError(
        f"Expected {EXPECTED_SELECTED_DIM} selected features, "
        f"but selector contains {SELECTED_DIM}."
    )

print("\nFeature dimensions")
print(f"  PepBERT         : {PEPBERT_DIM}")
print(f"  ChemBERTa       : {CHEMBERTA_DIM}")
print(f"  ProstT5         : {PROSTT5_DIM}")
print(f"  Handcrafted     : {GLOBAL_DIM}")
print(f"  Complementary   : {COMPLEMENTARY_DIM}")
print(f"  Selected        : {SELECTED_DIM}")


# ============================================================================
# Required files
# ============================================================================

required_files = [
    DATA_DIR / "train_smiles.csv",
    DATA_DIR / "train_pepbert.npy",
    DATA_DIR / "train_chemberta.npy",
    DATA_DIR / "train_prostt5.npy",
    DATA_DIR / "train_handcrafted_global.npy",
]

print("\nChecking required files...")

for path in required_files:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    print(f"  OK  {path.name}")


# ============================================================================
# Dataset
# ============================================================================

class TLHAFPDataset(Dataset):
    """Training dataset with precomputed peptide representations."""

    def __init__(self, data_dir: Path) -> None:

        self.pepbert = self._load_feature(
            data_dir / "train_pepbert.npy"
        )
        self.chemberta = self._load_feature(
            data_dir / "train_chemberta.npy"
        )
        self.prostt5 = self._load_feature(
            data_dir / "train_prostt5.npy"
        )
        self.global_feat = self._load_feature(
            data_dir / "train_handcrafted_global.npy"
        )

        csv_path = data_dir / "train_smiles.csv"

        df = pd.read_csv(csv_path)

        if "label" not in df.columns:
            raise ValueError(
                f"'label' column not found in {csv_path}"
            )

        self.labels = df["label"].astype(np.int64).to_numpy()

        self._validate()
        self._print_summary()

    @staticmethod
    def _load_feature(path: Path) -> np.ndarray:
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

    def _print_summary(self) -> None:
        positive = int(self.labels.sum())
        negative = len(self.labels) - positive

        print("\nTraining dataset")
        print(f"  Samples       : {len(self.labels)}")
        print(f"  PepBERT       : {self.pepbert.shape}")
        print(f"  ChemBERTa     : {self.chemberta.shape}")
        print(f"  ProstT5       : {self.prostt5.shape}")
        print(f"  Handcrafted   : {self.global_feat.shape}")
        print(f"  Positive      : {positive}")
        print(f"  Negative      : {negative}")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict:
        return {
            "pepbert": torch.from_numpy(self.pepbert[index]),
            "chemberta": torch.from_numpy(self.chemberta[index]),
            "prostt5": torch.from_numpy(self.prostt5[index]),
            "global": torch.from_numpy(self.global_feat[index]),
            "label": torch.tensor(
                self.labels[index],
                dtype=torch.long,
            ),
        }


def collate_fn(batch: list[dict]) -> dict:
    return {
        key: torch.stack([sample[key] for sample in batch])
        for key in batch[0]
    }


# ============================================================================
# Feature selection
# ============================================================================

def apply_feature_selection(
    pepbert: torch.Tensor,
    chemberta: torch.Tensor,
    prostt5: torch.Tensor,
    global_feat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:

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

    return (
        pepbert.float(),
        torch.from_numpy(selected).float(),
    )


# ============================================================================
# Data augmentation
# ============================================================================

def augment_view(
    pepbert: torch.Tensor,
    complementary: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:

    pepbert_aug = (
        pepbert
        + torch.randn_like(pepbert) * AUGMENT_NOISE_STD
    )

    complementary_aug = (
        complementary
        + torch.randn_like(complementary)
        * AUGMENT_NOISE_STD
    )

    pepbert_mask = (
        torch.rand_like(pepbert_aug)
        > AUGMENT_DROP_PROB
    ).float()

    complementary_mask = (
        torch.rand_like(complementary_aug)
        > AUGMENT_DROP_PROB
    ).float()

    pepbert_aug *= pepbert_mask
    complementary_aug *= complementary_mask

    return pepbert_aug, complementary_aug


# ============================================================================
# Encoder
# ============================================================================

class TLH_AFP_Encoder(nn.Module):
    """Shared representation encoder used during pre-training."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = HIDDEN_DIM,
        num_heads: int = NUM_HEADS,
        num_layers: int = NUM_TRANSFORMER_LAYERS,
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
                norm_first=False,
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
    ) -> torch.Tensor:

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

        z = self.projection(
            pooled
        )

        return F.normalize(
            z,
            dim=-1,
        )


# ============================================================================
# Supervised contrastive loss
# ============================================================================

class SupConLoss(nn.Module):
    """Supervised contrastive loss for two augmented views."""

    def __init__(
        self,
        temperature: float = TEMPERATURE,
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

        batch_size, n_views, _ = features.shape

        features = features.reshape(
            batch_size * n_views,
            -1,
        )

        logits = torch.matmul(
            features,
            features.T,
        ) / self.temperature

        logits_max, _ = logits.max(
            dim=1,
            keepdim=True,
        )

        logits = logits - logits_max.detach()

        labels = labels.contiguous().view(-1, 1)

        labels = labels.repeat(
            n_views,
            1,
        )

        mask = torch.eq(
            labels,
            labels.T,
        ).float()

        device = features.device

        logits_mask = (
            1.0
            - torch.eye(
                batch_size * n_views,
                device=device,
            )
        )

        mask *= logits_mask

        exp_logits = (
            torch.exp(logits)
            * logits_mask
        )

        log_prob = (
            logits
            - torch.log(
                exp_logits.sum(
                    dim=1,
                    keepdim=True,
                ) + 1e-12
            )
        )

        positive_count = mask.sum(dim=1)

        mean_log_prob_pos = (
            (mask * log_prob).sum(dim=1)
            / (positive_count + 1e-12)
        )

        valid = positive_count > 0

        if not valid.any():
            return torch.zeros(
                (),
                device=device,
                requires_grad=True,
            )

        return -mean_log_prob_pos[valid].mean()


# ============================================================================
# Dataset
# ============================================================================

dataset = TLHAFPDataset(
    DATA_DIR
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
    collate_fn=collate_fn,
)

print(f"\nBatches per epoch: {len(loader)}")


# ============================================================================
# Model
# ============================================================================

model = TLH_AFP_Encoder(
    input_dim=SELECTED_DIM,
    hidden_dim=HIDDEN_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_TRANSFORMER_LAYERS,
).to(DEVICE)

trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print(f"Trainable parameters: {trainable_parameters:,}")


# ============================================================================
# Optimizer and scheduler
# ============================================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
    eta_min=LR / 100,
)

criterion = SupConLoss(
    temperature=TEMPERATURE
)


# ============================================================================
# Resume checkpoint
# ============================================================================

start_epoch = 0
best_loss = float("inf")
patience_counter = 0

checkpoint_files = list(
    CHECKPOINT_DIR.glob("pretrain_epoch_*.pt")
)

if checkpoint_files:

    latest_checkpoint = max(
        checkpoint_files,
        key=lambda path: int(
            path.stem.split("_")[-1]
        ),
    )

    print(
        f"\nResuming from:\n{latest_checkpoint}"
    )

    checkpoint = torch.load(
        latest_checkpoint,
        map_location=DEVICE,
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

    start_epoch = int(
        checkpoint["epoch"]
    )

    best_loss = float(
        checkpoint["best_loss"]
    )

    patience_counter = int(
        checkpoint["patience_counter"]
    )

    print(
        f"Resume epoch : {start_epoch}"
    )
    print(
        f"Best loss    : {best_loss:.6f}"
    )


# ============================================================================
# Training
# ============================================================================

print("\nStarting supervised contrastive pre-training...")

for epoch in range(
    start_epoch,
    EPOCHS,
):

    model.train()

    epoch_loss = 0.0
    epoch_start = time.time()

    for batch in loader:

        pepbert, complementary = (
            apply_feature_selection(
                batch["pepbert"],
                batch["chemberta"],
                batch["prostt5"],
                batch["global"],
            )
        )

        pepbert_view1, comp_view1 = augment_view(
            pepbert,
            complementary,
        )

        pepbert_view2, comp_view2 = augment_view(
            pepbert,
            complementary,
        )

        pepbert_view1 = pepbert_view1.to(
            DEVICE,
            non_blocking=True,
        )

        comp_view1 = comp_view1.to(
            DEVICE,
            non_blocking=True,
        )

        pepbert_view2 = pepbert_view2.to(
            DEVICE,
            non_blocking=True,
        )

        comp_view2 = comp_view2.to(
            DEVICE,
            non_blocking=True,
        )

        labels = batch["label"].to(
            DEVICE,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        z1 = model(
            pepbert_view1,
            comp_view1,
        )

        z2 = model(
            pepbert_view2,
            comp_view2,
        )

        features = torch.stack(
            [z1, z2],
            dim=1,
        )

        loss = criterion(
            features,
            labels,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP,
        )

        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = (
        epoch_loss
        / max(len(loader), 1)
    )

    scheduler.step()

    elapsed = time.time() - epoch_start

    improved = avg_loss < best_loss

    if improved:
        best_loss = avg_loss
        patience_counter = 0
    else:
        patience_counter += 1

    current_lr = scheduler.get_last_lr()[0]

    print(
        f"Epoch {epoch + 1:03d}/{EPOCHS} | "
        f"SupCon Loss {avg_loss:.6f} | "
        f"Best {best_loss:.6f} | "
        f"LR {current_lr:.3e} | "
        f"{elapsed:.1f}s"
        + (" | BEST" if improved else "")
    )

    checkpoint = {
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss": avg_loss,
        "best_loss": best_loss,
        "patience_counter": patience_counter,
        "selected_features": SELECTED_DIM,
        "pepbert_dim": PEPBERT_DIM,
        "chemberta_dim": CHEMBERTA_DIM,
        "prostt5_dim": PROSTT5_DIM,
        "global_dim": GLOBAL_DIM,
        "seed": SEED,
    }

    checkpoint_path = (
        CHECKPOINT_DIR
        / f"pretrain_epoch_{epoch + 1}.pt"
    )

    torch.save(
        checkpoint,
        checkpoint_path,
    )

    if improved:
        torch.save(
            model.state_dict(),
            BEST_MODEL_PATH,
        )

        print(
            f"  Best model saved: {BEST_MODEL_PATH}"
        )

    if patience_counter >= PATIENCE:
        print(
            f"\nEarly stopping at epoch {epoch + 1}."
        )
        break


# ============================================================================
# Final export
# ============================================================================

torch.save(
    model.state_dict(),
    FINAL_MODEL_PATH,
)

print("\n" + "=" * 72)
print("TLH-AFP | PRE-TRAINING COMPLETED")
print("=" * 72)
print(f"Selected features : {SELECTED_DIM}")
print(f"Best loss         : {best_loss:.6f}")
print(f"Best model        : {BEST_MODEL_PATH}")
print(f"Final model       : {FINAL_MODEL_PATH}")
print("=" * 72)
