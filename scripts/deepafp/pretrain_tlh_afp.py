import os
import time
import random
import pickle
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
warnings.filterwarnings('ignore')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(ROOT, 'data', 'deepafp')
SCRIPTS_DIR = SCRIPT_DIR
CV_FS_DIR = os.path.join(SCRIPT_DIR, 'cv_feature_selection')
CV_PRETRAIN_DIR = os.path.join(SCRIPT_DIR, 'cv_pretraining_esm2_query')
os.makedirs(CV_PRETRAIN_DIR, exist_ok=True)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HIDDEN_DIM = 128
NUM_HEADS = 1
NUM_TRANSFORMER_LAYERS = 2
DROPOUT = 0.2
TRANSFORMER_DROPOUT = 0.2
CROSS_ATTENTION_DROPOUT = 0.1
TRANSFORMER_FEEDFORWARD_DIM = 4096
BATCH_SIZE = 8
EPOCHS = 100
LR = 1e-05
WEIGHT_DECAY = 0.05
TEMPERATURE = 0.07
PATIENCE = 20
GRAD_CLIP = 1.0
AUGMENT_DROP_PROB = 0.1
AUGMENT_NOISE_STD = 0.02
ESM2_MODEL_NAME = 'facebook/esm2_t30_150M_UR50D'
TRAIN_ESM2_TOKENS = os.path.join(DATA_DIR, 'DeepAFP_main_train_esm2_t30_150m_tokens.npy')
TRAIN_ESM2_LENGTHS = os.path.join(DATA_DIR, 'DeepAFP_main_train_esm2_t30_150m_lengths.npy')
print('=' * 80)
print('TLH-AFP 5-FOLD SUPERVISED CONTRASTIVE PRE-TRAINING')
print('SEQUENCE-AWARE ESM-2 QUERY + 3-TOKEN CROSS-ATTENTION')
print('=' * 80)
print(f'Project Root : {ROOT}')
print(f'Data Dir     : {DATA_DIR}')
print(f'CV FS Dir    : {CV_FS_DIR}')
print(f'CV PT Dir    : {CV_PRETRAIN_DIR}')
print(f'Device       : {DEVICE}')
print(f'ESM-2 Model  : {ESM2_MODEL_NAME}')
if torch.cuda.is_available():
    print(f'GPU          : {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory   : {torch.cuda.get_device_properties(0).total_memory / 1000000000.0:.2f} GB')
print('\n' + '=' * 80)
print('CHECKING REQUIRED FILES')
print('=' * 80)
required_global_files = {'train_smiles.csv': os.path.join(DATA_DIR, 'DeepAFP-main-train_with_smiles.csv'), 'DeepAFP_main_train_esm2_t30_150m_tokens.npy': TRAIN_ESM2_TOKENS, 'DeepAFP_main_train_esm2_t30_150m_lengths.npy': TRAIN_ESM2_LENGTHS, 'cv_splits.pkl': os.path.join(CV_FS_DIR, 'cv_splits.pkl')}
for name, path in required_global_files.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f'\n❌ Required file not found:\n{path}')
    print(f' {name}')
print('\n' + '=' * 80)
print('LOADING LABELS AND SEQUENCES')
print('=' * 80)
train_df = pd.read_csv(os.path.join(DATA_DIR, 'DeepAFP-main-train_with_smiles.csv'))
if 'label' not in train_df.columns:
    raise ValueError("Column 'label' not found in train_smiles.csv.")
if 'sequence' not in train_df.columns:
    raise ValueError("Column 'sequence' not found in train_smiles.csv.")
labels_all = train_df['label'].astype(np.int64).values
sequences_all = train_df['sequence'].astype(str).values
sequence_lengths_csv = np.asarray([len(seq) for seq in sequences_all], dtype=np.int64)
print(f'Samples  : {len(labels_all)}')
print(f'Positive : {np.sum(labels_all == 1)}')
print(f'Negative : {np.sum(labels_all == 0)}')
unique_labels = np.unique(labels_all)
if not np.array_equal(unique_labels, np.array([0, 1])):
    raise ValueError(f'Labels must be [0, 1], but found {unique_labels}')
print('\n' + '=' * 80)
print('LOADING ESM-2 QUERY EMBEDDINGS')
print('=' * 80)
esm2_all = np.load(TRAIN_ESM2_TOKENS, mmap_mode='r')
esm2_lengths_all = np.load(TRAIN_ESM2_LENGTHS, mmap_mode='r')
if esm2_all.ndim != 3:
    raise ValueError(f'ESM-2 token embeddings must be 3D. Found shape {esm2_all.shape}')
if esm2_lengths_all.ndim != 1:
    raise ValueError(f'ESM-2 lengths must be 1D. Found shape {esm2_lengths_all.shape}')
QUERY_SEQUENCE_LENGTH = esm2_all.shape[1]
QUERY_EMBEDDING_DIM = esm2_all.shape[2]
print(f'ESM-2 shape : {esm2_all.shape}')
print(f'Sequence axis : {QUERY_SEQUENCE_LENGTH}')
print(f'Embedding dim : {QUERY_EMBEDDING_DIM}')
if esm2_all.shape[0] != len(labels_all):
    raise ValueError('ESM-2 sample count does not match labels.')
if esm2_lengths_all.shape[0] != len(labels_all):
    raise ValueError('ESM-2 length count does not match labels.')
esm2_lengths_int = np.asarray(esm2_lengths_all).astype(np.int64)
if np.any(esm2_lengths_int <= 0):
    raise ValueError('ESM-2 sequence lengths must be > 0.')
if np.any(esm2_lengths_int > QUERY_SEQUENCE_LENGTH):
    raise ValueError('ESM-2 sequence length exceeds stored token dimension.')
if not np.array_equal(esm2_lengths_int, sequence_lengths_csv):
    mismatch = np.where(esm2_lengths_int != sequence_lengths_csv)[0]
    raise ValueError(f'ESM-2 lengths do not match actual sequence lengths.\nMismatches: {len(mismatch)}\nExamples: {mismatch[:10].tolist()}')
print(' ESM-2 lengths match CSV sequences.')
print(f'Min length : {np.min(esm2_lengths_int)}')
print(f'Max length : {np.max(esm2_lengths_int)}')
print(f'Mean length: {np.mean(esm2_lengths_int):.2f}')
print('\n Checking ESM-2 for NaN / Inf...')
for start in range(0, len(labels_all), 64):
    end = min(start + 64, len(labels_all))
    chunk = np.asarray(esm2_all[start:end])
    if not np.isfinite(chunk).all():
        raise ValueError(f'ESM-2 contains NaN/Inf in samples {start}:{end}')
print(' ESM-2 finite-value check passed.')
print('\n' + '=' * 80)
print('LOADING CV SPLITS')
print('=' * 80)
splits_path = os.path.join(CV_FS_DIR, 'cv_splits.pkl')
with open(splits_path, 'rb') as f:
    cv_splits = pickle.load(f)
if not isinstance(cv_splits, list):
    raise ValueError('cv_splits.pkl must contain a list of fold dictionaries.')
N_FOLDS = len(cv_splits)
if N_FOLDS < 2:
    raise ValueError(f'Invalid number of folds: {N_FOLDS}')
print(f'Detected folds : {N_FOLDS}')
for fold_idx, split in enumerate(cv_splits, start=1):
    if not isinstance(split, dict):
        raise ValueError(f'Fold {fold_idx} split must be a dictionary.')
    if 'train_idx' not in split or 'val_idx' not in split:
        raise KeyError(f'Fold {fold_idx} missing train_idx or val_idx.')
print('\n Checking feature-selection folds...')
for fold_number in range(1, N_FOLDS + 1):
    fold_dir = os.path.join(CV_FS_DIR, f'fold_{fold_number}')
    if not os.path.isdir(fold_dir):
        raise FileNotFoundError(f'\n❌ Fold directory not found:\n{fold_dir}')
    required_fold_files = ['feature_selector_rf.pkl', 'train_selected.npy', 'val_selected.npy', 'fold_indices.npz']
    for filename in required_fold_files:
        path = os.path.join(fold_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f'\n❌ Missing fold file:\n{path}')
    print(f' fold_{fold_number}')

def load_fold_indices(fold_dir):
    indices_path = os.path.join(fold_dir, 'fold_indices.npz')
    data = np.load(indices_path)
    if 'train_idx' not in data or 'val_idx' not in data:
        raise KeyError(f'Invalid fold indices file:\n{indices_path}')
    train_idx = data['train_idx'].astype(np.int64)
    val_idx = data['val_idx'].astype(np.int64)
    return (train_idx, val_idx)
print('\n' + '=' * 80)
print('VALIDATING CV FOLD INDICES')
print('=' * 80)
all_validation_indices = []
for fold_number in range(1, N_FOLDS + 1):
    fold_dir = os.path.join(CV_FS_DIR, f'fold_{fold_number}')
    train_idx_file, val_idx_file = load_fold_indices(fold_dir)
    train_idx_pkl = np.asarray(cv_splits[fold_number - 1]['train_idx']).astype(np.int64)
    val_idx_pkl = np.asarray(cv_splits[fold_number - 1]['val_idx']).astype(np.int64)
    if not np.array_equal(train_idx_file, train_idx_pkl):
        raise ValueError(f'Fold {fold_number}: fold_indices.npz and cv_splits.pkl train indices do not match.')
    if not np.array_equal(val_idx_file, val_idx_pkl):
        raise ValueError(f'Fold {fold_number}: fold_indices.npz and cv_splits.pkl validation indices do not match.')
    if len(train_idx_file) == 0:
        raise ValueError(f'Fold {fold_number} has empty training set.')
    if len(val_idx_file) == 0:
        raise ValueError(f'Fold {fold_number} has empty validation set.')
    if len(np.unique(train_idx_file)) != len(train_idx_file):
        raise ValueError(f'Fold {fold_number}: duplicate train indices.')
    if len(np.unique(val_idx_file)) != len(val_idx_file):
        raise ValueError(f'Fold {fold_number}: duplicate validation indices.')
    overlap = np.intersect1d(train_idx_file, val_idx_file)
    if len(overlap) != 0:
        raise ValueError(f'Fold {fold_number}: train/validation overlap detected.')
    if train_idx_file.min() < 0 or train_idx_file.max() >= len(labels_all):
        raise ValueError(f'Fold {fold_number}: train indices out of range.')
    if val_idx_file.min() < 0 or val_idx_file.max() >= len(labels_all):
        raise ValueError(f'Fold {fold_number}: validation indices out of range.')
    all_validation_indices.extend(val_idx_file.tolist())
    print(f'Fold {fold_number}: train={len(train_idx_file)} | val={len(val_idx_file)}')
all_validation_indices = np.asarray(all_validation_indices, dtype=np.int64)
if len(all_validation_indices) != len(labels_all):
    raise ValueError('Validation folds do not contain exactly all samples.')
if len(np.unique(all_validation_indices)) != len(labels_all):
    raise ValueError('Validation folds overlap.')
if not np.array_equal(np.sort(all_validation_indices), np.arange(len(labels_all))):
    raise ValueError('Validation folds do not form a complete partition.')
print(' CV validation partition verified.')

class CVPretrainDataset(Dataset):

    def __init__(self, esm2_tokens, sequence_lengths, chemberta, prostt5, handcrafted, labels):
        self.esm2_tokens = esm2_tokens
        self.sequence_lengths = sequence_lengths
        self.chemberta = chemberta
        self.prostt5 = prostt5
        self.handcrafted = handcrafted
        self.labels = labels
        n = len(labels)
        if esm2_tokens.shape[0] != n:
            raise ValueError('ESM-2 sample count mismatch.')
        if sequence_lengths.shape[0] != n:
            raise ValueError('Sequence-length count mismatch.')
        if chemberta.shape[0] != n:
            raise ValueError('ChemBERTa sample count mismatch.')
        if prostt5.shape[0] != n:
            raise ValueError('ProstT5 sample count mismatch.')
        if handcrafted.shape[0] != n:
            raise ValueError('Handcrafted sample count mismatch.')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {'esm2': torch.from_numpy(np.asarray(self.esm2_tokens[idx], dtype=np.float32)), 'length': torch.tensor(self.sequence_lengths[idx], dtype=torch.long), 'chemberta': torch.from_numpy(self.chemberta[idx]), 'prostt5': torch.from_numpy(self.prostt5[idx]), 'handcrafted': torch.from_numpy(self.handcrafted[idx]), 'label': torch.tensor(self.labels[idx], dtype=torch.long)}

def augment_view(esm2, lengths, chemberta, prostt5, handcrafted):
    esm2_aug = esm2.clone()
    chemberta_aug = chemberta.clone()
    prostt5_aug = prostt5.clone()
    handcrafted_aug = handcrafted.clone()
    batch_size = esm2.shape[0]
    seq_len = esm2.shape[1]
    positions = torch.arange(seq_len, device=esm2.device).unsqueeze(0)
    valid_mask = positions < lengths.unsqueeze(1)
    valid_mask_float = valid_mask.unsqueeze(-1).float()
    esm2_aug = esm2_aug + torch.randn_like(esm2_aug) * AUGMENT_NOISE_STD * valid_mask_float
    chemberta_aug = chemberta_aug + torch.randn_like(chemberta_aug) * AUGMENT_NOISE_STD
    prostt5_aug = prostt5_aug + torch.randn_like(prostt5_aug) * AUGMENT_NOISE_STD
    handcrafted_aug = handcrafted_aug + torch.randn_like(handcrafted_aug) * AUGMENT_NOISE_STD
    esm2_keep = (torch.rand_like(esm2_aug) > AUGMENT_DROP_PROB).float()
    esm2_aug *= esm2_keep
    esm2_aug *= valid_mask_float
    chem_keep = (torch.rand_like(chemberta_aug) > AUGMENT_DROP_PROB).float()
    chemberta_aug *= chem_keep
    prost_keep = (torch.rand_like(prostt5_aug) > AUGMENT_DROP_PROB).float()
    prostt5_aug *= prost_keep
    handcrafted_keep = (torch.rand_like(handcrafted_aug) > AUGMENT_DROP_PROB).float()
    handcrafted_aug *= handcrafted_keep
    return (esm2_aug, lengths, chemberta_aug, prostt5_aug, handcrafted_aug)

class TLH_AFP_Model_SequenceQuery(nn.Module):

    def __init__(self, query_dim, chem_dim, prost_dim, handcrafted_dim, hidden_dim=128, num_heads=1, num_layers=2):
        super().__init__()
        self.query_transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=query_dim, nhead=num_heads, dim_feedforward=TRANSFORMER_FEEDFORWARD_DIM, dropout=TRANSFORMER_DROPOUT, activation='gelu', batch_first=True, norm_first=False), num_layers=num_layers)
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.chemberta_proj = nn.Sequential(nn.Linear(chem_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(DROPOUT))
        self.prostt5_proj = nn.Sequential(nn.Linear(prost_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(DROPOUT))
        self.handcrafted_proj = nn.Sequential(nn.Linear(handcrafted_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(DROPOUT))
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=CROSS_ATTENTION_DROPOUT, batch_first=True)
        self.cross_attn_norm = nn.LayerNorm(hidden_dim)
        self.projection = nn.Sequential(nn.Linear(hidden_dim, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(DROPOUT), nn.Linear(64, 32))

    def forward(self, esm2, lengths, chemberta, prostt5, handcrafted):
        batch_size = esm2.shape[0]
        seq_len = esm2.shape[1]
        positions = torch.arange(seq_len, device=esm2.device).unsqueeze(0)
        valid_mask = positions < lengths.unsqueeze(1)
        padding_mask = ~valid_mask
        x = self.query_transformer(esm2, src_key_padding_mask=padding_mask)
        x = self.query_proj(x)
        x = self.query_norm(x)
        x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        chem_token = self.chemberta_proj(chemberta).unsqueeze(1)
        prost_token = self.prostt5_proj(prostt5).unsqueeze(1)
        handcrafted_token = self.handcrafted_proj(handcrafted).unsqueeze(1)
        kv_tokens = torch.cat([chem_token, prost_token, handcrafted_token], dim=1)
        attn_out, _ = self.cross_attn(query=x, key=kv_tokens, value=kv_tokens)
        attn_out = self.cross_attn_norm(x + attn_out)
        attn_out = attn_out.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        valid_float = valid_mask.unsqueeze(-1).float()
        summed = (attn_out * valid_float).sum(dim=1)
        valid_counts = valid_float.sum(dim=1).clamp(min=1.0)
        pooled = summed / valid_counts
        z = self.projection(pooled)
        z = F.normalize(z, dim=-1)
        return z

class SupConLoss(nn.Module):

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        n_views = features.shape[1]
        features = F.normalize(features, dim=-1)
        features_flat = features.reshape(batch_size * n_views, -1)
        logits = torch.matmul(features_flat, features_flat.T) / self.temperature
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        labels = labels.contiguous().view(-1)
        labels_views = labels.repeat_interleave(n_views).view(-1, 1)
        mask = torch.eq(labels_views, labels_views.T).float().to(device)
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size * n_views, device=device)
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
        positive_count = mask.sum(dim=1)
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (positive_count + 1e-12)
        loss = -mean_log_prob_pos.mean()
        return loss

def find_latest_checkpoint(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        return None
    candidates = []
    for filename in os.listdir(checkpoint_dir):
        if not filename.startswith('pretrain_epoch_'):
            continue
        if not filename.endswith('.pt'):
            continue
        try:
            epoch = int(filename.split('_')[-1].split('.')[0])
        except ValueError:
            continue
        candidates.append((epoch, os.path.join(checkpoint_dir, filename)))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]

def load_fold_feature_selection(fold_number):
    fold_fs_dir = os.path.join(CV_FS_DIR, f'fold_{fold_number}')
    selector_path = os.path.join(fold_fs_dir, 'feature_selector_rf.pkl')
    selected_train_path = os.path.join(fold_fs_dir, 'train_selected.npy')
    selected_val_path = os.path.join(fold_fs_dir, 'val_selected.npy')
    with open(selector_path, 'rb') as f:
        fs_data = pickle.load(f)
    if 'selected_mask' not in fs_data:
        raise KeyError(f'selected_mask not found in:\n{selector_path}')
    selected_mask = np.asarray(fs_data['selected_mask']).astype(bool)
    total_features_from_mask = len(selected_mask)
    if 'feature_ranges' not in fs_data:
        raise KeyError(f'feature_ranges not found in:\n{selector_path}')
    feature_ranges = fs_data['feature_ranges']
    if 'selected_feature_counts' in fs_data:
        saved_selected_counts = {key: int(value) for key, value in fs_data['selected_feature_counts'].items()}
    else:
        saved_selected_counts = None
    selected_counts = {}
    for name, (start, end) in feature_ranges.items():
        if start < 0 or end > total_features_from_mask or start >= end:
            raise ValueError(f'Invalid feature range for {name} in fold {fold_number}.')
        selected_counts[name] = int(np.sum(selected_mask[start:end]))
    selected_dim = int(np.sum(selected_mask))
    if selected_dim <= 0:
        raise ValueError(f'Fold {fold_number}: zero selected features.')
    if saved_selected_counts is not None:
        if selected_counts != saved_selected_counts:
            raise ValueError(f'Fold {fold_number}: saved selected-feature counts do not match selected mask.')
    selected_train = np.load(selected_train_path, mmap_mode='r')
    selected_val = np.load(selected_val_path, mmap_mode='r')
    if selected_train.ndim != 2:
        raise ValueError(f'Fold {fold_number}: train_selected.npy must be 2D.')
    if selected_val.ndim != 2:
        raise ValueError(f'Fold {fold_number}: val_selected.npy must be 2D.')
    if selected_train.shape[1] != selected_dim:
        raise ValueError(f'Fold {fold_number}: selected train dimension mismatch.\nMatrix: {selected_train.shape[1]}\nMask:   {selected_dim}')
    if selected_val.shape[1] != selected_dim:
        raise ValueError(f'Fold {fold_number}: selected validation dimension mismatch.')
    return {'fold_fs_dir': fold_fs_dir, 'selected_mask': selected_mask, 'feature_ranges': feature_ranges, 'selected_counts': selected_counts, 'selected_dim': selected_dim, 'selected_train': selected_train, 'selected_val': selected_val, 'selector_path': selector_path, 'selected_train_path': selected_train_path, 'selected_val_path': selected_val_path}

def split_selected_features(selected_train, selected_counts):
    chemberta_selected = selected_counts['ChemBERTa']
    prostt5_selected = selected_counts['ProstT5']
    handcrafted_selected = selected_counts['Handcrafted_488']
    chem_end = chemberta_selected
    prost_end = chemberta_selected + prostt5_selected
    chemberta_train = np.asarray(selected_train[:, :chem_end], dtype=np.float32)
    prostt5_train = np.asarray(selected_train[:, chem_end:prost_end], dtype=np.float32)
    handcrafted_train = np.asarray(selected_train[:, prost_end:], dtype=np.float32)
    if chemberta_train.shape[1] != chemberta_selected:
        raise RuntimeError('ChemBERTa selected dimension mismatch.')
    if prostt5_train.shape[1] != prostt5_selected:
        raise RuntimeError('ProstT5 selected dimension mismatch.')
    if handcrafted_train.shape[1] != handcrafted_selected:
        raise RuntimeError('Handcrafted selected dimension mismatch.')
    return (chemberta_train, prostt5_train, handcrafted_train)

def run_fold(fold_number):
    print('\n')
    print('#' * 80)
    print(f'PRE-TRAINING FOLD {fold_number}/{N_FOLDS}')
    print('#' * 80)
    fold_seed = SEED + fold_number
    random.seed(fold_seed)
    np.random.seed(fold_seed)
    torch.manual_seed(fold_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(fold_seed)
    fold_output_dir = os.path.join(CV_PRETRAIN_DIR, f'fold_{fold_number}')
    checkpoint_dir = os.path.join(fold_output_dir, 'checkpoints')
    os.makedirs(fold_output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    fs = load_fold_feature_selection(fold_number)
    selected_mask = fs['selected_mask']
    selected_counts = fs['selected_counts']
    selected_dim = fs['selected_dim']
    selected_train = fs['selected_train']
    fold_fs_dir = fs['fold_fs_dir']
    train_idx, val_idx = load_fold_indices(fold_fs_dir)
    if selected_train.shape[0] != len(train_idx):
        raise ValueError(f'Fold {fold_number}: train_selected row count mismatch.')
    labels_train = labels_all[train_idx].copy()
    esm2_train = np.asarray(esm2_all[train_idx], dtype=np.float32)
    lengths_train = esm2_lengths_int[train_idx].copy()
    chemberta_train, prostt5_train, handcrafted_train = split_selected_features(selected_train, selected_counts)
    if not np.isfinite(esm2_train).all():
        raise ValueError(f'Fold {fold_number}: ESM-2 train contains NaN/Inf.')
    if not np.isfinite(chemberta_train).all():
        raise ValueError(f'Fold {fold_number}: ChemBERTa selected features contain NaN/Inf.')
    if not np.isfinite(prostt5_train).all():
        raise ValueError(f'Fold {fold_number}: ProstT5 selected features contain NaN/Inf.')
    if not np.isfinite(handcrafted_train).all():
        raise ValueError(f'Fold {fold_number}: Handcrafted selected features contain NaN/Inf.')
    print(f'\nTrain samples : {len(train_idx)}')
    print(f'Validation samples : {len(val_idx)}')
    print(f'Train positive : {np.sum(labels_train == 1)}')
    print(f'Train negative : {np.sum(labels_train == 0)}')
    print('\nSelected complementary features:')
    print(f"  ChemBERTa   : {selected_counts['ChemBERTa']}")
    print(f"  ProstT5     : {selected_counts['ProstT5']}")
    print(f"  Handcrafted : {selected_counts['Handcrafted_488']}")
    print(f'  Total       : {selected_dim}')
    print('\nESM-2 Query:')
    print(f'  Train shape : {esm2_train.shape}')
    print(f'  Embedding dim : {QUERY_EMBEDDING_DIM}')
    print(f'  Sequence axis : {QUERY_SEQUENCE_LENGTH}')
    dataset = CVPretrainDataset(esm2_tokens=esm2_train, sequence_lengths=lengths_train, chemberta=chemberta_train, prostt5=prostt5_train, handcrafted=handcrafted_train, labels=labels_train)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0, pin_memory=torch.cuda.is_available())
    if len(loader) == 0:
        raise ValueError(f'Fold {fold_number}: zero batches.')
    print(f'\nBatches / epoch : {len(loader)}')
    print('\n Creating sequence-aware 3-token cross-attention model...')
    model = TLH_AFP_Model_SequenceQuery(query_dim=QUERY_EMBEDDING_DIM, chem_dim=selected_counts['ChemBERTa'], prost_dim=selected_counts['ProstT5'], handcrafted_dim=selected_counts['Handcrafted_488'], hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS, num_layers=NUM_TRANSFORMER_LAYERS).to(DEVICE)
    num_parameters = sum((p.numel() for p in model.parameters() if p.requires_grad))
    print(f'Parameters : {num_parameters:,}')
    print('\n Cross-Attention:')
    print(f'  Query : ESM-2 sequence tokens [B, {QUERY_SEQUENCE_LENGTH}, {HIDDEN_DIM}]')
    print('  K/V   : 3 complementary tokens')
    print('    1. ChemBERTa')
    print('    2. ProstT5')
    print('    3. Handcrafted')
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR / 100)
    criterion = SupConLoss(temperature=TEMPERATURE)
    start_epoch = 0
    best_loss = float('inf')
    patience_counter = 0
    loss_history = []
    latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
    if latest_checkpoint is not None:
        print('\n Existing checkpoint detected:')
        print(latest_checkpoint)
        checkpoint = torch.load(latest_checkpoint, map_location=DEVICE, weights_only=False)
        compatible = checkpoint.get('architecture', '') == 'three_token_cross_attention_sequence_query_cv' and int(checkpoint.get('fold_number', -1)) == fold_number and (int(checkpoint.get('query_embedding_dimension', -1)) == QUERY_EMBEDDING_DIM) and (int(checkpoint.get('query_sequence_length', -1)) == QUERY_SEQUENCE_LENGTH) and (int(checkpoint.get('selected_features', -1)) == selected_dim) and (int(checkpoint.get('chemberta_selected', -1)) == selected_counts['ChemBERTa']) and (int(checkpoint.get('prostt5_selected', -1)) == selected_counts['ProstT5']) and (int(checkpoint.get('handcrafted_selected', -1)) == selected_counts['Handcrafted_488'])
        if compatible:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = int(checkpoint['epoch'])
            best_loss = float(checkpoint['best_loss'])
            patience_counter = int(checkpoint['patience_counter'])
            loss_history = checkpoint.get('loss_history', [])
            print(f' Resumed from epoch {start_epoch}')
        else:
            print(' Existing checkpoint is incompatible with the current ESM-2 sequence-aware architecture.')
            print('➡ Starting from epoch 1.')
    print('\n Starting fold pre-training...')
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss = 0.0
        start_time = time.time()
        for batch in loader:
            esm2 = batch['esm2'].to(DEVICE, non_blocking=True)
            lengths = batch['length'].to(DEVICE, non_blocking=True)
            chemberta = batch['chemberta'].to(DEVICE, non_blocking=True)
            prostt5 = batch['prostt5'].to(DEVICE, non_blocking=True)
            handcrafted = batch['handcrafted'].to(DEVICE, non_blocking=True)
            labels = batch['label'].to(DEVICE, non_blocking=True)
            esm2_v1, lengths_v1, chem_v1, prost_v1, handcrafted_v1 = augment_view(esm2, lengths, chemberta, prostt5, handcrafted)
            esm2_v2, lengths_v2, chem_v2, prost_v2, handcrafted_v2 = augment_view(esm2, lengths, chemberta, prostt5, handcrafted)
            optimizer.zero_grad(set_to_none=True)
            z1 = model(esm2_v1, lengths_v1, chem_v1, prost_v1, handcrafted_v1)
            z2 = model(esm2_v2, lengths_v2, chem_v2, prost_v2, handcrafted_v2)
            features = torch.stack([z1, z2], dim=1)
            loss = criterion(features, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss detected in fold {fold_number}, epoch {epoch + 1}.')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        scheduler.step()
        elapsed = time.time() - start_time
        improved = avg_loss < best_loss
        if improved:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
        loss_history.append({'epoch': epoch + 1, 'loss': avg_loss, 'best_loss': best_loss, 'learning_rate': scheduler.get_last_lr()[0]})
        print(f'Fold {fold_number} | Epoch {epoch + 1:03d}/{EPOCHS} | SupCon Loss: {avg_loss:.6f} | Best: {best_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.3e} | Time: {elapsed:.1f}s' + ('  BEST' if improved else ''))
        checkpoint = {'architecture': 'three_token_cross_attention_sequence_query_cv', 'fold_number': fold_number, 'epoch': epoch + 1, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'loss': avg_loss, 'best_loss': best_loss, 'patience_counter': patience_counter, 'loss_history': loss_history, 'query_embedding_model': ESM2_MODEL_NAME, 'query_embedding_dimension': QUERY_EMBEDDING_DIM, 'query_sequence_length': QUERY_SEQUENCE_LENGTH, 'query_is_sequence_aware': True, 'query_feature_selection': False, 'chemberta_selected': selected_counts['ChemBERTa'], 'prostt5_selected': selected_counts['ProstT5'], 'handcrafted_selected': selected_counts['Handcrafted_488'], 'selected_features': selected_dim, 'num_kv_tokens': 3, 'hidden_dim': HIDDEN_DIM, 'num_heads': NUM_HEADS, 'num_transformer_layers': NUM_TRANSFORMER_LAYERS, 'transformer_feedforward_dim': TRANSFORMER_FEEDFORWARD_DIM, 'temperature': TEMPERATURE, 'batch_size': BATCH_SIZE, 'learning_rate': LR, 'weight_decay': WEIGHT_DECAY, 'augmentation_drop_prob': AUGMENT_DROP_PROB, 'augmentation_noise_std': AUGMENT_NOISE_STD, 'seed': fold_seed, 'train_samples': len(train_idx), 'validation_samples': len(val_idx), 'train_positive': int(np.sum(labels_train == 1)), 'train_negative': int(np.sum(labels_train == 0)), 'feature_selection_directory': fold_fs_dir, 'selected_mask_dimension': len(selected_mask), 'selected_feature_counts': selected_counts}
        checkpoint_path = os.path.join(checkpoint_dir, f'pretrain_epoch_{epoch + 1}.pt')
        torch.save(checkpoint, checkpoint_path)
        if improved:
            best_model_path = os.path.join(fold_output_dir, 'contrastive_pretrained_best.pt')
            torch.save(model.state_dict(), best_model_path)
            print(f'⭐ Best model saved:\n   {best_model_path}')
        if patience_counter >= PATIENCE:
            print(f'\n Early stopping at epoch {epoch + 1}')
            break
    final_model_path = os.path.join(fold_output_dir, 'contrastive_pretrained.pt')
    torch.save(model.state_dict(), final_model_path)
    loss_history_df = pd.DataFrame(loss_history)
    loss_history_path = os.path.join(fold_output_dir, 'pretraining_loss_history.csv')
    loss_history_df.to_csv(loss_history_path, index=False)
    metadata = {'architecture': 'three_token_cross_attention_sequence_query_cv', 'fold_number': fold_number, 'query_embedding_model': ESM2_MODEL_NAME, 'query_embedding_dimension': QUERY_EMBEDDING_DIM, 'query_sequence_length': QUERY_SEQUENCE_LENGTH, 'query_actual_min_length': int(np.min(lengths_train)), 'query_actual_max_length': int(np.max(lengths_train)), 'query_actual_mean_length': float(np.mean(lengths_train)), 'query_is_sequence_aware': True, 'query_feature_selection': False, 'chemberta_selected': selected_counts['ChemBERTa'], 'prostt5_selected': selected_counts['ProstT5'], 'handcrafted_selected': selected_counts['Handcrafted_488'], 'selected_features': selected_dim, 'num_kv_tokens': 3, 'hidden_dim': HIDDEN_DIM, 'num_heads': NUM_HEADS, 'num_transformer_layers': NUM_TRANSFORMER_LAYERS, 'transformer_feedforward_dim': TRANSFORMER_FEEDFORWARD_DIM, 'best_loss': best_loss, 'train_samples': len(train_idx), 'validation_samples': len(val_idx), 'train_positive': int(np.sum(labels_train == 1)), 'train_negative': int(np.sum(labels_train == 0)), 'temperature': TEMPERATURE, 'batch_size': BATCH_SIZE, 'learning_rate': LR, 'weight_decay': WEIGHT_DECAY, 'augmentation_drop_prob': AUGMENT_DROP_PROB, 'augmentation_noise_std': AUGMENT_NOISE_STD, 'seed': fold_seed, 'test_data_loaded': False, 'validation_data_used_for_pretraining': False, 'fold_specific_feature_selection': True, 'feature_selection_artifact': os.path.join(fold_fs_dir, 'feature_selector_rf.pkl')}
    metadata_path = os.path.join(fold_output_dir, 'pretrain_metadata.pkl')
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print('\n' + '=' * 80)
    print(f' FOLD {fold_number} PRE-TRAINING COMPLETED')
    print('=' * 80)
    print(f'Selected total : {selected_dim}')
    print(f"  ChemBERTa    : {selected_counts['ChemBERTa']}")
    print(f"  ProstT5      : {selected_counts['ProstT5']}")
    print(f"  Handcrafted  : {selected_counts['Handcrafted_488']}")
    print(f'Query         : ESM-2 {QUERY_EMBEDDING_DIM}D × {QUERY_SEQUENCE_LENGTH} tokens')
    print(f'Best loss     : {best_loss:.6f}')
    print(f"Best model    : {os.path.join(fold_output_dir, 'contrastive_pretrained_best.pt')}")
    print(f'Final model   : {final_model_path}')
    print(f'Loss history  : {loss_history_path}')
    print('=' * 80)
    return {'fold': fold_number, 'train_samples': len(train_idx), 'validation_samples': len(val_idx), 'chemberta_selected': selected_counts['ChemBERTa'], 'prostt5_selected': selected_counts['ProstT5'], 'handcrafted_selected': selected_counts['Handcrafted_488'], 'selected_total': selected_dim, 'query_embedding_dimension': QUERY_EMBEDDING_DIM, 'query_sequence_length': QUERY_SEQUENCE_LENGTH, 'best_loss': best_loss}
results = []
for fold_number in range(1, N_FOLDS + 1):
    result = run_fold(fold_number)
    results.append(result)
summary_df = pd.DataFrame(results)
summary_path = os.path.join(CV_PRETRAIN_DIR, 'cv_pretraining_summary.csv')
summary_df.to_csv(summary_path, index=False)
print('\n')
print('=' * 80)
print(' ALL FOLD PRE-TRAINING COMPLETED')
print('=' * 80)
print('\n Pre-training summary:')
print(summary_df.to_string(index=False))
print(f'\nSummary saved to:\n{summary_path}')
print('\n DATA ISOLATION:')
print('  Test data were NOT loaded.')
print('  Validation samples were NOT used for pre-training.')
print('  Each fold used only its own RF-selected training features.')
print('\n QUERY:')
print(f'  Model      : {ESM2_MODEL_NAME}')
print(f'  Dimension  : {QUERY_EMBEDDING_DIM}')
print(f'  Max tokens : {QUERY_SEQUENCE_LENGTH}')
print('  Sequence-aware : YES')
print('  Feature selection on Query : NO')
print('\n CROSS-ATTENTION:')
print('  Query = ESM-2 residue/token sequence')
print('  K/V   = 3 complementary modality tokens')
print('    1. ChemBERTa')
print('    2. ProstT5')
print('    3. Handcrafted')
print('\n Output directory:')
print(CV_PRETRAIN_DIR)
print('=' * 80)
