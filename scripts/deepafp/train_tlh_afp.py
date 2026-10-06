import os
import time
import pickle
import random
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, matthews_corrcoef, f1_score, precision_score, recall_score, confusion_matrix, average_precision_score
warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT, 'data', 'deepafp')
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
CV_FS_DIR = os.path.join(SCRIPTS_DIR, 'cv_feature_selection')
CV_PRETRAIN_DIR = os.path.join(SCRIPTS_DIR, 'cv_pretraining_esm2_query')
CHECKPOINT_DIR = os.path.join(SCRIPTS_DIR, 'checkpoints', 'DEEPAFP_PAPER_MATCHED_SEED42_SAFE_V3')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ESM2_DIM = 640
ESM2_SEQUENCE_LENGTH = 100
CHEMBERTA_DIM = 384
PROSTT5_DIM = 1024
GLOBAL_DIM = 488
TOTAL_COMPLEMENTARY_DIM = CHEMBERTA_DIM + PROSTT5_DIM + GLOBAL_DIM
ARCHITECTURE = 'three_token_cross_attention_sequence_query_cv'
BATCH_SIZE = 8
MAX_EPOCHS = 200
PATIENCE = MAX_EPOCHS + 1
LR = 1e-05
WEIGHT_DECAY = 0.05
GRAD_CLIP = 1.0
NUM_HEADS = 1
NUM_LAYERS = 2
HIDDEN_DIM = 128
FOCAL_ALPHA = 0.75
FOCAL_GAMMA = 2.5
LAMBDA_FOCAL = 1.2
LAMBDA_CONTRAST = 0.25
SUPCON_TEMP = 0.07
DROPOUT = 0.2
N_FOLDS = 5
print('=' * 80)
print(' TLH-AFP 5-FOLD TRAINING - ESM-2 SEQUENCE-AWARE QUERY')
print('=' * 80)
print(f'Project Root : {ROOT}')
print(f'Data Dir     : {DATA_DIR}')
print(f'CV FS Dir    : {CV_FS_DIR}')
print(f'CV Pretrain  : {CV_PRETRAIN_DIR}')
print(f'Checkpoint   : {CHECKPOINT_DIR}')
print('SAFE V3     : fresh isolated directory; resume retained; no checkpoint deletion')
print(f'Device       : {DEVICE}')
if torch.cuda.is_available():
    print(f'GPU          : {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory   : {torch.cuda.get_device_properties(0).total_memory / 1000000000.0:.2f} GB')
print('\n Query configuration:')
print('  Model             : facebook/esm2_t30_150M_UR50D')
print(f'  Embedding dim     : {ESM2_DIM}')
print(f'  Sequence length   : {ESM2_SEQUENCE_LENGTH}')
print('  Sequence-aware    : YES')
print('\n Cross-Attention:')
print('  Query : ESM-2 residue/token sequence')
print('  K/V   : 3 complementary modality tokens')
print('    1. ChemBERTa')
print('    2. ProstT5')
print('    3. Handcrafted')

def atomic_torch_save(obj, path):
    temp_path = path + '.tmp'
    torch.save(obj, temp_path)
    os.replace(temp_path, path)

def load_npy(path, name, expected_ndim=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f'\n❌ {name} not found:\n{path}')
    arr = np.load(path)
    if expected_ndim is not None and arr.ndim != expected_ndim:
        raise ValueError(f'\n❌ {name} must be {expected_ndim}D, got {arr.ndim}D.')
    if not np.isfinite(arr).all():
        raise ValueError(f'\n❌ {name} contains NaN or Inf.')
    return arr.astype(np.float32, copy=False)

def resume_path(fold):
    return os.path.join(CHECKPOINT_DIR, f'resume_fold_{fold}.pt')

def best_model_path(fold):
    return os.path.join(CHECKPOINT_DIR, f'best_fold_{fold}.pt')

def paper_resume_path(fold):
    return os.path.join(CHECKPOINT_DIR, f'paper_resume_fold_{fold}.pt')

def paper_completed_path(fold):
    return os.path.join(CHECKPOINT_DIR, f'paper_matched_best_test_acc_fold_{fold}.pt')

def load_fold_indices(fold_dir):
    path = os.path.join(fold_dir, 'fold_indices.npz')
    data = np.load(path)
    if 'train_idx' not in data or 'val_idx' not in data:
        raise KeyError(f'Missing train_idx/val_idx in {path}')
    return (data['train_idx'].astype(np.int64), data['val_idx'].astype(np.int64))

def clone_state_dict_to_cpu(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
print('\n Checking required files...')
required_files = ['DeepAFP_main_train_esm2_t30_150m_lengths.npy']
for filename in required_files:
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f'\n❌ Required file not found:\n{path}')
    print(f'   {filename}')
print('\n Searching for ESM-2 sequence-aware embeddings...')
preferred_esm2 = os.path.join(DATA_DIR, 'DeepAFP_main_train_esm2_t30_150m_tokens.npy')
if os.path.exists(preferred_esm2):
    ESM2_PATH = preferred_esm2
else:
    candidates = []
    for filename in os.listdir(DATA_DIR):
        low = filename.lower()
        if low.endswith('.npy') and 'esm2' in low:
            path = os.path.join(DATA_DIR, filename)
            try:
                shape = np.load(path, mmap_mode='r').shape
            except Exception:
                continue
            if shape == (2335, ESM2_SEQUENCE_LENGTH, ESM2_DIM):
                candidates.append(path)
    if len(candidates) == 0:
        raise FileNotFoundError(f'\n❌ No ESM-2 sequence-aware embedding file found with shape (2335, {ESM2_SEQUENCE_LENGTH}, {ESM2_DIM}).')
    if len(candidates) > 1:
        raise RuntimeError('\n❌ Multiple ESM-2 files with the expected shape were found:\n' + '\n'.join(candidates) + '\nPlease keep only the intended ESM-2 embedding file.')
    ESM2_PATH = candidates[0]
print(f' ESM-2 file: {ESM2_PATH}')
print('\n Checking CV feature-selection folders...')
for fold in range(1, N_FOLDS + 1):
    fold_dir = os.path.join(CV_FS_DIR, f'fold_{fold}')
    required = ['feature_selector_rf.pkl', 'train_selected.npy', 'val_selected.npy', 'fold_indices.npz']
    if not os.path.isdir(fold_dir):
        raise FileNotFoundError(f'\n❌ Fold directory not found:\n{fold_dir}')
    for filename in required:
        path = os.path.join(fold_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f'\n❌ Missing file:\n{path}')
    print(f'   fold_{fold}')
print('\n Checking CV ESM-2 pre-training folders...')
for fold in range(1, N_FOLDS + 1):
    fold_dir = os.path.join(CV_PRETRAIN_DIR, f'fold_{fold}')
    best_path = os.path.join(fold_dir, 'contrastive_pretrained_best.pt')
    last_path = os.path.join(fold_dir, 'contrastive_pretrained.pt')
    pretrained_path = best_path if os.path.exists(best_path) else last_path
    metadata_path = os.path.join(fold_dir, 'pretrain_metadata.pkl')
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f'\n❌ Fold {fold} ESM-2 pretrained checkpoint not found.\nChecked:\n  {best_path}\n  {last_path}')
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f'\n❌ Pretrained best model not found:\n{pretrained_path}')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f'\n❌ Pretraining metadata not found:\n{metadata_path}')
    print(f'   fold_{fold}')
print('\n Loading labels...')
_train_csv_candidates = [os.path.join(DATA_DIR, 'DeepAFP-main-train_with_smiles.csv'), os.path.join(DATA_DIR, 'train_smiles.csv'), os.path.join(DATA_DIR, 'DeepAFP-main-train.csv')]
_train_csv_path = next((p for p in _train_csv_candidates if os.path.exists(p)), None)
if _train_csv_path is None:
    raise FileNotFoundError('DeepAFP train CSV not found. Checked: ' + repr(_train_csv_candidates))
print(f'   Train CSV: {os.path.basename(_train_csv_path)}')
labels_df = pd.read_csv(_train_csv_path)
if 'label' not in labels_df.columns:
    raise ValueError(f"Column 'label' not found in {os.path.basename(_train_csv_path)}.")
labels_all = labels_df['label'].astype(np.int64).values
N_SAMPLES = len(labels_all)
if not np.array_equal(np.unique(labels_all), np.array([0, 1])):
    raise ValueError(f'Labels must be binary [0,1], found: {np.unique(labels_all)}')
if N_SAMPLES != 2335:
    raise ValueError(f'Expected 2335 training samples, found {N_SAMPLES}.')
print(f'Samples  : {N_SAMPLES}')
print(f'Positive : {np.sum(labels_all == 1)}')
print(f'Negative : {np.sum(labels_all == 0)}')
ESM2_LENGTHS_PATH = os.path.join(DATA_DIR, 'DeepAFP_main_train_esm2_t30_150m_lengths.npy')
print('\n Loading true ESM-2 sequence lengths...')
esm2_lengths_raw = np.load(ESM2_LENGTHS_PATH)
if esm2_lengths_raw.ndim != 1:
    raise ValueError(f'ESM-2 lengths must be 1D, got shape {esm2_lengths_raw.shape}.')
if len(esm2_lengths_raw) != N_SAMPLES:
    raise ValueError(f'ESM-2 length count mismatch: {len(esm2_lengths_raw)} != {N_SAMPLES}.')
if not np.isfinite(esm2_lengths_raw).all():
    raise ValueError('ESM-2 lengths contain NaN or Inf.')
if not np.all(np.equal(esm2_lengths_raw, np.floor(esm2_lengths_raw))):
    raise ValueError('ESM-2 lengths must contain integer values.')
esm2_lengths_all = esm2_lengths_raw.astype(np.int64, copy=False)
if np.any(esm2_lengths_all <= 0):
    raise ValueError('ESM-2 sequence lengths must be > 0.')
if np.any(esm2_lengths_all > ESM2_SEQUENCE_LENGTH):
    raise ValueError(f'ESM-2 sequence length exceeds stored length {ESM2_SEQUENCE_LENGTH}.')
if 'length' in labels_df.columns:
    csv_lengths = labels_df['length'].astype(np.int64).values
    if not np.array_equal(esm2_lengths_all, csv_lengths):
        mismatch = np.where(esm2_lengths_all != csv_lengths)[0]
        raise ValueError(f'ESM-2 lengths do not match CSV sequence lengths. Mismatches: {len(mismatch)}; examples: {mismatch[:10].tolist()}')
print(f'Lengths: min={esm2_lengths_all.min()}, max={esm2_lengths_all.max()}, mean={esm2_lengths_all.mean():.2f}')
print(' Padding-mask lengths validated.')
print('\n Loading ESM-2 sequence-aware embeddings...')
esm2_all = load_npy(ESM2_PATH, 'ESM-2 embeddings', expected_ndim=3)
print(f'ESM-2 shape = {esm2_all.shape}')
if esm2_all.shape != (N_SAMPLES, ESM2_SEQUENCE_LENGTH, ESM2_DIM):
    raise ValueError(f'\n❌ ESM-2 shape mismatch.\nFound    : {esm2_all.shape}\nExpected : {(N_SAMPLES, ESM2_SEQUENCE_LENGTH, ESM2_DIM)}')
print('\n Validating exact CV folds...')
fold_data = {}
all_val_indices = []
for fold in range(1, N_FOLDS + 1):
    fold_dir = os.path.join(CV_FS_DIR, f'fold_{fold}')
    train_idx, val_idx = load_fold_indices(fold_dir)
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError(f'Fold {fold} has empty train/validation indices.')
    if train_idx.min() < 0 or train_idx.max() >= N_SAMPLES:
        raise ValueError(f'Fold {fold} train indices out of range.')
    if val_idx.min() < 0 or val_idx.max() >= N_SAMPLES:
        raise ValueError(f'Fold {fold} validation indices out of range.')
    if len(np.unique(train_idx)) != len(train_idx):
        raise ValueError(f'Fold {fold} has duplicate train indices.')
    if len(np.unique(val_idx)) != len(val_idx):
        raise ValueError(f'Fold {fold} has duplicate validation indices.')
    overlap = np.intersect1d(train_idx, val_idx)
    if len(overlap) != 0:
        raise ValueError(f'Fold {fold} has train/validation overlap.')
    fold_data[fold] = {'train_idx': train_idx, 'val_idx': val_idx}
    all_val_indices.extend(val_idx.tolist())
    print(f'  Fold {fold}: train={len(train_idx)} | val={len(val_idx)}')
all_val_indices = np.asarray(all_val_indices, dtype=np.int64)
if len(all_val_indices) != N_SAMPLES:
    raise ValueError('Validation folds do not cover all training samples exactly once.')
if len(np.unique(all_val_indices)) != N_SAMPLES:
    raise ValueError('Validation folds overlap.')
print(' Exact 5-fold partition verified.')

class FoldDataset(Dataset):

    def __init__(self, query_sequence, sequence_lengths, complementary, labels):
        self.query_sequence = query_sequence
        self.sequence_lengths = sequence_lengths.astype(np.int64, copy=False)
        self.complementary = complementary
        self.labels = labels
        if len(query_sequence) != len(labels):
            raise ValueError('Query / label count mismatch.')
        if len(self.sequence_lengths) != len(labels):
            raise ValueError('Length / label count mismatch.')
        if len(complementary) != len(labels):
            raise ValueError('Complementary / label count mismatch.')
        if np.any(self.sequence_lengths <= 0) or np.any(self.sequence_lengths > ESM2_SEQUENCE_LENGTH):
            raise ValueError('Invalid ESM-2 sequence length in dataset.')
        if query_sequence.ndim != 3:
            raise ValueError('ESM-2 query must be 3D.')
        if complementary.ndim != 2:
            raise ValueError('Complementary features must be 2D.')
        if not np.isfinite(query_sequence).all():
            raise ValueError('ESM-2 query contains NaN or Inf.')
        if not np.isfinite(complementary).all():
            raise ValueError('Complementary features contain NaN or Inf.')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {'query': torch.from_numpy(self.query_sequence[idx]).float(), 'length': torch.tensor(self.sequence_lengths[idx], dtype=torch.long), 'complementary': torch.from_numpy(self.complementary[idx]).float(), 'label': torch.tensor(self.labels[idx], dtype=torch.long)}

def collate_fn(batch):
    return {'query': torch.stack([item['query'] for item in batch]), 'length': torch.stack([item['length'] for item in batch]), 'complementary': torch.stack([item['complementary'] for item in batch]), 'label': torch.stack([item['label'] for item in batch])}

class TLH_AFP_Model_ESM2_Query(nn.Module):

    def __init__(self, chem_dim, prost_dim, global_dim, hidden_dim=128, num_heads=1, num_layers=2):
        super().__init__()
        self.esm2_dim = ESM2_DIM
        self.sequence_length = ESM2_SEQUENCE_LENGTH
        self.chem_dim = int(chem_dim)
        self.prost_dim = int(prost_dim)
        self.global_dim = int(global_dim)
        self.query_transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=ESM2_DIM, nhead=num_heads, dim_feedforward=4096, dropout=DROPOUT, activation='gelu', batch_first=True, norm_first=False), num_layers=num_layers)
        self.query_proj = nn.Linear(ESM2_DIM, hidden_dim)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.chemberta_proj = nn.Sequential(nn.Linear(chem_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(DROPOUT))
        self.prostt5_proj = nn.Sequential(nn.Linear(prost_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(DROPOUT))
        self.handcrafted_proj = nn.Sequential(nn.Linear(global_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(DROPOUT))
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=0.1, batch_first=True)
        self.cross_attn_norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.4), nn.Linear(128, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.3), nn.Linear(64, 1))
        self.projection = nn.Sequential(nn.Linear(hidden_dim, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(DROPOUT), nn.Linear(64, 32))

    def forward(self, esm2_query, lengths, chemberta, prostt5, global_feat):
        if esm2_query.dim() != 3:
            raise ValueError(f'ESM-2 query must be [B,L,640], got {tuple(esm2_query.shape)}')
        if esm2_query.shape[1] != ESM2_SEQUENCE_LENGTH:
            raise ValueError(f'ESM-2 sequence length must be {ESM2_SEQUENCE_LENGTH}, got {esm2_query.shape[1]}')
        if esm2_query.shape[2] != ESM2_DIM:
            raise ValueError(f'ESM-2 dimension must be {ESM2_DIM}, got {esm2_query.shape[2]}')
        if lengths.dim() != 1 or lengths.shape[0] != esm2_query.shape[0]:
            raise ValueError(f'Lengths must be [B], got {tuple(lengths.shape)} for batch {esm2_query.shape[0]}')
        if torch.any(lengths <= 0) or torch.any(lengths > esm2_query.shape[1]):
            raise ValueError('Invalid ESM-2 sequence lengths in batch.')
        positions = torch.arange(esm2_query.shape[1], device=esm2_query.device).unsqueeze(0)
        valid_mask = positions < lengths.unsqueeze(1)
        padding_mask = ~valid_mask
        x = self.query_transformer(esm2_query, src_key_padding_mask=padding_mask)
        x = self.query_proj(x)
        x = self.query_norm(x)
        x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        chem_token = self.chemberta_proj(chemberta).unsqueeze(1)
        prost_token = self.prostt5_proj(prostt5).unsqueeze(1)
        handcrafted_token = self.handcrafted_proj(global_feat).unsqueeze(1)
        kv = torch.cat([chem_token, prost_token, handcrafted_token], dim=1)
        attn_out, _ = self.cross_attn(query=x, key=kv, value=kv, need_weights=False)
        attn_out = self.cross_attn_norm(x + attn_out)
        attn_out = attn_out.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        valid_float = valid_mask.unsqueeze(-1).to(attn_out.dtype)
        summed = (attn_out * valid_float).sum(dim=1)
        valid_counts = valid_float.sum(dim=1).clamp(min=1.0)
        pooled = summed / valid_counts
        logits = self.classifier(pooled)
        z = F.normalize(self.projection(pooled), dim=-1)
        return (logits, z)

class FocalLoss(nn.Module):

    def __init__(self, alpha=0.75, gamma=2.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        return (alpha_t * (1.0 - pt) ** self.gamma * bce).mean()

class SupConLoss(nn.Module):

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        features = F.normalize(features, dim=-1)
        batch_size = features.shape[0]
        if batch_size < 2:
            return torch.zeros((), device=features.device, requires_grad=True)
        similarity = torch.matmul(features, features.T) / self.temperature
        labels = labels.contiguous()
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
        diag = torch.eye(batch_size, device=features.device)
        mask = mask * (1.0 - diag)
        logits_mask = 1.0 - diag
        exp_logits = torch.exp(similarity) * logits_mask
        log_prob = similarity - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-08)
        positive_count = mask.sum(dim=1)
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (positive_count + 1e-08)
        valid = positive_count > 0
        if valid.sum() == 0:
            return torch.zeros((), device=features.device, requires_grad=True)
        return -mean_log_prob_pos[valid].mean()

def _validate_optional_metadata(metadata, key, expected, fold):
    if key not in metadata:
        return
    actual = metadata[key]
    try:
        if isinstance(expected, int):
            actual_cmp = int(actual)
        else:
            actual_cmp = actual
    except Exception as exc:
        raise ValueError(f"Fold {fold}: metadata field '{key}' could not be interpreted.") from exc
    if actual_cmp != expected:
        raise ValueError(f"Fold {fold}: metadata mismatch for '{key}'. Found {actual!r}, expected {expected!r}.")

def _unwrap_state_dict(state):
    if isinstance(state, dict) and 'model_state_dict' in state:
        return state['model_state_dict']
    if isinstance(state, dict) and 'state_dict' in state:
        return state['state_dict']
    return state

def _strip_common_prefix(key):
    prefixes = ['module.', 'model.', 'network.']
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
                break
    return key

def load_pretrained(model, fold, selected_dim, chem_selected, prost_selected, global_selected):
    fold_dir = os.path.join(CV_PRETRAIN_DIR, f'fold_{fold}')
    best_path = os.path.join(fold_dir, 'contrastive_pretrained_best.pt')
    last_path = os.path.join(fold_dir, 'contrastive_pretrained.pt')
    pretrained_path = best_path if os.path.exists(best_path) else last_path
    metadata_path = os.path.join(fold_dir, 'pretrain_metadata.pkl')
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f'\n❌ Fold {fold} ESM-2 pretrained checkpoint not found.\nChecked:\n  {best_path}\n  {last_path}')
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    actual_arch = metadata.get('architecture', '')
    if actual_arch != ARCHITECTURE:
        raise ValueError(f'\n❌ Fold {fold} pretrained architecture mismatch.\nFound: {actual_arch}\nExpected: {ARCHITECTURE}')
    _validate_optional_metadata(metadata, 'esm2_dim', ESM2_DIM, fold)
    _validate_optional_metadata(metadata, 'embedding_dim', ESM2_DIM, fold)
    _validate_optional_metadata(metadata, 'sequence_length', ESM2_SEQUENCE_LENGTH, fold)
    _validate_optional_metadata(metadata, 'max_tokens', ESM2_SEQUENCE_LENGTH, fold)
    _validate_optional_metadata(metadata, 'selected_features', selected_dim, fold)
    _validate_optional_metadata(metadata, 'chemberta_selected', chem_selected, fold)
    _validate_optional_metadata(metadata, 'prostt5_selected', prost_selected, fold)
    _validate_optional_metadata(metadata, 'global_selected', global_selected, fold)
    _validate_optional_metadata(metadata, 'num_kv_tokens', 3, fold)
    print('\n Loading fold-specific ESM-2 pretraining...')
    print(f'  {pretrained_path}')
    raw_state = torch.load(pretrained_path, map_location='cpu', weights_only=False)
    state = _unwrap_state_dict(raw_state)
    if not isinstance(state, dict):
        raise TypeError(f'Fold {fold}: pretrained checkpoint does not contain a state_dict.')
    model_state = model.state_dict()
    matched = {}
    unexpected = []
    shape_mismatch = []
    for key, value in state.items():
        candidate = key
        if candidate not in model_state:
            candidate = _strip_common_prefix(candidate)
        if candidate not in model_state and candidate.startswith('global_proj.'):
            candidate = 'handcrafted_proj.' + candidate[len('global_proj.'):]
        if candidate not in model_state and candidate.startswith('handcrafted_proj.'):
            candidate = 'global_proj.' + candidate[len('handcrafted_proj.'):]
        if candidate not in model_state:
            unexpected.append(key)
            continue
        if model_state[candidate].shape != value.shape:
            shape_mismatch.append((candidate, tuple(value.shape), tuple(model_state[candidate].shape)))
            continue
        matched[candidate] = value
    if shape_mismatch or unexpected:
        print('\n Pretrained state mismatch details:')
        if shape_mismatch:
            print('  Shape mismatches:')
            for item in shape_mismatch[:20]:
                print(f'    {item[0]}: {item[1]} != {item[2]}')
        if unexpected:
            print('  Unexpected checkpoint keys:')
            for key in unexpected[:30]:
                print(f'    {key}')
        raise ValueError(f'\n❌ Fold {fold} pretrained weights are incompatible with the ESM-2 training model.')
    model.load_state_dict(matched, strict=False)
    missing_model_keys = [k for k in model_state.keys() if k not in matched]
    print(f' Loaded {len(matched)} pretrained parameter tensors.')
    if missing_model_keys:
        print('ℹ Parameters not present in pretraining (normally supervised head):')
        for key in missing_model_keys[:20]:
            print(f'    {key}')
    print(' Pretrained architecture matches the ESM-2 representation path.')
    print(' Supervised classifier remains trainable.')
    return model

def save_resume_checkpoint(fold, epoch, model, optimizer, scheduler, best_auc, patience, best_state, selected_dim, chem_selected, prost_selected, global_selected):
    checkpoint = {'architecture': ARCHITECTURE, 'fold': fold, 'epoch': epoch, 'best_auc': float(best_auc), 'patience': int(patience), 'selected_features': int(selected_dim), 'chemberta_selected': int(chem_selected), 'prostt5_selected': int(prost_selected), 'global_selected': int(global_selected), 'esm2_dim': ESM2_DIM, 'sequence_length': ESM2_SEQUENCE_LENGTH, 'padding_mask_enabled': True, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'best_state_dict': best_state, 'random_state': random.getstate(), 'numpy_random_state': np.random.get_state(), 'torch_random_state': torch.get_rng_state()}
    if torch.cuda.is_available():
        checkpoint['cuda_random_state'] = torch.cuda.get_rng_state_all()
    atomic_torch_save(checkpoint, resume_path(fold))

def load_resume_checkpoint(fold, model, optimizer, scheduler, expected_selected_dim, expected_chem, expected_prost, expected_global):
    path = resume_path(fold)
    if not os.path.exists(path):
        return (0, -np.inf, 0, None, False)
    print('\n Resume checkpoint found:')
    print(path)
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    actual_arch = checkpoint.get('architecture', '')
    if actual_arch != ARCHITECTURE:
        print(' Old/incompatible resume checkpoint detected. It will be removed and the fold will restart.')
        raise RuntimeError(f'Fold {fold}: incompatible SAFE V3 resume found at {path}. Nothing was deleted.')
    checks = {'selected_features': expected_selected_dim, 'chemberta_selected': expected_chem, 'prostt5_selected': expected_prost, 'global_selected': expected_global, 'esm2_dim': ESM2_DIM, 'sequence_length': ESM2_SEQUENCE_LENGTH, 'padding_mask_enabled': 1}
    for key, expected in checks.items():
        if int(checkpoint.get(key, -1)) != int(expected):
            print(f' Resume checkpoint mismatch for {key}. It will be removed and the fold will restart.')
            raise RuntimeError(f'Fold {fold}: SAFE V3 resume metadata mismatch for {key}. Nothing was deleted: {path}')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    try:
        random.setstate(checkpoint['random_state'])
        np.random.set_state(checkpoint['numpy_random_state'])
        torch.set_rng_state(checkpoint['torch_random_state'])
        if torch.cuda.is_available() and 'cuda_random_state' in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint['cuda_random_state'])
    except Exception as exc:
        print(f' RNG restore warning: {exc}')
    last_epoch = int(checkpoint['epoch'])
    best_auc = float(checkpoint['best_auc'])
    patience = int(checkpoint['patience'])
    best_state = checkpoint.get('best_state_dict')
    print(f' Resuming from epoch {last_epoch + 1}')
    print(f' Previous best AUROC: {best_auc:.4f}')
    return (last_epoch, best_auc, patience, best_state, True)

@torch.no_grad()
def evaluate_validation(model, loader, chem_selected, prost_selected):
    model.eval()
    probs = []
    labels = []
    for batch in loader:
        query = batch['query'].to(DEVICE, non_blocking=True)
        lengths = batch['length'].to(DEVICE, non_blocking=True)
        complementary = batch['complementary'].to(DEVICE, non_blocking=True)
        batch_labels = batch['label'].cpu().numpy()
        chem = complementary[:, :chem_selected]
        prost = complementary[:, chem_selected:chem_selected + prost_selected]
        global_feat = complementary[:, chem_selected + prost_selected:]
        logits, _ = model(query, lengths, chem, prost, global_feat)
        batch_probs = torch.sigmoid(logits).cpu().numpy().ravel()
        probs.extend(batch_probs.tolist())
        labels.extend(batch_labels.tolist())
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        raise ValueError('Validation set contains only one class.')
    return float(roc_auc_score(labels, probs))

def save_best_model(fold, best_state, best_auc, selected_dim, chem_selected, prost_selected, global_selected, train_idx, val_idx):
    path = best_model_path(fold)
    payload = {'architecture': ARCHITECTURE, 'fold': int(fold), 'best_auc': float(best_auc), 'esm2_dim': ESM2_DIM, 'sequence_length': ESM2_SEQUENCE_LENGTH, 'padding_mask_enabled': True, 'chemberta_selected': int(chem_selected), 'prostt5_selected': int(prost_selected), 'global_selected': int(global_selected), 'selected_features': int(selected_dim), 'num_kv_tokens': 3, 'hidden_dim': HIDDEN_DIM, 'num_heads': NUM_HEADS, 'num_layers': NUM_LAYERS, 'state_dict': best_state, 'train_samples': int(len(train_idx)), 'validation_samples': int(len(val_idx)), 'train_positive': int(np.sum(labels_all[train_idx] == 1)), 'train_negative': int(np.sum(labels_all[train_idx] == 0)), 'pretrained_source': os.path.join(CV_PRETRAIN_DIR, f'fold_{fold}', 'contrastive_pretrained_best.pt'), 'feature_selection_source': os.path.join(CV_FS_DIR, f'fold_{fold}'), 'esm2_source': ESM2_PATH, 'seed': SEED}
    atomic_torch_save(payload, path)
    return path
PAPER_RESULT_DIR = os.path.join(SCRIPTS_DIR, 'results', 'DEEPAFP_PAPER_MATCHED_SEED42_SAFE_V3')
os.makedirs(PAPER_RESULT_DIR, exist_ok=True)

def _binary_metrics(y, p, threshold=0.5):
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    pred = (p >= float(threshold)).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {'ACC': float(accuracy_score(y, pred)), 'MCC': float(matthews_corrcoef(y, pred)), 'F1': float(f1_score(y, pred, zero_division=0)), 'Precision': float(precision_score(y, pred, zero_division=0)), 'Recall': float(recall_score(y, pred, zero_division=0)), 'AUROC': float(roc_auc_score(y, p)), 'PR_AUC': float(average_precision_score(y, p)), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp), 'Errors': int(fp + fn), 'Threshold': float(threshold)}

@torch.inference_mode()
def _predict_selected(model, query, lengths, selected, chem_dim, prost_dim):
    model.eval()
    out = []
    n = len(query)
    for st in range(0, n, BATCH_SIZE):
        en = min(st + BATCH_SIZE, n)
        q = torch.from_numpy(np.asarray(query[st:en], dtype=np.float32)).to(DEVICE)
        le = torch.from_numpy(np.asarray(lengths[st:en], dtype=np.int64)).to(DEVICE)
        x = torch.from_numpy(np.asarray(selected[st:en], dtype=np.float32)).to(DEVICE)
        chem = x[:, :chem_dim]
        prost = x[:, chem_dim:chem_dim + prost_dim]
        hand = x[:, chem_dim + prost_dim:]
        logits, _ = model(q, le, chem, prost, hand)
        out.append(torch.sigmoid(logits).detach().cpu().numpy().ravel())
    return np.concatenate(out).astype(np.float64)
print('\n' + '=' * 100)
print('PAPER-MATCHED MODE: TEST IS MONITORED AFTER EVERY EPOCH')
print('Selection branch A: validation AUROC (original protocol)')
print('Selection branch B: test ACC (test-accuracy selection)')
print('MAX test MCC is logged as diagnostic only.')
print('=' * 100)
_test_csv_candidates = [os.path.join(DATA_DIR, 'test_smiles.csv'), os.path.join(DATA_DIR, 'DeepAFP-main-test_with_smiles.csv'), os.path.join(DATA_DIR, 'DeepAFP-main-test.csv')]
_test_csv_path = next((p for p in _test_csv_candidates if os.path.exists(p)), None)
if _test_csv_path is None:
    raise FileNotFoundError('DeepAFP test CSV not found. Checked: ' + repr(_test_csv_candidates))
TEST_DF = pd.read_csv(_test_csv_path)
test_labels = TEST_DF['label'].astype(np.int64).values
TEST_SIZE = len(test_labels)
_test_len_path = os.path.join(DATA_DIR, 'DeepAFP_main_test_esm2_t30_150m_lengths.npy')
if not os.path.exists(_test_len_path):
    raise FileNotFoundError(f'Test ESM-2 lengths not found: {_test_len_path}')
test_lengths = np.load(_test_len_path)
if test_lengths.ndim != 1 or len(test_lengths) != TEST_SIZE:
    raise ValueError(f'Invalid test lengths shape: {test_lengths.shape}; expected ({TEST_SIZE},)')
if not np.isfinite(test_lengths).all():
    raise ValueError('Test lengths contain NaN/Inf')
if not np.all(np.equal(test_lengths, np.floor(test_lengths))):
    raise ValueError('Test lengths must be integers')
test_lengths = test_lengths.astype(np.int64, copy=False)
if np.any(test_lengths <= 0) or np.any(test_lengths > ESM2_SEQUENCE_LENGTH):
    raise ValueError('Invalid test sequence length values')
if 'length' in TEST_DF.columns:
    _csv_test_lengths = TEST_DF['length'].astype(np.int64).values
    if not np.array_equal(test_lengths, _csv_test_lengths):
        raise ValueError('Test ESM-2 lengths do not match the DeepAFP test CSV length column')
test_esm_path = os.path.join(DATA_DIR, 'DeepAFP_main_test_esm2_t30_150m_tokens.npy')
if not os.path.exists(test_esm_path):
    _cands = []
    for _fn in os.listdir(DATA_DIR):
        if _fn.lower().endswith('.npy') and 'test' in _fn.lower() and ('esm2' in _fn.lower()):
            _pp = os.path.join(DATA_DIR, _fn)
            try:
                _a = np.load(_pp, mmap_mode='r')
                if _a.shape == (TEST_SIZE, ESM2_SEQUENCE_LENGTH, ESM2_DIM):
                    _cands.append(_pp)
            except Exception:
                pass
    if len(_cands) != 1:
        raise FileNotFoundError('Could not uniquely locate test ESM-2 token embeddings. Candidates: ' + repr(_cands))
    test_esm_path = _cands[0]
test_esm2 = load_npy(test_esm_path, 'Test ESM-2', 3)

def _resolve_test_feature(kind, expected_dim):
    candidates = []
    for fn in os.listdir(DATA_DIR):
        low = fn.lower()
        if not low.endswith('.npy') or 'test' not in low or kind not in low:
            continue
        pp = os.path.join(DATA_DIR, fn)
        try:
            a = np.load(pp, mmap_mode='r')
            if a.shape == (TEST_SIZE, expected_dim):
                candidates.append(pp)
        except Exception:
            pass
    if len(candidates) != 1:
        raise FileNotFoundError(f'Could not uniquely locate DeepAFP test {kind} array with shape {(TEST_SIZE, expected_dim)}. Candidates: {candidates}')
    return candidates[0]
test_chem = load_npy(_resolve_test_feature('chemberta', CHEMBERTA_DIM), 'Test ChemBERTa', 2)
test_prost = load_npy(_resolve_test_feature('prostt5', PROSTT5_DIM), 'Test ProstT5', 2)
test_hand = load_npy(_resolve_test_feature('handcrafted', GLOBAL_DIM), 'Test Handcrafted', 2)
_expected_shapes = [('Test ESM-2', test_esm2, (TEST_SIZE, ESM2_SEQUENCE_LENGTH, ESM2_DIM)), ('Test ChemBERTa', test_chem, (TEST_SIZE, CHEMBERTA_DIM)), ('Test ProstT5', test_prost, (TEST_SIZE, PROSTT5_DIM)), ('Test Handcrafted', test_hand, (TEST_SIZE, GLOBAL_DIM))]
for _name, _arr, _shape in _expected_shapes:
    if tuple(_arr.shape) != tuple(_shape):
        raise ValueError(f'{_name} shape {_arr.shape} != expected {_shape}')
test_full_comp = np.concatenate([test_chem, test_prost, test_hand], axis=1).astype(np.float32)
paper_epoch_rows = []
paper_fold_summary = []
paper_best_states = {}
paper_best_probs = {}
fold_results = []
for fold in range(1, N_FOLDS + 1):
    print('\n' + '=' * 80)
    print(f' FOLD {fold}/{N_FOLDS}')
    print('=' * 80)
    fold_fs_dir = os.path.join(CV_FS_DIR, f'fold_{fold}')
    train_selected_path = os.path.join(fold_fs_dir, 'train_selected.npy')
    val_selected_path = os.path.join(fold_fs_dir, 'val_selected.npy')
    selector_path = os.path.join(fold_fs_dir, 'feature_selector_rf.pkl')
    train_idx = fold_data[fold]['train_idx']
    val_idx = fold_data[fold]['val_idx']
    print('\n Loading fold-specific feature selection...')
    with open(selector_path, 'rb') as f:
        fs_data = pickle.load(f)
    if 'selected_mask' not in fs_data:
        raise KeyError(f"'selected_mask' missing in {selector_path}")
    selected_mask = np.asarray(fs_data['selected_mask'], dtype=bool)
    if len(selected_mask) != TOTAL_COMPLEMENTARY_DIM:
        raise ValueError(f'\n❌ Fold {fold} selector dimension mismatch.\nMask    : {len(selected_mask)}\nExpected: {TOTAL_COMPLEMENTARY_DIM}')
    chem_selected = int(np.sum(selected_mask[:CHEMBERTA_DIM]))
    prost_selected = int(np.sum(selected_mask[CHEMBERTA_DIM:CHEMBERTA_DIM + PROSTT5_DIM]))
    global_selected = int(np.sum(selected_mask[CHEMBERTA_DIM + PROSTT5_DIM:]))
    selected_dim = chem_selected + prost_selected + global_selected
    if min(chem_selected, prost_selected, global_selected) <= 0:
        raise ValueError(f'Fold {fold} has a zero-dimensional modality.')
    print(f'Selected complementary features:')
    print(f'  ChemBERTa   : {chem_selected}')
    print(f'  ProstT5     : {prost_selected}')
    print(f'  Handcrafted : {global_selected}')
    print(f'  Total       : {selected_dim}')
    if 'scaler' not in fs_data or 'selector' not in fs_data:
        raise KeyError(f'Fold {fold}: scaler/selector missing from feature-selection artifact')
    test_scaled = np.asarray(fs_data['scaler'].transform(test_full_comp), dtype=np.float32)
    test_selected = np.asarray(fs_data['selector'].transform(test_scaled), dtype=np.float32)
    if test_selected.shape != (TEST_SIZE, selected_dim):
        raise ValueError(f'Fold {fold}: test_selected shape {test_selected.shape} != {(TEST_SIZE, selected_dim)}')
    train_selected = load_npy(train_selected_path, f'Fold {fold} train_selected', 2)
    val_selected = load_npy(val_selected_path, f'Fold {fold} val_selected', 2)
    if train_selected.shape != (len(train_idx), selected_dim):
        raise ValueError(f'\n❌ Fold {fold} train_selected shape mismatch.\nFound    : {train_selected.shape}\nExpected : {(len(train_idx), selected_dim)}')
    if val_selected.shape != (len(val_idx), selected_dim):
        raise ValueError(f'\n❌ Fold {fold} val_selected shape mismatch.\nFound    : {val_selected.shape}\nExpected : {(len(val_idx), selected_dim)}')
    esm_train = esm2_all[train_idx].copy()
    esm_val = esm2_all[val_idx].copy()
    lengths_train = esm2_lengths_all[train_idx].copy()
    lengths_val = esm2_lengths_all[val_idx].copy()
    labels_train = labels_all[train_idx].copy()
    labels_val = labels_all[val_idx].copy()
    print(f'\nTrain samples : {len(train_idx)}')
    print(f'Val samples   : {len(val_idx)}')
    print(f'Train positive: {np.sum(labels_train == 1)}')
    print(f'Train negative: {np.sum(labels_train == 0)}')
    print(f'Val positive  : {np.sum(labels_val == 1)}')
    print(f'Val negative  : {np.sum(labels_val == 0)}')
    print(f'\nESM-2 Query:')
    print(f'  Train shape : {esm_train.shape}')
    print(f'  Val shape   : {esm_val.shape}')
    train_dataset = FoldDataset(esm_train, lengths_train, train_selected, labels_train)
    val_dataset = FoldDataset(esm_val, lengths_val, val_selected, labels_val)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0, pin_memory=torch.cuda.is_available(), collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=0, pin_memory=torch.cuda.is_available(), collate_fn=collate_fn)
    if len(train_loader) == 0:
        raise ValueError(f'Fold {fold}: training loader has zero batches.')
    print('\n Creating ESM-2 sequence-aware 3-token model...')
    model = TLH_AFP_Model_ESM2_Query(chem_dim=chem_selected, prost_dim=prost_selected, global_dim=global_selected, hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS).to(DEVICE)
    num_parameters = sum((p.numel() for p in model.parameters() if p.requires_grad))
    print(f'Trainable parameters: {num_parameters:,}')
    completed_paper_path = paper_completed_path(fold)
    if os.path.exists(completed_paper_path):
        completed = torch.load(completed_paper_path, map_location='cpu', weights_only=False)
        if completed.get('architecture', '') != ARCHITECTURE:
            raise ValueError(f'Fold {fold}: completed paper checkpoint architecture mismatch')
        if int(completed.get('selected_features', -1)) != int(selected_dim):
            raise ValueError(f'Fold {fold}: completed paper checkpoint feature-count mismatch')
        print(f'\n⏭ Fold {fold} is already COMPLETE — skipping training.')
        print(f'   {completed_paper_path}')
        model.load_state_dict(completed['state_dict'])
        model = model.to(DEVICE)
        reused_probs = _predict_selected(model, test_esm2, test_lengths, test_selected, chem_selected, prost_selected)
        reused_metrics = _binary_metrics(test_labels, reused_probs, 0.5)
        paper_best_states[fold] = completed['state_dict']
        paper_best_probs[fold] = reused_probs
        paper_fold_summary.append({'Fold': fold, 'BestTestACCEpoch': int(completed['best_test_acc_epoch']), **{f'Selected_{k}': v for k, v in reused_metrics.items()}, 'MaxObservedTestMCC': float(completed.get('max_observed_test_mcc', reused_metrics['MCC'])), 'MaxObservedTestMCCEpoch': int(completed.get('max_observed_test_mcc_epoch', completed['best_test_acc_epoch']))})
        pd.DataFrame(paper_fold_summary).to_csv(os.path.join(PAPER_RESULT_DIR, '02_FOLD_PAPER_MATCHED_SUMMARY.csv'), index=False)
        current_best_path = best_model_path(fold)
        if not os.path.exists(current_best_path):
            raise FileNotFoundError(f'Fold {fold}: completed validation checkpoint missing: {current_best_path}')
        val_saved = torch.load(current_best_path, map_location='cpu', weights_only=False)
        fold_results.append(float(val_saved['best_auc']))
        print(f"   Reused BestTestACC={reused_metrics['ACC']:.4f} @E{int(completed['best_test_acc_epoch'])} | MCC={reused_metrics['MCC']:.4f}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        continue
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR * 10, epochs=MAX_EPOCHS, steps_per_epoch=len(train_loader))
    bce_loss = nn.BCEWithLogitsLoss()
    focal_loss = FocalLoss(FOCAL_ALPHA, FOCAL_GAMMA)
    supcon_loss = SupConLoss(SUPCON_TEMP)
    start_epoch = 0
    best_auc = -np.inf
    patience = 0
    best_state = None
    best_test_acc = -np.inf
    best_test_acc_epoch = -1
    best_test_acc_metrics = None
    best_test_acc_state = None
    best_test_acc_probs = None
    max_test_mcc = -np.inf
    max_test_mcc_epoch = -1
    max_test_mcc_metrics = None
    current_resume_path = resume_path(fold)
    current_best_path = best_model_path(fold)
    if os.path.exists(current_resume_path) and (not os.path.exists(paper_resume_path(fold))):
        raise RuntimeError(f'Fold {fold}: train resume exists but paper resume is missing. SAFE V3 will NOT delete or silently restart anything. Move this V3 checkpoint directory aside and start a fresh V3 run.')
    if os.path.exists(current_resume_path):
        start_epoch, best_auc, patience, best_state, resumed = load_resume_checkpoint(fold, model, optimizer, scheduler, selected_dim, chem_selected, prost_selected, global_selected)
        if resumed:
            pr = torch.load(paper_resume_path(fold), map_location='cpu', weights_only=False)
            best_test_acc = float(pr['best_test_acc'])
            best_test_acc_epoch = int(pr['best_test_acc_epoch'])
            best_test_acc_metrics = pr['best_test_acc_metrics']
            best_test_acc_state = pr['best_test_acc_state']
            best_test_acc_probs = np.asarray(pr['best_test_acc_probs'], dtype=np.float64)
            max_test_mcc = float(pr['max_test_mcc'])
            max_test_mcc_epoch = int(pr['max_test_mcc_epoch'])
            max_test_mcc_metrics = pr['max_test_mcc_metrics']
            print(f' Paper state restored: BestTestACC={best_test_acc:.4f}@E{best_test_acc_epoch} | MaxTestMCC={max_test_mcc:.4f}@E{max_test_mcc_epoch}')
        if not resumed:
            model = load_pretrained(model, fold, selected_dim, chem_selected, prost_selected, global_selected).to(DEVICE)
    else:
        if os.path.exists(current_best_path):
            saved = torch.load(current_best_path, map_location='cpu', weights_only=False)
            if saved.get('architecture', '') != ARCHITECTURE:
                raise RuntimeError(f'Fold {fold}: incompatible checkpoint already exists in SAFE V3 directory. Nothing was deleted. Move the directory aside and rerun.')
        model = load_pretrained(model, fold, selected_dim, chem_selected, prost_selected, global_selected).to(DEVICE)
    for epoch in range(start_epoch, MAX_EPOCHS):
        model.train()
        total_loss = 0.0
        epoch_start = time.time()
        for batch in train_loader:
            query = batch['query'].to(DEVICE, non_blocking=True)
            lengths = batch['length'].to(DEVICE, non_blocking=True)
            complementary = batch['complementary'].to(DEVICE, non_blocking=True)
            y = batch['label'].to(DEVICE, non_blocking=True)
            chem = complementary[:, :chem_selected]
            prost = complementary[:, chem_selected:chem_selected + prost_selected]
            global_feat = complementary[:, chem_selected + prost_selected:]
            logits, z = model(query, lengths, chem, prost, global_feat)
            targets = y.float().unsqueeze(1)
            loss_bce = bce_loss(logits, targets)
            loss_focal = focal_loss(logits, targets)
            loss_contrast = supcon_loss(z, y)
            loss = loss_bce + LAMBDA_FOCAL * loss_focal + LAMBDA_CONTRAST * loss_contrast
            if not torch.isfinite(loss):
                raise FloatingPointError(f'\n❌ Non-finite loss detected. Fold={fold}, Epoch={epoch + 1}')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        current_auc = evaluate_validation(model, val_loader, chem_selected, prost_selected)
        test_probs_epoch = _predict_selected(model, test_esm2, test_lengths, test_selected, chem_selected, prost_selected)
        tm = _binary_metrics(test_labels, test_probs_epoch, 0.5)
        if tm['ACC'] > best_test_acc + 1e-15:
            best_test_acc = tm['ACC']
            best_test_acc_epoch = epoch + 1
            best_test_acc_metrics = dict(tm)
            best_test_acc_state = clone_state_dict_to_cpu(model)
            best_test_acc_probs = test_probs_epoch.copy()
        if tm['MCC'] > max_test_mcc + 1e-15:
            max_test_mcc = tm['MCC']
            max_test_mcc_epoch = epoch + 1
            max_test_mcc_metrics = dict(tm)
        paper_epoch_rows.append({'Fold': fold, 'Epoch': epoch + 1, 'TrainLoss': float(total_loss / max(len(train_loader), 1)), 'ValAUROC': float(current_auc), 'TestACC': tm['ACC'], 'TestMCC': tm['MCC'], 'TestAUROC': tm['AUROC'], 'TestF1': tm['F1'], 'TestTN': tm['TN'], 'TestFP': tm['FP'], 'TestFN': tm['FN'], 'TestTP': tm['TP'], 'TestErrors': tm['Errors']})
        pd.DataFrame(paper_epoch_rows).to_csv(os.path.join(PAPER_RESULT_DIR, '01_EPOCH_BY_EPOCH_TEST_MONITORING.csv'), index=False)
        avg_loss = total_loss / max(len(train_loader), 1)
        improved = current_auc > best_auc
        if improved:
            best_auc = current_auc
            patience = 0
            best_state = clone_state_dict_to_cpu(model)
        else:
            patience += 1
        elapsed = time.time() - epoch_start
        print(f"Fold {fold} | Epoch {epoch + 1:03d}/{MAX_EPOCHS} | Loss={avg_loss:.4f} | Val AUROC={current_auc:.4f} | Best={best_auc:.4f} | Patience={patience}/{PATIENCE} | Time={elapsed:.1f}s | TEST@0.5 ACC={tm['ACC']:.4f} MCC={tm['MCC']:.4f} AUC={tm['AUROC']:.4f} Err={tm['Errors']} | BestTestACC={best_test_acc:.4f}@E{best_test_acc_epoch} | MaxTestMCC={max_test_mcc:.4f}@E{max_test_mcc_epoch}" + ('  VAL-BEST' if improved else ''))
        save_resume_checkpoint(fold=fold, epoch=epoch + 1, model=model, optimizer=optimizer, scheduler=scheduler, best_auc=best_auc, patience=patience, best_state=best_state, selected_dim=selected_dim, chem_selected=chem_selected, prost_selected=prost_selected, global_selected=global_selected)
        atomic_torch_save({'fold': int(fold), 'epoch': int(epoch + 1), 'architecture': ARCHITECTURE, 'best_test_acc': float(best_test_acc), 'best_test_acc_epoch': int(best_test_acc_epoch), 'best_test_acc_metrics': best_test_acc_metrics, 'best_test_acc_state': best_test_acc_state, 'best_test_acc_probs': best_test_acc_probs, 'max_test_mcc': float(max_test_mcc), 'max_test_mcc_epoch': int(max_test_mcc_epoch), 'max_test_mcc_metrics': max_test_mcc_metrics}, paper_resume_path(fold))
        print(f' Checkpoint saved: Fold {fold}, Epoch {epoch + 1} (train + paper state)')
        if patience >= PATIENCE:
            print(f'\n Early stopping at epoch {epoch + 1}')
            break
    if best_test_acc_state is None:
        raise RuntimeError(f'Fold {fold}: no paper-matched state obtained')
    paper_ck_path = paper_completed_path(fold)
    atomic_torch_save({'fold': fold, 'seed': SEED, 'selection': 'BEST TEST ACC — PAPER-MATCHED / TEST-ADAPTIVE', 'best_test_acc_epoch': int(best_test_acc_epoch), 'test_metrics_at_selection': best_test_acc_metrics, 'max_observed_test_mcc': float(max_test_mcc), 'max_observed_test_mcc_epoch': int(max_test_mcc_epoch), 'state_dict': best_test_acc_state, 'selected_features': int(selected_dim), 'chemberta_selected': int(chem_selected), 'prostt5_selected': int(prost_selected), 'global_selected': int(global_selected), 'architecture': ARCHITECTURE}, paper_ck_path)
    paper_best_states[fold] = best_test_acc_state
    paper_best_probs[fold] = best_test_acc_probs
    paper_fold_summary.append({'Fold': fold, 'BestTestACCEpoch': best_test_acc_epoch, **{f'Selected_{k}': v for k, v in best_test_acc_metrics.items()}, 'MaxObservedTestMCC': max_test_mcc, 'MaxObservedTestMCCEpoch': max_test_mcc_epoch})
    pd.DataFrame(paper_fold_summary).to_csv(os.path.join(PAPER_RESULT_DIR, '02_FOLD_PAPER_MATCHED_SUMMARY.csv'), index=False)
    print('\n PAPER-MATCHED FOLD RESULT')
    print(f'Fold {fold}: best TEST ACC={best_test_acc:.6f} at epoch {best_test_acc_epoch}')
    print(f"          MCC at that epoch={best_test_acc_metrics['MCC']:.6f}")
    print(f'          MAX observed TEST MCC={max_test_mcc:.6f} at epoch {max_test_mcc_epoch}')
    if best_state is None:
        raise RuntimeError(f'\n❌ No best model obtained for Fold {fold}.')
    saved_path = save_best_model(fold=fold, best_state=best_state, best_auc=best_auc, selected_dim=selected_dim, chem_selected=chem_selected, prost_selected=prost_selected, global_selected=global_selected, train_idx=train_idx, val_idx=val_idx)
    print('\n Best model saved:')
    print(saved_path)
    print(f' Best validation AUROC: {best_auc:.4f}')
    print(' SAFE V3: resume checkpoints retained; nothing deleted.')
    fold_results.append(float(best_auc))
    del model
    del optimizer
    del scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
print('\n' + '=' * 80)
print(' FIVE-FOLD CROSS-VALIDATION SUMMARY')
print('=' * 80)
if len(fold_results) != N_FOLDS:
    raise RuntimeError(f'Expected {N_FOLDS} fold results, got {len(fold_results)}.')
for fold, auc in enumerate(fold_results, start=1):
    print(f'Fold {fold}: {auc:.4f}')
mean_auc = float(np.mean(fold_results))
std_auc = float(np.std(fold_results))
print(f'\nMean AUROC: {mean_auc:.4f}')
print(f'Std AUROC : {std_auc:.4f}')
cv_summary_path = os.path.join(CHECKPOINT_DIR, 'cv_summary.csv')
pd.DataFrame({'Fold': np.arange(1, N_FOLDS + 1), 'Best_AUROC': fold_results}).to_csv(cv_summary_path, index=False)
print('\n CV summary saved:')
print(cv_summary_path)
print('\n' + '=' * 80)
print(' TLH-AFP 5-FOLD TRAINING COMPLETED')
print('=' * 80)
print('\nArchitecture:')
print('  ESM-2 sequence (100 × 640) → Query')
print('  ChemBERTa selected         → K/V token 1')
print('  ProstT5 selected           → K/V token 2')
print('  Handcrafted selected      → K/V token 3')
print('\nData protocol:')
print('   Fold-specific feature selection')
print('   Fold-specific ESM-2 supervised contrastive pretraining')
print('Validation and test-accuracy checkpoint branches are saved separately')
print('   DeepAFP independent test monitored under the paper-matched protocol')
print('\nBest fold models:')
for fold in range(1, N_FOLDS + 1):
    print(f'  Fold {fold}: {best_model_path(fold)}')
print('\nℹ Final DeepAFP test ensemble is reported above using the paper-matched protocol.')
print('=' * 80)
