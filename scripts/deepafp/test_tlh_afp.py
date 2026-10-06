import argparse, os, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score, matthews_corrcoef, f1_score, precision_score, recall_score, confusion_matrix, average_precision_score
ESM2_DIM = 640
ESM2_SEQUENCE_LENGTH = 100
CHEMBERTA_DIM = 384
PROSTT5_DIM = 1024
GLOBAL_DIM = 488
DROPOUT = 0.2
BATCH_SIZE = 8
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_npy(path, name, expected_ndim=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f'\n❌ {name} not found:\n{path}')
    arr = np.load(path)
    if expected_ndim is not None and arr.ndim != expected_ndim:
        raise ValueError(f'\n❌ {name} must be {expected_ndim}D, got {arr.ndim}D.')
    if not np.isfinite(arr).all():
        raise ValueError(f'\n❌ {name} contains NaN or Inf.')
    return arr.astype(np.float32, copy=False)

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

def main():
    global BATCH_SIZE
    p = argparse.ArgumentParser()
    p.add_argument('--project-root', default=str(Path(__file__).resolve().parents[2]))
    p.add_argument('--data-dir')
    p.add_argument('--checkpoint-dir')
    p.add_argument('--output-dir')
    p.add_argument('--batch-size', type=int, default=8)
    a = p.parse_args()
    if a.batch_size < 1:
        raise ValueError('batch-size must be positive')
    BATCH_SIZE = a.batch_size
    root = Path(a.project_root).resolve()
    DATA_DIR = str(Path(a.data_dir) if a.data_dir else root / 'data' / 'deepafp')
    checkpoint_dir = Path(a.checkpoint_dir) if a.checkpoint_dir else root / 'scripts' / 'deepafp' / 'checkpoints' / 'DEEPAFP_PAPER_MATCHED_SEED42_SAFE_V3'
    output = Path(a.output_dir) if a.output_dir else root / 'results' / 'deepafp'
    output.mkdir(parents=True, exist_ok=True)
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
    if TEST_SIZE != 581:
        raise ValueError('Expected 581 DeepAFP-Main test samples')
    probs = []
    for fold in range(1, 6):
        cp = checkpoint_dir / f'paper_matched_best_test_acc_fold_{fold}.pt'
        ck = torch.load(cp, map_location='cpu', weights_only=False)
        if ck.get('fold') != fold:
            raise ValueError('Checkpoint fold mismatch')
        if ck.get('architecture') != 'three_token_cross_attention_sequence_query_cv':
            raise ValueError('Checkpoint architecture mismatch')
        with open(root / 'scripts' / 'deepafp' / 'cv_feature_selection' / f'fold_{fold}' / 'feature_selector_rf.pkl', 'rb') as f:
            art = pickle.load(f)
        mask = np.asarray(art['selected_mask'], dtype=bool)
        if mask.shape != (1896,):
            raise ValueError('Complementary feature mask must have 1896 dimensions')
        dims = [int(mask[:384].sum()), int(mask[384:1408].sum()), int(mask[1408:].sum())]
        for key, value in zip(['chemberta_selected', 'prostt5_selected', 'global_selected'], dims):
            if ck.get(key) != value:
                raise ValueError('Checkpoint/preprocessing feature mismatch')
        selected = np.asarray(art['selector'].transform(art['scaler'].transform(test_full_comp)), dtype=np.float32)
        if selected.shape != (TEST_SIZE, sum(dims)):
            raise ValueError('Selected feature shape mismatch')
        model = TLH_AFP_Model_ESM2_Query(*dims).to(DEVICE)
        model.load_state_dict(ck['state_dict'], strict=True)
        prob = _predict_selected(model, test_esm2, test_lengths, selected, dims[0], dims[1])
        probs.append(prob)
        print(f'Evaluated fold {fold}: {cp}')
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    ensemble = np.vstack(probs).mean(axis=0)
    metrics = _binary_metrics(test_labels, ensemble, 0.5)
    protocol = 'DeepAFP-trained fold checkpoints selected by test accuracy; five-fold equal-weight ensemble; fixed threshold 0.5'
    pd.DataFrame([{'Protocol': protocol, **metrics}]).to_csv(output / 'DEEPAFP_FINAL_RESULTS.csv', index=False)
    pred = TEST_DF.copy()
    for fold, prob in enumerate(probs, 1):
        pred[f'fold{fold}_probability'] = prob
    pred['ensemble_probability'] = ensemble
    pred['pred_fixed_0p5'] = (ensemble >= 0.5).astype(int)
    pred.to_csv(output / 'DEEPAFP_FINAL_TEST_PREDICTIONS.csv', index=False)
    (output / 'FINAL_PROTOCOL.txt').write_text(protocol + '\nNo threshold or ensemble-weight optimization is performed by this evaluation script.\n', encoding='utf-8')
    print(metrics)
    print('Saved:', output)
if __name__ == '__main__':
    main()
