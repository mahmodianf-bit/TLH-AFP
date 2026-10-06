import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'deepafp')
SCRIPTS_DIR = SCRIPT_DIR
CV_OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'cv_feature_selection')
os.makedirs(CV_OUTPUT_DIR, exist_ok=True)
TRAIN_SMILES = os.path.join(DATA_DIR, 'DeepAFP-main-train_with_smiles.csv')
TRAIN_CHEMBERTA = os.path.join(DATA_DIR, 'train_chemberta_384.npy')
TRAIN_PROSTT5 = os.path.join(DATA_DIR, 'train_prostt5_1024.npy')
TRAIN_HANDCRAFTED = os.path.join(DATA_DIR, 'train_handcrafted_488.npy')
TRAIN_ESM2_TOKENS = os.path.join(DATA_DIR, 'DeepAFP_main_train_esm2_t30_150m_tokens.npy')
TRAIN_ESM2_LENGTHS = os.path.join(DATA_DIR, 'DeepAFP_main_train_esm2_t30_150m_lengths.npy')
N_SPLITS = 5
SEED = 42
RF_N_ESTIMATORS = 100
RF_RANDOM_STATE = 42
ESM2_MODEL_NAME = 'facebook/esm2_t30_150M_UR50D'

def check_file_exists(path, name):
    if not os.path.exists(path):
        raise FileNotFoundError(f'\n{name} not found:\n{path}')

def load_array(path, name, expected_ndim):
    check_file_exists(path, name)
    array = np.load(path, mmap_mode='r')
    if array.ndim != expected_ndim:
        raise ValueError(f'{name} must be {expected_ndim}D, but got shape {array.shape}')
    print(f'{name:<30} shape = {array.shape}')
    return array

def check_finite_2d(array, name):
    if not np.isfinite(array).all():
        raise ValueError(f'{name} contains NaN or Inf values.')

def check_finite_3d_chunked(array, name, chunk_size=64):
    n_samples = array.shape[0]
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        chunk = np.asarray(array[start:end])
        if not np.isfinite(chunk).all():
            raise ValueError(f'{name} contains NaN or Inf values in samples {start}:{end}')

def check_same_sample_count(arrays):
    counts = {name: arr.shape[0] for name, arr in arrays.items()}
    unique_counts = set(counts.values())
    if len(unique_counts) != 1:
        raise ValueError('Number of samples is not consistent:\n' + '\n'.join((f'{name}: {count}' for name, count in counts.items())))
    return next(iter(unique_counts))

def get_feature_ranges(feature_dimensions):
    feature_ranges = {}
    current_start = 0
    for name, dimension in feature_dimensions.items():
        current_end = current_start + dimension
        feature_ranges[name] = (current_start, current_end)
        current_start = current_end
    return feature_ranges

def get_selected_feature_counts(selected_mask, feature_ranges):
    counts = {}
    for name, (start, end) in feature_ranges.items():
        counts[name] = int(np.sum(selected_mask[start:end]))
    return counts

def feature_index_to_representation(feature_index, feature_ranges):
    for name, (start, end) in feature_ranges.items():
        if start <= feature_index < end:
            local_index = feature_index - start
            return (name, local_index)
    return ('Unknown', feature_index)
print('=' * 80)
print('RANDOM FOREST FEATURE SELECTION - 5-FOLD STRATIFIED CROSS-VALIDATION')
print('=' * 80)
print(f'Project Root : {PROJECT_ROOT}')
print(f'Data Dir     : {DATA_DIR}')
print(f'Output Dir   : {CV_OUTPUT_DIR}')
print(f'ESM-2 Model  : {ESM2_MODEL_NAME}')
print('\n' + '=' * 80)
print('CHECKING REQUIRED FILES')
print('=' * 80)
required_files = {'Train CSV': TRAIN_SMILES, 'Train ChemBERTa': TRAIN_CHEMBERTA, 'Train ProstT5': TRAIN_PROSTT5, 'Train Handcrafted': TRAIN_HANDCRAFTED, 'Train ESM-2 tokens': TRAIN_ESM2_TOKENS, 'Train ESM-2 lengths': TRAIN_ESM2_LENGTHS}
for name, path in required_files.items():
    check_file_exists(path, name)
    print(f' {name:<25} {os.path.basename(path)}')
print('\n' + '=' * 80)
print('LOADING TRAIN CSV')
print('=' * 80)
train_df = pd.read_csv(TRAIN_SMILES)
if 'label' not in train_df.columns:
    raise ValueError("Column 'label' was not found in train_smiles.csv")
if 'sequence' not in train_df.columns:
    raise ValueError("Column 'sequence' was not found in train_smiles.csv")
y = train_df['label'].astype(int).values
unique_labels = np.unique(y)
if not np.array_equal(unique_labels, np.array([0, 1])):
    raise ValueError(f'Labels must be binary [0, 1], but found {unique_labels}')
n_samples_csv = len(y)
print(f'Training samples : {n_samples_csv}')
print(f'Positive samples : {np.sum(y == 1)}')
print(f'Negative samples : {np.sum(y == 0)}')
print('\n' + '=' * 80)
print('CHECKING SEQUENCE INFORMATION')
print('=' * 80)
sequence_lengths_from_sequence = train_df['sequence'].astype(str).str.len().values
if 'length' in train_df.columns:
    csv_lengths = train_df['length'].astype(int).values
    if not np.array_equal(sequence_lengths_from_sequence, csv_lengths):
        raise ValueError("The 'length' column in train_smiles.csv does not match the actual sequence lengths.")
    print(" CSV 'length' matches sequence lengths.")
else:
    csv_lengths = sequence_lengths_from_sequence
    print("ℹ No 'length' column found; sequence lengths were calculated directly.")
print(f'Minimum sequence length : {np.min(sequence_lengths_from_sequence)}')
print(f'Maximum sequence length : {np.max(sequence_lengths_from_sequence)}')
print(f'Mean sequence length    : {np.mean(sequence_lengths_from_sequence):.2f}')
print('\n' + '=' * 80)
print('LOADING COMPLEMENTARY FEATURES')
print('=' * 80)
train_chemberta = load_array(TRAIN_CHEMBERTA, 'Train ChemBERTa', expected_ndim=2)
train_prostt5 = load_array(TRAIN_PROSTT5, 'Train ProstT5', expected_ndim=2)
train_handcrafted = load_array(TRAIN_HANDCRAFTED, 'Train Handcrafted', expected_ndim=2)
print('\n' + '=' * 80)
print('LOADING ESM-2 QUERY REPRESENTATION')
print('=' * 80)
train_esm2_tokens = load_array(TRAIN_ESM2_TOKENS, 'Train ESM-2 tokens', expected_ndim=3)
train_esm2_lengths = load_array(TRAIN_ESM2_LENGTHS, 'Train ESM-2 lengths', expected_ndim=1)
chemberta_dim = train_chemberta.shape[1]
prostt5_dim = train_prostt5.shape[1]
handcrafted_dim = train_handcrafted.shape[1]
esm2_sequence_length = train_esm2_tokens.shape[1]
esm2_embedding_dim = train_esm2_tokens.shape[2]
print('\n' + '=' * 80)
print('DYNAMICALLY DETECTED DIMENSIONS')
print('=' * 80)
print(f'ChemBERTa dimension       : {chemberta_dim}')
print(f'ProstT5 dimension         : {prostt5_dim}')
print(f'Handcrafted dimension     : {handcrafted_dim}')
print(f'ESM-2 token sequence dim  : {esm2_sequence_length}')
print(f'ESM-2 embedding dimension : {esm2_embedding_dim}')
print('\n' + '=' * 80)
print('CHECKING NaN / INF VALUES')
print('=' * 80)
check_finite_2d(train_chemberta, 'Train ChemBERTa')
print(' ChemBERTa finite-value check passed.')
check_finite_2d(train_prostt5, 'Train ProstT5')
print(' ProstT5 finite-value check passed.')
check_finite_2d(train_handcrafted, 'Train Handcrafted')
print(' Handcrafted finite-value check passed.')
check_finite_3d_chunked(train_esm2_tokens, 'Train ESM-2 tokens')
print(' ESM-2 token finite-value check passed.')
if not np.isfinite(train_esm2_lengths).all():
    raise ValueError('ESM-2 lengths contain NaN or Inf.')
print('\n' + '=' * 80)
print('SAMPLE-COUNT VALIDATION')
print('=' * 80)
feature_sample_counts = {'ChemBERTa': train_chemberta, 'ProstT5': train_prostt5, 'Handcrafted': train_handcrafted, 'ESM-2': train_esm2_tokens}
n_samples_features = check_same_sample_count(feature_sample_counts)
if n_samples_features != n_samples_csv:
    raise ValueError(f'Feature rows ({n_samples_features}) do not match CSV labels ({n_samples_csv}).')
if train_esm2_lengths.shape[0] != n_samples_csv:
    raise ValueError(f'ESM-2 length count ({train_esm2_lengths.shape[0]}) does not match training samples ({n_samples_csv}).')
print(f' All representations contain {n_samples_csv} samples.')
print('\n' + '=' * 80)
print('VALIDATING ESM-2 SEQUENCE LENGTHS')
print('=' * 80)
esm2_lengths = np.asarray(train_esm2_lengths).astype(int)
if np.any(esm2_lengths <= 0):
    raise ValueError('ESM-2 lengths must be greater than zero.')
if np.any(esm2_lengths > esm2_sequence_length):
    raise ValueError('At least one ESM-2 sequence length exceeds the stored token sequence dimension.')
if not np.array_equal(esm2_lengths, sequence_lengths_from_sequence):
    mismatched = np.where(esm2_lengths != sequence_lengths_from_sequence)[0]
    example_indices = mismatched[:10]
    raise ValueError(f'ESM-2 lengths do not match the actual sequence lengths in train_smiles.csv.\nNumber of mismatches: {len(mismatched)}\nExample indices: {example_indices.tolist()}')
print(' ESM-2 lengths match the input sequences.')
print(f'Stored token length : {esm2_sequence_length}')
print(f'Actual max length   : {np.max(esm2_lengths)}')
print(f'Actual min length   : {np.min(esm2_lengths)}')
print(f'Actual mean length  : {np.mean(esm2_lengths):.2f}')
print('\n' + '=' * 80)
print('BUILDING COMPLEMENTARY FEATURE SPACE')
print('=' * 80)
X = np.concatenate([np.asarray(train_chemberta), np.asarray(train_prostt5), np.asarray(train_handcrafted)], axis=1)
TOTAL_FEATURES = X.shape[1]
print(f'Combined X shape : {X.shape}')
print(f'Total complementary features : {TOTAL_FEATURES}')
if TOTAL_FEATURES <= 0:
    raise ValueError('No complementary features were found.')
feature_dimensions = {'ChemBERTa': chemberta_dim, 'ProstT5': prostt5_dim, 'Handcrafted_488': handcrafted_dim}
feature_ranges = get_feature_ranges(feature_dimensions)
print('\nFeature ranges:')
for name, (start, end) in feature_ranges.items():
    print(f'  {name:<20} [{start}:{end}] -> {end - start} features')
calculated_total = sum(feature_dimensions.values())
if calculated_total != TOTAL_FEATURES:
    raise ValueError('Dynamic feature dimension calculation does not match concatenated feature matrix.')
print('\n' + '=' * 80)
print('CREATING STRATIFIED 5-FOLD SPLITS')
print('=' * 80)
sample_indices = np.arange(n_samples_csv)
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
fold_splits = list(skf.split(sample_indices, y))
cv_splits = [{'train_idx': train_idx, 'val_idx': val_idx} for train_idx, val_idx in fold_splits]
if len(cv_splits) != N_SPLITS:
    raise RuntimeError('Unexpected number of CV folds.')
splits_path = os.path.join(CV_OUTPUT_DIR, 'cv_splits.pkl')
with open(splits_path, 'wb') as f:
    pickle.dump(cv_splits, f)
print(' CV splits saved:')
print(splits_path)
print(f'Number of folds: {len(cv_splits)}')
fold_summary = []
for fold_number, split in enumerate(cv_splits, start=1):
    train_idx = split['train_idx']
    val_idx = split['val_idx']
    print('\n' + '=' * 80)
    print(f'FOLD {fold_number}/{N_SPLITS}')
    print('=' * 80)
    X_fold_train = X[train_idx]
    y_fold_train = y[train_idx]
    X_fold_val = X[val_idx]
    y_fold_val = y[val_idx]
    print(f'Fold train samples : {len(train_idx)}')
    print(f'Fold validation samples : {len(val_idx)}')
    print(f'Train positive : {np.sum(y_fold_train == 1)}')
    print(f'Train negative : {np.sum(y_fold_train == 0)}')
    print(f'Val positive   : {np.sum(y_fold_val == 1)}')
    print(f'Val negative   : {np.sum(y_fold_val == 0)}')
    fold_dir = os.path.join(CV_OUTPUT_DIR, f'fold_{fold_number}')
    os.makedirs(fold_dir, exist_ok=True)
    print('\n Fitting StandardScaler on fold training data...')
    scaler = StandardScaler()
    X_fold_train_scaled = scaler.fit_transform(X_fold_train)
    X_fold_val_scaled = scaler.transform(X_fold_val)
    print(' Fold scaling completed.')
    print('\n Training Random Forest...')
    rf = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, random_state=RF_RANDOM_STATE, n_jobs=-1)
    rf.fit(X_fold_train_scaled, y_fold_train)
    print(' Random Forest training completed.')
    print(f'Trees : {rf.n_estimators}')
    importances = rf.feature_importances_
    if len(importances) != TOTAL_FEATURES:
        raise RuntimeError('Random Forest importance vector does not match total feature count.')
    mean_importance = float(np.mean(importances))
    print('\n Feature importance statistics:')
    print(f'Number of features : {len(importances)}')
    print(f'Mean importance    : {mean_importance:.12f}')
    print(f'Maximum importance : {np.max(importances):.12f}')
    print(f'Minimum importance : {np.min(importances):.12f}')
    selector = SelectFromModel(estimator=rf, threshold='mean', prefit=True)
    selected_mask = selector.get_support()
    n_selected = int(np.sum(selected_mask))
    if n_selected <= 0:
        raise RuntimeError(f'Fold {fold_number}: Feature selection removed all features.')
    print('\n' + '-' * 80)
    print(f'FEATURE SELECTION RESULT - FOLD {fold_number}')
    print('-' * 80)
    print(f'Total features    : {TOTAL_FEATURES}')
    print(f'Selected features : {n_selected}')
    print(f'Removed features  : {TOTAL_FEATURES - n_selected}')
    print(f'Selection ratio   : {100.0 * n_selected / TOTAL_FEATURES:.2f}%')
    X_fold_train_selected = selector.transform(X_fold_train_scaled)
    X_fold_val_selected = selector.transform(X_fold_val_scaled)
    print('\nSelected matrix shapes:')
    print(f'Fold train selected : {X_fold_train_selected.shape}')
    print(f'Fold val selected   : {X_fold_val_selected.shape}')
    selected_feature_counts = get_selected_feature_counts(selected_mask, feature_ranges)
    print('\nSelected features per representation:')
    for name, count in selected_feature_counts.items():
        total_for_representation = feature_dimensions[name]
        print(f'  {name:<20}: {count} / {total_for_representation}')
    sorted_feature_indices = np.argsort(importances)[::-1]
    print('\nTop 20 features:')
    for rank, feature_index in enumerate(sorted_feature_indices[:20], start=1):
        representation_name, local_index = feature_index_to_representation(feature_index, feature_ranges)
        print(f'{rank:>2}. Global={feature_index:<5} | {representation_name:<18} local={local_index:<5} | importance={importances[feature_index]:.10f}')
    train_selected_path = os.path.join(fold_dir, 'train_selected.npy')
    val_selected_path = os.path.join(fold_dir, 'val_selected.npy')
    np.save(train_selected_path, X_fold_train_selected)
    np.save(val_selected_path, X_fold_val_selected)
    fold_indices_path = os.path.join(fold_dir, 'fold_indices.npz')
    np.savez(fold_indices_path, train_idx=train_idx, val_idx=val_idx)
    artifact = {'method': 'Random Forest Feature Selection', 'paper_method': 'AFP-MVFL', 'selection_rule': 'features with importance greater than or equal to mean importance', 'cv': True, 'fold': fold_number, 'n_splits': N_SPLITS, 'cv_random_state': SEED, 'rf_n_estimators': RF_N_ESTIMATORS, 'rf_random_state': RF_RANDOM_STATE, 'scaler': scaler, 'random_forest': rf, 'selector': selector, 'selected_mask': selected_mask, 'feature_importances': importances, 'mean_importance': mean_importance, 'threshold': 'mean', 'total_features': TOTAL_FEATURES, 'selected_features': n_selected, 'representations': list(feature_dimensions.keys()), 'representation_dimensions': feature_dimensions, 'feature_ranges': feature_ranges, 'selected_feature_counts': selected_feature_counts, 'query_embedding_model': ESM2_MODEL_NAME, 'query_embedding_dimension': int(esm2_embedding_dim), 'query_sequence_length_dimension': int(esm2_sequence_length), 'query_actual_min_length': int(np.min(esm2_lengths)), 'query_actual_max_length': int(np.max(esm2_lengths)), 'query_actual_mean_length': float(np.mean(esm2_lengths)), 'query_feature_selection': False, 'query_is_sequence_aware': True, 'complementary_feature_selection': True, 'train_samples': len(train_idx), 'validation_samples': len(val_idx), 'train_indices': train_idx, 'validation_indices': val_idx, 'train_selected_file': 'train_selected.npy', 'validation_selected_file': 'val_selected.npy', 'fold_indices_file': 'fold_indices.npz'}
    selector_path = os.path.join(fold_dir, 'feature_selector_rf.pkl')
    with open(selector_path, 'wb') as f:
        pickle.dump(artifact, f)
    mask_path = os.path.join(fold_dir, 'selected_feature_mask.npy')
    np.save(mask_path, selected_mask)
    importance_records = []
    for feature_index in range(TOTAL_FEATURES):
        representation_name, local_index = feature_index_to_representation(feature_index, feature_ranges)
        importance_records.append({'global_feature_index': feature_index, 'representation': representation_name, 'local_feature_index': local_index, 'importance': float(importances[feature_index]), 'selected': bool(selected_mask[feature_index])})
    importance_df = pd.DataFrame(importance_records)
    importance_path = os.path.join(fold_dir, 'feature_importance.csv')
    importance_df.to_csv(importance_path, index=False)
    print('\n Fold artifacts saved:')
    print(f'  Selector          : {selector_path}')
    print(f'  Selected mask     : {mask_path}')
    print(f'  Train selected    : {train_selected_path}')
    print(f'  Validation selected: {val_selected_path}')
    print(f'  Fold indices      : {fold_indices_path}')
    print(f'  Importance table  : {importance_path}')
    fold_summary.append({'Fold': fold_number, 'Train_Samples': len(train_idx), 'Validation_Samples': len(val_idx), 'Total_Features': TOTAL_FEATURES, 'Selected_Features': n_selected, 'Removed_Features': TOTAL_FEATURES - n_selected, 'Selection_Ratio': 100.0 * n_selected / TOTAL_FEATURES, 'ChemBERTa_Total': feature_dimensions['ChemBERTa'], 'ChemBERTa_Selected': selected_feature_counts['ChemBERTa'], 'ProstT5_Total': feature_dimensions['ProstT5'], 'ProstT5_Selected': selected_feature_counts['ProstT5'], 'Handcrafted_Total': feature_dimensions['Handcrafted_488'], 'Handcrafted_Selected': selected_feature_counts['Handcrafted_488'], 'RF_Estimators': RF_N_ESTIMATORS, 'RF_Random_State': RF_RANDOM_STATE, 'Mean_Importance': mean_importance, 'ESM2_Query_Dimension': esm2_embedding_dim, 'ESM2_Query_Max_Sequence_Length': esm2_sequence_length, 'ESM2_Query_Feature_Selection': False})
summary_path = os.path.join(CV_OUTPUT_DIR, 'feature_selection_cv_summary.csv')
summary_df = pd.DataFrame(fold_summary)
summary_df.to_csv(summary_path, index=False)
protocol_metadata = {'experiment': 'TLH-AFP', 'feature_selection_method': 'Random Forest Feature Selection', 'paper_method': 'AFP-MVFL', 'cv_method': 'StratifiedKFold', 'n_splits': N_SPLITS, 'cv_random_state': SEED, 'rf_n_estimators': RF_N_ESTIMATORS, 'rf_random_state': RF_RANDOM_STATE, 'feature_selection_threshold': 'mean', 'complementary_representations': list(feature_dimensions.keys()), 'complementary_feature_dimensions': feature_dimensions, 'complementary_total_features': TOTAL_FEATURES, 'query_embedding_model': ESM2_MODEL_NAME, 'query_embedding_dimension': int(esm2_embedding_dim), 'query_sequence_length_dimension': int(esm2_sequence_length), 'query_actual_min_length': int(np.min(esm2_lengths)), 'query_actual_max_length': int(np.max(esm2_lengths)), 'query_actual_mean_length': float(np.mean(esm2_lengths)), 'query_feature_selection': False, 'query_is_sequence_aware': True, 'test_data_loaded': False, 'test_data_used_for_scaling': False, 'test_data_used_for_rf': False, 'test_data_used_for_feature_selection': False}
protocol_path = os.path.join(CV_OUTPUT_DIR, 'feature_selection_protocol.pkl')
with open(protocol_path, 'wb') as f:
    pickle.dump(protocol_metadata, f)
print('\n' + '=' * 80)
print(' 5-FOLD FEATURE SELECTION COMPLETED')
print('=' * 80)
print('\nFold summary:')
for row in fold_summary:
    print(f"Fold {row['Fold']}: {row['Selected_Features']} / {row['Total_Features']} ({row['Selection_Ratio']:.2f}%) | ChemBERTa={row['ChemBERTa_Selected']} | ProstT5={row['ProstT5_Selected']} | Handcrafted={row['Handcrafted_Selected']}")
print('\n Summary saved:')
print(summary_path)
print('\n Protocol metadata saved:')
print(protocol_path)
print('\n TEST DATA')
print('Test data were NOT loaded.')
print('Test data were NOT used for scaling.')
print('Test data were NOT used for Random Forest fitting.')
print('Test data were NOT used for feature selection.')
print('\n ESM-2 QUERY PATH')
print(f'Model      : {ESM2_MODEL_NAME}')
print(f'Embedding  : {esm2_embedding_dim} dimensions')
print(f'Token axis  : {esm2_sequence_length}')
print(f'Max length : {np.max(esm2_lengths)}')
print('Feature Selection on ESM-2: NO')
print('ESM-2 is reserved for the sequence-aware Query pathway.')
print('\n EACH FOLD CONTAINS:')
print('   - fold-specific StandardScaler')
print('   - fold-specific Random Forest')
print('   - fold-specific feature selector')
print('   - selected train matrix')
print('   - selected validation matrix')
print('   - fold indices')
print('   - selected-feature mask')
print('   - complete feature-importance table')
print('\n➡ These artifacts are ready for the next fold-specific Pre-training/Training stage.')
print('=' * 80)
