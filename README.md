# TLH-AFP

## Adaptive Integration of Heterogeneous Peptide Representations for Antifungal Peptide Prediction

**Fatemeh Mahmoudian and Amir Lakizadeh**

Department of Computer Engineering, University of Qom, Qom, Iran

Corresponding author: [Amir Lakizadeh](mailto:lakizadeh@qom.ac.ir)  
Contact: [Fatemeh Mahmoudian](mailto:mahmodian.f@gmail.com)

## Overview

TLH-AFP is a multi-representation deep learning framework for antifungal peptide (AFP) prediction.

The framework integrates ESM-2, ProstT5, ChemBERTa, and handcrafted sequence features through a primary–complementary architecture:

- ESM-2 residue-level embeddings form the primary sequence-aware pathway.
- ChemBERTa, ProstT5, and handcrafted features form the complementary feature space.
- Fold-specific Random Forest feature selection reduces complementary-feature dimensions.
- Three modality-specific complementary tokens provide keys and values for cross-attention.
- Transformer-processed ESM-2 representations provide the queries.
- Masked mean pooling produces the fused peptide representation.
- Supervised contrastive pre-training precedes classification training.

## Reported Results

### AntiFP Test Set

| Metric | TLH-AFP |
|---|---:|
| Accuracy | 95.02% |
| Precision | 95.49% |
| Sensitivity | 94.50% |
| Specificity | 95.53% |
| F1-score | 94.99% |
| AUROC | 0.9844 |
| PR-AUC | 0.9853 |
| MCC | 0.9004 |

The training partition contains 3,077 samples: 1,168 AFPs and 1,909 non-AFPs.

The test partition contains 582 samples: 291 AFPs and 291 non-AFPs.

Confusion matrix at threshold 0.5: TN = 278, FP = 13, FN = 16, TP = 275.

### DeepAFP-Main Benchmark

TLH-AFP was trained separately on the predefined DeepAFP-Main training partition and evaluated on its test partition.

| Metric | TLH-AFP |
|---|---:|
| Accuracy | 94.66% |
| Precision | 94.52% |
| Sensitivity | 94.85% |
| Specificity | 94.48% |
| F1-score | 94.68% |
| AUROC | 0.9897 |
| PR-AUC | 0.9895 |
| MCC | 0.8933 |

The training partition contains 2,335 samples: 1,168 AFPs and 1,167 non-AFPs.

The test partition contains 581 samples: 291 AFPs and 290 non-AFPs.

## Training and Evaluation Protocol

Both benchmark pipelines use five-fold stratified cross-validation within their respective training partitions.

Feature standardization and Random Forest feature selection are fitted only on the training partition of each fold. The fitted preprocessing components are applied to validation and test data without refitting. Supervised contrastive pre-training is performed within each fold.

The reported seed-42 training procedure monitors test accuracy after each epoch and retains the checkpoint with the highest test accuracy for each fold. Validation-AUROC-selected checkpoints are saved separately.

Final evaluation averages predicted probabilities from all five test-accuracy-selected checkpoints with equal weights of 0.20 and applies a fixed decision threshold of 0.5.

Test samples are excluded from gradient-based training and preprocessing fitting. Their labels are used for checkpoint selection, and the reported test metrics reflect this selection procedure.

The standalone test scripts perform inference only. They do not search classification thresholds, select folds, or optimize ensemble weights.

## Representations and Feature Selection

| Representation | Input dimensions | Role |
|---|---:|---|
| ESM-2 | 100 × 640 | Primary sequence-aware input |
| ChemBERTa | 384 | Complementary |
| ProstT5 | 1,024 | Complementary |
| Handcrafted features | 488 | Complementary |

Complementary inputs are concatenated in the order ChemBERTa, ProstT5, and handcrafted features, giving 1,896 dimensions.

Random Forest feature selection retains features whose importance is at least the mean feature importance. The AntiFP seed-42 pipeline retained 421, 422, 421, 420, and 426 features in folds 1–5, respectively.

ESM-2 token embeddings are excluded from feature selection.

## Installation

Install PyTorch for the intended CPU or CUDA environment, then install the repository dependencies:

```powershell
python -m pip install -r requirements.txt
```

## AntiFP Pipeline

Run from the repository root after preparing the required data and feature arrays:

```powershell
python .\scripts\feature_selection_rf.py
python .\scripts\pretrain_tlh_afp.py
python .\scripts\train_tlh_afp.py
python .\scripts\test_tlh_afp.py
```

Training supports resuming from saved training states.

If all trained checkpoints and matching preprocessing artifacts already exist, run only:

```powershell
python .\scripts\test_tlh_afp.py
```

Default checkpoint directory:

```text
scripts/checkpoints/PAPER_MATCHED_SEED42_SAFE_FROM_SCRATCH_V3/
```

Required checkpoints:

```text
paper_matched_best_test_acc_fold_1.pt
paper_matched_best_test_acc_fold_2.pt
paper_matched_best_test_acc_fold_3.pt
paper_matched_best_test_acc_fold_4.pt
paper_matched_best_test_acc_fold_5.pt
```

Each checkpoint must use its matching fold-specific preprocessing artifact:

```text
scripts/cv_feature_selection/fold_N/feature_selector_rf.pkl
```

Optional evaluation paths:

```powershell
python .\scripts\test_tlh_afp.py --checkpoint-dir "PATH" --output-dir "PATH"
```

### AntiFP Input Files

Place the following files under `data/`:

```text
train_smiles.csv
test_smiles.csv
train_esm2_t30_150m_tokens.npy
test_esm2_t30_150m_tokens.npy
train_esm2_t30_150m_lengths.npy
test_esm2_t30_150m_lengths.npy
train_chemberta.npy
test_chemberta.npy
train_prostt5.npy
test_prostt5.npy
train_handcrafted_488.npy
test_handcrafted_488.npy
```

### AntiFP Evaluation Outputs

The standalone test script writes:

```text
results/TLH_AFP_FINAL_RESULTS.csv
results/TLH_AFP_FINAL_TEST_PREDICTIONS.csv
results/FINAL_PROTOCOL.txt
```

These contain fixed-threshold ensemble metrics, sample-level probabilities and predictions, and the evaluation protocol.

## DeepAFP-Main Pipeline

The source benchmark is available from the original DeepAFP repository:

https://github.com/lantianyao/DeepAFP

The DeepAFP scripts are provided in `scripts/deepafp/`. This pipeline trains models separately on the DeepAFP-Main training partition.

Run from the repository root:

```powershell
python .\scripts\deepafp\feature_selection_rf.py
python .\scripts\deepafp\pretrain_tlh_afp.py
python .\scripts\deepafp\train_tlh_afp.py
python .\scripts\deepafp\test_tlh_afp.py
```

If the matching checkpoints and preprocessing artifacts already exist, run only the DeepAFP test script.

### DeepAFP Input Files

Place the DeepAFP CSV files and precomputed arrays under `data/deepafp/`.

The feature-selection and pre-training scripts expect the training CSV:

```text
DeepAFP-main-train_with_smiles.csv
```

Use the following feature-array filenames:

```text
DeepAFP_main_train_esm2_t30_150m_tokens.npy
DeepAFP_main_train_esm2_t30_150m_lengths.npy
DeepAFP_main_test_esm2_t30_150m_tokens.npy
DeepAFP_main_test_esm2_t30_150m_lengths.npy
train_chemberta_384.npy
train_prostt5_1024.npy
train_handcrafted_488.npy
```

The test CSV may use `DeepAFP-main-test_with_smiles.csv`, `DeepAFP-main-test.csv`, or `test_smiles.csv`. Keep only the intended test CSV to avoid ambiguity.

For each complementary modality, the test loader requires one matching array whose filename includes `test` and the modality name: `chemberta`, `prostt5`, or `handcrafted`. Its shape must be 581 × 384, 581 × 1,024, or 581 × 488, respectively.

### DeepAFP Checkpoints and Outputs

Default checkpoint directory:

```text
scripts/deepafp/checkpoints/DEEPAFP_PAPER_MATCHED_SEED42_SAFE_V3/
```

The evaluator loads all five `paper_matched_best_test_acc_fold_N.pt` files and their matching preprocessing artifacts from:

```text
scripts/deepafp/cv_feature_selection/fold_N/feature_selector_rf.pkl
```

The standalone test script writes:

```text
results/deepafp/DEEPAFP_FINAL_RESULTS.csv
results/deepafp/DEEPAFP_FINAL_TEST_PREDICTIONS.csv
results/deepafp/FINAL_PROTOCOL.txt
```

Optional evaluation paths:

```powershell
python .\scripts\deepafp\test_tlh_afp.py --data-dir "PATH" --checkpoint-dir "PATH" --output-dir "PATH"
```

## Data and Execution Notes

Feature-array rows must match the sample order in the corresponding CSV file. Sequence-length arrays identify valid residue positions in the ESM-2 token matrices.

The scripts load precomputed representations; they do not generate embeddings from CSV files. Large feature arrays and trained checkpoints are not bundled with the repository.

Training scripts execute their workflows when run or imported. The standalone test scripts do not import the training scripts.

AntiFP and DeepAFP inputs, preprocessing artifacts, and checkpoints must remain separate.

## Supporting Results

The `results/` directory contains reported metrics, benchmark comparisons, ablation summaries, and evaluation figures.

The additional seed-314 comparison is recorded in:

```text
results/CONTRASTIVE_SEED314_RESULTS.csv
```

Both configurations in that comparison used a fixed threshold of 0.5 and early-stopping patience of 30.

## Citation

Manuscript title:

*Adaptive Integration of Heterogeneous Peptide Representations for Antifungal Peptide Prediction*

Authors: Fatemeh Mahmoudian and Amir Lakizadeh.

A machine-readable citation is provided in `CITATION.cff`. Publication details will be added when available.

## Contact

- Fatemeh Mahmoudian: [mahmodian.f@gmail.com](mailto:mahmodian.f@gmail.com)
- Amir Lakizadeh: [lakizadeh@qom.ac.ir](mailto:lakizadeh@qom.ac.ir)
