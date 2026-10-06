# TLH-AFP

## Adaptive Integration of Heterogeneous Peptide Representations for Antifungal Peptide Prediction

**Fatemeh Mahmoudian and Amir Lakizadeh**

Department of Computer Engineering, University of Qom, Qom, Iran

Corresponding author: [Amir Lakizadeh](mailto:lakizadeh@qom.ac.ir)  
Contact: [Fatemeh Mahmoudian](mailto:mahmodian.f@gmail.com)

## Overview

TLH-AFP is a multi-representation deep learning framework for antifungal peptide (AFP) prediction.

The framework combines ESM-2, ProstT5, ChemBERTa, and handcrafted sequence features through a structured primary–complementary architecture:

- ESM-2 residue-level embeddings form the primary sequence-aware pathway.
- ProstT5, ChemBERTa, and handcrafted features form the complementary feature space.
- Fold-specific Random Forest feature selection reduces the complementary dimensions.
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

The test set contains 582 samples: 291 AFPs and 291 non-AFPs.

Confusion matrix: TN = 278, FP = 13, FN = 16, TP = 275.

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

The training partition contains 2,335 samples, and the test partition contains 581 samples.

## Training and Evaluation Protocol

The pipeline uses five-fold stratified cross-validation.

Feature standardization and Random Forest feature selection are fitted only on the training partition of each fold. The fitted preprocessing components are applied to the corresponding validation and test data without refitting.

The reported seed-42 training procedure monitors test accuracy after each epoch and retains the checkpoint with the highest test accuracy for each fold. Validation-AUROC-selected checkpoints are also saved separately.

Final evaluation uses all five test-accuracy-selected checkpoints, averages their predicted probabilities with equal weights of 0.20, and applies a fixed decision threshold of 0.5.

The test set is separate from gradient-based training, but its labels are used for checkpoint selection. The reported test metrics therefore reflect this selection procedure.

The standalone test script performs inference only. It does not search classification thresholds, select folds, or optimize ensemble weights.

## Representations and Feature Selection

| Representation | Input dimensions | Role |
|---|---:|---|
| ESM-2 | 100 × 640 | Primary sequence-aware input |
| ChemBERTa | 384 | Complementary |
| ProstT5 | 1,024 | Complementary |
| Handcrafted features | 488 | Complementary |

The complementary inputs are concatenated in the order ChemBERTa, ProstT5, and handcrafted features, giving 1,896 dimensions.

Random Forest feature selection retains features whose importance is at least the mean feature importance. The AntiFP seed-42 pipeline retained 421, 422, 421, 420, and 426 features in folds 1–5, respectively.

ESM-2 token embeddings are excluded from this feature-selection procedure.

## Running the AntiFP Pipeline

Run the following commands from the repository root after installing the dependencies and preparing the required feature arrays:

```powershell
python .\scripts\feature_selection_rf.py
python .\scripts\pretrain_tlh_afp.py
python .\scripts\train_tlh_afp.py
python .\scripts\test_tlh_afp.py
```

The training script supports resuming from its saved training state.

If the trained checkpoints and matching preprocessing artifacts already exist, run only:

```powershell
python .\scripts\test_tlh_afp.py
```

The default checkpoint directory is:

```text
scripts/checkpoints/PAPER_MATCHED_SEED42_SAFE_FROM_SCRATCH_V3/
```

The test script requires all five files:

```text
paper_matched_best_test_acc_fold_1.pt
paper_matched_best_test_acc_fold_2.pt
paper_matched_best_test_acc_fold_3.pt
paper_matched_best_test_acc_fold_4.pt
paper_matched_best_test_acc_fold_5.pt
```

Each checkpoint must be paired with the preprocessing artifact from its corresponding fold:

```text
scripts/cv_feature_selection/fold_N/feature_selector_rf.pkl
```

Optional evaluation paths can be specified as follows:

```powershell
python .\scripts\test_tlh_afp.py --checkpoint-dir "PATH" --output-dir "PATH"
```

Importing `train_tlh_afp.py` executes its training workflow. The standalone test script does not import the training script.

## Required Data

The AntiFP pipeline expects the following files under `data/`:

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

Feature-array rows must match the sample order in the corresponding CSV file.

Large feature arrays and trained checkpoints are not bundled with the repository. Running the pipeline requires these inputs; the CSV files alone are insufficient.

## Evaluation Outputs

The standalone AntiFP test script writes:

```text
results/TLH_AFP_FINAL_RESULTS.csv
results/TLH_AFP_FINAL_TEST_PREDICTIONS.csv
results/FINAL_PROTOCOL.txt
```

These files contain the fixed-threshold ensemble metrics, sample-level probabilities and predictions, and the evaluation protocol.

## DeepAFP-Main Data

The DeepAFP-Main benchmark originates from the DeepAFP study:

https://github.com/lantianyao/DeepAFP

The DeepAFP results above were obtained through a separate training and evaluation pipeline using that benchmark's predefined partitions. They are not outputs of the AntiFP test script.

## Citation

Manuscript title:

*Adaptive Integration of Heterogeneous Peptide Representations for Antifungal Peptide Prediction*

Authors: Fatemeh Mahmoudian and Amir Lakizadeh.

A machine-readable citation is provided in `CITATION.cff`. Publication details should be added when available.

## Contact

- Fatemeh Mahmoudian: [mahmodian.f@gmail.com](mailto:mahmodian.f@gmail.com)
- Amir Lakizadeh: [lakizadeh@qom.ac.ir](mailto:lakizadeh@qom.ac.ir)
