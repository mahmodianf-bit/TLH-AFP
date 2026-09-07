# TLH-AFP

## Learning Beyond Sequence: A Multi-Representation and Contrastive Learning Framework for Antifungal Peptide Prediction

**Fatemeh Mahmoudian¹, Amir Lakizadeh¹***
¹ Department of Computer Engineering, Artificial Intelligence and Robotics, University of Qom, Qom, Iran

*Corresponding author: [lakizadeh@qom.ac.ir](mailto:lakizadeh@qom.ac.ir)*
Fatemeh Mahmoudian: [mahmodian.f@gmail.com](mailto:mahmodian.f@gmail.com)

---

## Overview

**TLH-AFP** is a multi-representation deep learning framework for antifungal peptide (AFP) prediction.

The framework integrates **PepBERT, ChemBERTa, ProstT5, and handcrafted sequence features** to capture complementary peptide information. Random Forest-based feature selection is used to reduce feature redundancy, followed by single-head cross-attention for representation fusion. Supervised contrastive pre-training is further employed to improve the discriminability of AFP and non-AFP representations.

---

## Main Results

### Independent AntiFP Test Set

| Metric   |    TLH-AFP |
| -------- | ---------: |
| Accuracy | **94.33%** |
| F1-Score | **94.18%** |
| AUROC    | **0.9821** |
| PR-AUC   | **0.9805** |
| MCC      | **0.8878** |

### Independent DeepAFP-Main Test Set

| Metric    |    TLH-AFP |
| --------- | ---------: |
| Accuracy  | **93.46%** |
| Precision | **93.17%** |
| Recall    | **93.81%** |
| F1-Score  | **93.49%** |
| AUROC     | **0.9830** |
| PR-AUC    | **0.9851** |
| MCC       | **0.8692** |

---

## Model Components

TLH-AFP combines multiple representation and learning components:

* **PepBERT** for peptide sequence representation
* **ChemBERTa** for molecular representation
* **ProstT5** for protein-language representation
* **Handcrafted sequence features** for complementary physicochemical information
* **Random Forest feature selection** for feature reduction
* **Single-head cross-attention** for multimodal representation fusion
* **Supervised contrastive learning** for discriminative pre-training

---

## Experimental Design

The final model was developed using **stratified five-fold cross-validation** with a fixed random seed.

For the reported AntiFP evaluation, the final ensemble used folds **1–4**, with equal weights of **0.25** per fold and a fixed decision threshold of **0.50**.

For the independent DeepAFP-Main evaluation, all five folds were combined using equal weights of **0.20** per fold.

The independent test sets were reserved for final evaluation.

No test-set-based threshold optimization, ensemble-weight optimization, fold selection, or model-combination search was used for the reported final evaluations.

---

## Feature Selection

The complementary representation used for the AntiFP experiments contains **1,896 features**:

```text
ChemBERTa       : 384
ProstT5         : 1024
Handcrafted     : 488
--------------------------------
Total           : 1896
```

Random Forest-based feature selection reduces the complementary representation from **1,896 to 547 features**, corresponding to a **71.15% dimensionality reduction**.

---

## Reproducibility

The main computational pipelines are provided in the `scripts/` directory:

```text
scripts/
├── external_test_deepafp.py
├── feature_selection_rf.py
├── pretrain_tlh_afp.py
├── test_tlh_afp.py
└── train_tlh_afp.py
```

The scripts cover:

* Feature scaling and Random Forest feature selection
* Supervised contrastive pre-training
* Five-fold TLH-AFP training
* Final independent AntiFP evaluation
* External evaluation on DeepAFP-Main

Large feature matrices and trained model checkpoints are not included in the public repository.

---

## Data

The `data/` directory contains the AntiFP training and test sample files used in this study:

```text
data/
├── README.md
├── train_smiles.csv
└── test_smiles.csv
```

The independent **DeepAFP-Main** dataset was used for external evaluation and was obtained from the original DeepAFP study.

Please refer to the original DeepAFP publication and its associated repository for the source dataset and usage information.

---

## Results

Selected numerical results and evaluation figures are provided in the `results/` directory:

```text
results/
├── ABLATION_RESULTS.csv
├── COMPARISON_RESULTS.csv
├── DEEPAFP_EXTERNAL_RESULTS.csv
├── FINAL_PROTOCOL.txt
├── TLH_AFP_FINAL_Confusion_Matrix.png
├── TLH_AFP_FINAL_KDE.png
├── TLH_AFP_FINAL_RESULTS.csv
└── TLH_AFP_FINAL_ROC_PR.png
```

The repository includes the reported AntiFP test performance, external DeepAFP-Main results, ablation-study summary, benchmark comparison, and final evaluation figures.

---

## Repository Structure

```text
TLH-AFP/
├── data/
│   ├── README.md
│   ├── train_smiles.csv
│   └── test_smiles.csv
│
├── results/
│   ├── ABLATION_RESULTS.csv
│   ├── COMPARISON_RESULTS.csv
│   ├── DEEPAFP_EXTERNAL_RESULTS.csv
│   ├── FINAL_PROTOCOL.txt
│   ├── TLH_AFP_FINAL_Confusion_Matrix.png
│   ├── TLH_AFP_FINAL_KDE.png
│   ├── TLH_AFP_FINAL_RESULTS.csv
│   └── TLH_AFP_FINAL_ROC_PR.png
│
├── scripts/
│   ├── external_test_deepafp.py
│   ├── feature_selection_rf.py
│   ├── pretrain_tlh_afp.py
│   ├── test_tlh_afp.py
│   └── train_tlh_afp.py
│
├── .gitignore
├── CITATION.cff
├── README.md
└── requirements.txt
```

---

## Citation

If you use TLH-AFP in your research, please cite the associated publication:

> Mahmoudian, F., & Lakizadeh, A.
> *Learning Beyond Sequence: A Multi-Representation and Contrastive Learning Framework for Antifungal Peptide Prediction.*

A machine-readable citation file is provided in [`CITATION.cff`](CITATION.cff).

---

## Contact

**Fatemeh Mahmoudian**
Email: [mahmodian.f@gmail.com](mailto:mahmodian.f@gmail.com)

**Amir Lakizadeh**
Corresponding author: **Amir Lakizadeh**
Email: [lakizadeh@qom.ac.ir](mailto:lakizadeh@qom.ac.ir)
