# TLH-AFP

### Learning Beyond Sequence: A Multi-Representation and Contrastive Learning Framework for Antifungal Peptide Prediction

**Fatemeh Mahmoudian¹, Amir Lakizadeh¹***
¹ Department of Computer Engineering, Artificial Intelligence and Robotics, University of Qom, Qom, Iran
*Corresponding author: [lakizadeh@qom.ac.ir](mailto:lakizadeh@qom.ac.ir)
Fatemeh Mahmoudian: [mahmodian.f@gmail.com](mailto:mahmodian.f@gmail.com)

---

## Overview

**TLH-AFP** is a multi-representation deep learning framework for antifungal peptide (AFP) prediction.

The framework integrates **PepBERT, ChemBERTa, ProstT5, and handcrafted sequence features** to capture complementary peptide information. Random Forest-based feature selection is applied to reduce feature redundancy, followed by single-head cross-attention for structured representation fusion. Supervised contrastive pre-training is further employed to improve the discriminability of AFP and non-AFP representations.

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

TLH-AFP combines the following representation and learning components:

* **PepBERT** for peptide sequence representation
* **ChemBERTa** for molecular representation
* **ProstT5** for protein language representation
* **Handcrafted sequence features** for complementary physicochemical information
* **Random Forest feature selection** for redundancy reduction
* **Cross-attention fusion** for interaction between primary and complementary representations
* **Supervised contrastive learning** for discriminative representation pre-training

---

## Experimental Design

The final model was developed using **stratified five-fold cross-validation** with fixed random seeds.

Model selection was based on validation performance, while the independent test sets were reserved for final evaluation.

For the reported final AntiFP evaluation, the selected four-fold ensemble used:

* Folds **1–4**
* Equal fold weights of **0.25**
* Fixed decision threshold of **0.50**

For external evaluation on DeepAFP-Main, all five trained folds were combined using equal weights of **0.20** per fold.

No test-set-based threshold optimization, ensemble-weight optimization, or model-combination search was used for the reported final evaluations.

---

## Feature Selection

The complementary representation consists of **1,896 features**:

```text
ChemBERTa       : 384
ProstT5         : 1024
Handcrafted     : 488
--------------------------------
Total           : 1896
```

Random Forest-based feature selection reduces the complementary representation to **547 selected features**, corresponding to a **71.15% reduction** in dimensionality.

---

## Reproducibility

The repository provides the main computational pipelines required to reproduce the core TLH-AFP experiments:

```text
scripts/
├── feature_selection_rf.py
├── pretrain_tlh_afp.py
├── train_tlh_afp.py
├── test_tlh_afp.py
└── external_test_deepafp.py
```

The repository also contains the processed AntiFP training and test CSV files used by the experimental pipeline.

Large feature matrices and trained model checkpoints are not included in the public repository.

---

## Data

The `data/` directory contains the AntiFP training and test sample files used in this study.

The independent **DeepAFP-Main** dataset was used exclusively for external evaluation and was obtained from the original DeepAFP study.

Please refer to the original DeepAFP publication and its associated repository for the source dataset and usage information.

---

## Results

The `results/` directory contains selected numerical results and final evaluation figures, including:

* AntiFP test performance
* DeepAFP-Main external evaluation
* Ablation study results
* Final confusion matrices
* ROC and Precision–Recall curves
* Prediction distributions
* Evaluation protocols

---

## Repository Structure

```text
TLH-AFP/
├── data/
│   ├── train_smiles.csv
│   ├── test_smiles.csv
│   └── README.md
│
├── src/
│   └── README.md
│
├── scripts/
│   ├── feature_selection_rf.py
│   ├── pretrain_tlh_afp.py
│   ├── train_tlh_afp.py
│   ├── test_tlh_afp.py
│   └── external_test_deepafp.py
│
├── results/
│   ├── FINAL_TLH_AFP_TEST/
│   ├── TLH_AFP_FINAL_RESULTS.csv
│   ├── DEEPAFP_EXTERNAL_RESULTS.csv
│   ├── ABLATION_RESULTS.csv
│   └── ...
│
├── checkpoints/
├── README.md
├── requirements.txt
├── .gitignore
└── CITATION.cff
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
Corresponding author
Email: [lakizadeh@qom.ac.ir](mailto:lakizadeh@qom.ac.ir)
