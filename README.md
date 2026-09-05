# TLH-AFP

### Learning Beyond Sequence: A Multi-Representation and Contrastive Learning Framework for Antifungal Peptide Prediction

**Fatemeh Mahmoudian¹, Amir Lakizadeh¹***  
¹ Department of Computer Engineering, Artificial Intelligence and Robotics, University of Qom, Qom, Iran  
*Corresponding author: lakizadeh@qom.ac.ir  
Fatemeh Mahmoudian: mahmodian.f@gmail.com

---

## Overview

**TLH-AFP** is a multi-representation deep learning framework for antifungal peptide (AFP) prediction.

The framework integrates **PepBERT, ChemBERTa, ProstT5, and handcrafted sequence features**, followed by Random Forest-based feature selection and single-head cross-attention fusion. Supervised contrastive pre-training is further used to improve the discriminability of AFP and non-AFP representations.

---

## Main Results

### AntiFP Test Set

| Metric | TLH-AFP |
|---|---:|
| Accuracy | **94.33%** |
| F1-Score | **94.18%** |
| AUROC | **0.9821** |
| PR-AUC | **0.9805** |
| MCC | **0.8878** |

### DeepAFP-Main External Test Set

| Metric | TLH-AFP |
|---|---:|
| Accuracy | **93.46%** |
| Precision | **93.17%** |
| Recall | **93.81%** |
| F1-Score | **93.49%** |
| AUROC | **0.9830** |
| PR-AUC | **0.9851** |
| MCC | **0.8692** |

---

## Ablation Study

The contribution of the major components was evaluated through:

- **Representation ablation:** PepBERT, ChemBERTa, and ProstT5
- **Architectural ablation:** Feature Selection, Cross-Attention, and Supervised Contrastive Learning
- **Role-Swap analysis:** Primary-pathway assignment of PepBERT versus ProstT5

---

## Reproducibility

All experiments were performed using fixed random seeds and predefined evaluation protocols.

The repository provides code for:

- Feature preprocessing and selection
- Supervised contrastive pre-training
- Five-fold model training
- Ablation experiments
- Independent AntiFP evaluation
- External DeepAFP-Main evaluation
- Figure and result generation

No test-set-based threshold or ensemble-weight optimization was used for the reported final evaluations.

---

## Repository Structure

```text
TLH-AFP/
├── data/
├── src/
├── scripts/
├── results/
├── checkpoints/
├── README.md
├── requirements.txt
├── environment.yml
└── CITATION.cff
