# Data

This directory contains the datasets used in the TLH-AFP experiments.

## AntiFP Dataset

The `train_smiles.csv` and `test_smiles.csv` files contain the AntiFP training and test samples prepared for this study.

The corresponding peptide representations used by TLH-AFP include:

- PepBERT
- ChemBERTa
- ProstT5
- Handcrafted sequence features

The derived feature matrices are generated during the preprocessing pipeline and are not included as separate large binary files in this repository.

## External Dataset

DeepAFP-Main was used as an independent external dataset for evaluating the generalization capability of TLH-AFP.

The DeepAFP-Main dataset was obtained from the original DeepAFP study. Please refer to the original publication and its associated repository for the source dataset and usage information.

## Data Organization

```text
data/
├── train_smiles.csv
├── test_smiles.csv
└── README.md
