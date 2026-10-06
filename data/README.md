# Data

## AntiFP Dataset

This directory contains the AntiFP sample files prepared for the TLH-AFP experiments:

- `train_smiles.csv`: 3,077 samples, comprising 1,168 AFPs and 1,909 non-AFPs.
- `test_smiles.csv`: 582 samples, comprising 291 AFPs and 291 non-AFPs.

Labels are encoded as `1` for AFP and `0` for non-AFP.

## Peptide Representations

TLH-AFP uses four representation sources:

- ESM-2: token-level embeddings with shape 100 × 640 per sample.
- ChemBERTa: 384-dimensional molecular representations.
- ProstT5: 1,024-dimensional protein representations.
- Handcrafted sequence features: 488 dimensions.

The complementary feature vector contains 1,896 dimensions, concatenated in the order ChemBERTa, ProstT5, and handcrafted features. ESM-2 embeddings are retained separately as the primary sequence-aware input.

Large feature arrays are not included in this repository. The training and evaluation scripts load precomputed arrays; they do not generate embeddings from the CSV files.

## Required Feature Files

For each split (`train` and `test`), the AntiFP pipeline requires:

```text
{split}_esm2_t30_150m_tokens.npy
{split}_esm2_t30_150m_lengths.npy
{split}_chemberta.npy
{split}_prostt5.npy
{split}_handcrafted_488.npy
```

Array rows must follow the same sample order as the corresponding CSV file. Sequence lengths identify valid residue positions in the ESM-2 token matrices.

Standardization and feature selection are fitted separately on the training partition of each cross-validation fold. The fitted components are applied to validation and test data without refitting.

## DeepAFP-Main Benchmark

DeepAFP-Main was used as an additional benchmark with its predefined training and test partitions:

- Training: 2,335 samples, comprising 1,168 AFPs and 1,167 non-AFPs.
- Test: 581 samples, comprising 291 AFPs and 290 non-AFPs.

TLH-AFP was trained separately on the DeepAFP-Main training partition and evaluated on its test partition.

The source dataset and associated documentation are available from the original DeepAFP repository:

https://github.com/lantianyao/DeepAFP

## Data Organization

```text
data/
├── README.md
├── train_smiles.csv
└── test_smiles.csv
```

For checkpoint selection and ensemble evaluation details, see the main repository README.
