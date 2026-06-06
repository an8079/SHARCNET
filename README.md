# SHARCNet (HyperDNE-RC²)

A deep network embedding framework for protein-protein interaction (PPI) networks, integrating hypergraph semantic contrastive learning with Ricci curvature graph augmentation.

SHARCNet combines Graph Convolutional Networks (GCN), Hypergraph Neural Networks (HGNN), Ricci curvature-based augmentation, and multiple contrastive learning objectives to produce high-quality protein embeddings.

## Key Features

- **Dual Encoder Architecture**: GCN encodes raw PPI topology + HGNN encodes clique-based hypergraph structure
- **Ricci Curvature Graph Augmentation (RCGA)**: Uses Ollivier-Ricci curvature to prune noisy edges and build an augmented view
- **Multi-objective Contrastive Learning**:
  - TCL (Topological Contrastive Loss): original graph vs. Ricci-augmented graph
  - HSCL (Hypergraph Semantic Contrastive Loss): soft-clustering-based positive/negative pairs
  - Alignment loss: hypergraph view aligned with Ricci-augmented view
- **Structure & Feature Reconstruction**: Dual decoders ensure embeddings retain original information
- **Multiple Feature Sources**:
  - **ESM-2 protein language model**: Uses `facebook/esm2_t33_650M_UR50D` for sequence-based node features
  - **InterPro domain features**: Uses InterPro functional domain annotations (DPFunc dataset)
- **Auto Model Download**: Cascading fallback from HuggingFace to ModelScope for painless ESM setup

## Project Structure

```
sharc/
├── code/                          # Source code
│   ├── main.py                    # PPI dataset training entry point
│   ├── main_dpfunc.py             # DPFunc function prediction training entry point
│   ├── roc.py                     # Training entry point with ROC/PR curve output
│   ├── model.py                   # HyperDNE-RC² model definition
│   ├── dataset.py                 # PPI dataset loading & ESM feature generation
│   ├── dataset_dpfunc.py          # DPFunc dataset loading (InterPro domain features)
│   ├── parser.py                  # CLI argument definitions
│   ├── utils.py                   # Utility functions (losses, graph ops, etc.)
│   └── sensitivity_analysis.py    # Hyperparameter sensitivity analysis
├── data/                          # Datasets
│   ├── c_elegans/                 # C. elegans PPI network
│   │   ├── edge_list.csv          # Edge list (source, target)
│   │   └── protein_seq.tsv        # Protein sequences
│   ├── HuRI/                      # Human Reference Interactome
│   ├── yeast/                     # Yeast PPI network
│   ├── hy/                        # Custom dataset
│   └── data_dpfunc/               # Protein function prediction dataset
│       ├── id_map.pkl             # Protein ID mapping (60K+)
│       ├── all_protein_interpros.pkl  # InterPro domain annotations
│       ├── inter_idx.pkl          # InterPro domain index
│       └── {bp,cc,mf}_*.txt      # GO annotations (BP/CC/MF)
├── result/                        # Evaluation output
├── requirements.txt               # Python dependencies
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/an8079/SHARCNET.git
cd SHARCNET
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Core dependencies:
- PyTorch >= 2.0
- Transformers >= 4.30
- ModelScope >= 1.9 (for ESM model mirror download)
- NetworkX >= 3.0
- scikit-learn >= 1.3
- GraphRicciCurvature >= 0.5.3

### 3. ESM model download (automatic on first run)

On first run, the ESM-2 protein language model (~2.5 GB) is downloaded automatically:
- **HuggingFace**: Primary source. Downloaded directly if accessible.
- **ModelScope**: Automatic fallback if HuggingFace is unreachable (faster for users in China).
- **Manual**: Pre-download the model and pass the local path via `--esm_model_name`.

```bash
# Option 1: Automatic (default behavior)
# The model facebook/esm2_t33_650M_UR50D is downloaded automatically.

# Option 2: Manual local path
python main.py --esm_model_name /path/to/local/esm2_t33_650M_UR50D
```

## Usage

### PPI network datasets

```bash
cd code
python main.py --dataset_name c_elegans
```

Common options:

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset_name` | `c_elegans` | Dataset name: c_elegans, HuRI, yeast, hy |
| `--data_path` | `../data` | Dataset root directory (relative to code/) |
| `--epochs` | `10` | Number of training epochs |
| `--learning_rate` | `1e-3` | Learning rate |
| `--embedding_dim` | `128` | Protein embedding dimension |
| `--esm_model_name` | `facebook/esm2_t33_650M_UR50D` | ESM model ID or local path |
| `--device` | auto-detect | Device: cuda / cpu |
| `--use_ricci_augmentation` | `True` | Enable Ricci curvature augmentation |
| `--link_pred_n_trials` | `5` | Number of link prediction trials |

Run `python main.py --help` for the full parameter list.

### Training with ROC/PR curve output

```bash
cd code
python roc.py --dataset_name HuRI
```

### Hyperparameter sensitivity analysis

```bash
cd code
python sensitivity_analysis.py --base_data_path ../data
```

### DPFunc protein function prediction dataset

The `data_dpfunc` directory contains a large-scale protein function prediction dataset with **60,254 proteins** and **26,203 InterPro domains** across three Gene Ontology namespaces:

| Namespace | Ontology | Train | Valid | Test |
|-----------|----------|-------|-------|------|
| `bp` | Biological Process | 47,140 | 731 | 1,312 |
| `cc` | Cellular Component | 41,539 | 633 | 1,005 |
| `mf` | Molecular Function | 33,339 | 422 | 702 |

**Features**: Binary InterPro domain vectors (26,203-dim) reduced to 1,280 dimensions via TruncatedSVD, serving as an alternative to ESM sequence features.

**Graph**: k-NN similarity graph (k=10, cosine distance) built on the SVD-reduced feature space.

```bash
# Train on the default BP namespace
python main_dpfunc.py --dataset_name data_dpfunc

# Specify CC or MF namespace
python main_dpfunc.py --dataset_name data_dpfunc --dpfunc_namespace cc
python main_dpfunc.py --dataset_name data_dpfunc --dpfunc_namespace mf

# Tune graph construction parameters
python main_dpfunc.py --dataset_name data_dpfunc \
    --dpfunc_knn_k 15 \
    --dpfunc_svd_dim 1024 \
    --min_clique_size 3
```

DPFunc-specific arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--dpfunc_knn_k` | `10` | Number of neighbors in k-NN similarity graph |
| `--dpfunc_svd_dim` | `1280` | Target dimension for InterPro feature SVD reduction |
| `--dpfunc_namespace` | `bp` | GO annotation namespace (bp / cc / mf) |
| `--min_clique_size` | `2` | Minimum clique size for hyperedge (DPFunc default: 2) |

### Custom datasets

Create a new directory under `data/` with the following files:

1. `edge_list.csv` — PPI edge list with `source` and `target` columns
2. `protein_seq.tsv` — Protein sequences with `Entry` (or `VEuPathDB`) and `Sequence` columns

```bash
python main.py --dataset_name your_dataset_name
```

## Evaluation Metrics

Model performance is evaluated via link prediction, reporting the following metrics (mean ± std over 5 trials):

- **AUC** (Area under ROC curve)
- **AUPR** (Area under PR curve)
- **F1-Score**
- **Accuracy**

## Datasets

### PPI networks

| Dataset | Species | Nodes | Edges |
|---------|---------|-------|-------|
| c_elegans | *C. elegans* (nematode) | ~3,500 | ~8,000 |
| HuRI | *H. sapiens* (human) | ~8,000 | ~50,000 |
| yeast | *S. cerevisiae* (yeast) | ~5,000 | ~30,000 |

### DPFunc function prediction

| Property | Value |
|----------|-------|
| Proteins | 60,254 |
| InterPro domains | 26,203 |
| GO namespaces | BP, CC, MF |
| Feature source | InterPro domains + SVD |
| Graph type | k-NN similarity (cosine) |

## Citation

If you use this framework in your research, please cite the relevant work.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
