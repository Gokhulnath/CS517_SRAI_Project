# Getting Started with Fair GNN Disease Prediction

This guide will help you set up and run the fair GNN disease prediction project.

## Prerequisites

- Python 3.8+ 
- MIMIC-III Clinical Database access
- PyTorch 1.10+
- PyTorch Geometric

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Processing

1. Extract and process the MIMIC-III data:
```bash
python -m data.extract_mimic
```

This script will:
- Extract relevant patient features from MIMIC-III
- Construct patient similarity networks
- Create processed data files in `data/processed/`

## Demographic Analysis

1. Launch Jupyter Notebook:
```bash
jupyter notebook
```

2. Open and run the demographic analysis notebook:
```
notebooks/demographic_analysis.ipynb
```

This notebook will analyze the demographic distribution of the dataset and identify potential sources of bias.

## Model Training

To train the baseline GCN model:
```bash
python -m models.train --model gcn --epochs 100 --lr 0.01
```

To train the fair version with adversarial debiasing:
```bash
python -m models.train --model fair_gcn --epochs 100 --lr 0.01 --lambda_fairness 1.0
```

### Available Models
- `gcn`: Graph Convolutional Network
- `gat`: Graph Attention Network
- `fair_gcn`: Fair GCN with adversarial debiasing
- `fair_gat`: Fair GAT with adversarial debiasing

### Training Parameters
- `--epochs`: Number of training epochs
- `--lr`: Learning rate
- `--weight_decay`: Weight decay for regularization
- `--lambda_fairness`: Weight for fairness loss term
- `--test_size`: Proportion of data for testing
- `--output_dir`: Directory to save results

## Evaluation

After training, the model performance and fairness metrics will be saved in the `results/` directory. You'll find:

- Standard performance metrics (AUC, precision, recall, F1)
- Fairness metrics across demographic groups
- Visualizations of demographic parity and equalized odds

## Mid-Report Deliverables

For the mid-report, we will complete:
1. Data extraction and processing
2. Demographic analysis
3. Implementation of GNN architectures
4. Training of baseline models
5. Evaluation of model performance across demographic groups
