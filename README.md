# Fair GNN Models for Disease Prediction

This project develops fair Graph Neural Network (GNN) models for disease prediction using patient interaction networks derived from electronic health records. We address fairness issues in healthcare AI by ensuring disease progression predictions are equally accurate across demographic groups.

## Dataset
- MIMIC-III Clinical Database

## Project Structure
- `data/`: Scripts for data processing and network construction
- `models/`: GNN model implementations (GCN, GAT)
- `utils/`: Utility functions for data handling and evaluation
- `notebooks/`: Exploratory data analysis and visualization
- `results/`: Model outputs and fairness analyses

## Mid-Report Goals
- Extract relevant patient cohorts and features
- Construct patient similarity networks
- Analyze demographic distribution
- Implement standard GNN architectures
- Train baseline disease prediction models
- Evaluate performance across demographic groups
- Document performance disparities 