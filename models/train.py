#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training script for GNN models with fairness metrics

This script handles the training of different GNN architectures and evaluates
their performance with respect to both accuracy and fairness metrics.
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Import local modules
from models.gnn_models import load_model
from utils.fairness_metrics import demographic_parity, equalized_odds, predictive_parity


def load_data(data_dir='data/processed/'):
    """
    Load processed patient data and network structure
    
    Args:
        data_dir: Directory containing processed data
        
    Returns:
        tuple: (node_features, edge_index, targets, sensitive_attributes)
    """
    print(f"Loading data from {data_dir}")
    
    # Load patient features (node features)
    features_path = os.path.join(data_dir, 'patient_features.csv')
    patient_features = pd.read_csv(features_path)
    print(f"Loaded {len(patient_features)} patient feature records")
    
    # Extract ID column and convert to node features tensor
    subject_ids = patient_features['SUBJECT_ID'].values
    
    # Drop non-feature columns (ID and any sensitive attributes we'll handle separately)
    sensitive_columns = ['GENDER_F', 'GENDER_M']  # Example sensitive attributes
    
    # Get numeric columns only for node features
    numeric_columns = patient_features.select_dtypes(include=['int64', 'float64']).columns
    feature_columns = [col for col in numeric_columns 
                      if col != 'SUBJECT_ID' and col not in sensitive_columns]
    
    # Convert to tensor, ensuring all values are numeric
    node_features = torch.tensor(patient_features[feature_columns].values, dtype=torch.float)
    
    # Extract sensitive attributes
    sensitive_attrs = torch.tensor(patient_features[sensitive_columns].values, dtype=torch.float)
    
    # Load network structure (edges)
    edges_path = os.path.join(data_dir, 'network_edges.csv')
    edges = pd.read_csv(edges_path)
    print(f"Loaded {len(edges)} network edges")
    
    # Create mapping from SUBJECT_ID to node index
    id_to_idx = {id_val: idx for idx, id_val in enumerate(subject_ids)}
    
    # Convert edge list to edge index tensor [2, num_edges]
    edge_index = []
    for _, row in edges.iterrows():
        source_idx = id_to_idx.get(row['source'])
        target_idx = id_to_idx.get(row['target'])
        if source_idx is not None and target_idx is not None:
            edge_index.append([source_idx, target_idx])
            # Add reverse edge for undirected graph
            edge_index.append([target_idx, source_idx])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    
    # Load target conditions
    targets_path = os.path.join(data_dir, 'target_conditions.csv')
    target_conditions = pd.read_csv(targets_path)
    print(f"Loaded {len(target_conditions)} target condition records")
    
    # Create mapping from condition to index
    condition_to_idx = {
        'diabetes': 0,
        'heart_failure': 1,
        'respiratory_failure': 2,
        'kidney_disease': 3,
        'sepsis': 4
    }
    
    # Create target tensor (multi-label classification)
    targets = torch.zeros((len(subject_ids), len(condition_to_idx)), dtype=torch.float)
    
    for _, row in target_conditions.iterrows():
        subject_idx = id_to_idx.get(row['SUBJECT_ID'])
        condition_idx = condition_to_idx.get(row['CONDITION'])
        if subject_idx is not None and condition_idx is not None:
            targets[subject_idx, condition_idx] = row['HAS_CONDITION']
    
    return node_features, edge_index, targets, sensitive_attrs, subject_ids


def prepare_data_for_training(node_features, edge_index, targets, sensitive_attrs, test_size=0.2, val_size=0.1):
    """
    Prepare data for training by splitting into train/val/test sets
    
    Args:
        node_features: Node feature tensor
        edge_index: Edge index tensor
        targets: Target tensor
        sensitive_attrs: Sensitive attributes tensor
        test_size: Proportion of data for testing
        val_size: Proportion of training data for validation
    
    Returns:
        tuple: (train_data, val_data, test_data)
    """
    # Create a PyTorch Geometric Data object
    data = Data(x=node_features, edge_index=edge_index, y=targets, s=sensitive_attrs)
    
    # Split the nodes into train/val/test
    num_nodes = node_features.shape[0]
    indices = list(range(num_nodes))
    
    # First split into train+val / test
    train_val_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=42)
    
    # Then split train+val into train / val
    train_idx, val_idx = train_test_split(train_val_idx, test_size=val_size/(1-test_size), random_state=42)
    
    # Create train, val, test masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    
    return data


def train_model(model, data, optimizer, device, epochs=100, patience=10, lambda_fairness=1.0):
    """
    Train a GNN model
    
    Args:
        model: GNN model to train
        data: PyTorch Geometric Data object
        optimizer: Optimizer
        device: Device to run training on
        epochs: Number of training epochs
        patience: Early stopping patience
        lambda_fairness: Weight for fairness loss (only for FairGNN)
        
    Returns:
        tuple: (trained_model, train_losses, val_losses)
    """
    # Move data to device
    data = data.to(device)
    model = model.to(device)
    
    # For tracking training progress
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        
        # Forward pass depends on model type
        if hasattr(model, 'adversary'):  # FairGNN
            y_pred, s_pred, _ = model(data.x, data.edge_index)
            
            # Main task loss (for nodes in training set)
            task_loss = F.binary_cross_entropy_with_logits(
                y_pred[data.train_mask], data.y[data.train_mask]
            )
            
            # Adversarial loss (using sensitive attributes)
            s_true = torch.argmax(data.s[data.train_mask], dim=1)
            adv_loss = model.adversarial_loss(s_pred[data.train_mask], s_true)
            
            # Combined loss with gradient reversal for adversary
            loss = task_loss - lambda_fairness * adv_loss
        else:  # Standard GNN
            y_pred = model(data.x, data.edge_index)
            loss = F.binary_cross_entropy_with_logits(
                y_pred[data.train_mask], data.y[data.train_mask]
            )
        
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        
        # Validation
        model.eval()
        with torch.no_grad():
            if hasattr(model, 'adversary'):  # FairGNN
                y_pred, _, _ = model(data.x, data.edge_index)
            else:  # Standard GNN
                y_pred = model(data.x, data.edge_index)
            
            val_loss = F.binary_cross_entropy_with_logits(
                y_pred[data.val_mask], data.y[data.val_mask]
            )
            val_losses.append(val_loss.item())
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'models/best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    # Load best model
    model.load_state_dict(torch.load('models/best_model.pt'))
    
    return model, train_losses, val_losses


def evaluate_model(model, data, device, demographic_groups=None):
    """
    Evaluate model performance with both standard and fairness metrics
    
    Args:
        model: Trained GNN model
        data: PyTorch Geometric Data object
        device: Device to run evaluation on
        demographic_groups: List of column indices for demographic groups to analyze
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    model.eval()
    data = data.to(device)
    
    with torch.no_grad():
        # Get predictions
        if hasattr(model, 'adversary'):  # FairGNN
            y_pred, _, _ = model(data.x, data.edge_index)
        else:  # Standard GNN
            y_pred = model(data.x, data.edge_index)
        
        y_pred = torch.sigmoid(y_pred)
        y_pred_binary = (y_pred > 0.5).float()
    
    # Calculate standard metrics on test set
    y_true = data.y[data.test_mask].cpu().numpy()
    y_pred_prob = y_pred[data.test_mask].cpu().numpy()
    y_pred_binary = y_pred_binary[data.test_mask].cpu().numpy()
    
    # Calculate accuracy for each condition
    accuracy_scores = []
    for i in range(y_true.shape[1]):
        accuracy = np.mean(y_true[:, i] == y_pred_binary[:, i])
        accuracy_scores.append(accuracy)
    
    # Calculate AUC for each condition
    auc_scores = []
    for i in range(y_true.shape[1]):
        if np.sum(y_true[:, i]) > 0:  # Ensure there are positive samples
            auc = roc_auc_score(y_true[:, i], y_pred_prob[:, i])
            auc_scores.append(auc)
    
    # Calculate precision, recall, F1
    metrics = {}
    metrics['accuracy'] = np.mean(accuracy_scores)  # Average accuracy across conditions
    metrics['auc'] = np.mean(auc_scores)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred_binary, average='weighted',zero_division=0
    )
    
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1'] = f1
    
    # Fairness metrics
    if demographic_groups is not None:
        # Extract sensitive attributes
        sensitive_attrs = data.s[data.test_mask].cpu().numpy()
        
        # Calculate fairness metrics for each condition
        fairness_metrics = {}
        
        for i in range(y_true.shape[1]):
            condition_metrics = {}
            
            # Demographic parity
            dp = demographic_parity(y_pred_binary[:, i], sensitive_attrs)
            condition_metrics['demographic_parity'] = dp
            
            # Equalized odds
            eo = equalized_odds(y_pred_binary[:, i], y_true[:, i], sensitive_attrs)
            condition_metrics['equalized_odds'] = eo
            
            # Predictive parity
            pp = predictive_parity(y_pred_binary[:, i], y_true[:, i], sensitive_attrs)
            condition_metrics['predictive_parity'] = pp
            
            fairness_metrics[f'condition_{i}'] = condition_metrics
        
        metrics['fairness'] = fairness_metrics
    
    
    return metrics


def plot_results(metrics, train_losses, val_losses, output_dir='results/', model_type='gcn'):
    """
    Plot and save training results and fairness metrics
    
    Args:
        metrics: Dictionary of evaluation metrics
        train_losses: Training losses
        val_losses: Validation losses
        output_dir: Directory to save plots
        model_type: Type of model (gcn, gat, fair_gcn, fair_gat)
    """
    # Create model-specific subdirectory
    model_dir = os.path.join(output_dir, model_type)
    os.makedirs(model_dir, exist_ok=True)
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_type.upper()} Training and Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'training_curves.png'))
    
    # Plot standard metrics
    std_metrics = ['accuracy', 'auc', 'precision', 'recall', 'f1']
    # print("Metrics")
    # print(metrics[m] for m in std_metrics)
    plt.figure(figsize=(10, 6))
    plt.bar(std_metrics, [metrics[m] for m in std_metrics])
    plt.ylim(0, 1)
    plt.title(f'{model_type.upper()} Performance Metrics')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'performance_metrics.png'))
    
    # Plot fairness metrics if available
    if 'fairness' in metrics:
        for condition, condition_metrics in metrics['fairness'].items():
            plt.figure(figsize=(12, 6))
            
            # Demographic parity
            dp_values = condition_metrics['demographic_parity']
            sns.barplot(x=list(dp_values.keys()), y=list(dp_values.values()))
            plt.title(f'{model_type.upper()} Demographic Parity - {condition}')
            plt.ylabel('Prediction Rate')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(model_dir, f'{condition}_demographic_parity.png'))
            
            # Equalized odds
            eo_values = condition_metrics['equalized_odds']
            groups = list(eo_values.keys())
            tpr_list = [v['tpr'] for v in eo_values.values()]
            fpr_list = [v['fpr'] for v in eo_values.values()]
            
            plt.figure(figsize=(12, 6))
            width = 0.35
            x = np.arange(len(groups))
            plt.bar(x - width/2, tpr_list, width, label='TPR')
            plt.bar(x + width/2, fpr_list, width, label='FPR')
            plt.xlabel('Demographic Group')
            plt.ylabel('Rate')
            plt.title(f'{model_type.upper()} Equalized Odds - {condition}')
            plt.xticks(x, groups, rotation=45)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(model_dir, f'{condition}_equalized_odds.png'))
    
    plt.close('all')
    
    # Save metrics as CSV
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1'],
        'Value': [metrics['accuracy'], metrics['auc'], metrics['precision'], metrics['recall'], metrics['f1']]
    })
    metrics_df.to_csv(os.path.join(model_dir, 'metrics.csv'), index=False)


def main(args):
    """Main function to run the training pipeline"""
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f'Using device: {device}')
    
    # Load data
    node_features, edge_index, targets, sensitive_attrs, subject_ids = load_data(args.data_dir)
    
    # Prepare data for training
    data = prepare_data_for_training(
        node_features, edge_index, targets, sensitive_attrs, 
        test_size=args.test_size, val_size=args.val_size
    )
    
    # Initialize model
    num_features = node_features.shape[1]
    num_classes = targets.shape[1]
    num_sensitive_classes = sensitive_attrs.shape[1]
    
    model = load_model(args.model, num_features, num_classes, num_sensitive_classes)
    print(f'Using model: {args.model} with {num_features} features and {num_classes} target classes')
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Train model
    model, train_losses, val_losses = train_model(
        model, data, optimizer, device, 
        epochs=args.epochs, patience=args.patience, 
        lambda_fairness=args.lambda_fairness
    )
    
    # Evaluate model
    metrics = evaluate_model(model, data, device, demographic_groups=[0, 1])  # Assuming sensitive_attrs columns
    
    print(metrics)
    for x in metrics:
        print(x)
    # Create model-specific output directory
    model_output_dir = os.path.join(args.output_dir, args.model)
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Plot and save results
    plot_results(metrics, train_losses, val_losses, args.output_dir, args.model)
    
    # Save model in model-specific directory
    model_path = os.path.join(model_output_dir, f'{args.model}_final.pt')
    torch.save(model.state_dict(), model_path)
    
    print(f"Results saved to {model_output_dir}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    
    # Plot and save results
    plot_results(metrics, train_losses, val_losses, args.output_dir, args.model)
    
    # Save model
    torch.save(model.state_dict(), os.path.join(args.output_dir, f'{args.model}_final.pt'))
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train GNN models for disease prediction')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data/processed/',
                        help='Directory containing processed data')
    parser.add_argument('--output_dir', type=str, default='results/',
                        help='Directory to save results')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='gcn',
                        choices=['gcn', 'gat', 'fair_gcn', 'fair_gat'],
                        help='GNN model architecture')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--lambda_fairness', type=float, default=1.0,
                        help='Fairness loss weight')
    
    # Dataset arguments
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Test set size ratio')
    parser.add_argument('--val_size', type=float, default=0.1,
                        help='Validation set size ratio')
    
    # Misc arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA')
    
    args = parser.parse_args()
    main(args) 