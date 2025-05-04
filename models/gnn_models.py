#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GNN Models for Disease Prediction

This module implements various Graph Neural Network architectures for
disease prediction tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool


class GCNModel(torch.nn.Module):
    """
    Graph Convolutional Network (GCN) for disease prediction
    """
    def __init__(self, num_node_features, num_classes, hidden_dim=64, dropout_rate=0.5):
        """
        Initialize GCN model
        
        Args:
            num_node_features: Number of input features per node
            num_classes: Number of output classes (1 for binary classification)
            hidden_dim: Dimension of hidden layers
            dropout_rate: Dropout probability
        """
        super(GCNModel, self).__init__()
        
        # Graph Convolutional layers
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # Output layer for classification
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        # Regularization
        self.dropout = nn.Dropout(dropout_rate)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim)
    
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, num_node_features]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch vector for multiple graphs [num_nodes]
            
        Returns:
            torch.Tensor: Predictions
        """
        # First GCN layer with ReLU and dropout
        x = self.conv1(x, edge_index)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Second GCN layer
        x = self.conv2(x, edge_index)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # If dealing with multiple graphs, perform pooling
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        # Classification
        x = self.classifier(x)
        
        return x


class GATModel(torch.nn.Module):
    """
    Graph Attention Network (GAT) for disease prediction
    """
    def __init__(self, num_node_features, num_classes, hidden_dim=64, heads=8, dropout_rate=0.5):
        """
        Initialize GAT model
        
        Args:
            num_node_features: Number of input features per node
            num_classes: Number of output classes (1 for binary classification)
            hidden_dim: Dimension of hidden layers
            heads: Number of attention heads
            dropout_rate: Dropout probability
        """
        super(GATModel, self).__init__()
        
        # First GAT layer with multiple attention heads
        self.conv1 = GATConv(num_node_features, hidden_dim // heads, heads=heads, dropout=dropout_rate)
        
        # Second GAT layer that combines the heads
        self.conv2 = GATConv(hidden_dim, hidden_dim, heads=1, dropout=dropout_rate)
        
        # Output layer for classification
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        # Regularization
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, num_node_features]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch vector for multiple graphs [num_nodes]
            
        Returns:
            torch.Tensor: Predictions
        """
        # First GAT layer with ELU activation
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)
        
        # Second GAT layer
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)
        
        # If dealing with multiple graphs, perform pooling
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        # Classification
        x = self.classifier(x)
        
        return x


class FairGNN(torch.nn.Module):
    """
    Fair Graph Neural Network with adversarial debiasing
    """
    def __init__(self, base_model, num_node_features, num_classes, sensitive_dim,
                 hidden_dim=64, adversary_dim=32, lambda_fairness=1.0, dropout_rate=0.5):
        """
        Initialize Fair GNN model with adversarial debiasing
        
        Args:
            base_model: Base GNN model type ('gcn' or 'gat')
            num_node_features: Number of input features per node
            num_classes: Number of output classes (1 for binary classification)
            sensitive_dim: Dimension of sensitive attribute (e.g., gender, race)
            hidden_dim: Dimension of hidden layers
            adversary_dim: Dimension of adversary hidden layer
            lambda_fairness: Weight for adversarial loss
            dropout_rate: Dropout probability
        """
        super(FairGNN, self).__init__()
        
        # Initialize base GNN model
        if base_model == 'gcn':
            self.gnn = GCNModel(num_node_features, num_classes, hidden_dim, dropout_rate)
        elif base_model == 'gat':
            self.gnn = GATModel(num_node_features, num_classes, hidden_dim, heads=8, dropout_rate=dropout_rate)
        else:
            raise ValueError(f"Unknown base model: {base_model}")
        
        # Adversarial network to predict protected attributes
        self.adversary = nn.Sequential(
            nn.Linear(hidden_dim, adversary_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(adversary_dim, sensitive_dim)
        )
        
        # Fairness regularization weight
        self.lambda_fairness = lambda_fairness
        
        # Save model parameters for access in forward pass
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass of the fair GNN
        
        Returns the main model prediction and the adversary prediction
        """
        # Extract node embeddings from the base GNN (before classification layer)
        if isinstance(self.gnn, GCNModel):
            # Get the GCN embeddings before classification
            h = self.gnn.conv1(x, edge_index)
            h = self.gnn.batch_norm1(h)
            h = F.relu(h)
            h = self.gnn.dropout(h)
            
            h = self.gnn.conv2(h, edge_index)
            h = self.gnn.batch_norm2(h)
            h = F.relu(h)
            embeddings = self.gnn.dropout(h)
            
        elif isinstance(self.gnn, GATModel):
            # Get the GAT embeddings before classification
            h = self.gnn.conv1(x, edge_index)
            h = F.elu(h)
            h = self.gnn.dropout(h)
            
            h = self.gnn.conv2(h, edge_index)
            h = F.elu(h)
            embeddings = self.gnn.dropout(h)
        
        # If dealing with multiple graphs, perform pooling
        if batch is not None:
            embeddings_pooled = global_mean_pool(embeddings, batch)
        else:
            embeddings_pooled = embeddings
        
        # Main task prediction
        y_pred = self.gnn.classifier(embeddings_pooled)
        
        # Adversarial prediction of sensitive attributes
        s_pred = self.adversary(embeddings_pooled.detach())  # Detach to prevent backprop to GNN during adversary training
        
        return y_pred, s_pred, embeddings
    
    def adversarial_loss(self, s_pred, s_true):
        """
        Calculate adversarial loss for protected attribute prediction
        
        Returns binary cross entropy loss for adversary network
        """
        return F.cross_entropy(s_pred, s_true)


def load_model(model_type, num_features, num_classes, num_sensitive_classes=2):
    """
    Factory function to load the appropriate model
    
    Args:
        model_type: Type of model ('gcn', 'gat', 'fair_gcn', 'fair_gat')
        num_features: Number of input features
        num_classes: Number of output classes
        num_sensitive_classes: Number of sensitive attribute classes
        
    Returns:
        torch.nn.Module: Initialized model
    """
    if model_type == 'gcn':
        return GCNModel(num_features, num_classes)
    elif model_type == 'gat':
        return GATModel(num_features, num_classes)
    elif model_type == 'fair_gcn':
        return FairGNN('gcn', num_features, num_classes, num_sensitive_classes)
    elif model_type == 'fair_gat':
        return FairGNN('gat', num_features, num_classes, num_sensitive_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}") 