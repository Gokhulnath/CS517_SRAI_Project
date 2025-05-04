#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fairness Metrics for Model Evaluation

This module provides functions to calculate various fairness metrics for
evaluating model predictions across demographic groups.
"""

import numpy as np
from collections import defaultdict


def demographic_parity(y_pred, sensitive_attributes):
    """
    Calculate demographic parity (statistical parity) across demographic groups.
    
    Demographic parity requires that the prediction is independent of the
    protected attribute. It measures if the positive prediction rate is the
    same across different demographic groups.
    
    Args:
        y_pred: Binary prediction vector
        sensitive_attributes: Array of sensitive attributes
        
    Returns:
        dict: Dictionary mapping group names to their positive prediction rates
    """
    # Get unique demographic groups (assuming one-hot encoded)
    demographic_groups = {}
    
    # If sensitive attributes is one-hot encoded
    if len(sensitive_attributes.shape) > 1 and sensitive_attributes.shape[1] > 1:
        for i in range(sensitive_attributes.shape[1]):
            group_indices = np.where(sensitive_attributes[:, i] == 1)[0]
            demographic_groups[f'group_{i}'] = group_indices
    else:
        # If sensitive attributes is a single column
        unique_groups = np.unique(sensitive_attributes)
        for group in unique_groups:
            group_indices = np.where(sensitive_attributes == group)[0]
            demographic_groups[f'group_{group}'] = group_indices
    
    # Calculate positive prediction rate for each group
    parity_metrics = {}
    for group_name, indices in demographic_groups.items():
        if len(indices) > 0:
            positive_rate = np.mean(y_pred[indices])
            parity_metrics[group_name] = positive_rate
    
    return parity_metrics


def equalized_odds(y_pred, y_true, sensitive_attributes):
    """
    Calculate equalized odds across demographic groups.
    
    Equalized odds requires that the prediction and the protected attribute
    are independent conditional on the true label. It measures if TPR and FPR
    are the same across different demographic groups.
    
    Args:
        y_pred: Binary prediction vector
        y_true: Binary ground truth vector
        sensitive_attributes: Array of sensitive attributes
        
    Returns:
        dict: Dictionary mapping group names to their TPR and FPR
    """
    # Get unique demographic groups (assuming one-hot encoded)
    demographic_groups = {}
    
    # If sensitive attributes is one-hot encoded
    if len(sensitive_attributes.shape) > 1 and sensitive_attributes.shape[1] > 1:
        for i in range(sensitive_attributes.shape[1]):
            group_indices = np.where(sensitive_attributes[:, i] == 1)[0]
            demographic_groups[f'group_{i}'] = group_indices
    else:
        # If sensitive attributes is a single column
        unique_groups = np.unique(sensitive_attributes)
        for group in unique_groups:
            group_indices = np.where(sensitive_attributes == group)[0]
            demographic_groups[f'group_{group}'] = group_indices
    
    # Calculate TPR and FPR for each group
    odds_metrics = {}
    for group_name, indices in demographic_groups.items():
        if len(indices) > 0:
            # True Positive Rate (TPR): TP / (TP + FN)
            pos_indices = np.where(y_true[indices] == 1)[0]
            if len(pos_indices) > 0:
                tpr = np.mean(y_pred[indices][pos_indices])
            else:
                tpr = np.nan
            
            # False Positive Rate (FPR): FP / (FP + TN)
            neg_indices = np.where(y_true[indices] == 0)[0]
            if len(neg_indices) > 0:
                fpr = np.mean(y_pred[indices][neg_indices])
            else:
                fpr = np.nan
            
            odds_metrics[group_name] = {'tpr': tpr, 'fpr': fpr}
    
    return odds_metrics


def predictive_parity(y_pred, y_true, sensitive_attributes):
    """
    Calculate predictive parity across demographic groups.
    
    Predictive parity requires that the outcome and the protected attribute
    are independent conditional on the prediction. It measures if PPV 
    (precision) is the same across different demographic groups.
    
    Args:
        y_pred: Binary prediction vector
        y_true: Binary ground truth vector
        sensitive_attributes: Array of sensitive attributes
        
    Returns:
        dict: Dictionary mapping group names to their PPV (precision)
    """
    # Get unique demographic groups (assuming one-hot encoded)
    demographic_groups = {}
    
    # If sensitive attributes is one-hot encoded
    if len(sensitive_attributes.shape) > 1 and sensitive_attributes.shape[1] > 1:
        for i in range(sensitive_attributes.shape[1]):
            group_indices = np.where(sensitive_attributes[:, i] == 1)[0]
            demographic_groups[f'group_{i}'] = group_indices
    else:
        # If sensitive attributes is a single column
        unique_groups = np.unique(sensitive_attributes)
        for group in unique_groups:
            group_indices = np.where(sensitive_attributes == group)[0]
            demographic_groups[f'group_{group}'] = group_indices
    
    # Calculate Positive Predictive Value (PPV) for each group
    parity_metrics = {}
    for group_name, indices in demographic_groups.items():
        if len(indices) > 0:
            # Positive Predictive Value (PPV): TP / (TP + FP)
            pred_pos_indices = np.where(y_pred[indices] == 1)[0]
            if len(pred_pos_indices) > 0:
                ppv = np.mean(y_true[indices][pred_pos_indices])
            else:
                ppv = np.nan
            
            parity_metrics[group_name] = ppv
    
    return parity_metrics


def fairness_disparity(metrics_dict):
    """
    Calculate disparity between groups for a fairness metric.
    
    Disparity is measured as the maximum absolute difference between
    any two groups' metric values.
    
    Args:
        metrics_dict: Dictionary of metric values per group
        
    Returns:
        float: Maximum disparity between groups
    """
    values = list(metrics_dict.values())
    
    # Handle cases where metrics might be dictionaries (like equalized odds)
    if isinstance(values[0], dict):
        # Extract TPR and FPR values
        tpr_values = [v['tpr'] for v in values if not np.isnan(v['tpr'])]
        fpr_values = [v['fpr'] for v in values if not np.isnan(v['fpr'])]
        
        if len(tpr_values) > 1:
            tpr_disparity = max(tpr_values) - min(tpr_values)
        else:
            tpr_disparity = 0
            
        if len(fpr_values) > 1:
            fpr_disparity = max(fpr_values) - min(fpr_values)
        else:
            fpr_disparity = 0
            
        return {'tpr_disparity': tpr_disparity, 'fpr_disparity': fpr_disparity}
    else:
        # Filter out NaN values
        values = [v for v in values if not np.isnan(v)]
        
        if len(values) > 1:
            return max(values) - min(values)
        else:
            return 0


def fairness_ratio(metrics_dict):
    """
    Calculate the ratio between the minimum and maximum values of a metric.
    
    A ratio closer to 1 indicates more fairness.
    
    Args:
        metrics_dict: Dictionary of metric values per group
        
    Returns:
        float: Ratio between min and max values (min/max)
    """
    values = list(metrics_dict.values())
    
    # Handle cases where metrics might be dictionaries (like equalized odds)
    if isinstance(values[0], dict):
        # Extract TPR and FPR values
        tpr_values = [v['tpr'] for v in values if not np.isnan(v['tpr'])]
        fpr_values = [v['fpr'] for v in values if not np.isnan(v['fpr'])]
        
        if len(tpr_values) > 1 and max(tpr_values) > 0:
            tpr_ratio = min(tpr_values) / max(tpr_values)
        else:
            tpr_ratio = 1.0
            
        if len(fpr_values) > 1 and max(fpr_values) > 0:
            fpr_ratio = min(fpr_values) / max(fpr_values)
        else:
            fpr_ratio = 1.0
            
        return {'tpr_ratio': tpr_ratio, 'fpr_ratio': fpr_ratio}
    else:
        # Filter out NaN values
        values = [v for v in values if not np.isnan(v)]
        
        if len(values) > 1 and max(values) > 0:
            return min(values) / max(values)
        else:
            return 1.0


def fairness_summary(y_pred, y_true, sensitive_attributes):
    """
    Generate a comprehensive fairness summary across multiple metrics.
    
    Args:
        y_pred: Binary prediction vector
        y_true: Binary ground truth vector
        sensitive_attributes: Array of sensitive attributes
        
    Returns:
        dict: Dictionary of fairness metrics and disparities
    """
    # Calculate various fairness metrics
    dp = demographic_parity(y_pred, sensitive_attributes)
    eo = equalized_odds(y_pred, y_true, sensitive_attributes)
    pp = predictive_parity(y_pred, y_true, sensitive_attributes)
    
    # Calculate disparities
    dp_disparity = fairness_disparity(dp)
    eo_disparity = fairness_disparity(eo)
    pp_disparity = fairness_disparity(pp)
    
    # Calculate ratios
    dp_ratio = fairness_ratio(dp)
    eo_ratio = fairness_ratio(eo)
    pp_ratio = fairness_ratio(pp)
    
    # Compile summary
    summary = {
        'demographic_parity': {
            'values': dp,
            'disparity': dp_disparity,
            'ratio': dp_ratio
        },
        'equalized_odds': {
            'values': eo,
            'disparity': eo_disparity,
            'ratio': eo_ratio
        },
        'predictive_parity': {
            'values': pp,
            'disparity': pp_disparity,
            'ratio': pp_ratio
        }
    }
    
    return summary 