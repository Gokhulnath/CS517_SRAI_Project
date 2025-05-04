#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MIMIC-III Data Extraction Script

This script extracts patient data from the MIMIC-III database and performs
initial preprocessing for GNN-based disease prediction.
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# Define MIMIC data paths
# You'll need to update these paths to point to your MIMIC-III data location
MIMIC_PATH = 'data/raw/'

def load_patients(mimic_path=MIMIC_PATH):
    """
    Load patient demographic information from PATIENTS.csv and ADMISSIONS.csv
    
    Returns:
        pd.DataFrame: Patient demographic data
    """
    # Load data
    patients = pd.read_csv(os.path.join(mimic_path, 'PATIENTS.csv'))
    admissions = pd.read_csv(os.path.join(mimic_path, 'ADMISSIONS.csv'))
    
    # Convert dates to datetime
    patients['DOB'] = pd.to_datetime(patients['DOB'])
    admissions['ADMITTIME'] = pd.to_datetime(admissions['ADMITTIME'])
    
    # Calculate age at admission using a safer approach
    def calculate_age(row):
        try:
            # Use datetime's timedelta in days to avoid overflow
            delta = row['ADMITTIME'] - patients.loc[patients['SUBJECT_ID'] == row['SUBJECT_ID'], 'DOB'].iloc[0]
            return delta.days / 365.25
        except:
            return None
    
    # Apply age calculation row by row to avoid vectorized operation that causes overflow
    admissions['AGE_AT_ADMIT'] = admissions.apply(calculate_age, axis=1)
    
    # Option 1: Use age at first admission
    first_admissions = (admissions.sort_values('ADMITTIME')
                       .groupby('SUBJECT_ID')
                       .first()
                       .reset_index())
    
    # Option 2: Use median age across admissions
    # median_age = admissions.groupby('SUBJECT_ID')['AGE_AT_ADMIT'].median().reset_index()
    
    # Bin the ages
    first_admissions['AGE_GROUP'] = pd.cut(
        first_admissions['AGE_AT_ADMIT'],
        bins=[0, 20, 40, 60, 80, 150],
        labels=['0-20', '20-40', '40-60', '60-80', '80+'],
        include_lowest=True
    )
    
    # Merge with other demographic info
    patients_clean = patients[['SUBJECT_ID', 'GENDER']].merge(
        first_admissions[['SUBJECT_ID', 'AGE_GROUP', 'INSURANCE', 
                         'LANGUAGE', 'RELIGION', 'MARITAL_STATUS', 'ETHNICITY']],
        on='SUBJECT_ID',
        how='left'
    )
    
    return patients_clean

def load_admissions(mimic_path=MIMIC_PATH):
    """
    Load hospital admission data
    
    Returns:
        pd.DataFrame: Hospital admission data
    """
    admissions_path = os.path.join(mimic_path, 'ADMISSIONS.csv')
    print(f"Loading admissions data from {admissions_path}")
    
    admissions = pd.read_csv(admissions_path)
    print(f"Loaded {len(admissions)} admission records")
    
    return admissions

def load_diagnoses(mimic_path=MIMIC_PATH):
    """
    Load patient diagnosis data
    
    Returns:
        pd.DataFrame: Patient diagnosis data with ICD-9 codes
    """
    diagnoses_path = os.path.join(mimic_path, 'DIAGNOSES_ICD.csv')
    print(f"Loading diagnoses data from {diagnoses_path}")
    
    diagnoses = pd.read_csv(diagnoses_path)
    print(f"Loaded {len(diagnoses)} diagnosis records")
    
    # Map ICD codes to readable names
    d_icd_diagnoses_path = os.path.join(mimic_path, 'D_ICD_DIAGNOSES.csv')
    d_icd_diagnoses = pd.read_csv(d_icd_diagnoses_path)
    
    # Merge to get diagnosis names
    diagnoses = diagnoses.merge(d_icd_diagnoses, on='ICD9_CODE', how='left')
    
    return diagnoses

def extract_target_conditions(diagnoses_df):
    """
    Extract target conditions for prediction
    
    Args:
        diagnoses_df: DataFrame containing diagnoses
        
    Returns:
        pd.DataFrame: Patients with target conditions
    """
    # Define ICD-9 codes for common conditions we want to predict
    target_conditions = {
        'diabetes': ['250%'],  # Diabetes mellitus
        'heart_failure': ['428%'],  # Heart failure
        'respiratory_failure': ['518.81', '518.84'],  # Respiratory failure
        'kidney_disease': ['585%'],  # Chronic kidney disease
        'sepsis': ['995.91', '995.92'],  # Sepsis
    }
    
    patient_conditions = {}
    
    for condition, codes in target_conditions.items():
        # Get patients with this condition
        condition_patients = set()
        
        for code in codes:
            if code.endswith('%'):
                # Use string matching for wildcards
                code_base = code.replace('%', '')
                matches = diagnoses_df[diagnoses_df['ICD9_CODE'].astype(str).str.startswith(code_base)]
            else:
                # Exact match
                matches = diagnoses_df[diagnoses_df['ICD9_CODE'] == code]
            
            condition_patients.update(matches['SUBJECT_ID'].unique())
        
        print(f"Found {len(condition_patients)} patients with {condition}")
        patient_conditions[condition] = list(condition_patients)
    
    # Convert to DataFrame
    result = []
    for condition, patients in patient_conditions.items():
        for patient_id in patients:
            result.append({'SUBJECT_ID': patient_id, 'CONDITION': condition, 'HAS_CONDITION': 1})
    
    return pd.DataFrame(result)

def create_patient_feature_matrix(patients_df, diagnoses_df, admissions_df):
    """
    Create patient feature matrix for GNN input
    
    Returns:
        pd.DataFrame: Patient features
    """
    # Get patient demographics
    features = patients_df.copy()
    
    # Count number of admissions per patient
    admission_counts = admissions_df['SUBJECT_ID'].value_counts().reset_index()
    admission_counts.columns = ['SUBJECT_ID', 'ADMISSION_COUNT']
    features = features.merge(admission_counts, on='SUBJECT_ID', how='left')
    features['ADMISSION_COUNT'] = features['ADMISSION_COUNT'].fillna(0)
    
    # Get unique diagnoses per patient
    diagnoses_per_patient = diagnoses_df.groupby('SUBJECT_ID')['ICD9_CODE'].nunique().reset_index()
    diagnoses_per_patient.columns = ['SUBJECT_ID', 'DIAGNOSIS_COUNT']
    features = features.merge(diagnoses_per_patient, on='SUBJECT_ID', how='left')
    features['DIAGNOSIS_COUNT'] = features['DIAGNOSIS_COUNT'].fillna(0)
    
    # One-hot encode categorical variables
    cat_columns = ['GENDER', 'AGE_GROUP', 'INSURANCE', 'MARITAL_STATUS']
    features_encoded = pd.get_dummies(features, columns=cat_columns, drop_first=False)
    
    return features_encoded

def build_patient_similarity_network(patient_features, similarity_threshold=0.7):
    """
    Build patient similarity network based on feature similarity
    
    Args:
        patient_features: DataFrame with patient features
        similarity_threshold: Threshold for creating edges
        
    Returns:
        tuple: (nodes, edges) for network construction
    """
    from sklearn.metrics.pairwise import cosine_similarity
    import networkx as nx
    
    # Select numerical features for similarity calculation
    numerical_features = patient_features.select_dtypes(include=['int64', 'float64'])
    numerical_features = numerical_features.drop(columns=['SUBJECT_ID'], errors='ignore')
    
    # Normalize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(numerical_features)
    
    # Calculate similarities (this can be memory-intensive for large datasets)
    # For a large dataset, consider using approximate nearest neighbors or batching
    print("Calculating patient similarities...")
    patient_ids = patient_features['SUBJECT_ID'].values
    
    # For demo purposes with potentially large datasets, sample if needed
    if len(patient_ids) > 1000:
        sample_size = 1000
        print(f"Dataset too large, sampling {sample_size} patients for network construction. You can change this in extract_mimic.py.")
        sample_indices = np.random.choice(len(patient_ids), sample_size, replace=False)
        patient_ids = patient_ids[sample_indices]
        features_scaled = features_scaled[sample_indices]
    
    # Build similarity matrix
    similarities = cosine_similarity(features_scaled)
    
    # Create network
    G = nx.Graph()
    
    # Add nodes
    for i, patient_id in enumerate(patient_ids):
        G.add_node(patient_id)
    
    # Add edges based on similarity threshold
    edges = []
    for i in range(len(patient_ids)):
        for j in range(i+1, len(patient_ids)):
            if similarities[i, j] > similarity_threshold:
                G.add_edge(patient_ids[i], patient_ids[j], weight=similarities[i, j])
                edges.append((patient_ids[i], patient_ids[j], similarities[i, j]))
    
    print(f"Created network with {len(G.nodes)} nodes and {len(G.edges)} edges")
    
    # Convert to format suitable for PyTorch Geometric
    nodes = pd.DataFrame({'SUBJECT_ID': list(G.nodes())})
    edges_df = pd.DataFrame(edges, columns=['source', 'target', 'weight'])
    
    return nodes, edges_df

def main():
    """Main function to extract and process MIMIC-III data"""
    print("Starting MIMIC-III data extraction")
    
    # Check if MIMIC path exists
    if not os.path.exists(MIMIC_PATH):
        print(f"Error: MIMIC path {MIMIC_PATH} does not exist.")
        print("Please set the MIMIC_PATH environment variable or update the script.")
        return
    
    # Load data
    patients = load_patients()
    admissions = load_admissions()
    diagnoses = load_diagnoses()
    
    # Extract target conditions
    target_conditions = extract_target_conditions(diagnoses)
    
    # Create feature matrix
    patient_features = create_patient_feature_matrix(patients, diagnoses, admissions)
    
    # Build patient similarity network
    nodes, edges = build_patient_similarity_network(patient_features)
    
    # Save processed data
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed')
    os.makedirs(output_dir, exist_ok=True)
    
    patients.to_csv(os.path.join(output_dir, 'patients.csv'), index=False)
    target_conditions.to_csv(os.path.join(output_dir, 'target_conditions.csv'), index=False)
    patient_features.to_csv(os.path.join(output_dir, 'patient_features.csv'), index=False)
    nodes.to_csv(os.path.join(output_dir, 'network_nodes.csv'), index=False)
    edges.to_csv(os.path.join(output_dir, 'network_edges.csv'), index=False)
    
    print("Data extraction and processing complete")

if __name__ == "__main__":
    main() 