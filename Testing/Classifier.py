from sklearn.metrics import f1_score, precision_recall_curve
import os
import torch
import random
import logging
import argparse
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import SAGEConv, BatchNorm, GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_undirected,to_dense_adj
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score, precision_recall_fscore_support
from data_generating import load_dataset, make_pu_dataset
from Loss_functions import ContrastiveLossWithDiffusionReliableNegatives, NeighborhoodSimilarityLoss, LearnableDiffusionContrastiveLoss, NeighborhoodConsistencyLoss, AttentionBasedDGI, LabelPropagationLoss, AdjacencyBasedLoss
from Reliable_negatives import RocchioMethod, SpyMethod, PositiveEnlargement, KMeansMethod, GenerativePUMethod, OneDNFMethod
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Optional, Dict, List, Tuple, Any
from tqdm import tqdm
import numpy as np
import logging
from pathlib import Path
import json
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss
import numpy as np
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_recall_curve

def f1_eval(preds, dtrain):
    """
    Custom evaluation metric for F1 score.
    """
    labels = dtrain.get_label()
    preds_binary = (preds > 0.5).astype(int)  # Default threshold
    return 'f1', f1_score(labels, preds_binary)

def train_xgboost_classifier(train_embeddings, train_labels, val_embeddings, val_labels, num_epochs=100, patience=10):
    # Convert tensors to numpy arrays
    train_embeddings_np = train_embeddings.detach().cpu().numpy()
    val_embeddings_np = val_embeddings.detach().cpu().numpy()
    train_labels_np = train_labels.detach().cpu().numpy()
    val_labels_np = val_labels.detach().cpu().numpy()
        
    # Create DMatrix objects
    dtrain = xgb.DMatrix(train_embeddings_np, label=train_labels_np)
    dval = xgb.DMatrix(val_embeddings_np, label=val_labels_np)

    # XGBoost parameters
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',  # This will log "logloss" alongside the custom F1.
        'max_depth': 6,
        'eta': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42,
    }

    # Training with custom evaluation metric
    watchlist = [(dtrain, 'train'), (dval, 'eval')]
    model = xgb.train(
        params,
        dtrain,
        evals=watchlist,
        feval=f1_eval,  # Custom F1 metric
        maximize=True,  # We want to maximize F1
        num_boost_round=num_epochs,
        early_stopping_rounds=patience,
        verbose_eval=0
    )

    # Evaluate on validation set
    val_preds = model.predict(dval)

    # Find optimal threshold for F1
    precision, recall, thresholds = precision_recall_curve(val_labels_np, val_preds)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_threshold = thresholds[f1_scores.argmax()]

    val_preds_binary = (val_preds > best_threshold).astype(int)
    accuracy = accuracy_score(val_labels_np, val_preds_binary)
    f1 = f1_score(val_labels_np, val_preds_binary)
    logloss = log_loss(val_labels_np, val_preds)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    logger.info(f'Validation Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, Log Loss: {logloss:.4f}')

    return model, best_threshold

def evaluate_xgboost_classifier(model, embeddings, labels, mask, threshold=0.5):
    # Convert embeddings and labels to numpy arrays
    embeddings_np = embeddings[mask].detach().cpu().numpy()
    labels_np = labels[mask].cpu().numpy()
    
    # Create DMatrix for prediction
    dtest = xgb.DMatrix(embeddings_np)
    
    # Get predictions
    preds_prob = model.predict(dtest)
    preds = (preds_prob > threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(labels_np, preds)
    f1 = f1_score(labels_np, preds)
    recall = recall_score(labels_np, preds)
    auc = roc_auc_score(labels_np, preds_prob)
    
    return accuracy, f1, recall, auc

def train_random_forest_classifier(train_embeddings, train_labels, val_embeddings, val_labels, n_estimators=100, max_depth=None):
    # Convert tensors to numpy arrays
    train_embeddings_np = train_embeddings.detach().cpu().numpy()
    val_embeddings_np = val_embeddings.detach().cpu().numpy()
    train_labels_np = train_labels.detach().cpu().numpy()
    val_labels_np = val_labels.detach().cpu().numpy()

    # Train Random Forest
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(train_embeddings_np, train_labels_np)

    # Validate and find optimal threshold for F1
    val_preds_prob = model.predict_proba(val_embeddings_np)[:, 1]
    precision, recall, thresholds = precision_recall_curve(val_labels_np, val_preds_prob)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_threshold = thresholds[f1_scores.argmax()]

    val_preds_binary = (val_preds_prob > best_threshold).astype(int)
    accuracy = accuracy_score(val_labels_np, val_preds_binary)
    f1 = f1_score(val_labels_np, val_preds_binary)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    logger.info(f'Validation Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}')

    return model, best_threshold

def evaluate_random_forest_classifier(model, embeddings, labels, mask, threshold=0.5):
    # Convert embeddings and labels to numpy arrays
    embeddings_np = embeddings[mask].detach().cpu().numpy()
    labels_np = labels[mask].cpu().numpy()

    # Predict probabilities
    preds_prob = model.predict_proba(embeddings_np)[:, 1]
    preds = (preds_prob > threshold).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(labels_np, preds)
    f1 = f1_score(labels_np, preds)
    recall = recall_score(labels_np, preds)
    auc = roc_auc_score(labels_np, preds_prob)

    return accuracy, f1, recall, auc


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_curve, accuracy_score, f1_score, recall_score, roc_auc_score
import logging

def train_logistic_regression_classifier(train_embeddings, train_labels, val_embeddings, val_labels, C_values=None, cv_folds=5):
    """
    Train a logistic regression model with cross-validation to select the best value of C.
    
    Parameters:
    - train_embeddings: Tensor with the training embeddings.
    - train_labels: Tensor with the training labels.
    - val_embeddings: Tensor with the validation embeddings.
    - val_labels: Tensor with the validation labels.
    - C_values: List of regularization strengths (default: [0.01, 0.1, 1.0, 10.0]).
    - cv_folds: Number of folds for cross-validation (default: 5).
    
    Returns:
    - model: The trained Logistic Regression model.
    - best_threshold: The optimal threshold for F1 score based on validation set.
    """
    # Convert tensors to numpy arrays
    train_embeddings_np = train_embeddings.detach().cpu().numpy()
    val_embeddings_np = val_embeddings.detach().cpu().numpy()
    train_labels_np = train_labels.detach().cpu().numpy()
    val_labels_np = val_labels.detach().cpu().numpy()

    # If no C values are provided, use a default range
    if C_values is None:
        C_values = [0.01, 0.1, 1.0, 10.0, 100.0]

    # Initialize the Logistic Regression model
    model = LogisticRegression(solver='liblinear', random_state=42)

    # Set up GridSearchCV to search over different values of C with cross-validation
    param_grid = {'C': C_values}
    grid_search = GridSearchCV(model, param_grid, cv=cv_folds, scoring='f1', verbose=1)

    # Fit the model using cross-validation
    grid_search.fit(train_embeddings_np, train_labels_np)
    
    # Get the best value of C from GridSearchCV
    best_C = grid_search.best_params_['C']
    logger = logging.getLogger(__name__)
    logger.info(f"Best value of C: {best_C}")

    # Train the final model with the best C
    final_model = LogisticRegression(C=best_C, solver='liblinear', random_state=42)
    final_model.fit(train_embeddings_np, train_labels_np)

    # Validate and find the optimal threshold for F1
    val_preds_prob = final_model.predict_proba(val_embeddings_np)[:, 1]
    precision, recall, thresholds = precision_recall_curve(val_labels_np, val_preds_prob)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_threshold = thresholds[f1_scores.argmax()]

    val_preds_binary = (val_preds_prob > best_threshold).astype(int)
    accuracy = accuracy_score(val_labels_np, val_preds_binary)
    f1 = f1_score(val_labels_np, val_preds_binary)
    logger.info(f'Validation Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}')

    return final_model, best_threshold


def evaluate_logistic_regression_classifier(model, embeddings, labels, mask, threshold=0.5):
    # Convert embeddings and labels to numpy arrays
    embeddings_np = embeddings[mask].detach().cpu().numpy()
    labels_np = labels[mask].cpu().numpy()

    # Predict probabilities
    preds_prob = model.predict_proba(embeddings_np)[:, 1]
    preds = (preds_prob > threshold).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(labels_np, preds)
    f1 = f1_score(labels_np, preds)
    recall = recall_score(labels_np, preds)
    auc = roc_auc_score(labels_np, preds_prob)

    return accuracy, f1, recall, auc
