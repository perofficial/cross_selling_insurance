"""
Insurance Cross-Selling Prediction Model
=========================================
Author: Matteo Peroni
Company: AssurePredict
Description: Production-ready pipeline for predicting vehicle insurance 
             cross-selling opportunities for existing insurance customers.

Features:
    - Advanced feature engineering with interaction terms
    - Target encoding for high-cardinality categorical variables
    - Log transformation for skewed numerical features
    - Threshold optimization for imbalanced classification
    - Comprehensive model evaluation metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve
)

# Configuration
RANDOM_SEED = 42
TEST_SIZE = 0.2
DATASET_URL = "https://proai-datasets.s3.eu-west-3.amazonaws.com/insurance_cross_sell.csv"

# Plotting configuration
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def load_and_clean_data(url: str) -> pd.DataFrame:
    """
    Load dataset from URL and perform initial cleaning.
    
    Args:
        url: Dataset URL
        
    Returns:
        Cleaned DataFrame
    """
    print("=" * 80)
    print("1. DATA LOADING AND CLEANING")
    print("=" * 80)
    
    df = pd.read_csv(url)
    
    # Remove non-informative ID column
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
    
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Missing values:\n{df.isna().sum()}")
    
    return df


def analyze_and_transform_skewness(df: pd.DataFrame, 
                                   column: str = 'Annual_Premium') -> pd.DataFrame:
    """
    Analyze skewness and apply log transformation if needed.
    
    Args:
        df: Input DataFrame
        column: Column name to analyze
        
    Returns:
        DataFrame with transformed column
    """
    print("\n" + "=" * 80)
    print("2. SKEWNESS ANALYSIS AND TRANSFORMATION")
    print("=" * 80)
    
    original_skew = skew(df[column])
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Original distribution
    sns.histplot(df[column], kde=True, ax=axes[0], color='#e74c3c', bins=50)
    axes[0].set_title(f'Original Distribution\nSkewness: {original_skew:.2f}', 
                     fontsize=12, fontweight='bold')
    axes[0].set_xlabel(f'{column} (€)')
    
    # Log-transformed distribution
    log_values = np.log1p(df[column])
    log_skew_val = skew(log_values)
    
    sns.histplot(log_values, kde=True, ax=axes[1], color='#2ecc71', bins=50)
    axes[1].set_title(f'Log-Transformed Distribution\nSkewness: {log_skew_val:.2f}', 
                     fontsize=12, fontweight='bold')
    axes[1].set_xlabel(f'Log({column})')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Skewness - Original: {original_skew:.4f} -> Transformed: {log_skew_val:.4f}")
    print("Decision: Applying log transformation\n")
    
    # Apply transformation
    df[f'{column}_Log'] = np.log1p(df[column])
    df = df.drop(column, axis=1)
    
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered features before train-test split.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with engineered features
    """
    print("=" * 80)
    print("3. FEATURE ENGINEERING (Pre-Split)")
    print("=" * 80)
    
    # One-hot encoding for low-cardinality categorical variables
    categorical_cols = ['Gender', 'Vehicle_Age', 'Vehicle_Damage']
    df_processed = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Create interaction features
    if 'Vehicle_Damage_Yes' in df_processed.columns:
        df_processed['Age_x_Damage'] = (df_processed['Age'] * 
                                        df_processed['Vehicle_Damage_Yes'])
        print("✓ Interaction feature 'Age_x_Damage' created")
    
    print(f"Total features after engineering: {df_processed.shape[1] - 1}\n")
    
    return df_processed


def apply_target_encoding(X_train: pd.DataFrame, 
                         X_test: pd.DataFrame,
                         y_train: pd.Series,
                         columns: list) -> tuple:
    """
    Apply target encoding to high-cardinality categorical variables.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training target
        columns: Columns to encode
        
    Returns:
        Tuple of (X_train_encoded, X_test_encoded)
    """
    print("=" * 80)
    print("4. TARGET ENCODING (High-Cardinality Features)")
    print("=" * 80)
    
    X_train = X_train.copy()
    X_test = X_test.copy()
    
    for col in columns:
        # Calculate encoding on training set only
        encoding_map = y_train.groupby(X_train[col]).mean()
        global_mean = y_train.mean()
        
        # Apply encoding
        X_train[f'{col}_Encoded'] = X_train[col].map(encoding_map)
        X_test[f'{col}_Encoded'] = X_test[col].map(encoding_map).fillna(global_mean)
        
        # Drop original columns
        X_train = X_train.drop(col, axis=1)
        X_test = X_test.drop(col, axis=1)
        
        print(f"✓ {col} encoded and replaced")
    
    print()
    return X_train, X_test


def scale_features(X_train: pd.DataFrame, 
                  X_test: pd.DataFrame) -> tuple:
    """
    Standardize features using StandardScaler.
    
    Args:
        X_train: Training features
        X_test: Test features
        
    Returns:
        Tuple of (X_train_scaled, X_test_scaled, scaler)
    """
    print("=" * 80)
    print("5. FEATURE SCALING")
    print("=" * 80)
    
    scaler = StandardScaler()
    cols = X_train.columns
    
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), 
        columns=cols
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), 
        columns=cols
    )
    
    print("✓ Features standardized (mean=0, std=1)\n")
    
    return X_train_scaled, X_test_scaled, scaler


def train_model(X_train: pd.DataFrame, 
               y_train: pd.Series,
               C: float = 1.0) -> LogisticRegression:
    """
    Train logistic regression model with class balancing.
    
    Args:
        X_train: Training features
        y_train: Training target
        C: Regularization parameter (inverse of regularization strength)
        
    Returns:
        Trained model
    """
    print("=" * 80)
    print("6. MODEL TRAINING")
    print("=" * 80)
    
    model = LogisticRegression(
        solver='saga',
        class_weight='balanced',
        C=C,
        random_state=RANDOM_SEED,
        max_iter=5000
    )
    
    model.fit(X_train, y_train)
    
    print(f"✓ Logistic Regression trained")
    print(f"  - Regularization (C): {C}")
    print(f"  - Class weights: balanced")
    print(f"  - Solver: saga\n")
    
    return model


def optimize_threshold(y_true: pd.Series, 
                      y_pred_proba: np.ndarray) -> tuple:
    """
    Find optimal classification threshold by maximizing F1 score.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        
    Returns:
        Tuple of (optimal_threshold, best_f1_score)
    """
    print("=" * 80)
    print("7. THRESHOLD OPTIMIZATION")
    print("=" * 80)
    
    # Probability diagnostics
    print("Probability Distribution:")
    print(f"  Min:  {y_pred_proba.min():.4f}")
    print(f"  Mean: {y_pred_proba.mean():.4f}")
    print(f"  Max:  {y_pred_proba.max():.4f}\n")
    
    # Calculate precision-recall curve
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    # Align arrays (precision/recall have one extra element)
    precisions = precisions[:-1]
    recalls = recalls[:-1]
    
    # Calculate F1 scores
    with np.errstate(divide='ignore', invalid='ignore'):
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
        f1_scores = np.nan_to_num(f1_scores)
    
    # Find optimal threshold
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    print(f"Optimal Threshold: {best_threshold:.4f}")
    print(f"Best F1 Score: {best_f1:.4f}\n")
    
    # Visualization
    plt.figure(figsize=(10, 5))
    plt.plot(thresholds, f1_scores, label='F1 Score', color='purple', linewidth=2)
    plt.axvline(best_threshold, color='red', linestyle='--', 
                label=f'Optimal: {best_threshold:.2f}')
    plt.title('Threshold Optimization: F1 Score vs Decision Threshold', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Probability Threshold')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return best_threshold, best_f1


def evaluate_model(y_true: pd.Series, 
                  y_pred: np.ndarray,
                  y_pred_proba: np.ndarray,
                  threshold: float) -> dict:
    """
    Generate comprehensive model evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities
        threshold: Classification threshold used
        
    Returns:
        Dictionary with evaluation metrics
    """
    print("=" * 80)
    print("8. MODEL EVALUATION")
    print("=" * 80)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    # ROC-AUC Score
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    print(f"ROC-AUC Score: {roc_auc:.4f}\n")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                annot_kws={'size': 16})
    plt.title(f'Confusion Matrix (Threshold: {threshold:.2f})', 
              fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()
    
    # Prediction statistics
    print(f"Test Set Size: {len(y_true)}")
    print(f"Positive Predictions (Cross-Sell Yes): {y_pred.sum()}")
    print(f"Prediction Rate: {y_pred.sum() / len(y_true):.2%}\n")
    
    metrics = {
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'threshold': threshold
    }
    
    return metrics


def main():
    """Main execution pipeline."""
    
    print("\n" + "=" * 80)
    print("INSURANCE CROSS-SELLING PREDICTION MODEL")
    print("AssurePredict - Production Pipeline")
    print("=" * 80 + "\n")
    
    # 1. Load and clean data
    df = load_and_clean_data(DATASET_URL)
    
    # 2. Transform skewed features
    df = analyze_and_transform_skewness(df, 'Annual_Premium')
    
    # 3. Feature engineering (pre-split)
    df_processed = engineer_features(df)
    
    # 4. Prepare X and y
    X = df_processed.drop('Response', axis=1)
    y = df_processed['Response']
    
    # 5. Train-test split (stratified)
    print("=" * 80)
    print("TRAIN-TEST SPLIT")
    print("=" * 80)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_SEED, 
        stratify=y
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Class distribution (train): {y_train.value_counts().to_dict()}\n")
    
    # 6. Target encoding
    high_cardinality_cols = ['Region_Code', 'Policy_Sales_Channel']
    X_train, X_test = apply_target_encoding(
        X_train, X_test, y_train, high_cardinality_cols
    )
    
    # 7. Feature scaling
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # 8. Train model
    model = train_model(X_train_scaled, y_train, C=1.0)
    
    # 9. Generate predictions
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # 10. Optimize threshold
    best_threshold, best_f1 = optimize_threshold(y_test, y_pred_proba)
    
    # 11. Final predictions with optimal threshold
    y_pred = (y_pred_proba >= best_threshold).astype(int)
    
    # 12. Evaluate model
    metrics = evaluate_model(y_test, y_pred, y_pred_proba, best_threshold)
    
    print("=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)
    
    return model, scaler, metrics, best_threshold


if __name__ == "__main__":
    model, scaler, metrics, threshold = main()
