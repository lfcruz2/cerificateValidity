# Import necessary libraries
import pandas as pd
import numpy as np
import time
from difflib import SequenceMatcher
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

# Machine Learning algorithms
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Functions to calculate similarities
def string_similarity(a, b):
    """Calculate the similarity between two strings using SequenceMatcher."""
    if pd.isnull(a) or pd.isnull(b):
        return np.nan
    return SequenceMatcher(None, str(a), str(b)).ratio()

def numeric_similarity(a, b):
    """Calculate the normalized similarity between two numeric values."""
    if pd.isnull(a) or pd.isnull(b):
        return np.nan
    max_val = max(abs(a), abs(b))
    if max_val == 0:
        return 1.0
    return 1 - abs(a - b) / max_val

# Load or prepare your DataFrame 'df_pairs' and 'df_non_pairs'
# 'df_pairs' contains pairs that are known to be matches (is_pair=1)
# 'df_non_pairs' contains pairs that are known not to be matches (is_pair=0)
# For demonstration purposes, let's assume you have these DataFrames ready

# Example:
# df_pairs = pd.read_csv('pairs.csv')
# df_non_pairs = pd.read_csv('non_pairs.csv')

# Combine the DataFrames
df = pd.concat([df_pairs, df_non_pairs], ignore_index=True)

# List of columns to compare
columns_to_compare = ['col1', 'col2', 'col3']  # Replace with your column names

# Step 1: Calculate similarities and create features
print("Calculating similarities...")
for col in columns_to_compare:
    col_left = f"{col}_left"
    col_right = f"{col}_right"
    similarity_col = f"{col}_similarity"

    if df[col_left].dtype == 'object':
        # For string columns
        df[similarity_col] = df.apply(
            lambda row: string_similarity(row[col_left], row[col_right]), axis=1
        )
    elif np.issubdtype(df[col_left].dtype, np.number):
        # For numeric columns
        df[similarity_col] = df.apply(
            lambda row: numeric_similarity(row[col_left], row[col_right]), axis=1
        )
    else:
        # For other data types, check equality
        df[similarity_col] = df.apply(
            lambda row: 1.0 if row[col_left] == row[col_right] else 0.0, axis=1
        )

# Drop rows with NaN similarities
df.dropna(subset=[f"{col}_similarity" for col in columns_to_compare], inplace=True)

# Step 2: Prepare features and labels
feature_cols = [f"{col}_similarity" for col in columns_to_compare]
X = df[feature_cols]
y = df['is_pair']

# Optional: Scale features (important for some algorithms)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Split data into training and test sets
print("Splitting data into training and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Step 4: Define the models and hyperparameters to iterate over
models = [
    {
        'name': 'RandomForest',
        'estimator': RandomForestClassifier(random_state=42),
        'param_grid': {
            'n_estimators': [100, 200],
            'max_depth': [None, 5, 10],
        }
    },
    {
        'name': 'LogisticRegression',
        'estimator': LogisticRegression(random_state=42, max_iter=1000),
        'param_grid': {
            'C': [0.1, 1, 10],
            'penalty': ['l2'],
            'solver': ['lbfgs'],
        }
    },
    {
        'name': 'SVM',
        'estimator': SVC(random_state=42, probability=True),
        'param_grid': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
        }
    },
    {
        'name': 'KNN',
        'estimator': KNeighborsClassifier(),
        'param_grid': {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance'],
        }
    },
    {
        'name': 'DecisionTree',
        'estimator': DecisionTreeClassifier(random_state=42),
        'param_grid': {
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5],
        }
    },
    {
        'name': 'XGBoost',
        'estimator': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
        'param_grid': {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.01, 0.1],
        }
    },
]

# Step 5: Iterate over models, train, evaluate, and save results
results_list = []

for model_info in models:
    model_name = model_info['name']
    estimator = model_info['estimator']
    param_grid = model_info['param_grid']

    print(f"\nTraining model: {model_name}")

    # Measure training time
    start_time = time.time()

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)
    best_estimator = grid_search.best_estimator_
    train_time = time.time() - start_time

    # Measure prediction time
    start_time = time.time()
    y_pred = best_estimator.predict(X_test)
    y_proba = best_estimator.predict_proba(X_test)[:, 1]
    test_time = time.time() - start_time

    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)

    # Save results
    result = {
        'model': model_name,
        'best_params': grid_search.best_params_,
        'features_used': feature_cols,
        'train_time_sec': train_time,
        'test_time_sec': test_time,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc,
    }

    results_list.append(result)

    # Save individual model results to CSV
    result_df = pd.DataFrame([result])
    result_df.to_csv(f'results_{model_name}.csv', index=False)

    print(f"Model {model_name} trained and evaluated.")
    print(f"AUC-ROC: {auc:.4f}")
    print(f"Best hyperparameters: {grid_search.best_params_}")

# Step 6: Combine all results into a single DataFrame
print("\nCombining results from all models...")
all_results = pd.DataFrame(results_list)
all_results.to_csv('all_model_results.csv', index=False)

print("\nAll models have been trained and evaluated.")
print("Results saved to 'all_model_results.csv'.")
print("\nSummary of results:")
print(all_results)
