import pandas as pd
import numpy as np
import os
import time
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,
                             recall_score, f1_score)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from scipy.stats import randint, uniform

# Get the current timestamp and format it
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# Create a log file name with the timestamp
log_file_name = f'training_joint_kfold_only_{current_time}.log'

# Configure logging
logging.basicConfig(filename=log_file_name, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Set random seed for reproducibility
np.random.seed(42)

# ----------------------------------------
# Step 1: Load the Dataset
# ----------------------------------------

# Load the dataset and drop missing values
data = pd.read_csv('spatial_and_temporal_features_from_joint_data.csv').dropna()

# ----------------------------------------
# Step 2: Prepare Features and Labels
# ----------------------------------------

# Extract labels
y = LabelEncoder().fit_transform(data['ExerciseLabel'])

# Drop irrelevant columns
irrelevant_columns = ['SubjectID', 'ExerciseLabel', 'FrameNum']
X = data.drop(columns=irrelevant_columns)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#########################################
# Features without temporal features (Optional)
temporal_feature_prefixes = ['Delta']
non_temporal_features = [col for col in X.columns if not any(col.startswith(prefix) for prefix in temporal_feature_prefixes)]

X_non_temporal = X[non_temporal_features]
X_non_temporal_scaled = scaler.fit_transform(X_non_temporal)
#########################################

# ----------------------------------------
# Step 3: Split the Dataset
# ----------------------------------------

# Split the dataset into training and test sets
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42)

logging.info(f"Training set: {len(X_train_full)} samples")
logging.info(f"Test set: {len(X_test)} samples")

# ----------------------------------------
# Step 4: Feature Selection
# ----------------------------------------

# Feature selection using Random Forest
logging.info("Starting feature selection with Random Forest...")
rf_selector = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_selector.fit(X_train_full, y_train_full)

# Get feature importances and select top features
importances = rf_selector.feature_importances_
indices = np.argsort(importances)[::-1]  # Sort in descending order

# Select top N features
top_n = 70  # Adjust as needed
top_indices = indices[:top_n]
top_features = X.columns[top_indices]
logging.info(f"Top {top_n} features selected: {list(top_features)}")

# Use only top features for training
X_train_selected = X_train_full[:, top_indices]
X_test_selected = X_test[:, top_indices]

# ----------------------------------------
# Step 5: Initialize Models and Hyperparameters
# ----------------------------------------

# Define parameter distributions for RandomizedSearchCV
param_distributions = {
    'Logistic Regression': {
        'C': uniform(0.01, 100),
        'penalty': ['l2'],
        'solver': ['lbfgs'],
        'multi_class': ['multinomial'],
        'max_iter': [1000]
    },
    'Decision Tree': {
        'max_depth': [None] + list(range(5, 21)),
        'min_samples_split': randint(2, 11),
        'min_samples_leaf': randint(1, 5),
        'criterion': ['gini', 'entropy']
    },
    'K-Nearest Neighbors': {
        'n_neighbors': randint(3, 12),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    },
    'Support Vector Machine': {
        'C': uniform(0.1, 100),
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto']
    },
    'Naive Bayes': {
        'var_smoothing': uniform(1e-09, 1e-06)
    },
    'Random Forest': {
        'n_estimators': randint(100, 501),
        'max_depth': [None] + list(range(10, 31)),
        'min_samples_split': randint(2, 11),
        'min_samples_leaf': randint(1, 5),
        'max_features': ['auto', 'sqrt', 'log2'],
        'bootstrap': [True, False]
    },
    'XGBoost': {
        'max_depth': randint(3, 13),
        'learning_rate': uniform(0.01, 0.19),
        'n_estimators': randint(100, 501),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'gamma': uniform(0, 0.2),
        'min_child_weight': randint(1, 6),
        'reg_alpha': uniform(0, 0.5),
        'reg_lambda': uniform(1, 1)
    }
}

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Support Vector Machine': SVC(random_state=42),
    'Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(
        objective='multi:softmax',
        num_class=len(np.unique(y)),
        random_state=42,
        tree_method='hist'  # Use histogram-based method for efficiency
        # Uncomment the following line if you have GPU support
        # device='cuda'
    )
}

# ----------------------------------------
# Step 6: Set Up Cross-Validation
# ----------------------------------------

# Cross-validation setup
k_folds = 5
logging.info(f"Using {k_folds}-fold cross-validation.")

# Initialize Stratified K-Fold cross-validator
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

# ----------------------------------------
# Step 7: Create Directory for Results
# ----------------------------------------

# Main directory for saving results
results_dir = 'Results_joint_kfold'
os.makedirs(results_dir, exist_ok=True)

# Initialize a list to store results
results = []

# ----------------------------------------
# Step 8: Train and Evaluate Models
# ----------------------------------------

# Train and evaluate each model using hyperparameter tuning
for model_name, model in models.items():
    logging.info(f"Starting hyperparameter tuning for {model_name}...")
    start_time = time.time()

    # RandomizedSearchCV for hyperparameter tuning
    logging.info(f"Using RandomizedSearchCV for {model_name}")
    param_dist = param_distributions[model_name]
    param_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=50,
        cv=skf,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    param_search.fit(X_train_selected, y_train_full)

    # Best estimator and parameters
    best_model = param_search.best_estimator_
    best_params = param_search.best_params_
    best_score = param_search.best_score_

    training_time = time.time() - start_time
    logging.info(f"{model_name} hyperparameter tuning completed in {training_time:.2f} seconds.")
    logging.info(f"{model_name} Best Parameters: {best_params}")
    logging.info(f"{model_name} Best Cross-Validation Accuracy: {best_score * 100:.2f}%")
    print(f"{model_name} Best Cross-Validation Accuracy: {best_score * 100:.2f}%")
    print(f"{model_name} Best Parameters: {best_params}")

    # Evaluate on test set
    y_test_pred = best_model.predict(X_test_selected)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    logging.info(f"{model_name} Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"{model_name} Test Accuracy: {test_accuracy * 100:.2f}%")

    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_test_pred)
    logging.info(f"{model_name} Confusion Matrix:\n{conf_matrix}")

    # Compute additional metrics
    precision = precision_score(y_test, y_test_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_test_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_test_pred, average='macro', zero_division=0)

    # Compute specificity for each class and then average
    TN = []
    FP = []
    for i in range(len(conf_matrix)):
        tn = conf_matrix.sum() - (conf_matrix[i, :].sum() + conf_matrix[:, i].sum() - conf_matrix[i, i])
        fp = conf_matrix[:, i].sum() - conf_matrix[i, i]
        TN.append(tn)
        FP.append(fp)
    specificity_per_class = np.array(TN) / (np.array(TN) + np.array(FP) + 1e-6)  # Add small epsilon to avoid division by zero
    specificity = np.mean(specificity_per_class)

    # Log additional metrics
    logging.info(f"{model_name} Precision: {precision * 100:.2f}%")
    logging.info(f"{model_name} Recall: {recall * 100:.2f}%")
    logging.info(f"{model_name} F1 Score: {f1 * 100:.2f}%")
    logging.info(f"{model_name} Specificity: {specificity * 100:.2f}%")

    # Print additional metrics
    print(f"{model_name} Precision: {precision * 100:.2f}%")
    print(f"{model_name} Recall: {recall * 100:.2f}%")
    print(f"{model_name} F1 Score: {f1 * 100:.2f}%")
    print(f"{model_name} Specificity: {specificity * 100:.2f}%")

    # Append results to the list
    results.append({
        'Model': model_name,
        'Best Parameters': best_params,
        'Accuracy': test_accuracy * 100,
        'Precision': precision * 100,
        'Recall': recall * 100,
        'F1 Score': f1 * 100,
        'Specificity': specificity * 100
    })

    # Save confusion matrix to CSV file
    conf_matrix_csv_path = os.path.join(results_dir, f'{model_name.lower().replace(" ", "_")}_confusion_matrix.csv')
    conf_matrix_df = pd.DataFrame(conf_matrix)
    conf_matrix_df.to_csv(conf_matrix_csv_path, index=False)
    logging.info(f"{model_name} confusion matrix saved at '{conf_matrix_csv_path}'.")
    print(f"{model_name} confusion matrix saved at {conf_matrix_csv_path}.")

    logging.info("########################################################################################################################")

# ----------------------------------------
# Step 9: Save Final Results
# ----------------------------------------

# Save consolidated results to a CSV file in the Results directory
results_df = pd.DataFrame(results)
results_csv_path = os.path.join(results_dir, 'metrics.csv')
results_df.to_csv(results_csv_path, index=False)
print(f"Model results saved to '{results_csv_path}'")
logging.info(f"Model results saved to '{results_csv_path}'")
