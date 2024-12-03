import pandas as pd
import numpy as np
import time
import os
import logging
from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV, LeaveOneGroupOut, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,
                             recall_score, f1_score, roc_auc_score)
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
log_file_name = f'training_radar_loso_{current_time}.log'

# Configure logging
logging.basicConfig(filename=log_file_name, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Set random seed for reproducibility
np.random.seed(42)

# Load the dataset and drop missing values
data = pd.read_csv('spatial_and_temporal_features_from_radar_data.csv').dropna()

# Ensure 'SubjectID' is present for LOSO validation
if 'SubjectID' not in data.columns:
    raise ValueError("Dataset must contain a 'SubjectID' column for LOSO validation.")

# Extract labels
y = LabelEncoder().fit_transform(data['ExerciseLabel'])

# Extract groups for LOSO
groups = data['SubjectID'].values  # Groups for LOSO validation

# Drop irrelevant columns
irrelevant_columns = ['SubjectID', 'ExerciseLabel', 'FrameNum']
X = data.drop(columns=irrelevant_columns)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Directory for saving results
results_dir = 'Results_radar_loso'
os.makedirs(results_dir, exist_ok=True)  # Create 'Results_radar_loso' directory if it doesn't exist

# Cross-validation setup for hyperparameter tuning within LOSO
k_folds = 5  # Number of folds for inner k-fold CV
logging.info(f"Using {k_folds}-fold cross-validation within LOSO.")

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
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Support Vector Machine': SVC(probability=True, random_state=42),  # Enable probability output
    'Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(
        objective='multi:softmax',
        num_class=len(np.unique(y)),
        random_state=42,
        tree_method='hist'
    )
}

# Initialize LOSO cross-validator
logo = LeaveOneGroupOut()

# Initialize a list to store LOSO results for each iteration
loso_results = []
aggregate_results = []

# Start LOSO cross-validation
for fold, (train_index, test_index) in enumerate(logo.split(X_scaled, y, groups=groups), 1):
    # Extract training and testing data for this fold
    X_train_full, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train_full, y_test = y[train_index], y[test_index]
    group_train, group_test = groups[train_index], groups[test_index]

    subject_id = np.unique(group_test)[0]
    logging.info(f"LOSO iteration {fold}: Leaving out subject {subject_id}")

    # Feature selection using Random Forest on the training set
    rf_selector = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_selector.fit(X_train_full, y_train_full)

    # Get feature importances and select top features
    importances = rf_selector.feature_importances_
    indices = np.argsort(importances)[::-1]  # Sort in descending order

    # Select top N features
    top_n = 20
    top_indices = indices[:top_n]

    # Get feature names for top features
    feature_names = X.columns
    top_features = feature_names[top_indices]

    # Log the top features
    logging.info(f"Top {top_n} features selected for subject {subject_id}: {list(top_features)}")

    # Use only top features for training
    X_train_selected = X_train_full[:, top_indices]
    X_test_selected = X_test[:, top_indices]

    # Train and evaluate each model using hyperparameter tuning within this LOSO fold
    for model_name, model in models.items():
        logging.info(f"Starting hyperparameter tuning for {model_name} within LOSO (subject {subject_id})...")

        # Record the start time for hyperparameter tuning
        hyper_start_time = time.time()

        # Inner k-fold cross-validation for hyperparameter tuning
        param_dist = param_distributions[model_name]
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
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

        # Record the end time for hyperparameter tuning
        hyper_end_time = time.time()
        hyper_duration = hyper_end_time - hyper_start_time
        logging.info(f"Hyperparameter tuning for {model_name} on subject {subject_id} completed in {hyper_duration:.2f} seconds")

        # Best estimator and parameters
        best_model = param_search.best_estimator_
        best_params = param_search.best_params_
        logging.info(f"Best hyperparameters for {model_name} on subject {subject_id}: {best_params}")

        # Record the start time for training the best model
        train_start_time = time.time()

        # Refit the best model on the full training data
        best_model.fit(X_train_selected, y_train_full)

        # Record the end time for training
        train_end_time = time.time()
        train_duration = train_end_time - train_start_time
        logging.info(f"Training time for {model_name} on subject {subject_id}: {train_duration:.4f} seconds")

        # Record the start time for inference
        inference_start_time = time.time()

        # Evaluate on test set (left-out subject)
        y_test_pred = best_model.predict(X_test_selected)

        # Record the end time for inference
        inference_end_time = time.time()
        inference_duration = inference_end_time - inference_start_time
        logging.info(f"Inference time for {model_name} on subject {subject_id}: {inference_duration:.4f} seconds")

        # Save ROC data only for Random Forest and XGBoost
        if model_name in ['Random Forest', 'XGBoost']:
            y_prob = best_model.predict_proba(X_test_selected)  # Predicted probabilities
            roc_data_path = os.path.join(
                results_dir,
                f'{model_name.lower().replace(" ", "_")}_roc_data_subject_{subject_id}.csv'
            )
            roc_data = pd.DataFrame(y_prob, columns=[f'Prob_Class_{i}' for i in range(y_prob.shape[1])])
            roc_data['True_Label'] = y_test
            roc_data.to_csv(roc_data_path, index=False)
            logging.info(f"ROC data saved for {model_name} on subject {subject_id} at '{roc_data_path}'")

        # Calculate and log metrics
        accuracy = accuracy_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_test_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_test_pred, average='macro', zero_division=0)

        # Compute specificity for each class and then average
        conf_matrix = confusion_matrix(y_test, y_test_pred)
        TN, FP = [], []
        for i in range(len(conf_matrix)):
            tn = conf_matrix.sum() - (conf_matrix[i, :].sum() + conf_matrix[:, i].sum() - conf_matrix[i, i])
            fp = conf_matrix[:, i].sum() - conf_matrix[i, i]
            TN.append(tn)
            FP.append(fp)
        specificity_per_class = np.array(TN) / (np.array(TN) + np.array(FP) + 1e-6)  # Avoid division by zero
        specificity = np.mean(specificity_per_class)

        # Save confusion matrix to a CSV file
        conf_matrix_csv_path = os.path.join(
            results_dir,
            f'{model_name.lower().replace(" ", "_")}_confusion_matrix_subject_{subject_id}.csv'
        )
        conf_matrix_df = pd.DataFrame(conf_matrix)
        conf_matrix_df.to_csv(conf_matrix_csv_path, index=False)
        logging.info(f"Confusion matrix saved for {model_name} on subject {subject_id} at '{conf_matrix_csv_path}'")

        # Log metrics
        logging.info(f"{model_name} Metrics for Subject {subject_id}:")
        logging.info(f"  - Accuracy: {accuracy:.2f}")
        logging.info(f"  - Precision: {precision:.2f}")
        logging.info(f"  - Recall: {recall:.2f}")
        logging.info(f"  - F1 Score: {f1:.2f}")
        logging.info(f"  - Specificity: {specificity:.2f}")

        # Append per-subject results
        loso_results.append({
            'SubjectID': subject_id,
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'Specificity': specificity,
            'Best Parameters': str(best_params),
            'Top Features': ', '.join(top_features),
            'Hyperparameter Tuning Time (s)': hyper_duration,
            'Training Time (s)': train_duration,
            'Inference Time (s)': inference_duration
        })

# Save per-subject LOSO results to a CSV file
loso_results_df = pd.DataFrame(loso_results)
per_subject_csv_path = os.path.join(results_dir, 'loso_per_subject_metrics.csv')
loso_results_df.to_csv(per_subject_csv_path, index=False)
logging.info(f"Per-subject LOSO results saved to '{per_subject_csv_path}'")

# Aggregate results across all subjects
for model_name in models.keys():
    model_results = loso_results_df[loso_results_df['Model'] == model_name]
    if not model_results.empty:
        avg_accuracy = model_results['Accuracy'].mean()
        avg_precision = model_results['Precision'].mean()
        avg_recall = model_results['Recall'].mean()
        avg_f1_score = model_results['F1 Score'].mean()
        avg_specificity = model_results['Specificity'].mean()
        avg_hyper_time = model_results['Hyperparameter Tuning Time (s)'].mean()
        avg_train_time = model_results['Training Time (s)'].mean()
        avg_inference_time = model_results['Inference Time (s)'].mean()

        # Log aggregate results
        logging.info(f"Aggregate Results for {model_name}:")
        logging.info(f"  - Average Accuracy: {avg_accuracy:.2f}")
        logging.info(f"  - Average Precision: {avg_precision:.2f}")
        logging.info(f"  - Average Recall: {avg_recall:.2f}")
        logging.info(f"  - Average F1 Score: {avg_f1_score:.2f}")
        logging.info(f"  - Average Specificity: {avg_specificity:.2f}")
        logging.info(f"  - Average Hyperparameter Tuning Time: {avg_hyper_time:.2f} seconds")
        logging.info(f"  - Average Training Time: {avg_train_time:.4f} seconds")
        logging.info(f"  - Average Inference Time: {avg_inference_time:.4f} seconds")

        # Append to aggregate results
        aggregate_results.append({
            'Model': model_name,
            'Average Accuracy': avg_accuracy,
            'Average Precision': avg_precision,
            'Average Recall': avg_recall,
            'Average F1 Score': avg_f1_score,
            'Average Specificity': avg_specificity,
            'Average Hyperparameter Tuning Time (s)': avg_hyper_time,
            'Average Training Time (s)': avg_train_time,
            'Average Inference Time (s)': avg_inference_time
        })

# Save aggregated LOSO results to a CSV file
aggregate_results_df = pd.DataFrame(aggregate_results)
aggregate_csv_path = os.path.join(results_dir, 'loso_aggregate_metrics.csv')
aggregate_results_df.to_csv(aggregate_csv_path, index=False)
logging.info(f"Aggregate LOSO results saved to '{aggregate_csv_path}'")
