import pandas as pd
import numpy as np
import time
import os
import logging
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
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
log_file_name = f'training_radar_kfold_only_{current_time}.log'

# Configure logging
logging.basicConfig(filename=log_file_name, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Set random seed for reproducibility
np.random.seed(42)

# Load the dataset and drop missing values
data = pd.read_csv('spatial_and_temporal_features_from_radar_data.csv').dropna()

# Extract labels
y = LabelEncoder().fit_transform(data['ExerciseLabel'])

# Drop irrelevant columns
irrelevant_columns = ['SubjectID', 'ExerciseLabel', 'FrameNum']
X = data.drop(columns=irrelevant_columns)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Feature names for reference
feature_names = X.columns

# Main directory for saving results
results_dir = 'Results_radar_kfold'
os.makedirs(results_dir, exist_ok=True)  # Create 'Results_radar_kfold' directory if it doesn't exist

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
        tree_method='hist'  # Use histogram-based method for efficiency
        # device='cuda'  # Uncomment if you have GPU support
    )
}

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

# Initialize a list to store all results
all_results = []

# Outer cross-validation loop (10 folds)
outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
for fold_idx, (train_val_indices, test_indices) in enumerate(outer_cv.split(X_scaled, y)):
    logging.info(f"Starting fold {fold_idx + 1}/10")
    print(f"Starting fold {fold_idx + 1}/10")

    # Extract training and test data
    X_train_val, y_train_val = X_scaled[train_val_indices], y[train_val_indices]
    X_test, y_test = X_scaled[test_indices], y[test_indices]

    # Feature selection on training data
    logging.info("Starting feature selection with Random Forest...")
    rf_selector = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_selector.fit(X_train_val, y_train_val)

    # Get feature importances and select top features
    importances = rf_selector.feature_importances_
    indices = np.argsort(importances)[::-1]  # Sort in descending order

    # Select top N features
    top_n = 20
    top_indices = indices[:top_n]
    top_features = feature_names[top_indices]
    logging.info(f"Top {top_n} features selected: {list(top_features)}")

    # Use only top features for training
    X_train_selected = X_train_val[:, top_indices]
    X_test_selected = X_test[:, top_indices]

    # Inner cross-validation setup for hyperparameter tuning
    inner_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Initialize a list to store results for this fold
    fold_results = []

    # Directory for the current fold
    fold_dir = os.path.join(results_dir, f'fold_{fold_idx + 1}')
    os.makedirs(fold_dir, exist_ok=True)

    # Loop over models
    for model_name, model in models.items():
        logging.info(f"Starting hyperparameter tuning for {model_name} on fold {fold_idx + 1}")
        print(f"Starting hyperparameter tuning for {model_name} on fold {fold_idx + 1}")

        # Record the start time for hyperparameter tuning
        hyper_start_time = time.time()

        # RandomizedSearchCV
        logging.info(f"Using RandomizedSearchCV for {model_name}")
        param_dist = param_distributions[model_name]
        param_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=50,
            cv=inner_cv,
            scoring='accuracy',
            n_jobs=-1,
            random_state=42,
            verbose=0
        )
        param_search.fit(X_train_selected, y_train_val)

        # Record the end time for hyperparameter tuning
        hyper_end_time = time.time()
        hyper_duration = hyper_end_time - hyper_start_time
        logging.info(f"Hyperparameter tuning for {model_name} completed in {hyper_duration:.2f} seconds")

        # Best estimator
        best_model = param_search.best_estimator_
        best_params = param_search.best_params_
        best_score = param_search.best_score_

        logging.info(f"{model_name} Best Parameters: {best_params}")
        logging.info(f"{model_name} Best Cross-Validation Accuracy: {best_score * 100:.2f}%")
        print(f"{model_name} Best Cross-Validation Accuracy: {best_score * 100:.2f}%")
        print(f"{model_name} Best Parameters: {best_params}")

        # Record the start time for training
        train_start_time = time.time()

        # Refit the best model on the full training data
        best_model.fit(X_train_selected, y_train_val)

        # Record the end time for training
        train_end_time = time.time()
        train_duration = train_end_time - train_start_time
        logging.info(f"Training time for {model_name}: {train_duration:.4f} seconds")

        # Record the start time for inference
        inference_start_time = time.time()

        # Evaluate on test set
        y_test_pred = best_model.predict(X_test_selected)

        # Record the end time for inference
        inference_end_time = time.time()
        inference_duration = inference_end_time - inference_start_time
        logging.info(f"Inference time for {model_name}: {inference_duration:.4f} seconds")

        # Save ROC data
        y_prob = best_model.predict_proba(X_test_selected)  # Predicted probabilities
        roc_data_path = os.path.join(
            fold_dir,
            f'{model_name.lower().replace(" ", "_")}_roc_data_fold_{fold_idx + 1}.csv'
        )
        roc_data = pd.DataFrame(y_prob, columns=[f'Prob_Class_{i}' for i in range(y_prob.shape[1])])
        roc_data['True_Label'] = y_test
        roc_data.to_csv(roc_data_path, index=False)
        logging.info(f"ROC data saved for {model_name} at '{roc_data_path}'")
        print(f"ROC data saved for {model_name} at '{roc_data_path}'")

        # Calculate metrics
        test_accuracy = accuracy_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_test_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_test_pred, average='macro', zero_division=0)

        # Compute specificity
        conf_matrix = confusion_matrix(y_test, y_test_pred)
        TN, FP = [], []
        for i in range(len(conf_matrix)):
            tn = conf_matrix.sum() - (conf_matrix[i, :].sum() + conf_matrix[:, i].sum() - conf_matrix[i, i])
            fp = conf_matrix[:, i].sum() - conf_matrix[i, i]
            TN.append(tn)
            FP.append(fp)
        specificity_per_class = np.array(TN) / (np.array(TN) + np.array(FP) + 1e-6)
        specificity = np.mean(specificity_per_class)

        # Save confusion matrix
        conf_matrix_csv_path = os.path.join(
            fold_dir,
            f'{model_name.lower().replace(" ", "_")}_confusion_matrix_fold_{fold_idx + 1}.csv'
        )
        conf_matrix_df = pd.DataFrame(conf_matrix)
        conf_matrix_df.to_csv(conf_matrix_csv_path, index=False)
        logging.info(f"{model_name} confusion matrix saved as CSV at '{conf_matrix_csv_path}'.")
        print(f"{model_name} confusion matrix saved as CSV at '{conf_matrix_csv_path}'.")

        # Log and print metrics
        logging.info(f"{model_name} Metrics for fold {fold_idx + 1}:")
        logging.info(f"  - Test Accuracy: {test_accuracy * 100:.2f}%")
        logging.info(f"  - Precision: {precision * 100:.2f}%")
        logging.info(f"  - Recall: {recall * 100:.2f}%")
        logging.info(f"  - F1 Score: {f1 * 100:.2f}%")
        logging.info(f"  - Specificity: {specificity * 100:.2f}%")

        print(f"{model_name} Test Accuracy: {test_accuracy * 100:.2f}%")
        print(f"{model_name} Precision: {precision * 100:.2f}%")
        print(f"{model_name} Recall: {recall * 100:.2f}%")
        print(f"{model_name} F1 Score: {f1 * 100:.2f}%")
        print(f"{model_name} Specificity: {specificity * 100:.2f}%")

        # Append results to fold_results
        fold_results.append({
            'Model': model_name,
            'Fold': fold_idx + 1,
            'Best Parameters': str(best_params),
            'Accuracy': test_accuracy * 100,
            'Precision': precision * 100,
            'Recall': recall * 100,
            'F1 Score': f1 * 100,
            'Specificity': specificity * 100,
            'Hyperparameter Tuning Time (s)': hyper_duration,
            'Training Time (s)': train_duration,
            'Inference Time (s)': inference_duration,
            'Top Features': ', '.join(top_features)
        })

    # At the end of the fold, append fold_results to all_results
    all_results.extend(fold_results)

# After all folds are completed, save consolidated results
results_df = pd.DataFrame(all_results)
results_csv_path = os.path.join(results_dir, 'metrics_all_folds.csv')
results_df.to_csv(results_csv_path, index=False)
print(f"Model results saved to '{results_csv_path}'")
logging.info(f"Model results saved to '{results_csv_path}'")
