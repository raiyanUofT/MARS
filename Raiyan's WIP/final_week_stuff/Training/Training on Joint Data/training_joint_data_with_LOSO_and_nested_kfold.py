import pandas as pd
import numpy as np
import os
import time
import logging
from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV, LeaveOneGroupOut, StratifiedKFold
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
log_file_name = f'training_joint_loso_{current_time}.log'

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

# Ensure 'Subject' and 'Exercise' columns are present
if 'Subject' not in data.columns or 'Exercise' not in data.columns:
    raise ValueError("Dataset must contain 'Subject' and 'Exercise' columns for LOSO validation.")

# ----------------------------------------
# Step 2: Prepare Features and Labels
# ----------------------------------------

# Extract labels and groups
y = LabelEncoder().fit_transform(data['Exercise'])
groups = data['Subject'].values  # Groups for LOSO validation

# Drop irrelevant columns
irrelevant_columns = ['Subject', 'Exercise']
X = data.drop(columns=irrelevant_columns)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------------------
# Step 3: Initialize Models
# ----------------------------------------

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
        use_label_encoder=False,
        eval_metric='mlogloss',
        tree_method='hist'
    )
}

# ----------------------------------------
# Step 4: Define Hyperparameter Distributions
# ----------------------------------------

param_distributions = {
    'Logistic Regression': {
        'C': uniform(0.01, 100),
        'penalty': ['l2'],
        'solver': ['lbfgs'],
        'multi_class': ['multinomial']
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

# ----------------------------------------
# Step 5: Create Directory for Results
# ----------------------------------------

results_dir = 'Results_joint_loso'
os.makedirs(results_dir, exist_ok=True)

# Initialize a list to store LOSO results for each iteration
loso_results = []

# ----------------------------------------
# Step 6: Set up Leave-One-Subject-Out Cross-Validation
# ----------------------------------------

logo = LeaveOneGroupOut()

# Cross-validation setup for hyperparameter tuning within LOSO
k_folds = 5  # Number of folds for inner k-fold CV
logging.info(f"Using {k_folds}-fold cross-validation within LOSO.")

# ----------------------------------------
# Step 7: Start Leave-One-Subject-Out Testing
# ----------------------------------------

for fold, (train_idx, test_idx) in enumerate(logo.split(X_scaled, y, groups=groups), 1):
    train_subjects = np.unique(groups[train_idx])
    test_subject = np.unique(groups[test_idx])[0]
    logging.info(f"LOSO iteration {fold}: Training on subjects {train_subjects}")
    logging.info(f"LOSO iteration {fold}: Testing on subject {test_subject}")
    print(f"LOSO iteration {fold}: Training on subjects {train_subjects}")
    print(f"LOSO iteration {fold}: Testing on subject {test_subject}")

    X_train_full, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train_full, y_test = y[train_idx], y[test_idx]

    # ----------------------------------------
    # Step 8: Feature Selection using Random Forest
    # ----------------------------------------

    rf_selector = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_selector.fit(X_train_full, y_train_full)
    importances = rf_selector.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_n = 70  # Select top N features
    top_indices = indices[:top_n]
    top_features = [X.columns[i] for i in top_indices]
    logging.info(f"Top {top_n} features selected: {top_features}")

    # Use only top features for training
    X_train_selected = X_train_full[:, top_indices]
    X_test_selected = X_test[:, top_indices]

    # Initialize a dictionary to store results for this LOSO iteration
    loso_iteration_results = {
        'Subject': test_subject
    }

    # ----------------------------------------
    # Step 9: Train and Evaluate Each Model
    # ----------------------------------------

    for model_name, model in models.items():
        logging.info(f"Starting hyperparameter tuning for {model_name} within LOSO (subject {test_subject})...")
        print(f"Starting hyperparameter tuning for {model_name} within LOSO (subject {test_subject})...")
        start_time = time.time()

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

        # Best estimator and parameters
        best_model = param_search.best_estimator_
        best_params = param_search.best_params_
        best_score = param_search.best_score_

        training_time = time.time() - start_time
        logging.info(f"{model_name} hyperparameter tuning completed in {training_time:.2f} seconds.")
        print(f"{model_name} hyperparameter tuning completed in {training_time:.2f} seconds.")
        logging.info(f"{model_name} Best Parameters: {best_params}")
        logging.info(f"{model_name} Best Cross-Validation Accuracy: {best_score * 100:.2f}%")
        print(f"{model_name} Best Cross-Validation Accuracy: {best_score * 100:.2f}%")
        print(f"{model_name} Best Parameters: {best_params}")

        # Evaluate on test set (left-out subject)
        y_test_pred = best_model.predict(X_test_selected)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        logging.info(f"{model_name} Test Accuracy on subject {test_subject}: {test_accuracy * 100:.2f}%")
        print(f"{model_name} Test Accuracy on subject {test_subject}: {test_accuracy * 100:.2f}%")

        # Generate confusion matrix
        conf_matrix = confusion_matrix(y_test, y_test_pred)
        logging.info(f"{model_name} Confusion Matrix for subject {test_subject}:\n{conf_matrix}")

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

        # Append results to the LOSO iteration results
        loso_iteration_results[model_name] = {
            'Best Parameters': best_params,
            'Training Time (s)': training_time,
            'Accuracy': test_accuracy * 100,
            'Precision': precision * 100,
            'Recall': recall * 100,
            'F1 Score': f1 * 100,
            'Specificity': specificity * 100
        }

        # Save confusion matrix to CSV file
        conf_matrix_csv_path = os.path.join(results_dir, f'{model_name.lower().replace(" ", "_")}_confusion_matrix_subject_{test_subject}.csv')
        conf_matrix_df = pd.DataFrame(conf_matrix)
        conf_matrix_df.to_csv(conf_matrix_csv_path, index=False)
        logging.info(f"{model_name} confusion matrix saved at '{conf_matrix_csv_path}'.")
        print(f"{model_name} confusion matrix saved at {conf_matrix_csv_path}.")

        logging.info("--------------------------------------------------------")

    # Append results of this LOSO iteration
    loso_results.append(loso_iteration_results)
    logging.info(f"Completed LOSO iteration for subject {test_subject}")
    logging.info("########################################################################################################################")

# ----------------------------------------
# Step 10: Aggregate and Save Results
# ----------------------------------------

# Initialize lists to store per-subject results
per_subject_results = []

# Iterate over LOSO results to collect per-subject metrics
for result in loso_results:
    subject_id = result['Subject']
    for model_name in models.keys():
        model_result = result.get(model_name, {})
        if model_result:
            per_subject_results.append({
                'Subject': subject_id,
                'Model': model_name,
                'Training Time (s)': model_result.get('Training Time (s)', 0),
                'Accuracy': model_result.get('Accuracy', 0),
                'Precision': model_result.get('Precision', 0),
                'Recall': model_result.get('Recall', 0),
                'F1 Score': model_result.get('F1 Score', 0),
                'Specificity': model_result.get('Specificity', 0),
                'Best Parameters': model_result.get('Best Parameters', {})
            })

# Save per-subject LOSO results to a CSV file
per_subject_results_df = pd.DataFrame(per_subject_results)
per_subject_results_csv_path = os.path.join(results_dir, 'loso_per_subject_metrics.csv')
per_subject_results_df.to_csv(per_subject_results_csv_path, index=False)
print(f"Per-subject LOSO results saved to '{per_subject_results_csv_path}'")
logging.info(f"Per-subject LOSO results saved to '{per_subject_results_csv_path}'")

# Now aggregate the results across subjects for each model, including average training time
aggregate_results = []

for model_name in models.keys():
    model_results = per_subject_results_df[per_subject_results_df['Model'] == model_name]
    avg_training_time = model_results['Training Time (s)'].mean()
    avg_accuracy = model_results['Accuracy'].mean()
    avg_precision = model_results['Precision'].mean()
    avg_recall = model_results['Recall'].mean()
    avg_f1_score = model_results['F1 Score'].mean()
    avg_specificity = model_results['Specificity'].mean()

    # Log and print the aggregated results
    logging.info(f"{model_name} LOSO Average Training Time: {avg_training_time:.2f} seconds")
    logging.info(f"{model_name} LOSO Average Accuracy: {avg_accuracy:.2f}%")
    logging.info(f"{model_name} LOSO Average Precision: {avg_precision:.2f}%")
    logging.info(f"{model_name} LOSO Average Recall: {avg_recall:.2f}%")
    logging.info(f"{model_name} LOSO Average F1 Score: {avg_f1_score:.2f}%")
    logging.info(f"{model_name} LOSO Average Specificity: {avg_specificity:.2f}%")

    print(f"{model_name} LOSO Average Training Time: {avg_training_time:.2f} seconds")
    print(f"{model_name} LOSO Average Accuracy: {avg_accuracy:.2f}%")
    print(f"{model_name} LOSO Average Precision: {avg_precision:.2f}%")
    print(f"{model_name} LOSO Average Recall: {avg_recall:.2f}%")
    print(f"{model_name} LOSO Average F1 Score: {avg_f1_score:.2f}%")
    print(f"{model_name} LOSO Average Specificity: {avg_specificity:.2f}%")

    # Append to aggregate results
    aggregate_results.append({
        'Model': model_name,
        'Average Training Time (s)': avg_training_time,
        'LOSO Accuracy': avg_accuracy,
        'LOSO Precision': avg_precision,
        'LOSO Recall': avg_recall,
        'LOSO F1 Score': avg_f1_score,
        'LOSO Specificity': avg_specificity
    })

# Save aggregated LOSO results to a CSV file
aggregate_results_df = pd.DataFrame(aggregate_results)
aggregate_results_csv_path = os.path.join(results_dir, 'loso_aggregate_metrics.csv')
aggregate_results_df.to_csv(aggregate_results_csv_path, index=False)
print(f"Aggregated LOSO results saved to '{aggregate_results_csv_path}'")
logging.info(f"Aggregated LOSO results saved to '{aggregate_results_csv_path}'")
