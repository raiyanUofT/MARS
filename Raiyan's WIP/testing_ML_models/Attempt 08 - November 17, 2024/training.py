import pandas as pd
import numpy as np
import os
import time
import logging
from sklearn.model_selection import RandomizedSearchCV, LeaveOneGroupOut
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

# Configure logging
logging.basicConfig(filename='training_loso.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Set random seed for reproducibility
np.random.seed(42)

# ----------------------------------------
# Step 1: Load the Dataset
# ----------------------------------------

# Load the dataset and drop missing values
data = pd.read_csv('comprehensive_features_from_joint_data.csv').dropna()

# ----------------------------------------
# Step 2: Prepare Features and Labels
# ----------------------------------------

# Separate features and target
subjects = data['Subject']  # For LOSO grouping
X = data.drop(['Exercise', 'Subject'], axis=1)
y = LabelEncoder().fit_transform(data['Exercise'])

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

results_dir = 'Results_LOSO'
os.makedirs(results_dir, exist_ok=True)

# List to store final results
final_results = []

# ----------------------------------------
# Step 6: Set up Leave-One-Subject-Out Cross-Validation
# ----------------------------------------

logo = LeaveOneGroupOut()

# ----------------------------------------
# Step 7: Feature Scaling
# ----------------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------------------
# Step 8: Leave-One-Subject-Out Testing
# ----------------------------------------

for train_idx, test_idx in logo.split(X_scaled, y, groups=subjects):
    train_subjects = subjects.iloc[train_idx].unique()
    test_subject = subjects.iloc[test_idx].unique()[0]
    logging.info(f"Training on subjects: {train_subjects}")
    logging.info(f"Testing on subject: {test_subject}")
    
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # ----------------------------------------
    # Step 9: Feature Selection using Random Forest
    # ----------------------------------------

    rf_selector = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_selector.fit(X_train, y_train)
    importances = rf_selector.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_n = 20  # Select top N features
    top_indices = indices[:top_n]
    X_train_selected = X_train[:, top_indices]
    X_test_selected = X_test[:, top_indices]
    
    # ----------------------------------------
    # Step 10: Train and Evaluate Each Model
    # ----------------------------------------

    for model_name, model in models.items():
        logging.info(f"Hyperparameter tuning for {model_name} on Subject: {test_subject}")
        start_time = time.time()
        
        # RandomizedSearchCV for hyperparameter tuning
        param_dist = param_distributions[model_name]
        param_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=50,
            scoring='accuracy',
            n_jobs=-1,
            random_state=42,
            cv=3
        )
        param_search.fit(X_train_selected, y_train)
        
        # Evaluate on test set
        best_model = param_search.best_estimator_
        y_test_pred = best_model.predict(X_test_selected)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_test_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_test_pred, average='macro', zero_division=0)
        
        # Specificity calculation
        conf_matrix = confusion_matrix(y_test, y_test_pred, labels=np.unique(y))
        TN, FP = [], []
        for i in range(len(conf_matrix)):
            tn = conf_matrix.sum() - (conf_matrix[i, :].sum() + conf_matrix[:, i].sum() - conf_matrix[i, i])
            fp = conf_matrix[:, i].sum() - conf_matrix[i, i]
            TN.append(tn)
            FP.append(fp)
        specificity_per_class = np.array(TN) / (np.array(TN) + np.array(FP) + 1e-6)
        specificity = np.mean(specificity_per_class)
        
        # Log results
        logging.info(f"{model_name} Results for Subject {test_subject}: "
                     f"Accuracy={test_accuracy:.4f}, Precision={precision:.4f}, "
                     f"Recall={recall:.4f}, F1={f1:.4f}, Specificity={specificity:.4f}")
        
        # Save results
        final_results.append({
            'Subject': test_subject,
            'Model': model_name,
            'Accuracy': test_accuracy * 100,
            'Precision': precision * 100,
            'Recall': recall * 100,
            'F1 Score': f1 * 100,
            'Specificity': specificity * 100
        })

# ----------------------------------------
# Step 11: Save Final Results
# ----------------------------------------

results_df = pd.DataFrame(final_results)
results_csv_path = os.path.join(results_dir, 'loso_results.csv')
results_df.to_csv(results_csv_path, index=False)
print(f"LOSO results saved to '{results_csv_path}'")
