import pandas as pd
import numpy as np
import time
import xgboost as xgb
import os
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

# Configure logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(filename='logs/training_xgboost.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Set random seed for reproducibility
np.random.seed(42)

# Load the full dataset and drop missing values
data = pd.read_csv('allRadarDataWithComprehensiveFeatures.csv').dropna()

# Select relevant features and labels
relevant_features = [
    'NumPoints', 'MeanX', 'StdX', 'SkewX', 'KurtX', 'MeanY', 'StdY', 'SkewY', 'KurtY',
    'MeanZ', 'StdZ', 'SkewZ', 'KurtZ', 'MeanDoppler', 'StdDoppler', 'MeanIntensity',
    'StdIntensity', 'RangeX', 'RangeY', 'RangeZ', 'NumOccupiedVoxels', 'VoxelDensity',
    'MaxIntensity', 'MinIntensity', 'MeanRadialDist', 'StdRadialDist', 'VoxelEntropy',
    'PCA_DirectionX', 'PCA_DirectionY', 'PCA_DirectionZ', 'ExplainedVariance',
    'MeanGradX', 'MeanGradY', 'MeanGradZ', 'CovXY', 'CovXZ', 'CovYZ', 'CorrXY',
    'CorrXZ', 'CorrYZ', 'EigVal1', 'EigVal2', 'EigVal3', 'EigRatio1', 'EigRatio2',
    'SkewDoppler', 'KurtDoppler', 'SkewIntensity', 'KurtIntensity', 'MaxDoppler',
    'MinDoppler', 'MaxRadialDist', 'MinRadialDist', 'TotalIntensity', 'ConvexHullVolume',
    'DopplerEntropy', 'IntensityEntropy'
]

X = data[relevant_features]
y = LabelEncoder().fit_transform(data['ExerciseLabel'])

# Standardize features
X = StandardScaler().fit_transform(X)

# Split the dataset into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

logging.info(f"Training set: {len(X_train)} samples")
logging.info(f"Validation set: {len(X_val)} samples")
logging.info(f"Test set: {len(X_test)} samples")

# Step 1: Feature selection using Random Forest
logging.info("Starting feature selection with Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Get feature importances and select top features
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]  # Sort in descending order

# Select top N features
top_n = 20
top_features = [relevant_features[i] for i in indices[:top_n]]
logging.info(f"Top {top_n} features selected: {top_features}")

# Use only top features for training
X_train_selected = X_train[:, indices[:top_n]]
X_val_selected = X_val[:, indices[:top_n]]
X_test_selected = X_test[:, indices[:top_n]]

# Hyperparameter grid for full training
param_grid = {
    'max_depth': [9],
    'learning_rate': [0.1],
    'n_estimators': [300],
    'subsample': [0.8],
    'colsample_bytree': [1.0],
    'gamma': [0.1],
    'min_child_weight': [3],
    'reg_alpha': [0],  # L1 regularization
    'reg_lambda': [1]  # L2 regularization
}

# XGBoost model with GPU support
xgb_model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=len(np.unique(y)),
    random_state=42,
    tree_method='hist',  # Use histogram-based method
    device='cuda'        # Set device to GPU
)

# Stratified K-Fold Cross-Validation
k = 5
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
logging.info(f"Number of stratified cross-validation folds: {k} folds")

# Initialize GridSearchCV
param_search = GridSearchCV(
    estimator=xgb_model, param_grid=param_grid, scoring='accuracy', cv=skf, n_jobs=-1, verbose=0
)

# Start timing the training process
start_time = time.time()
logging.info("Starting full cross-validation and training with selected features...")

# Fit the model using only the selected features
param_search.fit(X_train_selected, y_train)

# Calculate time taken
end_time = time.time()
elapsed_time = end_time - start_time
logging.info(f"Training completed in {elapsed_time:.2f} seconds.")

# Log the best parameters and cross-validation accuracy
best_params = param_search.best_params_
best_score = param_search.best_score_ * 100

logging.info(f"Best Parameters: {best_params}")
logging.info(f"Best Cross-Validation Accuracy: {best_score:.2f}%")

print(f"Best Parameters: {best_params}")
print(f"Best Cross-Validation Accuracy: {best_score:.2f}%")

# Evaluate on validation set
y_val_pred = param_search.best_estimator_.predict(X_val_selected)
val_accuracy = accuracy_score(y_val, y_val_pred)
logging.info(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

# Evaluate on test set
y_test_pred = param_search.best_estimator_.predict(X_test_selected)
test_accuracy = accuracy_score(y_test, y_test_pred)
logging.info(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Print results to console
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Generate confusion matrix and additional metrics
conf_matrix = confusion_matrix(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred, average='macro', zero_division=0)
recall = recall_score(y_test, y_test_pred, average='macro', zero_division=0)
f1 = f1_score(y_test, y_test_pred, average='macro', zero_division=0)

# Calculate specificity
TN = []
FP = []
for i in range(len(conf_matrix)):
    tn = conf_matrix.sum() - (conf_matrix[i, :].sum() + conf_matrix[:, i].sum() - conf_matrix[i, i])
    fp = conf_matrix[:, i].sum() - conf_matrix[i, i]
    TN.append(tn)
    FP.append(fp)
specificity_per_class = np.array(TN) / (np.array(TN) + np.array(FP) + 1e-6)  # Avoid division by zero
specificity = np.mean(specificity_per_class)

# Log additional metrics
logging.info(f"Precision: {precision * 100:.2f}%")
logging.info(f"Recall: {recall * 100:.2f}%")
logging.info(f"F1 Score: {f1 * 100:.2f}%")
logging.info(f"Specificity: {specificity * 100:.2f}%")

# Print additional metrics
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")
print(f"Specificity: {specificity * 100:.2f}%")

# Create Results directory for XGBoost if it doesn't exist
results_dir = 'Results/XGBoost'
os.makedirs(results_dir, exist_ok=True)

# Save metrics and best parameters to a CSV file
metrics_df = pd.DataFrame({
    'Metric': ['Best Cross-Validation Accuracy', 'Validation Accuracy', 'Test Accuracy', 'Precision', 'Recall', 'F1 Score', 'Specificity'],
    'Value (%)': [best_score, val_accuracy * 100, test_accuracy * 100, precision * 100, recall * 100, f1 * 100, specificity * 100]
})
metrics_df.to_csv(os.path.join(results_dir, '../metrics_xgboost.csv'), index=False)
logging.info("Metrics and best parameters saved to 'Results/metrics_xgboost.csv'.")

# Save confusion matrix to CSV file for MATLAB
conf_matrix_df = pd.DataFrame(conf_matrix)
conf_matrix_csv_path = os.path.join(results_dir, 'xgboost_confusion_matrix.csv')
conf_matrix_df.to_csv(conf_matrix_csv_path, index=False)
logging.info(f"Confusion matrix saved as CSV for MATLAB at '{conf_matrix_csv_path}'.")
print("Confusion matrix CSV saved for MATLAB.")

###################################################################################################################################################
# 2024-11-02 07:39:17,260 - INFO - Top 20 features selected: ['MinRadialDist', 'MeanX', 'MeanRadialDist', 'CorrXY', 'MaxRadialDist', 'CovXY', 
# 'MeanY', 'StdX', 'StdRadialDist', 'RangeX', 'MeanZ', 'SkewX', 'MeanIntensity', 'StdDoppler', 'PCA_DirectionZ', 'PCA_DirectionY', 'PCA_DirectionX', 
# 'TotalIntensity', 'CovXZ', 'EigVal1']
# 2024-11-02 07:39:17,264 - INFO - Number of stratified cross-validation folds: 5 folds
# 2024-11-02 07:39:17,264 - INFO - Starting full cross-validation and training with selected features...
# 2024-11-02 07:43:45,474 - INFO - Training completed in 268.21 seconds.
# 2024-11-02 07:43:45,475 - INFO - Best Parameters: {'colsample_bytree': 1.0, 'gamma': 0.1, 'learning_rate': 0.1, 'max_depth': 15, 'min_child_weight': 3, 
# 'n_estimators': 1500, 'reg_alpha': 0, 'reg_lambda': 1, 'subsample': 0.8}
# 2024-11-02 07:43:45,475 - INFO - Best Cross-Validation Accuracy: 75.83%
# 2024-11-02 07:43:45,922 - INFO - Validation Accuracy: 75.95%
# 2024-11-02 07:43:46,368 - INFO - Test Accuracy: 75.87%
###################################################################################################################################################