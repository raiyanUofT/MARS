import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Configure logging
logging.basicConfig(filename='training.log', level=logging.INFO, 
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
top_n = 40
top_features = [relevant_features[i] for i in indices[:top_n]]
logging.info(f"Top {top_n} features selected: {top_features}")

# Use only top features for training
X_train_selected = X_train[:, indices[:top_n]]
X_val_selected = X_val[:, indices[:top_n]]
X_test_selected = X_test[:, indices[:top_n]]

# Hyperparameter grid for full training
# param_grid = {
#     'max_depth': [9, 12, 15],
#     'learning_rate': [0.05, 0.1],
#     'n_estimators': [300, 500, 1000],
#     'subsample': [0.75, 0.8],
#     'colsample_bytree': [0.6, 0.8, 1.0],
#     'gamma': [0, 0.1, 0.3],
#     'min_child_weight': [3, 5],
#     'reg_alpha': [0, 1, 5],  # L1 regularization
#     'reg_lambda': [1, 5]     # L2 regularization
# }

param_grid = {
    'max_depth': [9],
    'learning_rate': [0.1],
    'n_estimators': [300],
    'subsample': [0.8],
    'colsample_bytree': [1.0],
    'gamma': [0.1],
    'min_child_weight': [3],
    'reg_alpha': [1],  # L1 regularization
    'reg_lambda': [1]     # L2 regularization
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

# Initialize RandomizedSearchCV
# param_search = RandomizedSearchCV(
#     estimator=xgb_model, param_distributions=param_grid, scoring='accuracy', cv=skf, n_jobs=-1,
#     n_iter=5, verbose=0, random_state=42
# )

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

# Generate and save confusion matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)
logging.info(f"Confusion Matrix:\n{conf_matrix}")

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', 
            xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('XGBoost Confusion Matrix')
plt.savefig('full_confusion_matrix.png')
logging.info("Confusion matrix saved as 'full_confusion_matrix.png'.")
print("Confusion matrix saved as 'full_confusion_matrix.png'.")


###################################################################################################################################################
# 2024-10-31 14:00:33,360 - INFO - Top 40 features selected: ['MinRadialDist', 'MeanX', 'MeanRadialDist', 'CorrXY', 'MaxRadialDist', 'CovXY', 
# 'MeanY', 'StdX', 'StdRadialDist', 'RangeX', 'MeanZ', 'SkewX', 'MeanIntensity', 'StdDoppler', 'PCA_DirectionZ', 'PCA_DirectionY', 'PCA_DirectionX', 
# 'TotalIntensity', 'CovXZ', 'EigVal1', 'VoxelEntropy', 'CovYZ', 'CorrXZ', 'SkewY', 'StdZ', 'RangeY', 'ConvexHullVolume', 'RangeZ', 'StdY', 
# 'MeanGradX', 'CorrYZ', 'KurtX', 'EigVal2', 'MaxIntensity', 'StdIntensity', 'NumPoints', 'KurtDoppler', 'EigVal3', 'MeanDoppler', 'VoxelDensity']

# 2024-10-31 14:00:33,367 - INFO - Number of stratified cross-validation folds: 5 folds
# 2024-10-31 14:02:52,031 - INFO - Training completed in 138.66 seconds.

# 2024-10-31 14:02:52,033 - INFO - Best Parameters: {'colsample_bytree': 1.0, 'gamma': 0.1, 'learning_rate': 0.1, 'max_depth': 9, 'min_child_weight': 3, 
# 'n_estimators': 300, 'reg_alpha': 1, 'reg_lambda': 1, 'subsample': 0.8}

# 2024-10-31 14:02:52,034 - INFO - Best Cross-Validation Accuracy: 76.56%
# 2024-10-31 14:02:52,225 - INFO - Validation Accuracy: 77.00%
# 2024-10-31 14:02:52,388 - INFO - Test Accuracy: 76.44%
###################################################################################################################################################