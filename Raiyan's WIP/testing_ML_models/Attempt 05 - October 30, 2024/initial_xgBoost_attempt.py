import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import logging  # Logging for progress tracking

# Configure logging
logging.basicConfig(filename='full_training.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Set random seed for reproducibility
np.random.seed(42)

# Load the full dataset and drop missing values
data = pd.read_csv('allRadarData.csv').dropna()

# Select relevant features and labels
relevant_features = [
    'NumPoints', 'MeanX', 'StdX', 'MeanY', 'StdY', 'MeanZ', 'StdZ',
    'MeanDoppler', 'StdDoppler', 'MeanIntensity', 'StdIntensity',
    'RangeX', 'RangeY', 'RangeZ', 'SkewX', 'KurtX', 'SkewY', 
    'KurtY', 'SkewZ', 'KurtZ', 'NumOccupiedVoxels'
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

# Hyperparameter grid for full training (using single parameters for this run)
param_grid = {
    'max_depth': [9],              
    'learning_rate': [0.1],   
    'n_estimators': [150],      
    'subsample': [0.8],          
    'colsample_bytree': [1.0]
}

# XGBoost model with GPU support
xgb_model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=len(np.unique(y)),
    random_state=42,
    tree_method='hist',  # Use histogram-based method
    device='cuda'        # Set device to GPU
)

# Stratified K-Fold Cross-Validation (using 5 folds for better evaluation)
k = 5
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=xgb_model, param_grid=param_grid, scoring='accuracy', cv=skf, n_jobs=-1, verbose=0
)

# Start timing the training process
start_time = time.time()
logging.info("Starting full cross-validation and training...")

# Fit the model
grid_search.fit(X_train, y_train)

# Calculate time taken
end_time = time.time()
elapsed_time = end_time - start_time
logging.info(f"Training completed in {elapsed_time:.2f} seconds.")

# Log the best parameters and cross-validation accuracy
best_params = grid_search.best_params_
best_score = grid_search.best_score_ * 100

logging.info(f"Best Parameters: {best_params}")
logging.info(f"Best Cross-Validation Accuracy: {best_score:.2f}%")

print(f"Best Parameters: {best_params}")
print(f"Best Cross-Validation Accuracy: {best_score:.2f}%")

# Evaluate on validation set
y_val_pred = grid_search.best_estimator_.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
logging.info(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

# Evaluate on test set
y_test_pred = grid_search.best_estimator_.predict(X_test)
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
