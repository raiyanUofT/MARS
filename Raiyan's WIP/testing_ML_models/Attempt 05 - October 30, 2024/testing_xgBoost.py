import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm  # Progress bar
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

print(f"Training set: {len(X_train)} samples")
print(f"Validation set: {len(X_val)} samples")
print(f"Test set: {len(X_test)} samples")

# Hyperparameter grid for full training
param_grid = {
    'max_depth': [3, 6, 9],              
    'learning_rate': [0.01, 0.05, 0.1],   
    'n_estimators': [50, 100, 150],      
    'subsample': [0.6, 0.8, 1.0],          
    'colsample_bytree': [0.6, 0.8, 1.0]
}

# XGBoost model with updated GPU support
xgb_model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=len(np.unique(y)),
    random_state=42,
    tree_method='hist',         # Use GPU-friendly histogram-based algorithm
    device='cuda'               # Specify GPU device for training
)

# Stratified K-Fold Cross-Validation (using 5 folds for better evaluation)
k = 5
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

# Custom GridSearchCV with TQDM progress bar
class TQDMGridSearchCV(GridSearchCV):
    def fit(self, X, y, **kwargs):
        # Calculate the total number of iterations
        n_iter = k * np.prod([len(v) for v in self.param_grid.values()])

        with tqdm(total=n_iter, desc="Grid Search Progress", unit="iteration") as pbar:
            # Call parent fit and update progress within each iteration
            result = super().fit(X, y, **kwargs)
            pbar.update(n_iter)  # Update the progress bar with total iterations
            return result

# Initialize the GridSearchCV with TQDM wrapper
grid_search = TQDMGridSearchCV(
    estimator=xgb_model, param_grid=param_grid, scoring='accuracy', cv=skf, n_jobs=-1, verbose=0
)

print("Starting full cross-validation and training...")
grid_search.fit(X_train, y_train)

# Log and display the best parameters and accuracy
best_params = grid_search.best_params_
best_score = grid_search.best_score_ * 100

logging.info(f"Best Parameters: {best_params}")
logging.info(f"Best Cross-Validation Accuracy: {best_score:.2f}%")

print(f"Best Parameters: {best_params}")
print(f"Best Cross-Validation Accuracy: {best_score:.2f}%")

# Evaluate on validation set
y_val_pred = grid_search.best_estimator_.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

# Evaluate on test set
y_test_pred = grid_search.best_estimator_.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Generate and save confusion matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', 
            xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('XGBoost Confusion Matrix')
plt.savefig('full_confusion_matrix.png')
print("Confusion matrix saved as 'full_confusion_matrix.png'.")
