import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# Configure logging
logging.basicConfig(filename='training_other_models.log', level=logging.INFO,
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
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and test sets
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42)

logging.info(f"Training set: {len(X_train_full)} samples")
logging.info(f"Test set: {len(X_test)} samples")

# Feature selection using Random Forest
logging.info("Starting feature selection with Random Forest...")
rf_selector = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_selector.fit(X_train_full, y_train_full)

# Get feature importances and select top features
importances = rf_selector.feature_importances_
indices = np.argsort(importances)[::-1]  # Sort in descending order

# Select top N features
top_n = 20
top_indices = indices[:top_n]
top_features = [relevant_features[i] for i in top_indices]
logging.info(f"Top {top_n} features selected: {top_features}")

# Use only top features for training
X_train_selected = X_train_full[:, top_indices]
X_test_selected = X_test[:, top_indices]

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(
        multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Support Vector Machine': SVC(random_state=42),
    'Naive Bayes': GaussianNB()
}

# Cross-validation setup
k_folds = 5
logging.info(f"Using {k_folds}-fold cross-validation.")

# Train and evaluate each model using cross-validation
for model_name, model in models.items():
    logging.info(f"Training {model_name} with top {top_n} features...")
    start_time = time.time()
    # Perform cross-validation
    cv_scores = cross_val_score(model, X_train_selected, y_train_full, cv=k_folds, scoring='accuracy', n_jobs=-1)
    mean_cv_score = np.mean(cv_scores)
    training_time = time.time() - start_time
    logging.info(f"{model_name} cross-validation completed in {training_time:.2f} seconds.")
    logging.info(f"{model_name} Cross-Validation Accuracy: {mean_cv_score * 100:.2f}%")
    print(f"{model_name} Cross-Validation Accuracy: {mean_cv_score * 100:.2f}%")
    
    # Fit the model on the entire training data
    model.fit(X_train_selected, y_train_full)
    
    # Evaluate on test set
    y_test_pred = model.predict(X_test_selected)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    logging.info(f"{model_name} Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"{model_name} Test Accuracy: {test_accuracy * 100:.2f}%")
    
    # Generate and save confusion matrix
    conf_matrix = confusion_matrix(y_test, y_test_pred)
    logging.info(f"{model_name} Confusion Matrix:\n{conf_matrix}")
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d',
                xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'{model_name} Confusion Matrix')
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png')
    logging.info(f"{model_name} confusion matrix saved as "
                 f"'{model_name.lower().replace(' ', '_')}_confusion_matrix.png'.")
    plt.close()
    print(f"{model_name} confusion matrix saved.")
