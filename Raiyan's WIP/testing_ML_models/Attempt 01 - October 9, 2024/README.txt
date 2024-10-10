---------------------------------------------------------------------
extracting_features_for_ML_models.mlx
---------------------------------------------------------------------

This MATLAB script processes radar data from four subjects and normalizes the features across all exercises for each subject. The script then splits the data into training, validation, and test sets. Finally, the sets are saved into .mat files for future use in machine learning models.

The data is assumed to contain information on various exercises recorded from radar, with multiple time steps for each exercise. The script handles varying sequence lengths by either padding shorter sequences or truncating longer ones to a fixed length. Labels corresponding to each exercise are also stored for supervised learning tasks.

Prerequisites
Ensure you have the following files and directories properly set up before running the script:

Radar data stored in .mat format, organized by subject. There are four subjects (subject1, subject2, subject3, and subject4), with each subject's radar data stored in files named radar_data1.mat, radar_data2.mat, etc.
MATLAB installed with the necessary toolboxes (such as the Statistics and Machine Learning Toolbox, if applicable).
Input Data:
Each .mat file contains radar data from a specific exercise.
The exercise data has the following features:
Doppler Shift
Intensity
X Coordinate
Y Coordinate
Z Coordinate
Exercises:
The data represents 10 different exercises for each subject. However, for subject4, there are only 9 exercises. The script is designed to handle this difference.
Script Workflow
1. Loading Radar Data
The script loads radar data for each subject, iterating through each exercise. The data is stored as matrices where the rows represent different time steps, and the columns represent the five features (Doppler, Intensity, X, Y, Z).

2. Global Feature Normalization
Before normalization, the script collects all the features from all subjects and exercises to compute global statistics:

Global Mean and Standard Deviation are calculated for each feature (Doppler, Intensity, X, Y, Z).

Each feature is normalized using the formula:
normalized_value = (value - mean) / std_dev​
 
This ensures that the features are standardized across all subjects and exercises.

3. Padding or Truncating Sequences
To ensure that all data sequences have the same length (a requirement for many machine learning algorithms), the script:

Pads shorter sequences with zeros if the number of time steps is less than the defined max_time_steps.
Truncates sequences if they exceed the maximum length.
In this case, the maximum number of time steps is set to 200 (max_time_steps = 200).

4. Flattening Feature Matrices
For each exercise, the time-series data is flattened into a single row vector. This format allows the radar data to be easily fed into machine learning models, such as decision trees or neural networks, which often expect flat feature vectors.

5. Storing Labels
Each exercise is assigned a label corresponding to its exercise type. Labels are stored alongside the features and are used for supervised learning tasks. The label assignments are numeric, where each number corresponds to a specific exercise.

6. Splitting Data into Train, Validation, and Test Sets
The radar data is randomly split into three subsets:

60% Training Set: Used for training the machine learning model.
20% Validation Set: Used for tuning model hyperparameters and evaluating performance during the training process.
20% Test Set: Used for evaluating the final performance of the model.
The data is split randomly using MATLAB’s randperm function to ensure the randomness of the splits.

7. Saving Data to .mat Files
Once the data has been normalized and split into training, validation, and test sets, the script saves the following .mat files:

train_data_flat.mat: Contains the features and labels for the training set.
val_data_flat.mat: Contains the features and labels for the validation set.
test_data_flat.mat: Contains the features and labels for the test set.
These .mat files can later be loaded into MATLAB or other environments to train machine learning models.

Running the Script
Ensure that the radar data files are located in their respective subject folders, and the folder structure matches the expected format.
Run the script in MATLAB.
The script will process the radar data, normalize it, split it into training, validation, and test sets, and save these sets into .mat files.
The output files (train_data_flat.mat, val_data_flat.mat, test_data_flat.mat) will be created in the current directory.
Output
After running the script, you will have the following three output files:

train_data_flat.mat – Training set features and labels.
val_data_flat.mat – Validation set features and labels.
test_data_flat.mat – Test set features and labels.
These files can be directly used for training machine learning models on radar data classification tasks.

---------------------------------------------------------------------
trying_simple_mL_models.mlx
---------------------------------------------------------------------

This MATLAB script is designed to classify radar data into various exercise categories using multiple machine learning models. The radar data has been preprocessed into training, validation, and test sets. The script tries several models, evaluates their performance, and selects the best model based on validation accuracy.

The script covers the following steps:
1. **Loading the Data**: Loads preprocessed radar data from `.mat` files.
2. **Training Machine Learning Models**: Trains multiple machine learning models (Decision Tree, SVM, k-NN, Random Forest, and Neural Network) using the training set.
3. **Validation and Model Selection**: Evaluates each model on the validation set and selects the best-performing model.
4. **Final Test Evaluation**: Evaluates the selected best model on the test set.
5. **Confusion Matrix**: Displays the confusion matrix of the test results for the selected model.

## Prerequisites

Before running this script, ensure you have the following:
- MATLAB (with required toolboxes such as Statistics and Machine Learning Toolbox, Neural Network Toolbox).
- Preprocessed radar data stored in `.mat` files: `train_data_flat.mat`, `val_data_flat.mat`, and `test_data_flat.mat`.
  - Each of these files contains two variables:
    - `XTrain`, `XVal`, `XTest`: Flattened feature vectors of the radar data for training, validation, and test sets.
    - `YTrain`, `YVal`, `YTest`: Corresponding labels (exercise types) for the training, validation, and test sets.

## Models

This script trains the following machine learning models:
1. **Decision Tree**: A simple tree-based model that splits data based on feature thresholds.
2. **Support Vector Machine (SVM)**: A model for binary classification, extended to multi-class classification using Error-Correcting Output Codes (ECOC).
3. **k-Nearest Neighbors (k-NN)**: A simple distance-based classifier using 5 nearest neighbors.
4. **Random Forest**: An ensemble of decision trees trained using bagging.
5. **Neural Network (Multilayer Perceptron)**: A simple feedforward neural network with one hidden layer.