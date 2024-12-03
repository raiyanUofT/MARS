#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Navigate to the directory containing the radar data training scripts
cd "/Training on Radar Data"

echo "Starting Radar Data K-Fold Only Training..."
python3 training_radar_data_with_kfold_only.py > radar_kfold.log 2>&1
echo "Completed Radar Data K-Fold Only Training."

echo "Starting Radar Data LOSO and Nested K-Fold Training..."
python3 training_radar_data_with_LOSO_and_nested_kfold.py > radar_loso.log 2>&1
echo "Completed Radar Data LOSO and Nested K-Fold Training."

# Navigate to the directory containing the joint data training scripts
cd "../Training on Joint Data"

echo "Starting Joint Data K-Fold Only Training..."
python3 training_joint_data_with_kfold_only.py > joint_kfold.log 2>&1
echo "Completed Joint Data K-Fold Only Training."

echo "Starting Joint Data LOSO and Nested K-Fold Training..."
python3 training_joint_data_with_LOSO_and_nested_kfold.py > joint_loso.log 2>&1
echo "Completed Joint Data LOSO and Nested K-Fold Training."

echo "All training scripts have been executed successfully."
