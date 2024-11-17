import os
import numpy as np
import pandas as pd
import h5py
from keras.models import load_model

# Define the base folder where the subject folders are stored
BASE_FOLDER = '../../../synced_data/woutlier/'

# List of subjects
SUBJECT_FOLDERS = ['subject1', 'subject2', 'subject3', 'subject4']

# List of radar files and corresponding exercise labels
FILE_EXERCISE_MAPPING = [
    ('radar_data1.mat', 'exercise1'),
    ('radar_data2.mat', 'exercise2'),
    ('radar_data3.mat', 'exercise3'),
    ('radar_data4.mat', 'exercise4'),
    ('radar_data5.mat', 'exercise5'),
    ('radar_data6.mat', 'exercise6'),
    ('radar_data7.mat', 'exercise7'),
    ('radar_data8.mat', 'exercise8'),
    ('radar_data9.mat', 'exercise9'),
    ('radar_data10.mat', 'exercise10')
]

# Initialize containers for features, labels, and subjects
all_features = []
all_labels = []
all_subjects = []

# Loop through each subject folder
for subject_idx, subject in enumerate(SUBJECT_FOLDERS, start=1):
    subject_folder = os.path.join(BASE_FOLDER, subject)
    num_files = 9 if subject == 'subject4' else 10  # Subject 4 has only 9 radar files

    # Loop through radar data files for the current subject
    for i in range(num_files):
        radar_file, exercise_label = FILE_EXERCISE_MAPPING[i]
        radar_path = os.path.join(subject_folder, radar_file)

        # Load radar data
        # Load radar data using h5py for MATLAB v7.3 files
        with h5py.File(radar_path, 'r') as f:
            radar_data_cropped = np.array(f['radar_data_cropped'])  # Replace 'radar_data_cropped' with the actual dataset name in your .mat file
            radar_data_cropped = radar_data_cropped.T  # h5py loads data transposed, so we need to correct it

        # Extract point cloud data: X (3rd), Y (4th), Z (5th), Doppler (6th), Intensity (7th)
        point_cloud_data = radar_data_cropped[:, [2, 3, 4, 5, 6]]  # MATLAB indexing starts at 1

        # Get unique frame numbers
        frames = np.unique(radar_data_cropped[:, 0])

        # Loop through each frame
        for frame_num in frames:
            current_frame_data = point_cloud_data[radar_data_cropped[:, 0] == frame_num]

            # Filter valid points within the defined range
            x_range, y_range, z_range = [-1, 1], [0, 3], [-1, 1]
            valid_points = current_frame_data[
                (current_frame_data[:, 0] >= x_range[0]) & (current_frame_data[:, 0] <= x_range[1]) &
                (current_frame_data[:, 1] >= y_range[0]) & (current_frame_data[:, 1] <= y_range[1]) &
                (current_frame_data[:, 2] >= z_range[0]) & (current_frame_data[:, 2] <= z_range[1])
            ]

            # Sort points by X, then Y, then Z
            sorted_points = valid_points[np.lexsort((valid_points[:, 2], valid_points[:, 1], valid_points[:, 0]))]

            # Ensure there are exactly 64 points
            num_points = sorted_points.shape[0]
            if num_points < 64:
                # Pad with zeros
                sorted_points = np.vstack([sorted_points, np.zeros((64 - num_points, 5))])
            elif num_points > 64:
                # Truncate to 64 points
                sorted_points = sorted_points[:64, :]

            # Reshape to 8x8x5 feature map
            feature_map = sorted_points.reshape(8, 8, 5)

            # Append the feature map, label, and subject
            all_features.append(feature_map)
            all_labels.append(exercise_label)
            all_subjects.append(f"Subject{subject_idx}")

# Convert to numpy arrays
all_features = np.array(all_features)
all_labels = np.array(all_labels)
all_subjects = np.array(all_subjects)

# Load the pretrained MARS model (assuming it's a Keras model)
mars_model = load_model('../../../model/MARS.h5', compile=False)

# Predict joint positions using the MARS model
joint_predictions = []

total_frames = len(all_features)  # Total number of frames
print(f"Total frames to process: {total_frames}")

count = 0
for feature_map in all_features:
    feature_vector = feature_map.reshape(1, 8, 8, 5)  # Reshape to (1, 8, 8, 5)
    joint_positions = mars_model.predict(feature_vector, verbose=0)
    joint_predictions.append(joint_positions[0])  # Assuming model output is (1, n) shape

    # Temporary progress print statement
    count += 1
    if (count + 1) % 100 == 0 or (count + 1) == total_frames:
        print(f"Processed {count + 1}/{total_frames} frames...")

# Convert joint predictions to numpy array
joint_predictions = np.array(joint_predictions)

# Combine joint positions with exercise labels and subjects
num_joints = joint_predictions.shape[1] // 3  # Assuming x, y, z per joint
joint_columns = [f'Joint{j+1}_{axis}' for j in range(num_joints) for axis in ['X', 'Y', 'Z']]
output_df = pd.DataFrame(joint_predictions, columns=joint_columns)
output_df['Exercise'] = all_labels
output_df['Subject'] = all_subjects

# Save to CSV
output_df.to_csv('mars_predicted_joint_positions_with_labels.csv', index=False)

print("Joint positions, exercise labels, and subjects saved to 'mars_predicted_joint_positions_with_labels.csv'.")
