import os
import numpy as np
import pandas as pd
import h5py
from keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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

# Initialize containers for features and labels
all_features = []
all_labels = []

# New containers to keep track of subject, exercise, and frame number
all_subjects = []
all_exercises = []
all_frame_numbers = []

# Loop through each subject folder
for subject in SUBJECT_FOLDERS:
    subject_folder = os.path.join(BASE_FOLDER, subject)
    num_files = 9 if subject == 'subject4' else 10  # Subject 4 has only 9 radar files

    # Loop through radar data files for the current subject
    for i in range(num_files):
        radar_file, exercise_label = FILE_EXERCISE_MAPPING[i]
        radar_path = os.path.join(subject_folder, radar_file)

        # Load radar data
        # Load radar data using h5py for MATLAB v7.3 files
        with h5py.File(radar_path, 'r') as f:
            # Replace 'radar_data_cropped' with the actual dataset name in your .mat file
            radar_data_cropped = np.array(f['radar_data_cropped'])  
            radar_data_cropped = radar_data_cropped.T  # h5py loads data transposed, so we need to correct it

        # Extract point cloud data: X (3rd), Y (4th), Z (5th), Doppler (6th), Intensity (7th)
        point_cloud_data = radar_data_cropped[:, [2, 3, 4, 5, 6]]  # MATLAB indexing starts at 1

        # Get frame numbers
        frames = radar_data_cropped[:, 0]

        # Loop through each frame
        unique_frames = np.unique(frames)
        for frame_num in unique_frames:
            current_frame_data = point_cloud_data[frames == frame_num]

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

            # Append the feature map and labels
            all_features.append(feature_map)
            all_labels.append(exercise_label)
            all_subjects.append(subject)
            all_exercises.append(exercise_label)
            all_frame_numbers.append(int(frame_num))

# Convert to numpy arrays
all_features = np.array(all_features)
all_labels = np.array(all_labels)
all_subjects = np.array(all_subjects)
all_exercises = np.array(all_exercises)
all_frame_numbers = np.array(all_frame_numbers)

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
    if (count + 1) % 1000 == 0 or (count + 1) == total_frames:
        print(f"Processed {count + 1}/{total_frames} frames...")

# Convert joint predictions to numpy array
joint_predictions = np.array(joint_predictions)

# Combine joint positions with exercise labels and other metadata
num_joints = joint_predictions.shape[1] // 3  # Assuming x, y, z per joint
joint_columns = [f'Joint{j+1}_{axis}' for j in range(num_joints) for axis in ['X', 'Y', 'Z']]

# Create a DataFrame with joint positions and metadata
data = pd.DataFrame(joint_predictions, columns=joint_columns)
data['Subject'] = all_subjects
data['Exercise'] = all_exercises
data['Frame'] = all_frame_numbers

# Save to CSV (optional)
data.to_csv('joint_positions_with_labels.csv', index=False)
print("Joint positions and metadata saved to 'joint_positions_with_labels.csv'.")

# Calculate global axis limits across all joint positions
global_min = data[joint_columns].min().min()
global_max = data[joint_columns].max().max()

print(f"Global axis limits calculated: [{global_min}, {global_max}]")

# Define the skeleton structure (joint connections)
# This will depend on the specific joints used in MARS model
# Example: For a skeleton with 17 joints (like COCO dataset), connections are defined as follows
skeleton_connections = [
    (0, 1), (1, 2), (2, 3),  # Right arm
    (0, 4), (4, 5), (5, 6),  # Left arm
    (0, 7), (7, 8), (8, 9),  # Torso
    (7, 10), (10, 11), (11, 12),  # Right leg
    (7, 13), (13, 14), (14, 15)  # Left leg
    # Add more connections as per the actual joint indices
]

# Ensure the skeleton_connections are valid for your model's joints

# Create output directory for videos
output_folder = './videos'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Group the data by subject and exercise
grouped = data.groupby(['Subject', 'Exercise'])

# Generate videos for each subject and exercise
for (subject, exercise), group in grouped:
    frames = group.sort_values('Frame')
    frame_numbers = frames['Frame'].values
    num_frames = len(frame_numbers)
    joint_positions = frames[joint_columns].values.reshape(num_frames, num_joints, 3)

    # Define video file name
    video_name = f"{subject}_{exercise}.mp4"
    video_path = os.path.join(output_folder, video_name)

    # Create a figure for plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([global_min, global_max])
    ax.set_ylim([global_min, global_max])
    ax.set_zlim([global_min, global_max])
    ax.view_init(elev=30, azim=45)  # Set consistent view angle

    # Function to update the plot for each frame
    def update(num, joint_positions, lines, scatters):
        ax.collections.clear()
        joints = joint_positions[num]
        xs = joints[:, 0]
        ys = joints[:, 1]
        zs = joints[:, 2]

        # Plot the joints
        scatters = ax.scatter(xs, ys, zs, c='r', s=25)

        # Plot the bones
        for connection in skeleton_connections:
            joint1 = joints[connection[0]]
            joint2 = joints[connection[1]]
            line = ax.plot([joint1[0], joint2[0]],
                           [joint1[1], joint2[1]],
                           [joint1[2], joint2[2]], c='b')
        return scatters

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=num_frames, fargs=(joint_positions, [], []), interval=100)

    # Save the animation
    ani.save(video_path, writer='ffmpeg', fps=10)
    plt.close(fig)
    print(f"Created video: {video_name}")

print(f"{len(grouped)} videos created successfully in {output_folder}.")
