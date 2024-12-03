import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
from keras.models import load_model
from keras.losses import mean_squared_error  # Importing mse explicitly

# Set the subject and exercise to test
subject = 'subject1'
exercise_label = 'exercise3'
radar_file = 'radar_data3.mat'

# Define the base folder where the subject folder is stored
BASE_FOLDER = '../../../synced_data/woutlier/'

# Path to radar data
subject_folder = os.path.join(BASE_FOLDER, subject)
radar_path = os.path.join(subject_folder, radar_file)

# Load radar data
with h5py.File(radar_path, 'r') as f:
    # Replace 'radar_data_cropped' with the actual dataset name in your .mat file
    radar_data_cropped = np.array(f['radar_data_cropped'])  
    radar_data_cropped = radar_data_cropped.T  # h5py loads data transposed

# Extract point cloud data: X (3rd), Y (4th), Z (5th), Doppler (6th), Intensity (7th)
point_cloud_data = radar_data_cropped[:, [2, 3, 4, 5, 6]]  # MATLAB indexing starts at 1

# Get frame numbers
frames = radar_data_cropped[:, 0]

# Initialize containers for features
features = []
frame_numbers = []

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

    # Append the feature map and frame number
    features.append(feature_map)
    frame_numbers.append(int(frame_num))

# Convert to numpy arrays
features = np.array(features)
frame_numbers = np.array(frame_numbers)

# *** Add code to save the feature map ***
# Save the features to a .npy file
output_feature_map_path = 'fmap.npy'
np.save(output_feature_map_path, features)
print(f"Feature map saved as {output_feature_map_path}")

# Create the directory for saving plots if it doesn't exist
output_dir = './skeleton_plots'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

connections = [
    (0, 1),  # SpineBase to SpineMid
    (1, 2),  # SpineMid to Neck
    (2, 3),  # Neck to Head
    (2, 4),  # Neck to ShoulderLeft
    (2, 7),  # Neck to ShoulderRight
    (4, 5),  # ShoulderLeft to ElbowLeft
    (5, 6),  # ElbowLeft to WristLeft
    (7, 8),  # ShoulderRight to ElbowRight
    (8, 9),  # ElbowRight to WristRight
    (0, 14),  # SpineBase to HipRight
    (14, 15),  # HipRight to KneeRight
    (15, 16),  # KneeRight to AnkleRight
    (16, 17),  # AnkleRight to FootRight
    (0, 10),  # SpineBase to HipLeft
    (10, 11),  # HipLeft to KneeLeft
    (11, 12),  # KneeLeft to AnkleLeft
    (12, 13),  # AnkleLeft to FootLeft
    (2, 18),  # Neck to SpineShoulder
]

# Define keypoint colors
keypoint_colors = [
    "blue",  # SpineBase,
    "blue",  # SpineMid,
    "blue",  # Neck,
    "red",  # Head,
    "blue",  # ShoulderLeft,
    "green",  # ElbowLeft,
    "green",  # WristLeft,
    "blue",  # ShoulderRight,
    "green",  # ElbowRight,
    "green",  # WristRight,
    "blue",  # HipLeft,
    "green",  # KneeLeft,
    "green",  # AnkleLeft,
    "green",  # FootLeft,
    "blue",  # HipRight,
    "green",  # KneeRight,
    "green",  # AnkleRight,
    "green",  # FootRight,
    "blue",  # SpineShoulder
]

def plot_skeleton(reshaped_data, ax, color_default=True):
    for connection in connections:
        x_values = [reshaped_data[0][connection[0]], reshaped_data[0][connection[1]]]
        y_values = [reshaped_data[1][connection[0]], reshaped_data[1][connection[1]]]
        z_values = [reshaped_data[2][connection[0]], reshaped_data[2][connection[1]]]

        ax.plot(x_values, z_values, y_values, color="black")

    for keypoint_index in range(len(reshaped_data[0])):
        if color_default:
            color = keypoint_colors[keypoint_index]
        else:
            color = "gray"

        ax.scatter(
            reshaped_data[0][keypoint_index],
            reshaped_data[2][keypoint_index],
            reshaped_data[1][keypoint_index],
            c=color,
            marker="o",
            s=100 if keypoint_index == 3 else 50,  # Larger size for the head
        )

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Load MARS model
model = load_model("../../../model/MARS.h5", custom_objects={'mse': mean_squared_error})

# Add your own path of the testing data and labels
featuremap_test = np.load("fmap.npy")

predictions = model.predict(featuremap_test)

for predict_num, prediction in enumerate(predictions):
    ax.clear()  # Clear the plot before each iteration

    # NOTE: MARS outputs the keypoint coords as [x1, x2, ..., xN, z1, z2, ..., zN, y1, y2, ..., yN]
    reshaped_data = prediction.reshape(3, -1)
    plot_skeleton(reshaped_data, ax)

    # Set fixed axis scales
    ax.set_xlim(-2, 2)
    ax.set_ylim(0, 4)
    ax.set_zlim(0, 3)

    # Label axes clearly
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    # Set the view angle (elevation and azimuth)
    ax.view_init(elev=-90, azim=-90)

    # Save the plot to a file instead of showing it interactively
    plt.savefig(f"{output_dir}/skeleton_plot_frame_{predict_num}.png")