import pandas as pd
import numpy as np
import logging

# Set up logging to display the steps
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# ----------------------------------------
# Step 1: Read the CSV Data
# ----------------------------------------

logging.info('Reading the CSV data')
# Replace 'data.csv' with the path to your CSV file
df = pd.read_csv('mars_predicted_joint_positions_with_labels.csv')

# ----------------------------------------
# Step 2: Map Joint Names to Columns
# ----------------------------------------

logging.info('Mapping joint names to their corresponding columns')

# Define the mapping from joint numbers to joint names
joint_names = {
    1: 'SpineBase',
    2: 'SpineMid',
    3: 'Neck',
    4: 'Head',
    5: 'ShoulderLeft',
    6: 'ElbowLeft',
    7: 'WristLeft',
    8: 'ShoulderRight',
    9: 'ElbowRight',
    10: 'WristRight',
    11: 'HipLeft',
    12: 'KneeLeft',
    13: 'AnkleLeft',
    14: 'FootLeft',
    15: 'HipRight',
    16: 'KneeRight',
    17: 'AnkleRight',
    18: 'FootRight',
    19: 'SpineShoulder'
}

# Extract joint positions into a dictionary
joints = {}
for i in range(1, 20):  # Joints 1 to 19
    joint_name = joint_names[i]
    x_col = f'Joint{i}_X'
    y_col = f'Joint{i}_Y'
    z_col = f'Joint{i}_Z'
    joints[joint_name] = df[[x_col, y_col, z_col]].values

# ----------------------------------------
# Step 3: Normalize Joint Positions
# ----------------------------------------

logging.info('Normalizing joint positions relative to SpineBase')

# Get the SpineBase positions
spine_base = joints['SpineBase']  # N x 3 array

# Normalize joint positions relative to SpineBase
norm_joints = {}
for joint_name, positions in joints.items():
    norm_positions = positions - spine_base  # Relative positions
    norm_joints[joint_name] = norm_positions

logging.info('Normalizing joint positions by body height')

# Compute body height (distance from SpineBase to Head)
body_height_vector = joints['Head'] - joints['SpineBase']  # N x 3 array
body_height = np.linalg.norm(body_height_vector, axis=1)  # N x 1 array

# Avoid division by zero
body_height[body_height == 0] = 1e-6

# Scale normalized positions by body height
norm_joints_scaled = {}
for joint_name, positions in norm_joints.items():
    norm_positions_scaled = positions / body_height[:, np.newaxis]  # N x 3 array
    norm_joints_scaled[joint_name] = norm_positions_scaled

# ----------------------------------------
# Step 4: Define Function to Calculate Angles
# ----------------------------------------

def calculate_angle(a, b, c):
    """
    Calculates the angle at point b formed by the points a, b, and c.
    a, b, c are N x 3 arrays.
    Returns an array of angles in degrees.
    """
    ba = a - b
    bc = c - b
    cosine_angle = np.einsum('ij,ij->i', ba, bc) / (
        np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1) + 1e-6)
    # Ensure cosine values are within -1 to 1 to avoid NaNs
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

# ----------------------------------------
# Step 5: Compute Joint Angles
# ----------------------------------------

# **Justification:** Angles are invariant to position and scale, capturing posture and movement.

logging.info('Computing joint angles')

# Left Elbow Angle
logging.info('Computing Left Elbow Angle')
shoulder_left = joints['ShoulderLeft']
elbow_left = joints['ElbowLeft']
wrist_left = joints['WristLeft']
left_elbow_angle = calculate_angle(shoulder_left, elbow_left, wrist_left)
df['LeftElbowAngle'] = left_elbow_angle

# Right Elbow Angle
logging.info('Computing Right Elbow Angle')
shoulder_right = joints['ShoulderRight']
elbow_right = joints['ElbowRight']
wrist_right = joints['WristRight']
right_elbow_angle = calculate_angle(shoulder_right, elbow_right, wrist_right)
df['RightElbowAngle'] = right_elbow_angle

# Left Knee Angle
logging.info('Computing Left Knee Angle')
hip_left = joints['HipLeft']
knee_left = joints['KneeLeft']
ankle_left = joints['AnkleLeft']
left_knee_angle = calculate_angle(hip_left, knee_left, ankle_left)
df['LeftKneeAngle'] = left_knee_angle

# Right Knee Angle
logging.info('Computing Right Knee Angle')
hip_right = joints['HipRight']
knee_right = joints['KneeRight']
ankle_right = joints['AnkleRight']
right_knee_angle = calculate_angle(hip_right, knee_right, ankle_right)
df['RightKneeAngle'] = right_knee_angle

# Left Shoulder Angle
logging.info('Computing Left Shoulder Angle')
spine_shoulder = joints['SpineShoulder']
left_shoulder_angle = calculate_angle(spine_shoulder, shoulder_left, elbow_left)
df['LeftShoulderAngle'] = left_shoulder_angle

# Right Shoulder Angle
logging.info('Computing Right Shoulder Angle')
right_shoulder_angle = calculate_angle(spine_shoulder, shoulder_right, elbow_right)
df['RightShoulderAngle'] = right_shoulder_angle

# Left Hip Angle
logging.info('Computing Left Hip Angle')
left_hip_angle = calculate_angle(spine_base, hip_left, knee_left)
df['LeftHipAngle'] = left_hip_angle

# Right Hip Angle
logging.info('Computing Right Hip Angle')
right_hip_angle = calculate_angle(spine_base, hip_right, knee_right)
df['RightHipAngle'] = right_hip_angle

# ----------------------------------------
# Step 6: Compute Distances Between Joints
# ----------------------------------------

# **Justification:** Distances capture body proportions and are simple to compute.

logging.info('Computing distances between key joints')

# Left Arm Length
logging.info('Computing Left Arm Length')
left_arm_length = np.linalg.norm(wrist_left - shoulder_left, axis=1)
df['LeftArmLength'] = left_arm_length

# Right Arm Length
logging.info('Computing Right Arm Length')
right_arm_length = np.linalg.norm(wrist_right - shoulder_right, axis=1)
df['RightArmLength'] = right_arm_length

# Left Leg Length
logging.info('Computing Left Leg Length')
left_leg_length = np.linalg.norm(ankle_left - hip_left, axis=1)
df['LeftLegLength'] = left_leg_length

# Right Leg Length
logging.info('Computing Right Leg Length')
right_leg_length = np.linalg.norm(ankle_right - hip_right, axis=1)
df['RightLegLength'] = right_leg_length

# Shoulder Width
logging.info('Computing Shoulder Width')
shoulder_width = np.linalg.norm(shoulder_right - shoulder_left, axis=1)
df['ShoulderWidth'] = shoulder_width

# Hip Width
logging.info('Computing Hip Width')
hip_width = np.linalg.norm(hip_right - hip_left, axis=1)
df['HipWidth'] = hip_width

# ----------------------------------------
# Step 7: Compute Symmetry Features
# ----------------------------------------

# **Justification:** Symmetry features help distinguish exercises involving unilateral movements.

logging.info('Computing symmetry features')

# Angle Differences
df['ElbowAngleDiff'] = df['LeftElbowAngle'] - df['RightElbowAngle']
df['KneeAngleDiff'] = df['LeftKneeAngle'] - df['RightKneeAngle']
df['ShoulderAngleDiff'] = df['LeftShoulderAngle'] - df['RightShoulderAngle']
df['HipAngleDiff'] = df['LeftHipAngle'] - df['RightHipAngle']

# Angle Ratios
df['ElbowAngleRatio'] = df['LeftElbowAngle'] / (df['RightElbowAngle'] + 1e-6)
df['KneeAngleRatio'] = df['LeftKneeAngle'] / (df['RightKneeAngle'] + 1e-6)
df['ShoulderAngleRatio'] = df['LeftShoulderAngle'] / (df['RightShoulderAngle'] + 1e-6)
df['HipAngleRatio'] = df['LeftHipAngle'] / (df['RightHipAngle'] + 1e-6)

# Distance Differences
df['ArmLengthDiff'] = df['LeftArmLength'] - df['RightArmLength']
df['LegLengthDiff'] = df['LeftLegLength'] - df['RightLegLength']

# Distance Ratios
df['ArmLengthRatio'] = df['LeftArmLength'] / (df['RightArmLength'] + 1e-6)
df['LegLengthRatio'] = df['LeftLegLength'] / (df['RightLegLength'] + 1e-6)

# ----------------------------------------
# Step 8: Estimate Center of Mass
# ----------------------------------------

# **Justification:** Center of Mass (CoM) provides insights into balance and stability.

logging.info('Estimating Center of Mass')

# Simple estimate using SpineBase, SpineMid, and SpineShoulder
com_positions = (joints['SpineBase'] + joints['SpineMid'] + joints['SpineShoulder']) / 3.0
df['CoM_X'] = com_positions[:, 0]
df['CoM_Y'] = com_positions[:, 1]
df['CoM_Z'] = com_positions[:, 2]

# Vertical displacement of CoM relative to SpineBase
df['CoM_Height'] = com_positions[:, 1] - spine_base[:, 1]

# ----------------------------------------
# Step 9: Compute Custom Exercise-Specific Features
# ----------------------------------------

# **Justification:** These features capture unique movements of specific exercises.

logging.info('Computing custom exercise-specific features')

# Left Wrist Elevation relative to ShoulderLeft
logging.info('Computing Left Wrist Elevation')
df['LeftWristElevation'] = wrist_left[:, 1] - shoulder_left[:, 1]

# Right Wrist Elevation relative to ShoulderRight
logging.info('Computing Right Wrist Elevation')
df['RightWristElevation'] = wrist_right[:, 1] - shoulder_right[:, 1]

# Vertical position of SpineBase
logging.info('Computing SpineBase Vertical Position')
df['SpineBase_Y'] = spine_base[:, 1]

# Knee Angle Deviations from 90 degrees
logging.info('Computing Knee Angle Deviations from 90 degrees')
df['LeftKneeAngleDeviation'] = np.abs(df['LeftKneeAngle'] - 90)
df['RightKneeAngleDeviation'] = np.abs(df['RightKneeAngle'] - 90)

# Boolean flags indicating if knee angles are close to 90 degrees
df['LeftKneeAt90'] = (df['LeftKneeAngleDeviation'] < 10).astype(int)
df['RightKneeAt90'] = (df['RightKneeAngleDeviation'] < 10).astype(int)

# ----------------------------------------
# Step 10: Additional Features
# ----------------------------------------

# **Justification:** Enhance feature set to capture more posture details.

logging.info('Computing additional features')

# Distance Between Wrists
logging.info('Computing Distance Between Wrists')
wrist_distance = np.linalg.norm(wrist_left - wrist_right, axis=1)
df['WristDistance'] = wrist_distance

# Distance Between Ankles
logging.info('Computing Distance Between Ankles')
ankle_left = joints['AnkleLeft']
ankle_right = joints['AnkleRight']
logging.info('Computing Distance Between Ankles')
ankle_distance = np.linalg.norm(ankle_left - ankle_right, axis=1)
df['AnkleDistance'] = ankle_distance

# Left Hand Over Head Feature
logging.info('Computing Left Hand Over Head Feature')
head = joints['Head']
df['LeftHandOverHead'] = (wrist_left[:, 1] > head[:, 1]).astype(int)

# Right Hand Over Head Feature
logging.info('Computing Right Hand Over Head Feature')
df['RightHandOverHead'] = (wrist_right[:, 1] > head[:, 1]).astype(int)

# Left Arm Vertical Angle
logging.info('Computing Left Arm Vertical Angle')
vertical_vector = np.array([0, 1, 0])  # Upward direction
left_arm_vector = wrist_left - shoulder_left
left_arm_vector_norm = np.linalg.norm(left_arm_vector, axis=1)[:, np.newaxis]
left_arm_vector_unit = left_arm_vector / (left_arm_vector_norm + 1e-6)
cos_theta_left = np.dot(left_arm_vector_unit, vertical_vector)
cos_theta_left = np.clip(cos_theta_left, -1.0, 1.0)
left_arm_vertical_angle = np.arccos(cos_theta_left)
df['LeftArmVerticalAngle'] = np.degrees(left_arm_vertical_angle)

# Right Arm Vertical Angle
logging.info('Computing Right Arm Vertical Angle')
right_arm_vector = wrist_right - shoulder_right
right_arm_vector_norm = np.linalg.norm(right_arm_vector, axis=1)[:, np.newaxis]
right_arm_vector_unit = right_arm_vector / (right_arm_vector_norm + 1e-6)
cos_theta_right = np.dot(right_arm_vector_unit, vertical_vector)
cos_theta_right = np.clip(cos_theta_right, -1.0, 1.0)
right_arm_vertical_angle = np.arccos(cos_theta_right)
df['RightArmVerticalAngle'] = np.degrees(right_arm_vertical_angle)

# Torso Inclination Angle
logging.info('Computing Torso Inclination Angle')
torso_vector = spine_shoulder - spine_base
torso_vector_norm = np.linalg.norm(torso_vector, axis=1)[:, np.newaxis]
torso_vector_unit = torso_vector / (torso_vector_norm + 1e-6)
cos_theta_torso = np.dot(torso_vector_unit, vertical_vector)
cos_theta_torso = np.clip(cos_theta_torso, -1.0, 1.0)
torso_inclination_angle = np.arccos(cos_theta_torso)
df['TorsoInclinationAngle'] = np.degrees(torso_inclination_angle)

# Upper Leg Angle Difference
logging.info('Computing Upper Leg Angle Difference')
left_upper_leg_vector = knee_left - hip_left
right_upper_leg_vector = knee_right - hip_right
cos_theta_legs = np.einsum('ij,ij->i', left_upper_leg_vector, right_upper_leg_vector) / (
    np.linalg.norm(left_upper_leg_vector, axis=1) * np.linalg.norm(right_upper_leg_vector, axis=1) + 1e-6)
cos_theta_legs = np.clip(cos_theta_legs, -1.0, 1.0)
leg_angle_difference = np.arccos(cos_theta_legs)
df['UpperLegAngleDifference'] = np.degrees(leg_angle_difference)

# ----------------------------------------
# Step 11: Prepare Final DataFrame
# ----------------------------------------

logging.info('Preparing the final DataFrame with selected features')

# List of feature columns
feature_columns = [
    'LeftElbowAngle', 'RightElbowAngle', 'LeftKneeAngle', 'RightKneeAngle',
    'LeftShoulderAngle', 'RightShoulderAngle', 'LeftHipAngle', 'RightHipAngle',
    'LeftArmLength', 'RightArmLength', 'LeftLegLength', 'RightLegLength',
    'ShoulderWidth', 'HipWidth',
    'ElbowAngleDiff', 'KneeAngleDiff', 'ShoulderAngleDiff', 'HipAngleDiff',
    'ElbowAngleRatio', 'KneeAngleRatio', 'ShoulderAngleRatio', 'HipAngleRatio',
    'ArmLengthDiff', 'LegLengthDiff', 'ArmLengthRatio', 'LegLengthRatio',
    'CoM_X', 'CoM_Y', 'CoM_Z', 'CoM_Height',
    'LeftWristElevation', 'RightWristElevation', 'SpineBase_Y',
    'LeftKneeAngleDeviation', 'RightKneeAngleDeviation',
    'LeftKneeAt90', 'RightKneeAt90',
    'WristDistance', 'AnkleDistance',
    'LeftHandOverHead', 'RightHandOverHead',
    'LeftArmVerticalAngle', 'RightArmVerticalAngle',
    'TorsoInclinationAngle', 'UpperLegAngleDifference',
    'Exercise', 'Subject'
]

# Ensure all columns exist
missing_columns = [col for col in feature_columns if col not in df.columns]
if missing_columns:
    logging.warning(f'The following columns are missing and will be ignored: {missing_columns}')
    feature_columns = [col for col in feature_columns if col in df.columns]

# Create the final DataFrame
df_final = df[feature_columns]

# ----------------------------------------
# Step 12: Save to CSV
# ----------------------------------------

logging.info('Saving the final DataFrame to CSV')

# Save the DataFrame to a CSV file
df_final.to_csv('comprehensive_features_from_joint_data.csv', index=False)

logging.info('Feature extraction completed and data saved to comprehensive_features_from_joint_data.csv')
