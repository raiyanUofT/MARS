% Set up logging
fprintf('Starting feature extraction from joint data\n');

% ----------------------------------------
% Step 1: Read the CSV Data
% ----------------------------------------

fprintf('Reading the CSV data\n');
% Replace 'mars_predicted_joint_positions_with_labels.csv' with your CSV file path
data = readtable('mars_predicted_joint_positions_with_labels.csv');

% ----------------------------------------
% Step 2: Map Joint Names to Columns
% ----------------------------------------

fprintf('Mapping joint names to their corresponding columns\n');

% Define the mapping from joint numbers to joint names
joint_names = {
    'SpineBase', 'SpineMid', 'Neck', 'Head', 'ShoulderLeft', ...
    'ElbowLeft', 'WristLeft', 'ShoulderRight', 'ElbowRight', 'WristRight', ...
    'HipLeft', 'KneeLeft', 'AnkleLeft', 'FootLeft', 'HipRight', ...
    'KneeRight', 'AnkleRight', 'FootRight', 'SpineShoulder'};

% Initialize structure to hold joint positions
joints = struct();

for i = 1:length(joint_names)
    joint_name = joint_names{i};
    x_col = sprintf('Joint%d_X', i);
    y_col = sprintf('Joint%d_Y', i);
    z_col = sprintf('Joint%d_Z', i);
    if ismember(x_col, data.Properties.VariableNames)
        joints.(joint_name) = [data{:, x_col}, data{:, y_col}, data{:, z_col}];
    else
        error('Column %s not found in data.', x_col);
    end
end

% ----------------------------------------
% Step 3: Normalize Joint Positions
% ----------------------------------------

fprintf('Normalizing joint positions relative to SpineBase\n');

% Get the SpineBase positions
spine_base = joints.SpineBase;  % N x 3 array

% Normalize joint positions relative to SpineBase
norm_joints = struct();
joint_fields = fieldnames(joints);
for i = 1:length(joint_fields)
    joint_name = joint_fields{i};
    positions = joints.(joint_name);
    norm_positions = positions - spine_base;  % Relative positions
    norm_joints.(joint_name) = norm_positions;
end

fprintf('Normalizing joint positions by body height\n');

% Compute body height (distance from SpineBase to Head)
body_height_vector = joints.Head - joints.SpineBase;  % N x 3 array
body_height = sqrt(sum(body_height_vector.^2, 2));    % N x 1 array

% Avoid division by zero
body_height(body_height == 0) = 1e-6;

% Scale normalized positions by body height
norm_joints_scaled = struct();
for i = 1:length(joint_fields)
    joint_name = joint_fields{i};
    positions = norm_joints.(joint_name);
    norm_positions_scaled = positions ./ body_height;  % Element-wise division
    norm_joints_scaled.(joint_name) = norm_positions_scaled;
end

% ----------------------------------------
% Step 4: Define Function to Calculate Angles
% ----------------------------------------

calculate_angle = @(a, b, c) rad2deg(acosd( ...
    sum((a - b) .* (c - b), 2) ./ (vecnorm(a - b, 2, 2) .* vecnorm(c - b, 2, 2) + 1e-6)));

% ----------------------------------------
% Step 5: Compute Joint Angles
% ----------------------------------------

fprintf('Computing joint angles\n');

% Left Elbow Angle
fprintf('Computing Left Elbow Angle\n');
shoulder_left = joints.ShoulderLeft;
elbow_left = joints.ElbowLeft;
wrist_left = joints.WristLeft;
left_elbow_angle = calculate_angle(shoulder_left, elbow_left, wrist_left);
data.LeftElbowAngle = left_elbow_angle;

% Right Elbow Angle
fprintf('Computing Right Elbow Angle\n');
shoulder_right = joints.ShoulderRight;
elbow_right = joints.ElbowRight;
wrist_right = joints.WristRight;
right_elbow_angle = calculate_angle(shoulder_right, elbow_right, wrist_right);
data.RightElbowAngle = right_elbow_angle;

% Left Knee Angle
fprintf('Computing Left Knee Angle\n');
hip_left = joints.HipLeft;
knee_left = joints.KneeLeft;
ankle_left = joints.AnkleLeft;
left_knee_angle = calculate_angle(hip_left, knee_left, ankle_left);
data.LeftKneeAngle = left_knee_angle;

% Right Knee Angle
fprintf('Computing Right Knee Angle\n');
hip_right = joints.HipRight;
knee_right = joints.KneeRight;
ankle_right = joints.AnkleRight;
right_knee_angle = calculate_angle(hip_right, knee_right, ankle_right);
data.RightKneeAngle = right_knee_angle;

% Left Shoulder Angle
fprintf('Computing Left Shoulder Angle\n');
spine_shoulder = joints.SpineShoulder;
left_shoulder_angle = calculate_angle(spine_shoulder, shoulder_left, elbow_left);
data.LeftShoulderAngle = left_shoulder_angle;

% Right Shoulder Angle
fprintf('Computing Right Shoulder Angle\n');
right_shoulder_angle = calculate_angle(spine_shoulder, shoulder_right, elbow_right);
data.RightShoulderAngle = right_shoulder_angle;

% Left Hip Angle
fprintf('Computing Left Hip Angle\n');
left_hip_angle = calculate_angle(spine_base, hip_left, knee_left);
data.LeftHipAngle = left_hip_angle;

% Right Hip Angle
fprintf('Computing Right Hip Angle\n');
right_hip_angle = calculate_angle(spine_base, hip_right, knee_right);
data.RightHipAngle = right_hip_angle;

% ----------------------------------------
% Step 6: Compute Distances Between Joints
% ----------------------------------------

fprintf('Computing distances between key joints\n');

% Left Arm Length
fprintf('Computing Left Arm Length\n');
left_arm_length = vecnorm(wrist_left - shoulder_left, 2, 2);
data.LeftArmLength = left_arm_length;

% Right Arm Length
fprintf('Computing Right Arm Length\n');
right_arm_length = vecnorm(wrist_right - shoulder_right, 2, 2);
data.RightArmLength = right_arm_length;

% Left Leg Length
fprintf('Computing Left Leg Length\n');
left_leg_length = vecnorm(ankle_left - hip_left, 2, 2);
data.LeftLegLength = left_leg_length;

% Right Leg Length
fprintf('Computing Right Leg Length\n');
right_leg_length = vecnorm(ankle_right - hip_right, 2, 2);
data.RightLegLength = right_leg_length;

% Shoulder Width
fprintf('Computing Shoulder Width\n');
shoulder_width = vecnorm(shoulder_right - shoulder_left, 2, 2);
data.ShoulderWidth = shoulder_width;

% Hip Width
fprintf('Computing Hip Width\n');
hip_width = vecnorm(hip_right - hip_left, 2, 2);
data.HipWidth = hip_width;

% ----------------------------------------
% Step 7: Compute Symmetry Features
% ----------------------------------------

fprintf('Computing symmetry features\n');

% Angle Differences
data.ElbowAngleDiff = data.LeftElbowAngle - data.RightElbowAngle;
data.KneeAngleDiff = data.LeftKneeAngle - data.RightKneeAngle;
data.ShoulderAngleDiff = data.LeftShoulderAngle - data.RightShoulderAngle;
data.HipAngleDiff = data.LeftHipAngle - data.RightHipAngle;

% Angle Ratios
data.ElbowAngleRatio = data.LeftElbowAngle ./ (data.RightElbowAngle + 1e-6);
data.KneeAngleRatio = data.LeftKneeAngle ./ (data.RightKneeAngle + 1e-6);
data.ShoulderAngleRatio = data.LeftShoulderAngle ./ (data.RightShoulderAngle + 1e-6);
data.HipAngleRatio = data.LeftHipAngle ./ (data.RightHipAngle + 1e-6);

% Distance Differences
data.ArmLengthDiff = data.LeftArmLength - data.RightArmLength;
data.LegLengthDiff = data.LeftLegLength - data.RightLegLength;

% Distance Ratios
data.ArmLengthRatio = data.LeftArmLength ./ (data.RightArmLength + 1e-6);
data.LegLengthRatio = data.LeftLegLength ./ (data.RightLegLength + 1e-6);

% ----------------------------------------
% Step 8: Estimate Center of Mass
% ----------------------------------------

fprintf('Estimating Center of Mass\n');

% Simple estimate using SpineBase, SpineMid, and SpineShoulder
com_positions = (joints.SpineBase + joints.SpineMid + joints.SpineShoulder) / 3.0;
data.CoM_X = com_positions(:, 1);
data.CoM_Y = com_positions(:, 2);
data.CoM_Z = com_positions(:, 3);

% Vertical displacement of CoM relative to SpineBase
data.CoM_Height = com_positions(:, 2) - spine_base(:, 2);

% ----------------------------------------
% Step 9: Compute Custom Exercise-Specific Features
% ----------------------------------------

fprintf('Computing custom exercise-specific features\n');

% Left Wrist Elevation relative to ShoulderLeft
fprintf('Computing Left Wrist Elevation\n');
data.LeftWristElevation = wrist_left(:, 2) - shoulder_left(:, 2);

% Right Wrist Elevation relative to ShoulderRight
fprintf('Computing Right Wrist Elevation\n');
data.RightWristElevation = wrist_right(:, 2) - shoulder_right(:, 2);

% Vertical position of SpineBase
fprintf('Computing SpineBase Vertical Position\n');
data.SpineBase_Y = spine_base(:, 2);

% Knee Angle Deviations from 90 degrees
fprintf('Computing Knee Angle Deviations from 90 degrees\n');
data.LeftKneeAngleDeviation = abs(data.LeftKneeAngle - 90);
data.RightKneeAngleDeviation = abs(data.RightKneeAngle - 90);

% Boolean flags indicating if knee angles are close to 90 degrees
data.LeftKneeAt90 = double(data.LeftKneeAngleDeviation < 10);
data.RightKneeAt90 = double(data.RightKneeAngleDeviation < 10);

% ----------------------------------------
% Step 10: Additional Features
% ----------------------------------------

fprintf('Computing additional features\n');

% Distance Between Wrists
fprintf('Computing Distance Between Wrists\n');
wrist_distance = vecnorm(wrist_left - wrist_right, 2, 2);
data.WristDistance = wrist_distance;

% Distance Between Ankles
fprintf('Computing Distance Between Ankles\n');
fprintf('Computing Distance Between Ankles\n');
ankle_distance = vecnorm(ankle_left - ankle_right, 2, 2);
data.AnkleDistance = ankle_distance;

% Left Hand Over Head Feature
fprintf('Computing Left Hand Over Head Feature\n');
head = joints.Head;
data.LeftHandOverHead = double(wrist_left(:, 2) > head(:, 2));

% Right Hand Over Head Feature
fprintf('Computing Right Hand Over Head Feature\n');
data.RightHandOverHead = double(wrist_right(:, 2) > head(:, 2));

% Left Arm Vertical Angle
fprintf('Computing Left Arm Vertical Angle\n');
vertical_vector = [0, 1, 0];  % Upward direction
left_arm_vector = wrist_left - shoulder_left;
left_arm_vector_norm = vecnorm(left_arm_vector, 2, 2);
left_arm_vector_unit = left_arm_vector ./ (left_arm_vector_norm + 1e-6);
cos_theta_left = left_arm_vector_unit * vertical_vector';
cos_theta_left = min(max(cos_theta_left, -1.0), 1.0);
left_arm_vertical_angle = acosd(cos_theta_left);
data.LeftArmVerticalAngle = left_arm_vertical_angle;

% Right Arm Vertical Angle
fprintf('Computing Right Arm Vertical Angle\n');
right_arm_vector = wrist_right - shoulder_right;
right_arm_vector_norm = vecnorm(right_arm_vector, 2, 2);
right_arm_vector_unit = right_arm_vector ./ (right_arm_vector_norm + 1e-6);
cos_theta_right = right_arm_vector_unit * vertical_vector';
cos_theta_right = min(max(cos_theta_right, -1.0), 1.0);
right_arm_vertical_angle = acosd(cos_theta_right);
data.RightArmVerticalAngle = right_arm_vertical_angle;

% Torso Inclination Angle
fprintf('Computing Torso Inclination Angle\n');
torso_vector = spine_shoulder - spine_base;
torso_vector_norm = vecnorm(torso_vector, 2, 2);
torso_vector_unit = torso_vector ./ (torso_vector_norm + 1e-6);
cos_theta_torso = torso_vector_unit * vertical_vector';
cos_theta_torso = min(max(cos_theta_torso, -1.0), 1.0);
torso_inclination_angle = acosd(cos_theta_torso);
data.TorsoInclinationAngle = torso_inclination_angle;

% Upper Leg Angle Difference
fprintf('Computing Upper Leg Angle Difference\n');
left_upper_leg_vector = knee_left - hip_left;
right_upper_leg_vector = knee_right - hip_right;
cos_theta_legs = sum(left_upper_leg_vector .* right_upper_leg_vector, 2) ./ ...
    (vecnorm(left_upper_leg_vector, 2, 2) .* vecnorm(right_upper_leg_vector, 2, 2) + 1e-6);
cos_theta_legs = min(max(cos_theta_legs, -1.0), 1.0);
leg_angle_difference = acosd(cos_theta_legs);
data.UpperLegAngleDifference = leg_angle_difference;

% ----------------------------------------
% Step 11: Compute Temporal Features
% ----------------------------------------

fprintf('Computing temporal features\n');

% List of feature columns to compute temporal differences for
original_feature_columns = {
    'LeftElbowAngle', 'RightElbowAngle', 'LeftKneeAngle', 'RightKneeAngle', ...
    'LeftShoulderAngle', 'RightShoulderAngle', 'LeftHipAngle', 'RightHipAngle', ...
    'LeftArmLength', 'RightArmLength', 'LeftLegLength', 'RightLegLength', ...
    'ShoulderWidth', 'HipWidth', ...
    'ElbowAngleDiff', 'KneeAngleDiff', 'ShoulderAngleDiff', 'HipAngleDiff', ...
    'ElbowAngleRatio', 'KneeAngleRatio', 'ShoulderAngleRatio', 'HipAngleRatio', ...
    'ArmLengthDiff', 'LegLengthDiff', 'ArmLengthRatio', 'LegLengthRatio', ...
    'CoM_X', 'CoM_Y', 'CoM_Z', 'CoM_Height', ...
    'LeftWristElevation', 'RightWristElevation', 'SpineBase_Y', ...
    'LeftKneeAngleDeviation', 'RightKneeAngleDeviation', ...
    'LeftKneeAt90', 'RightKneeAt90', ...
    'WristDistance', 'AnkleDistance', ...
    'LeftHandOverHead', 'RightHandOverHead', ...
    'LeftArmVerticalAngle', 'RightArmVerticalAngle', ...
    'TorsoInclinationAngle', 'UpperLegAngleDifference'
};

% Compute temporal differences
for i = 1:length(original_feature_columns)
    col = original_feature_columns{i};
    delta_col_name = ['Delta', col];
    data.(delta_col_name) = [0; diff(data{:, col})];  % Set first frame difference to 0
end

% ----------------------------------------
% Step 12: Prepare Final DataFrame
% ----------------------------------------

fprintf('Preparing the final data table with original and temporal features\n');

% Combine original features and temporal features
feature_columns_with_deltas = [original_feature_columns, strcat('Delta', original_feature_columns)];

% Include 'Exercise' and 'Subject' labels if they exist
label_columns = {};
if ismember('Exercise', data.Properties.VariableNames)
    label_columns{end+1} = 'Exercise';
end
if ismember('Subject', data.Properties.VariableNames)
    label_columns{end+1} = 'Subject';
end

% Create the final data table
df_final = data(:, [feature_columns_with_deltas, label_columns]);

% ----------------------------------------
% Step 13: Save to CSV
% ----------------------------------------

fprintf('Saving the final data table to CSV\n');

% Save the data table to a CSV file
writetable(df_final, 'spatial_and_temporal_features_from_joint_data.csv');

fprintf('Feature extraction completed and data saved to spatial_and_temporal_features_from_joint_data.csv\n');
