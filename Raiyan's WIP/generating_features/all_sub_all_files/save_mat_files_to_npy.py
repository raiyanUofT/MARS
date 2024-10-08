import scipy.io as sio
import numpy as np

# Load the .mat files
train_data = sio.loadmat('train_features_all_sub_all_files_wooutlier.mat')
val_data = sio.loadmat('val_features_all_sub_all_files_wooutlier.mat')
test_data = sio.loadmat('test_features_all_sub_all_files_wooutlier.mat')

# Extract the feature arrays
train_features = train_data['train_features']
val_features = val_data['val_features']
test_features = test_data['test_features']

# Save the data as .npy files
np.save('train_features.npy', train_features)
np.save('val_features.npy', val_features)
np.save('test_features.npy', test_features)

print('MAT files successfully converted to .npy format.')