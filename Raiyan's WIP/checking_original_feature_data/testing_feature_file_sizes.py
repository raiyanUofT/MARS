import numpy as np

# Load the feature maps for train, validate, and test sets
data_train = np.load('feature/featuremap_train.npy')
data_validate = np.load('feature/featuremap_validate.npy')
data_test = np.load('feature/featuremap_test.npy')

# Get the number of samples in each set (assuming the first dimension represents samples)
train_size = np.shape(data_train)[0]
validate_size = np.shape(data_validate)[0]
test_size = np.shape(data_test)[0]

# Compute total number of samples
total_size = train_size + validate_size + test_size

# Calculate the ratio for each set
train_ratio = train_size / total_size
validate_ratio = validate_size / total_size
test_ratio = test_size / total_size

# Print the shapes of the arrays
print("Train Shape:", np.shape(data_train))
print("Validate Shape:", np.shape(data_validate))
print("Test Shape:", np.shape(data_test))

# Print the ratios
print(f"Train Ratio: {train_ratio:.2f}")
print(f"Validate Ratio: {validate_ratio:.2f}")
print(f"Test Ratio: {test_ratio:.2f}")

# Output from running:
# Train Shape: (24066, 8, 8, 5)
# Validate Shape: (8033, 8, 8, 5)
# Test Shape: (7984, 8, 8, 5)
# Train Ratio: 0.60
# Validate Ratio: 0.20
# Test Ratio: 0.20