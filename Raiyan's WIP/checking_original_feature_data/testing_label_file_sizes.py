import numpy as np

# Load the label arrays for train, validate, and test sets
labels_train = np.load('feature/labels_train.npy')
labels_validate = np.load('feature/labels_validate.npy')
labels_test = np.load('feature/labels_test.npy')

# Get the number of samples in each label set (assuming the first dimension represents samples)
train_size = np.shape(labels_train)[0]
validate_size = np.shape(labels_validate)[0]
test_size = np.shape(labels_test)[0]

# Compute total number of samples
total_size = train_size + validate_size + test_size

# Calculate the ratio for each set
train_ratio = train_size / total_size
validate_ratio = validate_size / total_size
test_ratio = test_size / total_size

# Print the shapes of the arrays
print("Train Shape:", np.shape(labels_train))
print("Validate Shape:", np.shape(labels_validate))
print("Test Shape:", np.shape(labels_test))

# Calculate the ratio for each set
train_ratio = train_size / total_size
validate_ratio = validate_size / total_size
test_ratio = test_size / total_size

# Print the ratios
print(f"Train Labels Ratio: {train_ratio:.2f}")
print(f"Validate Labels Ratio: {validate_ratio:.2f}")
print(f"Test Labels Ratio: {test_ratio:.2f}")

# Output from running:
# Train Shape: (24066, 57)
# Validate Shape: (8033, 57)
# Test Shape: (7984, 57)
# Train Labels Ratio: 0.60
# Validate Labels Ratio: 0.20
# Test Labels Ratio: 0.20