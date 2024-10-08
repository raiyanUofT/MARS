import numpy as np

# Load the feature map array
data_train = np.load('feature/featuremap_train.npy')

# Get the first entry (first sample) in the array
first_entry = data_train[700]
random_entry = data_train[1000]

# Save each 2D slice of the first entry to a text file
with open('first_entry_values.txt', 'w') as f:
    f.write(f"Shape of first entry: {first_entry.shape}\n\n")
    for i in range(first_entry.shape[2]):
        f.write(f"Slice {i + 1} (8x8 matrix for channel {i + 1}):\n")
        np.savetxt(f, first_entry[:, :, i], fmt="%.6f")
        f.write("\n")

print("First entry values saved to 'first_entry_values.txt'.")

with open('random_entry_values.txt', 'w') as f:
    f.write(f"Shape of random entry: {random_entry.shape}\n\n")
    for i in range(random_entry.shape[2]):
        f.write(f"Slice {i + 1} (8x8 matrix for channel {i + 1}):\n")
        np.savetxt(f, random_entry[:, :, i], fmt="%.6f")
        f.write("\n")

print("Random entry values saved to 'random_entry_values.txt'.")