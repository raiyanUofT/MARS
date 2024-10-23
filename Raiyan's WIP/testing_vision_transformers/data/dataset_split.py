import torch
from sklearn.model_selection import train_test_split

# Load the saved features and labels tensors
features = torch.load('raw/all_features.pt', weights_only=True)  # Shape: (num_frames, 1, 64, 5)
labels = torch.load('raw/all_labels.pt', weights_only=True)      # Shape: (num_frames,)

# Print loaded data shapes (for confirmation)
print(f"Loaded features shape: {features.shape}, Loaded labels shape: {labels.shape}")

# Step 1: Split into train (60%) and temp (40%)
X_train, X_temp, y_train, y_temp = train_test_split(
    features, labels, test_size=0.4, random_state=42
)

# Step 2: Split temp into validation (50% of 40%) and test (50% of 40%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# Print shapes to verify splits
print(f"Train features: {X_train.shape}, Train labels: {y_train.shape}")
print(f"Validation features: {X_val.shape}, Validation labels: {y_val.shape}")
print(f"Test features: {X_test.shape}, Test labels: {y_test.shape}")

# Save the split datasets
torch.save((X_train, y_train), 'processed/train_data.pt')
torch.save((X_val, y_val), 'processed/val_data.pt')
torch.save((X_test, y_test), 'processed/test_data.pt')

print("Dataset splitting complete and saved to train_data.pt, val_data.pt, and test_data.pt")

######################################
# Output of the above code snippet:

# Loaded features shape: torch.Size([40083, 1, 64, 5]), Loaded labels shape: torch.Size([40083])
# Train features: torch.Size([24049, 1, 64, 5]), Train labels: torch.Size([24049])
# Validation features: torch.Size([8017, 1, 64, 5]), Validation labels: torch.Size([8017])
# Test features: torch.Size([8017, 1, 64, 5]), Test labels: torch.Size([8017])
# Dataset splitting complete and saved to train_data.pt, val_data.pt, and test_data.pt

######################################