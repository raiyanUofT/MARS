import numpy as np
import torch
import torch.nn as nn

# Load the padded radar point cloud data
rpc_data = np.load('../../../feature/featuremap_train.npy')  # Shape: (batch_size, 64, 5)
vpc_data = np.load('../../../feature/labels_train.npy')  # Virtual data (teacher) for guidance

# Convert to PyTorch tensors
rpc_tensor = torch.tensor(rpc_data, dtype=torch.float32)
vpc_tensor = torch.tensor(vpc_data, dtype=torch.float32)

# Check if the data contains zero-padding
print(f"RPC shape: {rpc_tensor.shape}, VPC shape: {vpc_tensor.shape}")

# Reshape RPC to flatten the 8x8 grid into 64 points
rpc_tensor = rpc_tensor.view(rpc_tensor.size(0), -1, 5)  # Shape: (batch_size, 64, 5)

# Coarse Pose Estimation Network
class CoarsePoseNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CoarsePoseNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Fine Pose Refinement Network
class FinePoseNet(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(FinePoseNet, self).__init__()
        self.fc1 = nn.Linear(output_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, coarse_pose, teacher_vpc):
        out = self.fc1(coarse_pose)
        out = self.relu(out)
        refined_output = self.fc2(out + teacher_vpc)  # Refine coarse pose with VPC data
        return refined_output

# Masked loss function that ignores zero-padded points
def masked_loss(output, target, mask):
    output = output[mask]  # Apply mask to the output
    target = target[mask]  # Apply mask to the target (VPC)
    return criterion(output, target)

# Create a mask to identify non-zero points (Shape: (batch_size, 64))
mask = torch.any(rpc_tensor != 0, dim=-1)

# Hyperparameters
input_size = 64 * 5  # 64 points, each with (X, Y, Z, Doppler, Intensity)
hidden_size = 128
output_size = 57  # 19 joints, each with 3D coordinates (57 = 19 * 3)
learning_rate = 0.01
num_epochs = 1

# Instantiate models
coarse_model = CoarsePoseNet(input_size, hidden_size, output_size)
fine_model = FinePoseNet(hidden_size, output_size)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(list(coarse_model.parameters()) + list(fine_model.parameters()), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    optimizer.zero_grad()

    # Flatten the input data (Shape: (batch_size, 320))
    rpc_flattened = rpc_tensor.view(rpc_tensor.size(0), -1)

    # Coarse pose estimation
    coarse_output = coarse_model(rpc_flattened)  # Coarse estimation from RPC
    
    # Fine refinement using VPC as teacher
    fine_output = fine_model(coarse_output, vpc_tensor)

    # Compute masked losses
    coarse_loss = masked_loss(coarse_output, vpc_tensor, mask)  # Loss between coarse output and VPC
    fine_loss = masked_loss(fine_output, vpc_tensor, mask)      # Loss between refined output and VPC
    
    total_loss = coarse_loss + fine_loss
    total_loss.backward()  # Backpropagation
    optimizer.step()  # Update weights
    
    if epoch % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss.item():.4f}')
