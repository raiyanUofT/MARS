import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models.lh_vit_single_head import LHVITSingleHead
from models.lh_vit_multi_head import LHVITMultiHead

# Load pre-split data
X_train, y_train = torch.load('../data/processed/train_data.pt')
X_val, y_val = torch.load('../data/processed/val_data.pt')

# Use a small subset of data to quickly validate the pipeline
X_train = X_train[:128]  # Take the first 128 samples for training
y_train = y_train[:128]
X_val = X_val[:32]  # Take 32 samples for validation
y_val = y_val[:32]

# Create DataLoaders with small batch sizes
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=16)

# Initialize the model, loss function, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LHVITMultiHead(in_channels=1, num_classes=10).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Quick training loop (2 epochs to verify)
print("Starting quick training...")
for epoch in range(2):  # Run for 2 epochs
    model.train()
    total_loss = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)
        
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")

# Quick validation check
model.eval()
with torch.no_grad():
    val_correct = 0
    val_total = 0
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        val_correct += (predicted == labels).sum().item()
        val_total += labels.size(0)

    val_accuracy = val_correct / val_total
    print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

print("Quick training complete.")

######################################
# Output of the above code snippet:

# Starting quick training...
# Epoch 1, Loss: 6.7575
# Epoch 2, Loss: 5.6634
# Validation Accuracy: 3.12%
# Quick training complete.

######################################
