import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models.lh_vit_single_head import LHVITSingleHead
from models.lh_vit_multi_head import LHVITMultiHead

# Load datasets
X_train, y_train = torch.load('../data/processed/train_data.pt')
X_val, y_val = torch.load('../data/processed/val_data.pt')
X_test, y_test = torch.load('../data/processed/test_data.pt')

# Create DataLoaders
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)

# Initialize the model, loss function, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LHVITMultiHead(in_channels=1, num_classes=10).to(device)  # Adjust num_classes if needed

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(30):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    val_accuracy = sum((model(x.to(device)).argmax(1) == y.to(device)).sum() for x, y in val_loader) / len(y_val)
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}, Val Accuracy: {val_accuracy:.2f}")

# Test the model
model.eval()
test_accuracy = sum((model(x.to(device)).argmax(1) == y.to(device)).sum() for x, y in test_loader) / len(y_test)
print(f"Test Accuracy: {test_accuracy:.2f}")
