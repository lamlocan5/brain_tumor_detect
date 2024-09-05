import os
import cv2
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

INPUT_SIZE = 256
image_directory = r'E:/Project/2024 Project/BrainTumor_Lam/datasets/'

no_tumor_images = os.listdir(image_directory + 'sort_crop_no/')
yes_tumor_images = os.listdir(image_directory + 'sort_crop_yes/')
no_H = os.listdir(image_directory + 'test_data_no/')
yes_H = os.listdir(image_directory + 'test_data_yes/')
dataset = []
labels = []

# Define transformations
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor()
])

test_transforms = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor()
])

# Load images and labels
import os
import cv2
import numpy as np
from PIL import Image

def load_images(image_paths, label):
    images = []
    for image_name in image_paths:
        if image_name.endswith('.jpg'):
            image_path = os.path.join(image_directory, image_name)
            image = cv2.imread(image_path)
            
            # Check if the image was loaded properly
            if image is None:
                print(f"Warning: Unable to load image at {image_path}")
                continue
            
            image = Image.fromarray(image, 'RGB')
            image = train_transforms(image) if label == 0 else test_transforms(image)
            images.append(np.array(image))
            labels.append(label)
    return images

# Debug prints
print("Loading images...")
dataset += load_images(no_tumor_images, 0)
dataset += load_images(yes_tumor_images, 1)
dataset += load_images(no_H, 0)
dataset += load_images(yes_H, 1)
print("Finished loading images.")


print(len(dataset), len(labels))

dataset = np.array(dataset)
labels = np.array(labels)

# Split data
x_train, x_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, random_state=42)

x_train = torch.tensor(x_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)  # Change to float for BCE Loss
y_test = torch.tensor(y_test, dtype=torch.float32)    # Change to float for BCE Loss

# Create DataLoader
train_data = torch.utils.data.TensorDataset(x_train, y_train)
test_data = torch.utils.data.TensorDataset(x_test, y_test)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

class BrainTumorNet(nn.Module):
    def __init__(self):
        super(BrainTumorNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * (INPUT_SIZE // 8) * (INPUT_SIZE // 8), 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.5)  # Increased dropout rate for regularization

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(-1, 64 * (INPUT_SIZE // 8) * (INPUT_SIZE // 8))
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x

model = BrainTumorNet()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Training setup
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # Using SGD with momentum
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True, min_lr=1e-6)

# Training loop
num_epochs = 30
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs).squeeze()  # Forward pass
        
        # Ensure outputs and labels have the same shape
        if outputs.dim() != labels.dim():
            outputs = outputs.view(-1)  # Flatten if needed
            labels = labels.view(-1)  # Flatten if needed
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze()
            
            # Ensure outputs and labels have the same shape
            if outputs.dim() != labels.dim():
                outputs = outputs.view(-1)  # Flatten if needed
                labels = labels.view(-1)  # Flatten if needed
            
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f"Validation Loss: {val_loss/len(test_loader):.4f}, Val_Accuracy: {100 * correct / total:.2f}%")
    
    # Adjust learning rate after validation phase
    scheduler.step(val_loss)

# Save the model
# save_path = 'E:/Project/2024 Project/BrainTumor_Lam/models/brain_tumor_pytorch_v1.pth'
# torch.save(model.state_dict(), save_path)
model.eval()
y_pred = []
y_true = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs).squeeze()
        predicted = (outputs > 0.5).cpu().numpy().astype(int)
        y_pred.extend(predicted)
        y_true.extend(labels.numpy())

f1 = f1_score(y_true, y_pred)
print(f"F1 Score: {f1}")