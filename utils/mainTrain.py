# import cv2  
# import os
# import log
# from PIL import Image
# import tensorflow as tf
# import numpy as np
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
# from sklearn.metrics import f1_score


# INPUT_SIZE = 64
# image_directory = r'E:/Project/2024 Project/BrainTumor_Lam/datasets/'

# no_tumor_images = os.listdir(image_directory + 'sort_crop_no/')
# yes_tumor_images = os.listdir(image_directory + 'sort_crop_yes/')
# dataset = []
# label = []

# for image_name in no_tumor_images:
#     if image_name.endswith('.jpg'):
#         image = cv2.imread(image_directory + 'sort_crop_no/' + image_name)
#         image = Image.fromarray(image, 'RGB')
#         image = image.resize((INPUT_SIZE, INPUT_SIZE))
#         dataset.append(np.array(image))
#         label.append(0)

# for image_name in yes_tumor_images:
#     if image_name.endswith('.jpg'):
#         image = cv2.imread(image_directory + 'sort_crop_yes/' + image_name)
#         image = Image.fromarray(image, 'RGB')
#         image = image.resize((INPUT_SIZE, INPUT_SIZE))
#         dataset.append(np.array(image))
#         label.append(1)

# dataset = np.array(dataset)
# label = np.array(label)

# x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2)

# x_train = np.array(x_train, dtype=float)
# x_test = np.array(x_test, dtype=float)

# x_train_normalized = x_train / 255
# x_test_normalized = x_test / 255

# # No need for to_categorical since we have binary labels
# # y_train = to_categorical(y_train, num_classes=2)
# # y_test = to_categorical(y_test, num_classes=2)

# # Model Building
# model = Sequential([
#     Conv2D(32, (3, 3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3)),
#     Activation('relu'),
#     MaxPooling2D(pool_size=(2, 2)),
#     Conv2D(32, (3, 3), kernel_initializer='he_uniform'),
#     Activation('relu'),
#     MaxPooling2D(pool_size=(2, 2)),
#     Conv2D(64, (3, 3), kernel_initializer='he_uniform'),
#     Activation('relu'),
#     MaxPooling2D(pool_size=(2, 2)),
#     Flatten(),
#     Dense(64),
#     Activation('relu'),
#     Dropout(0.2),
#     Dense(1),  # Single output neuron for binary classification
#     Activation('sigmoid')  # Sigmoid activation for binary classification
# ])

# optimizer1 = tf.keras.optimizers.Adam(learning_rate=1e-3)
# model.compile(loss='binary_crossentropy', optimizer=optimizer1, metrics=['accuracy'])

# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-6)
# early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
# # Assuming the log.FileLoggingCallback() is correctly defined elsewhere in your code
# file_logging_callback = log.FileLoggingCallback()

# history = model.fit(
#     x_train_normalized, 
#     y_train, 
#     batch_size=16, 
#     epochs=10, 
#     verbose=1, 
#     validation_data=(x_test_normalized, y_test), 
#     shuffle=True,
#     callbacks=[file_logging_callback, reduce_lr, early_stopping]
# )
# save_path = 'E:/Project/2024 Project/BrainTumor_Lam/models/brain_tumor_tensorflow_v2.h5'
# # Save model
# model.save(save_path)
# # Calculate F1 Score
# y_pred = model.predict(x_test_normalized)
# y_pred_classes = (y_pred > 0.5).astype("int32").flatten()
# # y_true should not be categorical, should match the output of y_pred_classes
# f1 = f1_score(y_test, y_pred_classes)
# print(f"F1 Score: {f1}")



import os
import cv2
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Dataset preparation
INPUT_SIZE = 256
image_directory = r'E:/Project/2024 Project/BrainTumor_Lam/datasets/'

no_tumor_images = os.listdir(image_directory + 'sort_crop_no/')
yes_tumor_images = os.listdir(image_directory + 'sort_crop_yes/')
dataset = []
labels = []

for image_name in no_tumor_images:
    if image_name.endswith('.jpg'):
        image = cv2.imread(image_directory + 'sort_crop_no/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        labels.append(0)

for image_name in yes_tumor_images:
    if image_name.endswith('.jpg'):
        image = cv2.imread(image_directory + 'sort_crop_yes/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        labels.append(1)

dataset = np.array(dataset)
labels = np.array(labels)

x_train, x_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, random_state=42)

x_train = torch.tensor(x_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

x_train = x_train.permute(0, 3, 1, 2) / 255.0  # Permute dimensions for PyTorch (N, C, H, W)
x_test = x_test.permute(0, 3, 1, 2) / 255.0

train_data = torch.utils.data.TensorDataset(x_train, y_train)
test_data = torch.utils.data.TensorDataset(x_test, y_test)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

# Model definition
class BrainTumorNet(nn.Module):
    def __init__(self):
        super(BrainTumorNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * (INPUT_SIZE // 8) * (INPUT_SIZE // 8), 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.3)  # Increased dropout rate to 0.3 for regularization

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(-1, 64 * (INPUT_SIZE // 8) * (INPUT_SIZE // 8))
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x

model = BrainTumorNet()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True, min_lr=1e-6)

# Training loop
num_epochs = 30
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device, dtype=torch.float32)
        
        optimizer.zero_grad()
        
        outputs = model(inputs).squeeze()
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
            inputs, labels = inputs.to(device), labels.to(device, dtype=torch.float32)
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f"Validation Loss: {val_loss/len(test_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")
    
    # Adjust learning rate
    scheduler.step(val_loss)

save_path = 'E:/Project/2024 Project/BrainTumor_Lam/models/brain_tumor_pytorch_v1.pth'
# Save the model
torch.save(model.state_dict(), save_path)

# Calculate F1 Score
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
