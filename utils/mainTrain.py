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
from sklearn.model_selection import StratifiedKFold
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
print(f'Dataset size: {len(dataset)}, Labels size: {len(labels)}')

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
        x = torch.sigmoid(self.fc2(x))  # Use sigmoid if using BCELoss
        return x

# K-Fold Cross Validation
k_folds = 5
kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
fold_f1_scores = []

for fold, (train_index, val_index) in enumerate(kf.split(dataset, labels)):
    print(f'Fold {fold + 1}/{k_folds}')
    
    try:
        # Debugging information
        print(f"Train indices: {train_index}")
        print(f"Validation indices: {val_index}")
        print(f"Labels length: {len(labels)}")
        print(f"Train index range: {train_index.min()} - {train_index.max()}")
        print(f"Validation index range: {val_index.min()} - {val_index.max()}")

        # Check data slice sizes
        print(f"Dataset shape before slicing: {dataset.shape}")
        print(f"Labels shape before slicing: {labels.shape}")
        print(f"x_train shape before slicing: {dataset.shape}")
        print(f"x_train shape after slicing: {dataset[train_index].shape}")
        print(f"x_val shape after slicing: {dataset[val_index].shape}")

        # Prepare training and validation data
        x_train, x_val = dataset[train_index], dataset[val_index]
        y_train, y_val = labels[train_index], labels[val_index]

        # Debugging tensor shapes
        print(f"x_train tensor shape: {x_train.shape}")
        print(f"x_val tensor shape: {x_val.shape}")
        print(f"y_train tensor shape: {y_train.shape}")
        print(f"y_val tensor shape: {y_val.shape}")
        
        # Convert to tensor
        x_train = torch.tensor(x_train, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
        x_val = torch.tensor(x_val, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
        y_train = torch.tensor(y_train, dtype=torch.float32)  # For BCELoss
        y_val = torch.tensor(y_val, dtype=torch.float32)
        
        # Prepare DataLoader
        train_data = torch.utils.data.TensorDataset(x_train, y_train)
        val_data = torch.utils.data.TensorDataset(x_val, y_val)
        
        # Debugging DataLoader
        print(f"Train data size: {len(train_data)}")
        print(f"Validation data size: {len(val_data)}")

        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=16, shuffle=False)

        # Model, loss, optimizer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = BrainTumorNet().to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, min_lr=1e-6)
        
        num_epochs = 10
        # Training
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

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
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs).squeeze()
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    predicted = (outputs > 0.5).float()
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            print(f"Validation Loss: {val_loss/len(val_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")

            # Adjust learning rate
            scheduler.step(val_loss)

        # F1 Score Calculation
        y_pred = []
        y_true = []

        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                outputs = model(inputs).squeeze()
                predicted = (outputs > 0.5).cpu().numpy().astype(int)
                y_pred.extend(predicted)
                y_true.extend(labels.numpy())

        f1 = f1_score(y_true, y_pred)
        fold_f1_scores.append(f1)
        print(f"Fold {fold+1} F1 Score: {f1:.4f}")
    
    except Exception as e:
        print(f"An error occurred during fold {fold + 1}: {e}")

# Average F1 score of all folds
mean_f1_score = np.mean(fold_f1_scores)
print(f"Mean F1 Score over {k_folds} folds: {mean_f1_score:.4f}")

# Save the model
save_path = 'E:/Project/2024 Project/BrainTumor_Lam/models/brain_tumor_pytorch_v3_kfold.pth'
torch.save(model.state_dict(), save_path)

