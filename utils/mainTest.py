# import cv2
# from keras.models import load_model
# from PIL import Image
# import numpy as np

# model_path = r'E:\Project\2024 Project\BrainTumor_Lam\models\brain_tumor_tensorflow_v2.h5'
# image_path = r'E:\Project\2024 Project\BrainTumor_Lam\datasets\sort_crop_no\no1.jpg'

# model = load_model(model_path)
# image = cv2.imread(image_path)

# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# img = Image.fromarray(image_rgb)
# img = img.resize((64, 64))
# img = np.array(img)
# img = img.astype('float32') / 255.0
# # Expand dimensions to match the input of the model
# input_img = np.expand_dims(img, axis=0)

# result = model.predict(input_img)

# # Since it's binary classification, apply a threshold (e.g., 0.5) to determine the class
# predicted_class = (result > 0.5).astype("int32")

# print(f"Prediction: {predicted_class[0][0]}")

import cv2
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# Define your model architecture (should match the one used during training)
class BrainTumorNet(nn.Module):
    def __init__(self):
        super(BrainTumorNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * (256 // 8) * (256 // 8), 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(-1, 64 * (256 // 8) * (256 // 8))
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x

# Load the model
model_path = r'E:\Project\2024 Project\BrainTumor_Lam\models\brain_tumor_pytorch_v1.pth'
model = BrainTumorNet()
model.load_state_dict(torch.load(model_path))
model.eval()  # Set the model to evaluation mode

# Image preprocessing
image_path = r'E:\Project\2024 Project\BrainTumor_Lam\datasets\sort_crop_no\no1.jpg'

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize image to match model input size
    transforms.ToTensor(),        # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with mean and std used during training
])

# Load and preprocess the image
image = Image.open(image_path).convert('RGB')  # Ensure image is in RGB format
input_img = transform(image).unsqueeze(0)  # Add batch dimension

# Perform inference
with torch.no_grad():
    outputs = model(input_img)
    prediction = (outputs > 0.5).float().item()  # Apply threshold to get class

print(f"Prediction: {int(prediction)}")
