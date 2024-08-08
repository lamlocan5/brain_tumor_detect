import cv2  
import os
from PIL import Image
from tensorflow import keras
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from tensorflow.keras.utils import to_categorical


INPUT_SIZE = 64
image_directory = 'datasets/'



# Kiểm tra thư mục và các tệp ảnh
def check_directory(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"The directory {path} does not exist.")
    images = os.listdir(path)
    if len(images) == 0:
        raise FileNotFoundError(f"No images found in directory {path}.")
    return images

no_tumor_images=os.listdir(image_directory+ 'no/')
yes_tumor_images=os.listdir(image_directory+ 'yes/')
dataset=[]
label=[]
# print(no_tumor_images)
# print(yes_tumor_images)

for i , image_name in enumerate(no_tumor_images) :
    if(image_name.split('.')[1] == 'jpg') :
        image=cv2.imread(image_directory+ 'no/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(0)
        break

for i , image_name in enumerate(yes_tumor_images) :
    if(image_name.split('.')[1] == 'jpg') :
        image=cv2.imread(image_directory+ 'yes/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE,INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(1)
        break

np.array(dataset)
label = np.array(label)

x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size = 0.2 )

print(x_train, x_test, y_train, y_test)
print(type(x_train))
print
cv2.imshow("hung",x_train[0])
cv2.waitKey(0)
print(x_train[0].shape)
print(y_train[0])

# print(x_test.shape)
# print(y_test.shape)


