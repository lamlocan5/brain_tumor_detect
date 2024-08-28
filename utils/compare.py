import cv2
import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from keras.utils import normalize
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential 
from keras.layers import (
    Conv2D,
    MaxPooling2D,
    Activation,
    Dropout,
    Flatten,
    Dense
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, MaxPooling2D, Dropout
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3, MobileNet, DenseNet121
from tensorflow.keras.models import Model
import numpy as np


no_dir = os.listdir('./datasets/sort_crop_no/')
yes_dir = os.listdir('./datasets/sort_crop_yes/')

dataset,label = [],[]
for i,cur_img_dir in enumerate(no_dir):
    if cur_img_dir.split('.')[1]=='jpg':
        img = cv2.imread('./datasets/sort_crop_no/'+cur_img_dir)
        img = Image.fromarray(img,'RGB')
        img = img.resize((64,64))
        dataset.append(np.array(img))
        label.append(0)

for i,cur_img_dir in enumerate(yes_dir):
    if cur_img_dir.split('.')[1]=='jpg':
        img = cv2.imread('./datasets/sort_crop_yes/'+cur_img_dir)
        img = Image.fromarray(img,'RGB')
        img = img.resize((64,64))
        dataset.append(np.array(img))
        label.append(1)

dataset = np.array(dataset)
label = np.array(label)

x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size = 0.2 )        

x_train = np.array(x_train, dtype=float)
x_test= np.array(x_test, dtype=float)

x_train_normalized=x_train/255
x_test_normalized=x_test/255

y_train=to_categorical(y_train, num_classes=2)
y_test=to_categorical(y_test, num_classes=2)

print(x_train_normalized, x_test_normalized, y_train, y_test)


from keras.models import load_model
def create_original_model():
    model=load_model('Auto_Braintumor10EpochsCategorical.h5')
    return model


INPUT_SIZE = 64

def create_lenet5():
    model = Sequential([
        Conv2D(6, kernel_size=(5, 5), activation='sigmoid', input_shape=(INPUT_SIZE, INPUT_SIZE, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(16, kernel_size=(5, 5), activation='sigmoid'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(120, activation='sigmoid'),
        Dense(84, activation='sigmoid'),
        Dense(2, activation='softmax')
    ])
    return model

def create_alexnet():
    model = Sequential([
        Conv2D(96, kernel_size=(11, 11), strides=4, activation='relu', input_shape=(INPUT_SIZE, INPUT_SIZE, 3)),
        MaxPooling2D(pool_size=(3, 3), strides=2),
        Conv2D(256, kernel_size=(5, 5), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(3, 3), strides=2),
        Conv2D(384, kernel_size=(3, 3), padding='same', activation='relu'),
        Conv2D(384, kernel_size=(3, 3), padding='same', activation='relu'),
        Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2), strides=2),
        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.02),
        Dense(4096, activation='relu'),
        Dropout(0.02),
        Dense(2, activation='softmax')
    ])
    return model

def create_mobilenet():
    base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(INPUT_SIZE, INPUT_SIZE, 3))
    
    x = base_model.output
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(2, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model


from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def compile_and_train(model, x_train_normalized, y_train, x_test_normalized, y_test):
    model.compile(
        loss='categorical_crossentropy',  
        optimizer='adam',  
        metrics=['accuracy']  
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',  
        patience=5,  
        restore_best_weights=True  
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',  
        factor=0.1,  
        patience=3,  
        min_lr=1e-6  
    )
    

    history = model.fit(
        x_train_normalized, 
        y_train, 
        batch_size=16, 
        epochs=10,  
        validation_data=(x_test_normalized, y_test),  
        shuffle=True,  
        callbacks=[early_stopping, reduce_lr]  
    )


    loss, accuracy = model.evaluate(x_test_normalized, y_test, verbose=1)

    return accuracy, history


models = [
    ('Original', create_original_model()),
    ('LeNet-5', create_lenet5()),
    ('AlexNet', create_alexnet()),
    ('MobileNet', create_mobilenet()),
]
accuracies = []


for name, model in models:
    accuracy = compile_and_train(model, x_train_normalized, y_train, x_test_normalized, y_test)
    accuracies.append((name, accuracy))


# Sort models by accuracy in descending order
accuracies.sort(key=lambda x: x[1], reverse=True)

# Print models and their accuracies
print("list of models sorted by accuracy")
for name, accuracy in accuracies:
    print(f'Model: {name}, Accuracy: {accuracy}')

# Choose the model with the highest accuracy
best_model_name, best_accuracy = accuracies[0]
print(f'Best Model: {best_model_name}, Accuracy: {best_accuracy}')