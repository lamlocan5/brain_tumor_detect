import cv2  
import os
import log
from PIL import Image
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from sklearn.metrics import f1_score


INPUT_SIZE = 256
image_directory = 'datasets/'

no_tumor_images = os.listdir(image_directory + 'sort_crop_no/')
yes_tumor_images = os.listdir(image_directory + 'sort_crop_yes/')
dataset = []
label = []

for image_name in no_tumor_images:
    if image_name.endswith('.jpg'):
        image = cv2.imread(image_directory + 'sort_crop_no/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(0)

for image_name in yes_tumor_images:
    if image_name.endswith('.jpg'):
        image = cv2.imread(image_directory + 'sort_crop_yes/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(1)

dataset = np.array(dataset)
label = np.array(label)

x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2)

x_train = np.array(x_train, dtype=float)
x_test = np.array(x_test, dtype=float)

x_train_normalized = x_train / 255
x_test_normalized = x_test / 255

# No need for to_categorical since we have binary labels
# y_train = to_categorical(y_train, num_classes=2)
# y_test = to_categorical(y_test, num_classes=2)

# Model Building
model = Sequential([
    Conv2D(32, (3, 3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3)),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(32, (3, 3), kernel_initializer='he_uniform'),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), kernel_initializer='he_uniform'),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(64),
    Activation('relu'),
    Dropout(0.2),
    Dense(1),  # Single output neuron for binary classification
    Activation('sigmoid')  # Sigmoid activation for binary classification
])

optimizer1 = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(loss='binary_crossentropy', optimizer=optimizer1, metrics=['accuracy'])

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
# Assuming the log.FileLoggingCallback() is correctly defined elsewhere in your code
file_logging_callback = log.FileLoggingCallback()

history = model.fit(
    x_train_normalized, 
    y_train, 
    batch_size=16, 
    epochs=10, 
    verbose=1, 
    validation_data=(x_test_normalized, y_test), 
    shuffle=True,
    callbacks=[file_logging_callback, reduce_lr, early_stopping]
)

# Save model
model.save('Auto_Braintumor10EpochsCategorical.h5')

# Calculate F1 Score
y_pred = model.predict(x_test_normalized)
y_pred_classes = (y_pred > 0.5).astype("int32").flatten()
# y_true should not be categorical, should match the output of y_pred_classes
f1 = f1_score(y_test, y_pred_classes)
print(f"F1 Score: {f1}")
