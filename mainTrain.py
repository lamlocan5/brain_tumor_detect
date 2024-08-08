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

for i , image_name in enumerate(yes_tumor_images) :
    if(image_name.split('.')[1] == 'jpg') :
        image=cv2.imread(image_directory+ 'yes/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE,INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(1)

np.array(dataset)
label = np.array(label)

x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size = 0.2 )

# print(x_train, x_test, y_train, y_test)
#Reshape = (n, image_width, image_height, n_channel)

#print
# print(x_train.shape)
# print(y_train.shape)

# print(x_test.shape)
# print(y_test.shape)

print(tf.__version__)

from sklearn.preprocessing import normalize
import numpy as np
x_train = np.array(x_train, dtype=float)
x_test= np.array(x_test, dtype=float)
print(x_train.shape)

x_train_normalized=x_train/255
x_test_normalized=x_test/255

y_train=to_categorical(y_train, num_classes=2)
y_test=to_categorical(y_test, num_classes=2)


print(x_train)
print(x_train_normalized)

#Model Building
#64,64,3
model=Sequential()
model.add(Conv2D(32, (3, 3),input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3),kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))

#BinaryCrossEntryopy = 1, sigmoid
#CateCrossEntryopy = 2, softmax

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
import log
file_logging_callback  = log.FileLoggingCallback()

model.fit(x_train_normalized, y_train, batch_size=16, verbose=1, epochs=15, 
          validation_data=(x_test_normalized,y_test), shuffle=False
          ,callbacks=[file_logging_callback ])


model.save('Braintumor10EpochsCategorical.h5')




