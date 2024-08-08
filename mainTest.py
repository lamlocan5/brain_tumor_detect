import cv2
from keras import load_model
from PIL import Image
import numpy as np

model=load_model('BarinTumor10Epochs.h5')

image=cv2.imread('D:\\DeepLearningProject\Brain Tumor Image Classification\\pred\pred0.jpg')

img=Image.fromarray(image)

img=img.resize((64, 64))

img=np.array(img)

print(img)

input_img=np.expand_dims(img, axis=0)

result=model.predict_classes(img)
print(result)
