import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model=load_model('Braintumor10EpochsCategorical.h5')

image=cv2.imread(r'C:/Users/ADMIN/OneDrive/Máy tính/Project/2024 Project/BrainTumor/BRT350 - Copy/brain_tumor/test.jpg')

img=Image.fromarray(image)

img=img.resize((64, 64))

img=np.array(img)

print(img)

input_img=np.expand_dims(img, axis=0)

result=model.predict_classes(input_img)
print(result)
