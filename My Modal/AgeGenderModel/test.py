from keras.models import load_model
import numpy as np
import cv2
from keras.layers import *
from keras.models import *
import os
from sklearn.model_selection import train_test_split


model = load_model('model.h5')
age_model = load_model('age_model.h5')

test_image = cv2.imread('test_image.png', 0)
test_image = cv2.resize(test_image, dsize=(64, 64))
test_image = test_image.reshape((1, 64, 64, 1))
test_image = test_image / 255

age_p = age_model(test_image)
predicted_age_idx = np.argmax(age_p)
if predicted_age_idx == 0:
    # Age group 0-18
    predicted_age = np.random.randint(1, 19)
elif predicted_age_idx == 1:
    # Age group 18-30
    predicted_age = np.random.randint(18, 31)
elif predicted_age_idx == 2:
    # Age group 30-80
    predicted_age = np.random.randint(30, 81)
else:
    # Age group 80+
    predicted_age = np.random.randint(80, 101)

print('Predicted age:', predicted_age)
predicted_gender = model.predict(test_image)
print('Predicted gender:', 'male' if predicted_gender[0][0] > 0.5 else 'female')

