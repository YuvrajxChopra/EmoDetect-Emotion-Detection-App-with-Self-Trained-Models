import numpy as np
import cv2
from keras.models import load_model

model = load_model('model.h5')
age_model = Model(inputs=model.input, outputs=model.layers[9].output)  # extract age model from the full model

test_image = cv2.imread('test_image.png', 0)
test_image = cv2.resize(test_image, dsize=(64, 64))
test_image = test_image.reshape((1, 64, 64, 1))
test_image = test_image / 255

age_p = age_model.predict(test_image)
predicted_age_idx = np.argmax(age_p)
predicted_age = int((predicted_age_idx * 116 / 7) + 1)  # calculate predicted age from index
predicted_gender = model.predict(test_image)[1][0]

print('Predicted age:', predicted_age)
print('Predicted gender:', 'male' if predicted_gender > 0.5 else 'female')