import cv2
import numpy as np
from keras.models import model_from_json
from keras.utils import load_img, img_to_array 

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

model.load_weights('model.h5')

face_cascade = cv2.CascadeClassifier('haarcascade.xml')

img = cv2.imread('test_image2.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)

emotions = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']

for (x, y, w, h) in faces:
    roi_gray = gray[y:y+h, x:x+w]
    roi_gray = cv2.resize(roi_gray, (48, 48))
    img_pixels = img_to_array(roi_gray)
    img_pixels = np.expand_dims(img_pixels, axis=0)
    img_pixels /= 255.0
    predictions = model.predict(img_pixels)
    max_index = np.argmax(predictions[0])
    predicted_emotion = emotions[max_index]
    print("Emotions: ", end="")
    for i, emotion in enumerate(emotions):
        print(f"{emotion}: {predictions[0][i]:.2f}", end=", ")
    print()
    cv2.putText(img, predicted_emotion, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), 2)

cv2.imshow('Facial Emotion Recognition', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
