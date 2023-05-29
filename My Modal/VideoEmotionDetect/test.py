import cv2
import numpy as np
from keras.models import model_from_json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights('model.h5')
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

cap = cv2.VideoCapture("test.mp4")
frame_rate = cap.get(5)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output.mp4", fourcc, frame_rate, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48), interpolation = cv2.INTER_AREA)
    expanded = np.expand_dims(resized, axis = 2)
    expanded = np.expand_dims(expanded, axis = 0)
    prediction = model.predict(expanded)
    max_index = np.argmax(prediction)
    emotion = emotion_dict[max_index]
    cv2.putText(frame, emotion, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    out.write(frame)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
