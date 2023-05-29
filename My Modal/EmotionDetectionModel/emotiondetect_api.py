from flask import Flask, request, jsonify
import cv2
import numpy as np
from keras.models import model_from_json
from keras.utils import load_img, img_to_array 

app = Flask(__name__)

@app.route('/api/emotions', methods=['POST'])
def predict_emotion():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    model.load_weights('model.h5')

    face_cascade = cv2.CascadeClassifier('haarcascade.xml')

    file = request.files['image']
    img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    emotions = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']

    result = {}

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        img_pixels = img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255.0
        predictions = model.predict(img_pixels)
        max_index = np.argmax(predictions[0])
        predicted_emotion = emotions[max_index]
        
        for i, emotion in enumerate(emotions):
            result[emotion] = float(predictions[0][i])
        
        result['main_emotion'] = predicted_emotion

        break

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
