from flask import Flask, request, jsonify
import cv2
import numpy as np
from keras.models import model_from_json, load_model
from keras.utils import load_img, img_to_array 
import base64

app = Flask(__name__)


img_emotions_json_file = open('models/img_emotions_model.json', 'r')
loaded_img_emotions_json_file = img_emotions_json_file.read()
img_emotions_json_file.close()
img_emotions_model = model_from_json(loaded_img_emotions_json_file)

img_emotions_model.load_weights('models/img_emotions_model.h5')

gender_model = load_model("models/gender_model.h5")

age_model = load_model('models/age_model.h5')

face_cascade = cv2.CascadeClassifier("haarcascade.xml")

@app.route('/')
def index():
    return "Hello World!"

@app.route('/api/emotions/image', methods=['POST'])
def image_predict_emotion():
    fileBase64 = request.json['file']

    with open("file.jpeg", "wb") as f:
        f.write(base64.b64decode(fileBase64))

    file = open("file.jpeg", "rb")

    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    emotions = ['neutral', 'happy', 'surprise', 'sadness', 'anger', 'disgust', 'fear']

    result = {}

    for (x, y, w, h) in faces:
        _gray = gray[y:y+h, x:x+w]
        emotion_img = cv2.resize(_gray, (48, 48))
        emotion_img_arr = img_to_array(emotion_img)
        emotion_img_arr = np.expand_dims(emotion_img_arr, axis=0)
        emotion_img_arr /= 255.0
        predictions = img_emotions_model.predict(emotion_img_arr)
        max_index = np.argmax(predictions[0])
        predicted_emotion = emotions[max_index]
        
        for i, emotion in enumerate(emotions):
            result[emotion] = format(predictions[0][i], ".4f")
        
        result['main_emotion'] = predicted_emotion

        gender_img = cv2.resize(_gray, (64, 64))
        gender_img_arr = img_to_array(gender_img)
        gender_img_arr = gender_img_arr.reshape((1, 64, 64, 1))
        gender_img_arr /= 255

        g, gender_pred = gender_model.predict(gender_img_arr)

        age_p = age_model(gender_img_arr)
        predicted_age_idx = np.argmax(age_p)
        if predicted_age_idx == 0:
            predicted_age = np.random.randint(1, 19)
        elif predicted_age_idx == 1:
            predicted_age = np.random.randint(19, 31)
        elif predicted_age_idx == 2:
            predicted_age = np.random.randint(31, 81)
        else:
            predicted_age = np.random.randint(81, 101)
        result["age"] = predicted_age

        print(gender_pred)
        result["gender"] = 'Female' if gender_pred > 0.5 else 'Male' 

        break

    return result

@app.route('/api/video', methods=['POST'])
def process_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400
    
    video_file = request.files['video']
    
    cap = cv2.VideoCapture(video_file)
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
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    with open('output.mp4', 'rb') as f:
        video_data = io.BytesIO(f.read())

    return send_file(video_data, attachment_filename='output.mp4', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
