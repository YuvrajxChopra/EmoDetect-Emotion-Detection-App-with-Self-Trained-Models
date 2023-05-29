import cv2
import numpy as np
from keras_vggface.vggface import VGGFace
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from keras.utils import to_categorical
from keras.utils.layer_utils import get_source_inputs


# Define VGGFace2 model architecture
vggface = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3))
last_layer = vggface.get_layer('avg_pool').output
x = Flatten(name='flatten')(last_layer)
x = Dense(1024, activation='relu', name='fc6')(x)
x = Dense(1024, activation='relu', name='fc7')(x)
out = Dense(7, activation='softmax', name='fc8')(x)
model = Model(vggface.input, out)

# Load pre-trained weights
model.load_weights('vggface2_weights.h5')

# Define emotion labels
emotions = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']

# Load video
cap = cv2.VideoCapture('test.mp4')

# Define variables for majority voting
window_size = 5
frame_buffer = []
emotion_counts = [0] * len(emotions)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Resize frame to 224x224 for VGGFace2 model input
    frame = cv2.resize(frame, (224, 224))

    # Predict emotions for each face in the frame
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (224, 224))
        face_img = np.expand_dims(face_img, axis=0)
        predictions = model.predict(face_img)
        max_index = np.argmax(predictions[0])
        predicted_emotion = emotions[max_index]
        emotion_counts[max_index] += 1  # Add to emotion count
        cv2.putText(frame, predicted_emotion, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)

    # Append current frame to buffer
    frame_buffer.append(emotion_counts.copy())

    # If buffer size exceeds window size, remove oldest entry
    if len(frame_buffer) > window_size:
        oldest_frame_emotions = frame_buffer.pop(0)
        for i in range(len(emotions)):
            emotion_counts[i] -= oldest_frame_emotions[i]

    # If buffer size equals window size, get majority emotion and reset counts
    if len(frame_buffer) == window_size:
        majority_index = np.argmax(emotion_counts)
        majority_emotion = emotions[majority_index]
        print("Emotion: ", majority_emotion)
        emotion_counts = [0] * len(emotions)

    cv2.imshow('Facial Emotion Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
