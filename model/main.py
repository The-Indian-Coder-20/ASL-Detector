import numpy as np
import tensorflow as tf
import cv2 as cv
from tensorflow.keras.models import load_model

cnn_model = load_model("model/asl_detector.keras")

class_names = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
               "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
               "U", "V", "W", "X", "Y", "Z", "space", "nothing"]

capture = cv.VideoCapture(0)

while True:
    ret, frame = capture.read()
    if not ret:
        break

    h, w, _ = frame.shape

    x_min, y_min = w//2 - 150, h//2 - 150
    x_max, y_max = w//2 + 150, h//2 + 150
    hand_img = frame[y_min:y_max, x_min:x_max]

    img = cv.resize(hand_img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = cnn_model.predict(img)
    label = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    cv.putText(frame, f"{label} ({confidence:.1f}%)", (10, 40),
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

    cv.imshow('ASL-Detector', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv.destroyAllWindows()
