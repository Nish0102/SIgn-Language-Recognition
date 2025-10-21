# Placeholder for Sign Language Recognition project
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
# You can replace this with your own trained model
model = tf.keras.models.load_model('asl_model.h5')

# Define class labels (modify as per your dataset)
labels = ['A', 'B', 'C', 'D', 'E']

# Initialize MediaPipe for hand detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for natural interaction
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get landmarks
            landmarks = []
            for lm in hand_landmarks.landmark:
                lmx = int(lm.x * w)
                lmy = int(lm.y * h)
                landmarks.append([lmx, lmy])

            # Convert to numpy array for prediction
            landmarks = np.array(landmarks).flatten()
            landmarks = landmarks / np.linalg.norm(landmarks)  # normalize
            
            # Pad or trim to fixed size (depending on model input)
            input_data = np.zeros((42,))
            input_data[:min(len(landmarks), 42)] = landmarks[:min(len(landmarks), 42)]
            input_data = np.expand_dims(input_data, axis=0)

            # Predict gesture
            prediction = model.predict(input_data)
            class_id = np.argmax(prediction)
            sign = labels[class_id]

            # Display result
            cv2.putText(frame, f'Sign: {sign}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                        2, (255, 0, 0), 3, cv2.LINE_AA)

    cv2.imshow("Sign Language Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

