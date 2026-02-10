import cv2
import mediapipe as mp
import time
import math
from recognitions.finger_names import nameFingers
import joblib

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
)
model = joblib.load("thumb_model.pkl")
cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    gesture_text = "None"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )
        
        data = []
        for lm in hand_landmarks.landmark:
            data.extend([lm.x, lm.y, lm.z])

        gesture_text = model.predict([data])[0]

        frame = nameFingers(frame, hand_landmarks)

        if gesture_text:
            cv2.putText(frame, gesture_text, 
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1.2, 
                        (0, 255, 0), 
                        3)
           
    cv2.imshow("Hands", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()
