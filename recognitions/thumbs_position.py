import cv2

def detectThumbs(frame, results,mp_drawing, mp_hands):
    hand = results.multi_hand_landmarks[0]  # Solo primera mano
    lm_thumb_tip = hand.landmark[4]
    lm_thumb_base = hand.landmark[2]
    gesture_text = ""
        
    if lm_thumb_tip.y < lm_thumb_base.y:
        gesture_text = "Pulgar Arriba 👍"
    elif lm_thumb_tip.y > lm_thumb_base.y:
        gesture_text = "Pulgar Abajo 👎"

    mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    if gesture_text:
        cv2.putText(frame, gesture_text, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    return frame, gesture_text
