import cv2
from typing import Tuple


def nameFingers(frame, 
                hand_landmarks, 
                language: str = "eng", 
                font_scale: float = 0.6, 
                thickness: int = 2,
                bg_color: Tuple[int, int, int] = (0, 0, 0), 
                text_color: Tuple[int, int, int] = (255, 255, 255)):
    
    h, w, _ = frame.shape
    if language == "es":
        fingers = {4: "Pulgar", 8: "Indice", 12: "Medio", 16: "Anular", 20: "Meñique"}
    else:
        fingers = {4: "Thumb", 8: "Index", 12: "Middle", 16: "Ring", 20: "Pinky"}

    for idx, name in fingers.items():
        lm = hand_landmarks.landmark[idx]
        x = int(lm.x * w)
        y = int(lm.y * h)

        (text_w, text_h), baseline = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        rect_x1 = max(x - text_w // 2 - 4, 0)
        rect_y1 = max(y - text_h - 12, 0)
        rect_x2 = min(x + text_w // 2 + 4, w - 1)
        rect_y2 = min(y - 4, h - 1)

        cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), bg_color, -1)
        cv2.putText(frame, name, (rect_x1 + 2, rect_y2 - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)

    return frame
