import cv2
from .base_filter import BaseFilter

class FiltroNariz(BaseFilter):
    def __init__(self, facedetector):
        self.face_detector = facedetector

    def apply(self, frame, landmarks):
        if not landmarks:
            return frame

        face_landmark = landmarks[0].landmark
        nose_landmark = face_landmark[1]
        x, y = self.face_detector.landmark_to_coordinate(frame, nose_landmark)
        cv2.circle(frame, (x, y), 8, (0, 0, 250), -1)
        return frame