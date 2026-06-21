import cv2
from .base_filter import BaseFilter

class FiltroNariz(BaseFilter):
    def apply(self, frame, landmarks):
        if landmarks and len(landmarks)>0:
            pass