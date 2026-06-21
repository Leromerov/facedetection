import cv2

class CameraManager:
    def __init__(self, camera_index: int = 0):

        """Inicializa la cámara por defecto usando OpenCV."""

        self.camera_index = camera_index
        
        self.capture = cv2.VideoCapture(self.camera_index)

        if not self.capture.isOpened():
            raise RuntimeError(f"No se pudo abrir la cámara con índice {self.camera_index}")

    @staticmethod
    def list_available(max_cameras: int = 5):
        """Lista índices de cámaras disponibles hasta un máximo dado."""
        available = []
        for index in range(max_cameras):
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                available.append(index)
                cap.release()
        return available

    #Obtiene el frame del objeto de OpenCV  
    def get_frame(self):
        ret, frame = self.capture.read()
        return frame

    #Liberar la camera
    def release(self):
        self.capture.release()