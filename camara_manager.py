import cv2

class CameraManager:
    def __init__(self, camera_index: int = 0):

        """Abre la cámara seleccionada usando OpenCV.

        El índice recibe el número del dispositivo que quieres usar:
        0 suele ser la cámara principal, 1 una secundaria, y así sucesivamente.
        """

        self.camera_index = camera_index

        # VideoCapture es el objeto de OpenCV que administra la cámara real.
        self.capture = cv2.VideoCapture(self.camera_index)

        if not self.capture.isOpened():
            raise RuntimeError(f"No se pudo abrir la cámara con índice {self.camera_index}")

    # Obtiene el frame actual de la cámara.
    # Si la lectura falla, devolvemos None para que el flujo principal
    # pueda salir de forma ordenada.
    def get_frame(self):
        ret, frame = self.capture.read()
        if not ret:
            return None
        return frame

    # Libera la cámara cuando ya no se va a usar.
    def release(self):
        self.capture.release()