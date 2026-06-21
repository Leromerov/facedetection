import cv2
import numpy as np
from .base_filter import BaseFilter
from pathlib import Path

class MustacheFilter(BaseFilter):
    NOSE_TIP = 1

    def __init__(self, facedetector, png_path=None, offset_x=0, offset_y=0, target=140):
        self.face_detector = facedetector
        if png_path is None:
            png_path = Path(__file__).resolve().parent.parent / "assets" / "Bigote.png"

        self.mustache_image = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
        if self.mustache_image is None:
            raise FileNotFoundError(f"No se encontró la imagen del bigote en {png_path}")

        self.offset_x = offset_x
        self.offset_y = offset_y
        self.target = target
    
    def landmark_to_xy(self, frame, landmark):
        return self.face_detector.landmark_to_coordinate(frame,landmark)
    
    def apply(self, frame, landmarks):
        if not landmarks:
            return frame
        
        nose = landmarks[0].landmark[self.NOSE_TIP]
        #Cambiamos de landmarks a coordenadas de pixeles
        x_nose, y_nose = self.landmark_to_xy(frame, nose)

        #Obtenemos el ancho y el alto de la imagen (solo del color)
        oh, ow = self.mustache_image.shape[:2]

        #Calculamos la escala dinamicamente para crecer el bigote segun la cara
        scale = self.target/ow

        #Cambiamos el ancho y alto a los nuevos
        new_h = max(int(oh*scale),1)
        new_w = max(int(ow*scale),1)
        
        #Reescalamos nuestro bigote
        resized = cv2.resize(self.mustache_image,((new_w,new_h)))

        #Calculamos las coodenadas donde se va a poner el bigote
        x = x_nose - new_w // 2
        y = y_nose + self.offset_y - new_h //2
        self.overlay_rgba_on_rgb(frame, resized,x,y)

        #Regresamos la imagen 
        return frame

    def overlay_rgba_on_rgb(self,frame,bigote,x,y):
        #Obtener alto y ancho del frame principal
        fh,fw = frame.shape[:2]

        #Obtener alto y ancho del bigote
        bh,bw = bigote.shape[:2]

        #Calcular limites reales del frame
        x1 = max(0,x)
        y1 = max(0,y)
        x2 = min(fw, x + bw)
        y2 = min(fh, y + bh)

        if  x1 >= x2 or y1 >= y2:
            return
        
        #Region of Interest
        roi = frame[y1:y2, x1:x2]

        #Extraer parte visible del bigote
        rx1 = x1 - x
        ry1 = y1 - y
        rx2 = rx1 + (x2-x1)
        ry2 = ry1 + (y2-y1)

        crop = bigote[ry1:ry2,rx1:rx2]

        #Separemos RGB de A
        rgb = crop[..., :3].astype(np.float32)
        roi_f = roi.astype(np.float32)

        if crop.shape[2] >= 4:
            alpha = crop[..., 3:4].astype(np.float32) / 255.0
        else:
            # Si no hay canal alpha, usa opacidad total
            alpha = np.ones((crop.shape[0], crop.shape[1], 1), dtype=np.float32)

        blended = alpha * rgb + (1.0 - alpha) * roi_f
        roi[:] = blended.astype(np.uint8)
















