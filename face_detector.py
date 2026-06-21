#libreria: mediapipe
#--solutions
#----hands
#----pose
#----face_mesh
#----------Facemesh
#----------------------process
#----------Facemesh_Tesselation
#----------Facemesh_Contours

import cv2 
import mediapipe as mp

class FaceDetection:
    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_faces: int = 1,
        refine_landmarks: bool = False,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        """Inicializa el detector de FaceMesh con los parámetros de MediaPipe."""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.drawing_utils = mp.solutions.drawing_utils
    
    def detect_face(self,image):
        # Cambia de BGR a RGB -> OpenCV nos da la imagen en BGR
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Guarda el resultado del procesamiento del módulo FaceMesh
        results = self.face_mesh.process(rgb_image)

        # Retorna las caras que detectamos
        return results.multi_face_landmarks

    def draw_face(self,image):
        # Revisa cuántas caras tenemos
        faces = self.detect_face(image)
        # la imagen original no se afecta
        out_img = image.copy()

        if faces is None:
            return out_img

        # Recorremos las caras identificadas
        for face in faces:
            self.drawing_utils.draw_landmarks(
                image=out_img,
                landmark_list=face,
                connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)
            )
        return out_img
    
    def get_landmark_to_coordinates(self, image, face_landmarks):
        height, width = image.shape[:2] #(120,300,canales(RGB))

        coordinates = []

        for landmark in face_landmarks:
            #Tranformación de un punto a una coordenada
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            coordinates.append((x,y))

        return coordinates

