import os

import cv2
import mediapipe as mp


class _LandmarkListAdapter:
    """Adapta la salida de MediaPipe Tasks al formato .landmark usado por filtros."""

    def __init__(self, landmarks):
        self.landmark = landmarks

class FaceDetection:
    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_faces: int = 1,
        refine_landmarks: bool = False,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        """Inicializa el detector con API clásica (solutions) o nueva (tasks)."""
        self._use_solutions_api = hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh")

        if self._use_solutions_api:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=static_image_mode,
                max_num_faces=max_num_faces,
                refine_landmarks=refine_landmarks,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            return

        # Compatibilidad con mediapipe reciente (sin mp.solutions).
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        model_path = os.path.join(
            os.path.dirname(__file__), "assets", "face_landmarker.task"
        )
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No se encontró el modelo de Face Landmarker en: {model_path}"
            )

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=max_num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)
    
    def detect_face(self, image):
        # Cambiar de BGR a RGB -> OpenCV nos da la imagen en BGR
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self._use_solutions_api:
            # Guardamos el resultado del procesamiento del módulo FaceMesh
            results = self.face_mesh.process(rgb_image)

            # Retornamos las caras que detectamos
            return results.multi_face_landmarks

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        result = self.face_landmarker.detect(mp_image)

        if not result.face_landmarks:
            return None

        return [_LandmarkListAdapter(landmarks) for landmarks in result.face_landmarks]
    
    def get_landmark_to_coordinates(self, image, face_landmarks):
        height, width = image.shape[:2] #(120,300,canales(RGB))

        coordinates = []

        for landmark in face_landmarks:
            #Tranformación de un punto a una coordenada
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            coordinates.append((x,y))

        return coordinates

    def landmark_to_coordinate(self, image, landmark):
        height, width = image.shape[:2]
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        return x, y
