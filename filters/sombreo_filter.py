import cv2
import numpy as np
from pathlib import Path

from .base_filter import BaseFilter


class SombreroFilter(BaseFilter):
	# Landmarks de Face Mesh para estimar frente y ancho de cabeza
	FOREHEAD_TOP = 10
	LEFT_TEMPLE = 234
	RIGHT_TEMPLE = 454

	def __init__(self, facedetector, png_path=None, offset_x=0, offset_y=0, target=260):
		self.face_detector = facedetector
		if png_path is None:
			png_path = Path(__file__).resolve().parent.parent / "assets" / "sombrero.png"

		self.hat_image = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
		if self.hat_image is None:
			raise FileNotFoundError(f"No se encontró la imagen del sombrero en {png_path}")

		# Garantiza imagen BGR para mezclar usando mascara RGB
		if self.hat_image.ndim == 2:
			self.hat_image = cv2.cvtColor(self.hat_image, cv2.COLOR_GRAY2BGR)

		self.offset_x = offset_x
		self.offset_y = offset_y
		self.target = target

	def landmark_to_xy(self, frame, landmark):
		return self.face_detector.landmark_to_coordinate(frame, landmark)

	def apply(self, frame, landmarks):
		if not landmarks:
			return frame

		face = landmarks[0].landmark
		forehead = face[self.FOREHEAD_TOP]
		left_temple = face[self.LEFT_TEMPLE]
		right_temple = face[self.RIGHT_TEMPLE]

		x_forehead, y_forehead = self.landmark_to_xy(frame, forehead)
		x_left, _ = self.landmark_to_xy(frame, left_temple)
		x_right, _ = self.landmark_to_xy(frame, right_temple)

		center_x = (x_left + x_right) // 2
		face_width = max(abs(x_right - x_left), 1)

		# El sombrero suele ser más ancho que la cara
		target_width = max(int(face_width * 1.35), self.target)

		oh, ow = self.hat_image.shape[:2]
		scale = target_width / ow
		new_w = max(int(ow * scale), 1)
		new_h = max(int(oh * scale), 1)

		resized = cv2.resize(self.hat_image, (new_w, new_h), interpolation=cv2.INTER_AREA)

		x = center_x - new_w // 2 + self.offset_x
		# Coloca la parte inferior del sombrero cerca de la frente
		y = y_forehead - int(new_h * 0.78) + self.offset_y

		self.overlay_rgba_on_rgb(frame, resized, x, y)
		return frame

	def overlay_rgba_on_rgb(self, frame, overlay, x, y):
		fh, fw = frame.shape[:2]
		oh, ow = overlay.shape[:2]

		x1 = max(0, x)
		y1 = max(0, y)
		x2 = min(fw, x + ow)
		y2 = min(fh, y + oh)

		if x1 >= x2 or y1 >= y2:
			return

		roi = frame[y1:y2, x1:x2]

		rx1 = x1 - x
		ry1 = y1 - y
		rx2 = rx1 + (x2 - x1)
		ry2 = ry1 + (y2 - y1)

		crop = overlay[ry1:ry2, rx1:rx2]
		if crop.size == 0 or crop.ndim < 3:
			return

		rgb = crop[..., :3].astype(np.float32)
		alpha = (np.max(rgb, axis=2, keepdims=True) < 245).astype(np.float32)

		blended = alpha * rgb + (1.0 - alpha) * roi
		roi[:] = blended.astype(np.uint8)


# Alias para mantener compatibilidad con el nombre antiguo del archivo/clase
SombreoFilter = SombreroFilter
