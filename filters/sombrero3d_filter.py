import cv2
import numpy as np
from pathlib import Path

from .base_filter import BaseFilter


class Sombrero3DFilter(BaseFilter):
	# Landmarks principales para orientar y escalar el sombrero 3D
	FOREHEAD_TOP = 10
	CHIN = 152
	LEFT_TEMPLE = 234
	RIGHT_TEMPLE = 454
	LEFT_EYE_OUTER = 33
	RIGHT_EYE_OUTER = 263

	def __init__(self, facedetector, png_path=None, offset_x=0, offset_y=0, min_target=240):
		self.face_detector = facedetector
		if png_path is None:
			png_path = Path(__file__).resolve().parent.parent / "assets" / "sombrero2.png"

		self.hat_image = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
		if self.hat_image is None:
			raise FileNotFoundError(f"No se encontro la imagen del sombrero 3D en {png_path}")

		if self.hat_image.ndim == 2:
			self.hat_image = cv2.cvtColor(self.hat_image, cv2.COLOR_GRAY2BGRA)
		elif self.hat_image.shape[2] == 3:
			alpha = np.full(self.hat_image.shape[:2] + (1,), 255, dtype=np.uint8)
			self.hat_image = np.concatenate([self.hat_image, alpha], axis=2)

		self.offset_x = offset_x
		self.offset_y = offset_y
		self.min_target = min_target

	def landmark_to_xy(self, frame, landmark):
		return self.face_detector.landmark_to_coordinate(frame, landmark)

	def apply(self, frame, landmarks):
		if not landmarks:
			return frame

		face = landmarks[0].landmark
		forehead = face[self.FOREHEAD_TOP]
		chin = face[self.CHIN]
		left_temple = face[self.LEFT_TEMPLE]
		right_temple = face[self.RIGHT_TEMPLE]
		left_eye = face[self.LEFT_EYE_OUTER]
		right_eye = face[self.RIGHT_EYE_OUTER]

		x_forehead, y_forehead = self.landmark_to_xy(frame, forehead)
		x_chin, y_chin = self.landmark_to_xy(frame, chin)
		x_left, y_left = self.landmark_to_xy(frame, left_temple)
		x_right, y_right = self.landmark_to_xy(frame, right_temple)
		x_eye_l, y_eye_l = self.landmark_to_xy(frame, left_eye)
		x_eye_r, y_eye_r = self.landmark_to_xy(frame, right_eye)

		face_width = max(abs(x_right - x_left), 1)
		face_height = max(abs(y_chin - y_forehead), 1)

		center_x = (x_left + x_right) // 2
		roll_angle = np.degrees(np.arctan2((y_eye_r - y_eye_l), (x_eye_r - x_eye_l + 1e-6)))

		# Simula profundidad 3D: con cabeza mas frontal se usa mas altura.
		ratio_h_w = np.clip(face_height / face_width, 0.8, 1.5)
		depth_scale = np.clip((ratio_h_w - 0.8) / 0.7, 0.0, 1.0)

		target_width = max(int(face_width * (1.45 + 0.2 * depth_scale)), self.min_target)

		oh, ow = self.hat_image.shape[:2]
		scale = target_width / max(ow, 1)
		new_w = max(int(ow * scale), 1)
		new_h = max(int(oh * scale * (0.9 + 0.3 * depth_scale)), 1)

		resized = cv2.resize(self.hat_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
		rotated = self.rotate_rgba(resized, roll_angle * 0.85)

		rh, rw = rotated.shape[:2]

		x = int(center_x - rw // 2 + self.offset_x)
		# Ancla la base del sombrero a la frente para que quede arriba de la cabeza.
		y = int(y_forehead - rh * (0.80 + 0.06 * depth_scale) + self.offset_y)

		self.overlay_rgba(frame, rotated, x, y)
		return frame

	def rotate_rgba(self, image, angle):
		h, w = image.shape[:2]
		center = (w / 2.0, h / 2.0)

		matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
		abs_cos = abs(matrix[0, 0])
		abs_sin = abs(matrix[0, 1])

		new_w = int(h * abs_sin + w * abs_cos)
		new_h = int(h * abs_cos + w * abs_sin)

		matrix[0, 2] += new_w / 2 - center[0]
		matrix[1, 2] += new_h / 2 - center[1]

		return cv2.warpAffine(
			image,
			matrix,
			(new_w, new_h),
			flags=cv2.INTER_LINEAR,
			borderMode=cv2.BORDER_CONSTANT,
			borderValue=(0, 0, 0, 0),
		)

	def overlay_rgba(self, frame, overlay, x, y):
		fh, fw = frame.shape[:2]
		oh, ow = overlay.shape[:2]

		x1 = max(0, x)
		y1 = max(0, y)
		x2 = min(fw, x + ow)
		y2 = min(fh, y + oh)

		if x1 >= x2 or y1 >= y2:
			return

		rx1 = x1 - x
		ry1 = y1 - y
		rx2 = rx1 + (x2 - x1)
		ry2 = ry1 + (y2 - y1)

		crop = overlay[ry1:ry2, rx1:rx2]
		if crop.size == 0:
			return

		rgb = crop[..., :3].astype(np.float32)
		alpha = (crop[..., 3:4].astype(np.float32) / 255.0)

		roi = frame[y1:y2, x1:x2].astype(np.float32)
		blended = alpha * rgb + (1.0 - alpha) * roi
		frame[y1:y2, x1:x2] = blended.astype(np.uint8)


# Alias para compatibilidad por si se usa otro estilo de nombre.
Sombrero3dFilter = Sombrero3DFilter
