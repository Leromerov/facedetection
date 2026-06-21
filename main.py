"""Punto de entrada de la aplicación.

Este archivo conecta tres piezas:
- CameraManager: abre la cámara que elijas.
- FaceDetection: calcula y dibuja la malla facial.
- OpenCV: muestra el video en una ventana.

La idea es que aquí se vea el flujo completo de forma simple y legible.
"""

import argparse

import cv2

from camara_manager import CameraManager
from face_detector import FaceDetection


def parse_args() -> argparse.Namespace:
	"""Lee los argumentos de línea de comandos.

	Usamos un índice de cámara para poder cambiar fácilmente entre webcam,
	cámara virtual o cualquier dispositivo que OpenCV detecte.
	"""

	parser = argparse.ArgumentParser(
		description="Detecta el rostro y dibuja la malla facial sobre la cámara seleccionada."
	)
	parser.add_argument(
		"--camera-index",
		type=int,
		default=0,
		help="Índice de la cámara a abrir. Ejemplo: 0, 1, 2...",
	)
	return parser.parse_args()


def main() -> None:
	"""Arranca la cámara, procesa cada frame y muestra la malla facial."""

	args = parse_args()

	# Abrimos la cámara elegida por el usuario.
	# Si el índice no existe, CameraManager lanzará un error claro.
	camera = CameraManager(camera_index=args.camera_index)

	# Creamos el detector de FaceMesh con la configuración por defecto.
	# Aquí se genera la malla facial que luego pintaremos sobre el video.
	face_detection = FaceDetection()

	# Ventana donde se mostrará el resultado final.
	window_name = f"Face Mesh - Camara {args.camera_index}"

	try:
		while True:
			# Leemos un frame nuevo desde la cámara seleccionada.
			frame = camera.get_frame()

			# Si por algún motivo no llega imagen, no seguimos procesando.
			if frame is None:
				print("No se pudo leer un frame de la cámara.")
				break

			# Dibujamos la malla facial sobre una copia del frame original.
			mesh_frame = face_detection.draw_face(frame)

			# Mostramos la imagen procesada en una ventana.
			cv2.imshow(window_name, mesh_frame)

			# Esperamos una tecla por 1 ms.
			# Si el usuario presiona 'q', salimos del ciclo.
			key = cv2.waitKey(1) & 0xFF
			if key == ord("q"):
				break
	finally:
		# Siempre liberamos la cámara y cerramos ventanas,
		# incluso si ocurre un error o el usuario sale con 'q'.
		camera.release()
		cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
