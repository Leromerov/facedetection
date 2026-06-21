import cv2
from camera_manager import CameraManager
from face_detector import FaceDetection


def main():
    cameras = CameraManager.list_available(5)
    if not cameras:
        print("No se detectaron cámaras activas.")
        return

    print("Cámaras detectadas:")
    print("  ", " ".join(str(cam) for cam in cameras))

    try:
        choice = input("Elige el número de la cámara (enter = primera): ").strip()
    except EOFError:
        print("Entrada no interactiva detectada. Se usará la primera cámara.")
        choice = ""

    if choice == "":
        selected = cameras[0]
    else:
        try:
            selected = int(choice)
        except ValueError:
            print(f"Entrada inválida: '{choice}'. Debes escribir un número.")
            return

    if selected not in cameras:
        print(f"Cámara {selected} no está en la lista.")
        return

    camera = CameraManager(selected)
    detector = FaceDetection()

    print(f"Cámara {selected} seleccionada y abierta.")
    print("Presiona 'q' para cerrar la vista previa.")

    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                print("No se pudo leer el frame de la cámara.")
                break

            output = detector.draw_face(frame)
            cv2.imshow("Malla facial", output)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()