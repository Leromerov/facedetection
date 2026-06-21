import cv2
from camera_manager import CameraManager
from face_detector import FaceDetection
from filters.filtro_nariz import FiltroNariz
from filters.lentes_filter import LentesFilter
from filters.mostache_filter import MustacheFilter
from filters.sombreo_filter import SombreroFilter
from filters.sombrero3d_filter import Sombrero3DFilter


def main():
    cameras = CameraManager.list_available(5)
    if not cameras:
        print("No se detectaron cámaras activas.")
        return

    print("Cámaras detectadas:")
    print("  ", " ".join(str(cam) for cam in cameras))

    if len(cameras) == 1:
        selected = cameras[0]
        print(f"Solo hay una cámara disponible. Se usará la cámara {selected}.")
    else:
        while True:
            try:
                choice = input("Elige el número de la cámara: ").strip()
            except EOFError:
                print("Entrada no interactiva detectada. No se pudo elegir cámara.")
                return
            except KeyboardInterrupt:
                print("\nSelección cancelada por el usuario.")
                return

            if choice == "":
                print("Debes elegir una cámara válida.")
                continue

            try:
                selected = int(choice)
            except ValueError:
                print(f"Entrada inválida: '{choice}'. Debes escribir un número.")
                continue

            if selected not in cameras:
                print(f"Cámara {selected} no está en la lista. Elige una de: {' '.join(str(cam) for cam in cameras)}")
                continue

            break

    camera = CameraManager(selected)
    detector = FaceDetection()

    print("\nFiltros disponibles:")
    print("  1) Nariz")
    print("  2) Bigote")
    print("  3) Lentes")
    print("  4) Sombrero")
    print("  5) Sombrero3D")
    while True:
        try:
            filter_choice = input("Elige el filtro (1/2/3/4/5): ").strip()
        except EOFError:
            print("Entrada no interactiva detectada. No se pudo elegir filtro.")
            camera.release()
            return
        

        if filter_choice in ("1", "2", "3", "4", "5"):
            break

        print("Opción inválida. Debes elegir 1, 2, 3 , 4 o 5.")

    if filter_choice == "1":
        active_filter = FiltroNariz(detector)
        filter_name = "Nariz"
    elif filter_choice == "2":
        active_filter = MustacheFilter(detector)
        filter_name = "Bigote"
    elif filter_choice == "3":
        active_filter = LentesFilter(detector)
        filter_name = "Lentes"
    elif filter_choice == "4":
        active_filter = SombreroFilter(detector)
        filter_name = "Sombrero"
    elif filter_choice == "5":
        active_filter = Sombrero3DFilter(detector)
        filter_name = "Sombrero3D"

    print(f"Cámara {selected} seleccionada y abierta.")
    print(f"Filtro activo: {filter_name}.")
    print("Presiona 'q' para cerrar la vista previa.")

    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                print("No se pudo leer el frame de la cámara.")
                break

            faces = detector.detect_face(frame)
            output = active_filter.apply(frame, faces)
            cv2.imshow("Malla facial", output)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        print("\nInterrupción detectada (Ctrl+C). Cerrando la cámara...")
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()