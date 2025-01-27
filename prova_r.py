import cv2
import os
from ultralytics import YOLO

# Cargar el modelo YOLOv8
model = YOLO('xlarge1280.pt')

# Definir una lista de colores distinguibles en formato BGR
palette = [
    (255, 0, 0),    # Rojo
    (0, 255, 0),    # Verde
    (0, 255, 255),  # Amarillo
    (0, 0, 255),    # Azul
    (255, 255, 0),  # Cian
    (255, 0, 255),  # Magenta
    (128, 0, 128),  # Púrpura
    (128, 128, 0),  # Oliva
    (0, 128, 128),  # Teal
    (0, 0, 128)     # Azul Oscuro
]

# Función para dibujar keypoints con colores en un frame
def draw_colored_keypoints(frame, keypoints, palette):
    for i, keypoint in enumerate(keypoints):
        try:
            x, y = keypoint[:2]
            color = palette[i % len(palette)]
            cv2.circle(frame, (int(x), int(y)), 5, color, -1)
        except Exception as e:
            print(f"Error al dibujar keypoint {i}: {e}")

# Crear una carpeta para guardar los frames
output_folder = 'frames_with_keypoints'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Carpeta '{output_folder}' creada para guardar los frames.")

# Leer el video
cap = cv2.VideoCapture('SARTI_posi.mp4')
if not cap.isOpened():
    print("Error: No se puede abrir el video.")
else:
    print("El video se ha abierto correctamente.")

frame_counter = 0  # Inicializar el contador de frames

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("No se pueden leer más frames del video o se ha llegado al final.")
        break

    # Realizar la inferencia en el frame actual
    results = model(frame)

    # Flag para indicar si se deben guardar el frame
    save_frame = False

    # Procesar resultados y extraer keypoints
    for result in results:
        keypoints = result.keypoints
        if keypoints is not None and keypoints.xy is not None:
            keypoints_xy = keypoints.xy[0]  # Tomar el primer set de keypoints
            if keypoints_xy.shape[0] > 0:  # Comprobar que hay keypoints
                draw_colored_keypoints(frame, keypoints_xy, palette)
                save_frame = True

    # Guardar el frame si contiene al menos un keypoint
    if save_frame:
        frame_filename = os.path.join(output_folder, f'frame_{frame_counter:05d}.jpg')
        cv2.imwrite(frame_filename, frame)
        print(f"Frame guardado como: {frame_filename}")
        frame_counter += 1

    # Mostrar el frame con los keypoints
    cv2.imshow('Video with Keypoints', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

