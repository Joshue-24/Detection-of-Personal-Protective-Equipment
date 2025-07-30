from ultralytics import YOLO
import cv2

# Cargar los dos modelos entrenados
modelo1 = YOLO("best.pt")
modelo2 = YOLO("EPPS.pt")

# Capturar video desde la cámara
cap = cv2.VideoCapture(2)
ret, imagen = cap.read()
cap.release()

if not ret:
    print("Error al capturar la imagen")
    exit()

# Obtener predicciones de ambos modelos
resultados1 = modelo1(imagen)
resultados2 = modelo2(imagen)

# Dibujar las detecciones en la imagen original
for result in [resultados1, resultados2]:
    for box in result[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordenadas del bounding box
        label = f"{result[0].names[int(box.cls)]}: {box.conf.item():.2f}"
        cv2.rectangle(imagen, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(imagen, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Mostrar la imagen con las detecciones combinadas
cv2.imshow("Detecciones Fusionadas", imagen)
cv2.waitKey(0)
cv2.destroyAllWindows()