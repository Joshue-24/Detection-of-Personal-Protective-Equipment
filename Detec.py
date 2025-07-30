from ultralytics import YOLO
import cv2

model = YOLO("EPPS.pt")

with open("etiquetas.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]
num_classes = len(model.model.names)
model.model.names = labels

cap = cv2.VideoCapture(2)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo leer el frame de la cámara")
        break

    resultados = model.predict(frame, imgsz=416, conf=0.8) 

    anotaciones = resultados[0].plot()
    cv2.imshow("DETECCION EPPS", anotaciones)

    if cv2.waitKey(1) == 27: 
        break

cap.release()
cv2.destroyAllWindows()