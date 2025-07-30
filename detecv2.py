from ultralytics import YOLO
import cv2
import csv
import time
from datetime import datetime
import numpy as np
from sort import Sort
import torch
from ultralytics.nn.tasks import DetectionModel
from torch.nn.modules.container import Sequential
from ultralytics.nn.modules import Conv

# Añade las clases a la lista de globals seguros
torch.serialization.add_safe_globals([DetectionModel, Sequential, Conv])

model = YOLO("EPPS.pt")

with open("etiquetas.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]
num_classes = len(model.model.names)
model.model.names = labels

cap = cv2.VideoCapture(1)

# Inicializa el rastreador SORT
tracker = Sort()

# Abre el archivo CSV para escritura
with open("detecciones.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    # Escribe la cabecera del CSV
    writer.writerow(["ID", "Clase", "Confianza", "Coordenadas", "Hora"])

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo leer el frame de la cámara")
            break

        resultados = model.predict(frame, imgsz=416, conf=0.5) 

        anotaciones = resultados[0].plot()
        cv2.imshow("DETECCION EPPS", anotaciones)

        detections = []
        for result in resultados:
            for det in result.boxes:
                x1, y1, x2, y2 = det.xyxy
                confianza = det.conf
                detections.append([x1, y1, x2, y2, confianza])

        # Actualiza el rastreador con las detecciones actuales
        tracks = tracker.update(np.array(detections))

        # Guarda los resultados en el archivo CSV cada 5 segundos
        if time.time() - start_time >= 5:  # 5 segundos
            for track in tracks:
                track_id = int(track[4])
                x1, y1, x2, y2 = track[:4]
                clase = "persona"  # Asumiendo que solo se detectan personas
                confianza = track[5]
                coordenadas = [x1, y1, x2, y2]
                hora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                writer.writerow([track_id, clase, confianza, coordenadas, hora])
            start_time = time.time()  # Reinicia el temporizador

        if cv2.waitKey(1) == 27: 
            break

cap.release()
cv2.destroyAllWindows()