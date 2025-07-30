import numpy as np
import cv2
from ultralytics import YOLO
from sort import Sort

# Cargar etiquetas desde el archivo
with open('/Users/jhos/Documents/Visual code/EPPS/etiquetas.txt', 'r') as f:
    etiquetas = [line.strip() for line in f.readlines()]

# Asignar un ID común a las etiquetas relacionadas
etiquetas_comunes = {
    5: ['casco', 'mascarilla', 'no casco', 'sin mascarilla', 'no chaleco seguridad', 'persona', 'chaleco de seguridad']
}

if __name__ == '__main__':
    cap = cv2.VideoCapture(2)

    model = YOLO("EPPS.pt")

    tracker = Sort()

    while cap.isOpened():
        status, frame = cap.read()

        if not status:
            break

        results = model(frame, stream=True)

        for res in results:
            # Filtrar detecciones con confianza > 0.5
            filtered_indices = np.where(res.boxes.conf.cpu().numpy() > 0.5)[0]
            boxes = res.boxes.xyxy.cpu().numpy()[filtered_indices].astype(int)
            classes = res.boxes.cls.cpu().numpy()[filtered_indices].astype(int)
            tracks = tracker.update(boxes)
            tracks = tracks.astype(int)
            
            for i, (xmin, ymin, xmax, ymax, track_id) in enumerate(tracks):
                class_id = classes[i]
                if class_id == 5:  # ID común para las subcategorías
                    subcategoria = etiquetas[class_id]
                    label = f"Id: {track_id} {subcategoria}"
                else:
                    label = f"Id: {track_id} {etiquetas[class_id]}"
                cv2.putText(img=frame, text=label, org=(xmin, ymin-10), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(0,255,0), thickness=2)
                cv2.rectangle(img=frame, pt1=(xmin, ymin), pt2=(xmax, ymax), color=(0, 255, 0), thickness=2)

        # frame = results[0].plot()

        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()