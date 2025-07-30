from ultralytics import YOLO
import cv2

model = YOLO("EPPS.pt")
cap = cv2.VideoCapture(2)
model.info()
print(model.info)
while True:
    ret, frame = cap.read()
    resultados = model.predict(frame, imgsz = 640, conf = 0.8)

    anotaciones = resultados[0].plot()
    cv2.imshow("DETECCION Y SEGMENTACION", anotaciones)
    if cv2.waitKey(1) == 27:
        break

        

cap.release()
cv2.destroyAllWindows()