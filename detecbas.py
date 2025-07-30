from ultralytics import YOLO
import cv2
import pandas as pd

def cuadros_intersectan(cuadro1, cuadro2):
    x1_min, y1_min, x1_max, y1_max = cuadro1
    x2_min, y2_min, x2_max, y2_max = cuadro2

    return not (x1_max < x2_min or x1_min > x2_max or y1_max < y2_min or y1_min > y2_max)

modelo = YOLO("EPPS.pt")

with open("etiquetas.txt", "r") as f:
    etiquetas = [linea.strip() for linea in f.readlines()]
num_clases = len(modelo.model.names)
modelo.model.names = etiquetas

captura = cv2.VideoCapture(2)

# Lista para almacenar las detecciones
detecciones = []
personas = {}

while True:
    ret, fotograma = captura.read()
    if not ret:
        print("Error: No se pudo leer el fotograma de la cámara")
        break

    resultados = modelo.predict(fotograma, imgsz=416, conf=0.8) 

    # Procesar las detecciones
    for resultado in resultados:
        for det in resultado.boxes:
            clase = etiquetas[int(det.cls)]
            confianza = det.conf
            cuadro = det.xyxy.tolist()  # Asegurarse de que cuadro sea una lista

            if len(cuadro) != 4:
                continue  # Ignorar cuadros que no tengan el formato correcto

            if clase == "persona":
                id_persona = len(personas) + 1
                personas[id_persona] = {"casco": False, "mascarilla": False, "chaleco": False, "cuadro": cuadro}
            else:
                id_persona = None

            if clase in ["casco", "mascarilla", "chaleco de seguridad", "no casco", "sin mascarilla", "no chaleco seguridad"]:
                for id_p, atributos in personas.items():
                    if cuadros_intersectan(cuadro, atributos["cuadro"]):
                        id_persona = id_p
                        if clase == "casco":
                            atributos["casco"] = True
                        elif clase == "mascarilla":
                            atributos["mascarilla"] = True
                        elif clase == "chaleco de seguridad":
                            atributos["chaleco"] = True
                        elif clase == "no casco":
                            atributos["casco"] = False
                        elif clase == "sin mascarilla":
                            atributos["mascarilla"] = False
                        elif clase == "no chaleco seguridad":
                            atributos["chaleco"] = False

            if id_persona is not None:
                detecciones.append([id_persona, clase, confianza, cuadro])

    anotaciones = resultados[0].plot()
    cv2.imshow("DETECCION EPPS", anotaciones)
    if cv2.waitKey(1) == 27: 
        break

captura.release()
cv2.destroyAllWindows()

# Guardar las detecciones en un archivo CSV
df = pd.DataFrame(detecciones, columns=["ID Persona", "Clase", "Confianza", "Cuadro"])
df.to_csv("detecciones.csv", index=False)