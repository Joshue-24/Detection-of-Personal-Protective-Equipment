# Detección de Equipos de Protección Personal (EPP)

Este proyecto implementa un sistema de detección de Equipos de Protección Personal (EPP) utilizando YOLO (You Only Look Once), un modelo de detección de objetos en tiempo real. El sistema es capaz de identificar diferentes tipos de EPP en imágenes o flujos de video.

## Características

- Detección de múltiples EPP en tiempo real
- Procesamiento de imágenes y video
- Exportación de resultados a CSV y Excel
- Análisis estadístico de las detecciones
- Interfaz de línea de comandos fácil de usar

## Requisitos

- Python 3.7 o superior
- OpenCV
- PyTorch
- Ultralytics (YOLOv8)
- scikit-learn
- openpyxl
- NumPy
- Matplotlib

## Instalación

1. Clona este repositorio:
   ```bash
   git clone [URL_DEL_REPOSITORIO]
   cd EPPS
   ```

2. Crea un entorno virtual (recomendado):
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```

3. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## Uso

### Detección en imágenes
```bash
python Detec3.py --source imagen.jpg
```

### Detección en video
```bash
python Detec3.py --source video.mp4
```

### Detección en tiempo real desde cámara web
```bash
python Detec3.py --source 0
```

### Parámetros opcionales
- `--conf`: Umbral de confianza (por defecto: 0.25)
- `--save`: Guardar resultados
- `--show`: Mostrar resultados en tiempo real

## Estructura del Proyecto

- `EPPS.pt` - Modelo YOLO pre-entrenado para la detección de EPP
- `data.yaml` - Configuración del conjunto de datos
- `etiquetas.txt` - Lista de etiquetas de EPP detectables
- `Detec3.py` - Script principal de detección
- `sort.py` - Implementación del algoritmo SORT para seguimiento de objetos
- `train/`, `valid/`, `test/` - Directorios para entrenamiento, validación y pruebas
- `detecciones_epps.csv` - Archivo de salida con los resultados de las detecciones

## Entrenamiento del Modelo

Para entrenar tu propio modelo:

1. Prepara tu conjunto de datos en formato YOLO
2. Configura el archivo `data.yaml` con las rutas de tus datos
3. Ejecuta el entrenamiento:
   ```bash
   yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=100 imgsz=640
   ```

## Ejemplos

A continuación se muestran ejemplos de la detección de EPP en acción:

### Ejemplo 1
![Detección de EPP 1](epps_1.png)

### Ejemplo 2
![Detección de EPP 2](epps_2.png)

### Ejemplo 3
![Detección de EPP 3](epps_3.png)

## Resultados

El sistema generará:
- Archivos de detección en formato CSV y Excel
- Gráficos de rendimiento
- Imágenes/videos con las detecciones marcadas

## Contribución

Las contribuciones son bienvenidas. Por favor, abre un issue primero para discutir los cambios que te gustaría hacer.

## Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## Contacto

Para consultas o soporte, por favor contacta al equipo de desarrollo.
