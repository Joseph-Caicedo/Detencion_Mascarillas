from datetime import datetime
import RPi.GPIO as GPIO
import time
from picamera import PiCamera
from time import sleep
import tflite_runtime.interpreter as tflite
import numpy as np
from PIL import Image
import shutil
import os

# Guardar las rutas necesarias
model_path = "/home/pi/Detencion_Mascarillas/model_1.tflite"
images_path = "/home/pi/Detencion_Mascarillas/images"
image_path = "/home/pi/Detencion_Mascarillas/images/cam_image.jpg"
image_infra_path = "/home/pi/Detencion_Mascarillas/images/infractores"
image_pos_path = "/home/pi/Detencion_Mascarillas/images/positivos"

# Crear carpetas
os.makedirs(images_path, exist_ok=True)
os.makedirs(image_infra_path, exist_ok=True)
os.makedirs(image_pos_path, exist_ok=True)

# Crear un objeto Picamera
camera = PiCamera()

# Establecer la salida de la tarjeta
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(23, GPIO.OUT)

while True:
    # Tomar captura
    print("Capturando imagen...")
    camera.resolution = (300, 300)
    camera.framerate = 30
    camera.brightness = 50
    camera.start_preview()
    sleep(5)
    camera.capture(image_path)
    camera.stop_preview()
    print("Captura tomada correctamente\n")
    print("Detectando mascarilla...")

    # Importar el modleo de tflite
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Procesamiento de imagen
    floating_model = input_details[0]['dtype'] == np.float32

    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    img = Image.open(image_path).resize((width, height))

    input_data = np.expand_dims(img, axis=0)
    
    if floating_model:
        input_data = np.array(input_data, dtype=np.float32)
    
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    # Realizar predición
    output_data = interpreter.get_tensor(output_details[0]['index'])
    result = np.squeeze(output_data)
    scalar_result = int(round(np.asscalar(result)))
    # Tomar lectura de la fecha y hora
    now = datetime.now()
    
    # Revisar condición
    if (scalar_result):
        print("No se detecta mascarilla\n")
        shutil.copy(image_path, image_infra_path + "/{}-{}-{}_{}:{}:{}.jpg".format(now.year, now.month, now.day, now.hour, now.minute, now.second))
        GPIO.output(23, False)
        time.sleep(5)
        GPIO.output(23, False)
    else:
        print("Mascarilla detectada\n")
        shutil.copy(image_path, image_pos_path + "/{}-{}-{}_{}:{}:{}.jpg".format(now.year, now.month, now.day, now.hour, now.minute, now.second))
        GPIO.output(23, True)
        time.sleep(5)
        GPIO.output(23, False)
