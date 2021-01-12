import argparse
import tflite_runtime.interpreter as tflite
import numpy as np
from PIL import Image
import shutil
from datetime import datetime
import RPi.GPIO as GPIO
import time

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(23, GPIO.OUT)

directory = "/home/pi/tflite1/images/cam_image.jpg"
now = datetime.now()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--image',
        help='Imagen cam')
    parser.add_argument(
        '-m',
        '--model_file',
        help='.tflite model')
    parser.add_argument(
        '--input_mean',
        default=127.5, type=float,
        help='input_mean')
    parser.add_argument(
        '--input_std',
        default=127.5, type=float,
        help='input standard deviation')
    args = parser.parse_args()
    
    interpreter = tflite.Interpreter(model_path=args.model_file)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    floating_model = input_details[0]['dtype'] == np.float32
    
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    img = Image.open(args.image).resize((width, height))
    
    input_data = np.expand_dims(img, axis=0)
    
    if floating_model:
        input_data = (np.float32(input_data) - args.input_mean) / args.input_std
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    result = np.squeeze(output_data)
    scalar_result = int(round(np.asscalar(result)))
    if not (scalar_result):
        print("No se detecta mascarilla")
        shutil.copy(directory, "/home/pi/tflite1/images/infractores/{}-{}-{}_{}:{}:{}.jpg".format(now.year, now.month, now.day, now.hour, now.minute, now.second))
        GPIO.output(23, False)
        time.sleep(5)
        GPIO.cleanup()   
    else:
        print("Mascarilla detectada")
        shutil.copy(directory, "/home/pi/tflite1/images/positivos/{}-{}-{}_{}:{}:{}.jpg".format(now.year, now.month, now.day, now.hour, now.minute, now.second))
        GPIO.output(23, True)
        time.sleep(5)
        GPIO.cleanup()

    