from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import shutil
import os
import io
import numpy as np
import RPi.GPIO as GPIO
import time
import picamera

from PIL import Image
from tflite_runtime.interpreter import Interpreter

def set_input_tensor(interpreter, image):
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image
  
def classify_image(interpreter, image):
    set_input_tensor(interpreter, image)
    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    output = np.squeeze(interpreter.get_tensor(output_details['index']))

    if output_details['dtype'] == np.uint8:
        scale, zero_point = output_details['quantization']
        output = scale * (output - zero_point)
    scalar_output = int(round(np.asscalar(output)))
    
    if scalar_output:
        result = 'No se detecta mascarilla\n'
    else:
        result = 'Mascarilla detectada\n'
    return result

def main():

    model_path = '/home/pi/Detencion_Mascarillas/model_1.tflite'    
    interpreter = Interpreter(model_path)
    interpreter.allocate_tensors()
    _, height, width, _ = interpreter.get_input_details()[0]['shape']

    with picamera.PiCamera(resolution=(640, 480), framerate=30) as camera:
        camera.start_preview()
        try:
            stream = io.BytesIO()
            for _ in camera.capture_continuous(stream, format='jpeg', use_video_port=True):
                stream.seek(0)
                image = Image.open(stream).convert('RGB').resize((width, height), Image.ANTIALIAS)
                start_time = time.time()
                result = classify_image(interpreter, image)
                elapsed_ms = (time.time() - start_time) * 1000
                stream.seek(0)
                stream.truncate()
                print('%s%.1fms' % (result, elapsed_ms))
                camera.annotate_text = '%s \n%.1fms' % (result, elapsed_ms)
        finally:
            camera.stop_preview()

if __name__ == '__main__':
  main()