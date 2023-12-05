"""
Run object detection on images, Press ESC to exit the program
For Raspberry PI, please use `import tflite_runtime.interpreter as tflite` instead
"""
import re
import cv2
import numpy as np
import time
from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import classify


#import tensorflow.lite as tflite

# import tflite_runtime.interpreter as tflite

def load_model(model_path):
    print('before here')
    #interpreter = tflite.Interpreter(model_path=model_path,experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
    interpreter = edgetpu.make_interpreter(model_path, device = 'usb')
    print('got here')
    interpreter.allocate_tensors()
    return interpreter

if __name__ == "__main__":

    model_path = 'lidar_model_quantized.tflite'

    try:
        interpreter = edgetpu.make_interpreter(model_path, device='usb')
        interpreter.allocate_tensors()
    except Exception as e:
        print(f"Error during interpreter initialization: {e}")

    # interpreter = edgetpu.make_interpreter(model_path, device='usb')

    # # Allocate tensor memory
    # interpreter.allocate_tensors()

    # interpreter = load_model(model_path)
    
    # # input_details = interpreter.get_input_details()
    # input_details = interpreter.get_input_details()
    # output_details = interpreter.get_output_details()

    # # Set input tensor data
    # fake_lidar_data = [1000] * 360

    # # Preprocess the input image
    # input_tensor = common.input_tensor(interpreter)
    # common.set_input(input_tensor, fake_lidar_data)

    # # Run inference
    # interpreter.invoke()

    # # Get the output tensor
    # output_tensor = common.output_tensor(interpreter)

    # print(output_tensor)



    