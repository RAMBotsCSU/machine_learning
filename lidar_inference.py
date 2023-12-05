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

if __name__ == "__main__":

    model_path = 'lidar_model_quantized_edgetpu.tflite'

    interpreter = edgetpu.make_interpreter(model_path, device='usb')
    interpreter.allocate_tensors()

    print("Allocated tensors")
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("Got details")

    # Set input tensor data
    fake_lidar_data = [1000] * 360

    # Preprocess the input image
    common.set_input(interpreter, fake_lidar_data)

    print("Preprocess complete")

    # Run inference
    interpreter.invoke()

    print("Inference complete")

    # Get the output tensor
    output_tensor = common.output_tensor(interpreter)

    # Get the output values as a NumPy array
    output_values = np.array(output_tensor())

    print("Output:", output_values)



    
