import re
import cv2
import numpy as np
import time
import math
from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import classify


#import tensorflow.lite as tflite

import tflite_runtime.interpreter as tflite

from PIL import Image

CAMERA_WIDTH = 640  #640 to fill whole screen, 320 for GUI component
CAMERA_HEIGHT = 480 #480 to fill whole screen, 240 for GUI component
INPUT_WIDTH_AND_HEIGHT = 224

def load_model(model_path):
    r"""Load TFLite model, returns a Interpreter instance."""
    
    interpreter = edgetpu.make_interpreter(model_path, device = 'usb')
    print('got here')
    interpreter.allocate_tensors()
    return interpreter

def process_image(interpreter, image, input_index):
    r"""Process an image, Return a list of detected class ids and positions"""
    input_data = (np.array(image)).astype(np.uint8)
    input_data = input_data.reshape((1, 224, 224, 3))

    # Process
    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()

    # Get outputs
    output_details = interpreter.get_output_details()
    
    
    #print(output_details)
    #output_details[0] - position
    # output_details[1] - class id
    # output_details[2] - score
    # output_details[3] - count

    process_image.prevAreaPos = getattr(process_image, "prevAreaPos", 0)

    positions = (interpreter.get_tensor(output_details[0]['index']))
    conf = (interpreter.get_tensor(output_details[1]['index'])/255)
    result = []

    for idx, score in enumerate(conf):
        pos = positions[0]
        areaPos = area(pos)
        if score > 0.99 and  (350 <= areaPos < 50176) and process_image.prevAreaPos > 400:
            result.append({'pos': positions[idx]})
            print("Area: ", areaPos)
        process_image.prevAreaPos = areaPos  # Update prevAreaPos for the next iteration


    return result


def distance(point1, point2):
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def area(pos):
    side_length = distance((pos[0], pos[1]), (pos[2], pos[3]))
    return side_length ** 2

def display_result(result, frame, start_time):
    r"""Display Detected Objects"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    size = 0.6
    color = (255, 255, 0)  # Blue color
    thickness = 2

    # position = [ymin, xmin, ymax, xmax]
    # x * CAMERA_WIDTH
    # y * CAMERA_HEIGHT
    for obj in result:
        pos = obj['pos']
        scale_x = CAMERA_WIDTH / INPUT_WIDTH_AND_HEIGHT
        scale_y = CAMERA_HEIGHT / INPUT_WIDTH_AND_HEIGHT
        x1 = int(pos[0] * scale_x)
        y1 = int(pos[1] * scale_y)
        x2 = int(pos[2] * scale_x)
        y2 = int(pos[3] * scale_y)

        print(x1, y1)

        cv2.putText(frame, 'Tennis Ball', (x1, y1), font, size, color, thickness)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        center = bboxCenterPoint(x1, y1, x2, y2)
        calculate_direction(center[0], start_time)

    cv2.imshow('Object Detection', frame)

def bboxCenterPoint(x1, y1, x2, y2):
    bbox_center_x = int((x1 + x2) / 2)
    bbox_center_y = int((y1 + y2) / 2)

    return [bbox_center_x, bbox_center_y]

def calculate_direction(X, start, frame_width=CAMERA_WIDTH):
    increment = frame_width / 3
    if ((2*increment) <= X <= frame_width):
        end = time.perf_counter()
        #print("Turn Right! Latency: %s seconds", (end - start))
    elif (0 <= X < increment):
        end = time.perf_counter()
        #print("Turn Lefft! Latency: %s seconds", (end - start))
    elif (increment <= X < (2*increment)):
        end = time.perf_counter()
        #print("Centered! Latency: %s seconds", (end - start))


if __name__ == "__main__":

    model_path = 'tennisBall/BallTrackingModelQuant_edgetpu.tflite'

    # label_path = 'data/coco_labels.txt'
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    interpreter = load_model(model_path)
    
    #labels = load_labels(label_path)
    #labels = dataset.read_label_file(label_path)
    
    #input_details = common.input_size(interpreter)
    input_details = interpreter.get_input_details()

    # Get Width and Height
    input_shape = input_details[0]['shape']
    height = input_shape[1]
    width = input_shape[2]
    print(height)
    print(width)

    # Get input index
    input_index = input_details[0]['index']

    # Process Stream
    while True:
        start_time = time.perf_counter()

        ret, frame = cap.read()
        
        if not ret:
            print('Capture failed')
            break
        
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = image.resize((width, height))

        top_result = process_image(interpreter, image, input_index)

        display_result(top_result, frame, start_time)
        
        key = cv2.waitKey(1)
        if key == 27:  # esc
            break

    cap.release()
    cv2.destroyAllWindows()
