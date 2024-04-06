import re
import cv2
import numpy as np
import time
from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import classify
from PIL import Image

FRAME_COUNT = 10
STD_DEV_FACTOR = 2                                              # Threshold factor to determine if the object is detected or not
CENTER_X = 320

#import tensorflow.lite as tflite
import tflite_runtime.interpreter as tflite

CAMERA_WIDTH = 640  #640 to fill whole screen, 320 for GUI component
CAMERA_HEIGHT = 480 #480 to fill whole screen, 240 for GUI component
INPUT_WIDTH_AND_HEIGHT = 224

def load_model(model_path):                             # Load TFLite model, returns a Interpreter instance.
    interpreter = edgetpu.make_interpreter(model_path, device = 'usb')
    print('got here')
    interpreter.allocate_tensors()
    return interpreter

def process_image(interpreter, image, input_index):     # Process an image, Return a list of detected class ids and positions
    input_data = (np.array(image)).astype(np.uint8)
    input_data = input_data.reshape((1, 224, 224, 3))

    interpreter.set_tensor(input_index, input_data)     # Process
    interpreter.invoke()

    output_details = interpreter.get_output_details()   # Get outputs

    conf = (interpreter.get_tensor(output_details[0]['index'])/255)
    positions = (interpreter.get_tensor(output_details[1]['index']))
    print(conf)
    print(positions)
    print('\n')
    result = []

    for idx, score in enumerate(conf):
        if score > 0.99:
            result.append({'pos': positions[idx]})

    return result

def rescale_position(positions):
    for obj in positions:
        pos = obj['pos']
        scale_x = CAMERA_WIDTH / INPUT_WIDTH_AND_HEIGHT
        scale_y = CAMERA_HEIGHT / INPUT_WIDTH_AND_HEIGHT
        x1 = int(pos[0] * scale_x)
        y1 = int(pos[1] * scale_y)
        x2 = int(pos[2] * scale_x)
        y2 = int(pos[3] * scale_y)
        coords = [x1, y1, x2, y2]
    return  coords

def add_coordinates(coords):
    if len(coords) != 4:
        raise ValueError("Each row must contain exactly 4 values")
    coordinates_matrix.append(coords) 

def filter_coordinates(coordinates_matrix):
    mean_x0 = np.mean([coord[0] for coord in coordinates_matrix])
    mean_y0 = np.mean([coord[1] for coord in coordinates_matrix])
    mean_x1 = np.mean([coord[2] for coord in coordinates_matrix])
    mean_y1 = np.mean([coord[3] for coord in coordinates_matrix])

    std_x0 = np.std([coord[0] for coord in coordinates_matrix])
    std_y0 = np.std([coord[1] for coord in coordinates_matrix])
    std_x1 = np.std([coord[2] for coord in coordinates_matrix])
    std_y1 = np.std([coord[3] for coord in coordinates_matrix])

    filtered_coordinates = []
    for coord in coordinates_matrix:
        x0, y0, x1, y1 = coord
        if abs(x0 - mean_x0) < STD_DEV_FACTOR * std_x0 and abs(y0 - mean_y0) < STD_DEV_FACTOR * std_y0 \
            and abs(x1 - mean_x1) < STD_DEV_FACTOR * std_x1 and abs(y1 - mean_y1) < STD_DEV_FACTOR * std_y1:
            filtered_coordinates.append(coord)

    final_x0 = np.mean([coord[0] for coord in filtered_coordinates])
    final_y0 = np.mean([coord[1] for coord in filtered_coordinates])
    final_x1 = np.mean([coord[2] for coord in filtered_coordinates])
    final_y1 = np.mean([coord[3] for coord in filtered_coordinates])

    return final_x0, final_y0, final_x1, final_y1

def bbox_x_direction_center_point(x0, x1):
    return int((x0 + x1) / 2)

def display_result(positions, center, frame):       #Display Detected Objects
    font = cv2.FONT_HERSHEY_SIMPLEX
    size = 0.6
    color = (255, 255, 0)  # Blue color
    thickness = 2

    center_coordinates = (int(center[0]), int(center[1]))
    x0 = positions[0]
    y0 = positions[1]
    x1 = positions[2]
    y1 = positions[3]

    cv2.putText(frame, 'Tennis Ball', (x0, y0), font, size, color, thickness)
    cv2.rectangle(frame, (x0, y0), (x1, y1), color, thickness)
    cv2.circle(frame, center_coordinates, 1, (0, 0, 255), 5)

    cv2.imshow('Object Detection', frame)

if __name__ == "__main__":
    coordinates_matrix = []                            # List to store the coordinates of the detected object
    top_result = []
    model_path = 'tennisBall/BallTrackingModel_edgetpu.tflite'

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    interpreter = load_model(model_path)
    input_details = interpreter.get_input_details()     # input_details = common.input_size(interpreter)

    input_shape = input_details[0]['shape']             # Get Width and Height
    height = input_shape[1]
    width = input_shape[2]
    print(height)
    print(width)

    input_index = input_details[0]['index']             # Get input index

    while True:                                         # Process Stream
        for frame in range(FRAME_COUNT):
            ret, frame = cap.read()
            if not ret:
                print('Capture failed')
                break
            
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image = image.resize((width, height))

            result = process_image(interpreter, image, input_index)

            rescale_position(result)
            x0, y0, x1, x0 = filter_coordinates()
            top_result = [x0, y0, x1, y0]
            bbox_center_x = bbox_x_direction_center_point(x0, x1)

        display_result(top_result, frame)

        key = cv2.waitKey(1)
        if key == 27:  # esc
            break
    cap.release()
    cv2.destroyAllWindows() 