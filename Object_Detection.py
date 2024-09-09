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

import tflite_runtime.interpreter as tflite

from PIL import Image

CAMERA_WIDTH = 320  #640 to fill whole screen, 320 for GUI component
CAMERA_HEIGHT = 480 #480 to fill whole screen, 240 for GUI component
 
#this is a test
def load_labels(label_path):
    r"""Returns a list of labels"""
    with open(label_path) as f:
        labels = {}
        for line in f.readlines():
            m = re.match(r"(\d+)\s+(\w+)", line.strip())
            labels[int(m.group(1))] = m.group(2)
        return labels


def load_model(model_path):
    r"""Load TFLite model, returns a Interpreter instance."""
    
    #interpreter = tflite.Interpreter(model_path=model_path,experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
    interpreter = edgetpu.make_interpreter(model_path, device = 'usb')
    print('got here')
    interpreter.allocate_tensors()
    return interpreter


def process_image(interpreter, image, input_index):
    r"""Process an image, Return a list of detected class ids and positions"""
    input_data = np.expand_dims(image, axis=0)  # expand to 4-dim

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

    positions = np.squeeze(interpreter.get_tensor(output_details[0]['index']))
    classes = np.squeeze(interpreter.get_tensor(output_details[1]['index']))
    scores = np.squeeze(interpreter.get_tensor(output_details[2]['index']))

    result = []

    for idx, score in enumerate(scores):
        if score > 0.5:
            result.append({'pos': positions[idx], '_id': classes[idx]})

    return result


def display_result(result, frame, labels):
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
        _id = obj['_id']

        x1 = int(pos[1] * CAMERA_WIDTH)
        x2 = int(pos[3] * CAMERA_WIDTH)
        y1 = int(pos[0] * CAMERA_HEIGHT)
        y2 = int(pos[2] * CAMERA_HEIGHT)

        cv2.putText(frame, labels[_id], (x1, y1), font, size, color, thickness)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        print(labels[_id])

    cv2.imshow('Object Detection', frame)


if __name__ == "__main__":

    #model_path = 'data/detect.tflite'
    model_path = 'data/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite'
    #model_path = 'data/ssd_mobilenet_v1_coco_quant_postprocess_edgetpu.tflite'

    label_path = 'data/coco_labels.txt'
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 24)

    interpreter = load_model(model_path)
    
    #labels = load_labels(label_path)
    labels = dataset.read_label_file(label_path)
    
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
    
    start_time = 0

    # Process Stream
    while True:
        ret, frame = cap.read()
        
        
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = image.resize((width, height))

        top_result = process_image(interpreter, image, input_index)

        end = time.time()
        display_result(top_result, frame, labels)
        fps = round(1/(end-start_time),2)
        #if(round(time.time()) % 2 == 0):
            #print('FPS: ' + str(fps))
        
        start_time = end
        
        key = cv2.waitKey(1)
        if key == 27:  # esc
            break

    cap.release()
    cv2.destroyAllWindows()
