# RamBOTs Machine Learning Repo
                   
This is the official CSU RAMBots machine learning repository. 
Visit us at our [website](https://projects-web.engr.colostate.edu/ece-sr-design/AY22/RamBOTs).

Use Tensorflow (Lite) + OpenCV to perform basic object detection (human, sportsball, phone, etc.).
The tested model can be downloaded [here](https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip)

<img src="https://user-images.githubusercontent.com/112744753/196563382-2745e707-77d6-42d5-98a0-a29530e21c9a.png" width=50% height=50%>

Directories:
------

| Directories        | Description           |
| ------------- |-------------|
| data        |  machine learning model and labels     |
| tennisBall | Tennis ball learning model
| test_programs        | test programs used in creating object_detection.py      |


Files:
------

| Files       | Description           |
| ------------- |-------------|
| README      | this file |
| object_detection.py        | python object detection using webcam and Google Coral      |
| lidar_inference.py | Takes the LiDAR information from the LiDAR sensor and outputs the data |
| lidar_model.tflite |  |
| lidar_model_quantized.tflite | |
| lidar_model_quantized_edgetpu.tflite | |
| output.jpg | | 
| tenBall1stDraft.py | Legacy tennis ball tracking |
| tenBall2ndDrift.py | Legacy tennis ball tracking |
| tennisBallTracing.py | Tennis ball object detection |
| test_cameraNEW.py | Object Detection |
| youreWatching.py | Object Detection |



  
History:
--------
  
 **2022-10-12:**  
 <pre>Imported last year's machine learning work into this repository</pre>  

 **2023-03-01:**  
 <pre>Reorganized repo and renamed files, added some comments</pre>  

## How to Run the Program:
- Navigate to the base directory of this repository
- Ensure the USB webcam, Google Coral, and Pi fan are connected, 
- **python3 test_camera.py**
