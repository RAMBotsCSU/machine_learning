import tensorflow as tf
import tflite_runtime.interpreter as tflite
import cv2

interpreter = tflite.Interpreter(model_path = "detect.tflite")

