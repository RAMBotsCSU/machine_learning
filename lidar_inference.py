import numpy as np
from pycoral.utils import edgetpu
from sklearn.preprocessing import MinMaxScaler


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
    scaler_X = MinMaxScaler()
    normalized_lidar_view = scaler_X.fit_transform(fake_lidar_data)

    # Preprocess the input image
    input_tensor = interpreter.tensor(interpreter.get_input_details()[0]['index'])
    input_tensor()[0] = normalized_lidar_view

    print("Preprocess complete")

    # Run inference
    interpreter.invoke()

    print("Inference complete")

    # Get the output tensor
    output_tensor = interpreter.tensor(output_details[0]['index'])

    # Get the output values as a NumPy array
    output_values = np.array(output_tensor())

    print("Output:", output_values)



    
