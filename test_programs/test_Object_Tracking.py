import numpy as np

CONF_THRESHOLD = 0.9                                                # Confidence threshold  
FRAME_COUNT = 10                                                    # Number of frames to to collect by calling camera script
STD_DEV_FACTOR = 2                                              # Threshold factor to determine if the object is detected or not
CENTER_X = 320
coordinates_matrix = []                                             # List to store the coordinates of the detected object

def check_object_is_detected(conf_score, coords):                   # Check if the confidence score is above the threshold
    if conf_score > CONF_THRESHOLD:                                 # to determine if the object is detcted or not 
        add_coordinates(coords)
        return True
    else:
        return False

def add_coordinates(coords):
    if len(coords) != 4:
        raise ValueError("Each row must contain exactly 4 values")
    coordinates_matrix.append(coords)

def filter_coordinates():
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
    final_x1 = np.mean([coord[2] for coord in filtered_coordinates])

    return final_x0, final_x1
    
def bbox_x_direction_center_point(x0, x1):
    return int((x0 + x1) / 2)


if __name__ == "__main__":
    for i in range(FRAME_COUNT):
        #single_frame_conf_score, single_frame_coords = # CALL CAMERA SCRIPT
        single_frame_conf_score = 0.91 
        single_frame_coords = [100, 30, 105, 35]

        check_object_is_detected(single_frame_conf_score, single_frame_coords)
        x0, x1 = filter_coordinates()
        bbox_center_x = bbox_x_direction_center_point(x0, x1)