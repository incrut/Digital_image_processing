import cv2 
import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise


# set up the blob detector
detector_params = cv2.SimpleBlobDetector_Params()

detector_params.filterByColor = True
# detector_params.blobColor = 190
detector_params.minThreshold = 0
detector_params.maxThreshold = 250
detector_params.filterByCircularity = True
detector_params.minCircularity = 0.5
detector_params.maxCircularity = 1.0

detector_params.filterByConvexity = True
detector_params.minConvexity = 0.8
detector_params.maxConvexity = 1.0

detector_params.filterByInertia = True
detector_params.minInertiaRatio = 0.5
detector_params.maxInertiaRatio = 1.0


detector_params.filterByArea = True
detector_params.minArea = 500
detector_params.maxArea = 4000

detector = cv2.SimpleBlobDetector_create(detector_params)

# open the video file
vid = cv2.VideoCapture("cell.mp4")
if (vid.isOpened()== False): 
  print("Error opening video file")


# frame parameters
fps = vid.get(cv2.CAP_PROP_FPS)
frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
new_width = int(frame_width/1.5)
new_height = int(frame_height/1.5)

# setting the video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
frame_size = (frame_width, frame_height)
out = cv2.VideoWriter('DetectedCells.mp4', fourcc, fps, frame_size, True)   

cells = []
cell_id = 0  # initialize the first cell ID

# font and text color
font = cv2.FONT_HERSHEY_SIMPLEX
color = (0, 255, 0)  # green

# Kalman filter setup
dt = 1.0/fps  # time step
F = np.array([[1, dt, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, dt],
              [0, 0, 0, 1]])  # state transition matrix
H = np.array([[1, 0, 0, 0],
              [0, 0, 1, 0]])  # measurement matrix
Q = Q_discrete_white_noise(dim=4, dt=dt, var=0.01)  # process noise covariance
R = np.array([[10.0, 0],
              [0, 10.0]])  # measurement noise covariance
kalman_filters = []

while (vid.isOpened()):
    # Capture frame-by-frame
    ret, frame = vid.read()
    if ret:
    
        #resize the frame
        resized_frame = cv2.resize(frame, (new_width, new_height)) 

        # Detect the blood cell blobs
        gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

        # Apply histogram equalization to increase contrast
        equ = cv2.equalizeHist(gray_frame)

        rgb = cv2.cvtColor(equ, cv2.COLOR_GRAY2RGB)

        # Apply color thresholding to detect blood cells
        lower = np.array([0, 0, 0])
        upper = np.array([40, 40, 40])
        mask = cv2.inRange(rgb, lower, upper)

        # Set detected blood cells to red
        red = np.zeros_like(resized_frame)
        red[mask > 0] = (0, 190, 0)

        # Combine grayscale background with red cells
        result = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
        result += red

        # Create an empty list to store the estimated cell positions
        estimated_cells = []

        # Loop over the detected cells
        for cell in cells:

            # Get the cell position and radius
            cell_x, cell_y = cell[0]
            cell_r = cell[1]

            # Create a Kalman filter for the cell
            kalman = cv2.KalmanFilter(4, 2)
            kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0]], np.float32)
            kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                [0, 1, 0, 1],
                                                [0, 0, 1, 0],
                                                [0, 0, 0, 1]], np.float32)
            kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                            [0, 1, 0, 0],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1]], np.float32) * 0.03

            # Initialize the state of the Kalman filter with the cell position and zero velocity
            kalman.statePre = np.array([[cell_x],
                                        [cell_y],
                                        [0],
                                        [0]], np.float32)

            # Initialize the measurement noise covariance matrix
            kalman.measurementNoiseCov = np.array([[kalman.processNoiseCov[0][0] / 10, 0],
                                                [0, kalman.processNoiseCov[1][1] / 10]], np.float32)

            # Predict the position of the cell using the Kalman filter
            prediction = kalman.predict()
            # Store the predicted position in the list of estimated cells
            estimated_cells.append(prediction)

            # Update the state of the Kalman filter using the actual position of the cell
            measurement = np.array([[cell_x],
                                    [cell_y]], np.float32)
            kalman.correct(measurement)

            # Draw a circle around the cell with the estimated position
            x_est, y_est = prediction[0][0], prediction[1][0]
            cv2.circle(result, (int(x_est), int(y_est)), cell_r, (255, 0, 0), thickness=2)

        # Display the resulting image
        cv2.imshow('frame', result)

        # Write the frame to output video
        out.write(result)

        # Exit if 'q' key is pressed
        if cv2.waitKey(round(1000 / fps)) & 0xFF == ord('q'):
            break
    else:
        break

# Release the video capture object and the video writer
vid.release()
out.release()

 
# Closes all the frames
cv2.destroyAllWindows()
