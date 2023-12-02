import cv2 
import numpy as np

# set up the Kalman filter
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03
kalman.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 0.001

# open the video file
vid = cv2.VideoCapture("cell.mp4")
if (vid.isOpened()== False): 
  print("Error opening video stream or file")

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
out = cv2.VideoWriter('TrackedCells.mp4', fourcc, fps, frame_size, True)   

###############################

cells = []
cell_id = 0  # initialize the first cell ID

# font and text color
font = cv2.FONT_HERSHEY_SIMPLEX
color = (0, 255, 0)  # green

while (vid.isOpened()):
    # Capture frame-by-frame
    ret, frame = vid.read()
    if ret:
    
        #resize the frame
        resized_frame = cv2.resize(frame, (new_width, new_height)) 

        # Apply histogram equalization to increase contrast
        gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        equ = cv2.equalizeHist(gray_frame)

        # Predict the next state of each cell using the Kalman filter
        predicted = kalman.predict()

        # Draw the predicted state of each cell as a green circle
        for i in range(len(cells)):
            x, y = cells[i][0]
            r = cells[i][1]
            predicted_state = np.array([x, y], np.float32)
            predicted_state = np.reshape(predicted_state, (2, 1))
            predicted = kalman.predict()
            predicted = np.reshape(predicted, (2, 1))
            cv2.circle(resized_frame, (predicted[0], predicted[1]), r, color, thickness=2)

        # Display the resulting tracking frame
    cv2.imshow("Tracking", resized_frame)

    # Exit if the user presses 'q'
    if cv2.waitKey(round(1000/fps)) & 0xFF == ord('q'):
        break

# Release the capture and destroy any OpenCV windows
vid.release()
cv2.destroyAllWindows()
