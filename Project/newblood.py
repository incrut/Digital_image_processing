import cv2 
import numpy as np


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

        keypoints = detector.detect(gray_frame)

        # Loop over the blobs
        for keypoint in keypoints:
            # Calculate the blob's position and size
            x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
            r = int(keypoint.size / 2)
            cv2.circle(result, (x, y), r, color, thickness=2)
                        # Check if this cell is close to any existing cell
            too_close = False
            for cell in cells:
                if np.linalg.norm(np.array(cell[0]) - np.array([x, y])) < cell[1] / 2:
                    too_close = True
                    break

            # If the cell is not too close to any existing cells, add it to the list
            if not too_close:
                cells.append([(x, y), r])

            # Add cell ID to image
            cell_id = str(len(cells))
            cv2.putText(result, cell_id, (x, y), font, 1, color, 2)

        # Display the resulting image
        cv2.imshow('frame', result)

        # Write the frame to output video
        out.write(result)

        # Exit if 'q' key is pressed
        if cv2.waitKey(round(1000 / fps)) & 0xFF == ord('q'):
            break
    else:
        break

vid.release()
out.release()
cv2.destroyAllWindows()