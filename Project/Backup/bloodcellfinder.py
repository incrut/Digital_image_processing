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


cell_id = 1
cell_ids = {}

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
            # Draw a circle around the blob
            cv2.circle(result, (x, y), r, color, thickness=2)
            
            # Write the cell ID on top of the circle
            if keypoint not in cell_ids:
                cell_ids[keypoint] = cell_id
                cell_id += 1
            cv2.putText(result, str(cell_ids[keypoint]), (x-r, y-r), font, fontScale=1, color=color, thickness=2)
        # Write the frame to the output video
        out.write(result)

        # Display the resulting frame
        cv2.imshow('Frame', result)
 
        # Press Q on keyboard to  exit
        if cv2.waitKey(round(1000 / fps)) & 0xFF == ord('q'):
            break
 
    # Break the loop
    else: 
        break
 
# Release the video capture object and the video writer
vid.release()
out.release()

# cv2.imshow('Frame', result)
# cv2.waitKey(0)

 
# Closes all the frames
cv2.destroyAllWindows()









#     # Display the resulting frame
#     cv2.imshow('Frame', resized_frame)

#     # Press Q on keyboard to  exit
#     if cv2.waitKey(22) & 0xFF == ord('q'):
#         break

# #   # Break the loop
# #   else: 
# #     break
 
# # When everything done, release the video capture object
# vid.release()
 
# # Closes all the frames
# cv2.destroyAllWindows()