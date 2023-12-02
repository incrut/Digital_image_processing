# # # import cv2
# # # import numpy as np
# # # import PySimpleGUI as sg

# # # def add_contrast_filter(image, correction):
# # #     L = 256
# # #     b = np.zeros(L, dtype=np.uint8)
# # #     image_rows, image_cols, channels = image.shape
# # #     lAB = 0

# # #     for i in range(image_rows):
# # #         for j in range(image_cols):
# # #             valueB = image[i, j, 0]
# # #             valueG = image[i, j, 1]
# # #             valueR = image[i, j, 2]
# # #             lAB += int(valueR * 0.299 + valueG * 0.587 + valueB * 0.114)

# # #     lAB /= image_rows * image_cols
# # #     k = 1.0 + correction / 100.0

# # #     for i in range(L):
# # #         delta = i - lAB
# # #         temp = int(lAB + k * delta)
# # #         if temp < 0:
# # #             temp = 0
# # #         if temp >= 255:
# # #             temp = 255
# # #         b[i] = temp

# # #     for i in range(image_rows):
# # #         for j in range(image_cols):
# # #             value = image[i, j, 0]
# # #             image[i, j, 0] = b[value]
# # #             value = image[i, j, 1]
# # #             image[i, j, 1] = b[value]
# # #             value = image[i, j, 2]
# # #             image[i, j, 2] = b[value]

# # #     return image

# # # # Load an image
# # # img = cv2.imread("test1.jpg")

# # # # Resize the image to half its original size
# # # img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))

# # # # Create the PySimpleGUI layout
# # # layout = [
# # #     [sg.Image(filename="", key="-IMAGE-")],
# # #     [sg.Text(f"Edit contrast"), sg.Slider(range=(-100, 100), key="slider", default_value=0, size=(20, 20), enable_events=True, orientation="horizontal")]
# # # ]

# # # # Create the PySimpleGUI window
# # # window = sg.Window("Contrast Enhancement", layout)

# # # # Display the resized image
# # # window["-IMAGE-"].update(data=cv2.imencode(".png", img)[1].tobytes())

# # # # Main event loop
# # # while True:
# # #     event, values = window.read()
# # #     if event == sg.WINDOW_CLOSED:
# # #         break

# # #     # Get the slider value
# # #     slider_value = values["slider"]

# # #     # Enhance the contrast using the custom function
# # #     img_contrast = add_contrast_filter(img.copy(), slider_value)

# # #     # Update the image in the PySimpleGUI window
# # #     window["-IMAGE-"].update(data=cv2.imencode(".png", img_contrast)[1].tobytes())

# # # # Close the window and destroy the OpenCV window
# # # window.close()
# # # cv2.destroyAllWindows()


# # import cv2
# # import PySimpleGUI as sg
# # import numpy as np

# # # Load an image
# # img = cv2.imread("test1.jpg")

# # # Resize the image to half its original size
# # img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))

# # # Create the PySimpleGUI layout
# # layout = [
# #     [sg.Image(filename="", key="-IMAGE-")],
# #     [sg.Text(f"Edit contrast"), sg.Slider(range=(0, 255), key="contrast_slider", default_value=0, size=(20, 20), enable_events=True, orientation="horizontal")],
# #     [sg.Text(f"Remove noise"), sg.Slider(range=(3, 21, 2), key="filter_slider", default_value=3, size=(20, 20), enable_events=True, orientation="horizontal")],
# #     [sg.Checkbox('Logarithmic transformation', key='log_checkbox', default=False, enable_events=True), sg.Checkbox('Gamma transformation', key='gamma_checkbox', default=False, enable_events=True)],
# #     [sg.Text('Select Color Space'), sg.Combo(['RGB', 'HSI'], key='color_space', default_value='RGB', enable_events=True)],
# #     [sg.Text('Color Range'), sg.Slider(range=(0, 255), key='color_range', default_value=50, size=(20, 20), enable_events=True, orientation="horizontal")],
# # ]

# # # Create the PySimpleGUI window
# # window = sg.Window("Image Processing", layout)

# # # Main event loop
# # while True:
# #     event, values = window.read()
# #     if event == sg.WINDOW_CLOSED:
# #         break

# #     # Get the slider values
# #     contrast_value = values["contrast_slider"]
# #     filter_size = int(values["filter_slider"]) # Convert filter size to integer

# #     # Apply a median filter to remove noise
# #     img_filtered = cv2.medianBlur(img, filter_size)
    
# #     # Ensure that the filter_size is odd
# #     if filter_size % 2 == 0:
# #         filter_size += 1

# #     # Apply logarithmic transformation if checkbox is checked
# #     if values['log_checkbox']:
# #         img_filtered = np.log10(img_filtered + 1) * 255 / np.log10(256)
# #         img_filtered = np.uint8(img_filtered)

# #     # Apply gamma transformation if checkbox is checked
# #     if values['gamma_checkbox']:
# #         gamma = 0.5 # Change this value to adjust gamma
# #         img_filtered = np.power(img_filtered / 255, gamma) * 255
# #         img_filtered = np.uint8(img_filtered)
        
# #     # Apply color thresholding based on selected color space
# #     if values['color_space'] == 'RGB':
# #         color_range = values['color_range']
# #         lower_range = np.array([0, 0, 0])
# #         upper_range = np.array([color_range, color_range, color_range])
# #         mask = cv2.inRange(img_filtered, lower_range, upper_range)
# #         img_contrast = cv2.bitwise_and(img_filtered, img_filtered, mask=mask)
        
# #     elif values['color_space'] == 'HSI':
# #         color_range = values['color_range']
# #         img_hsi = cv2.cvtColor(img_filtered, cv2.COLOR_BGR2HSV)
# #         lower_range = np.array([0, 0, 0])
# #         upper_range = np.array([180, 255, color_range])
# #         mask = cv2.inRange(img_hsi, lower_range, upper_range)
# #         img_contrast = cv2.bitwise_and(img_filtered, img_filtered, mask=mask)

# #     # Enhance the contrast using OpenCV
# #     img_contrast = cv2.convertScaleAbs(img_contrast, alpha=1 + contrast_value/255, beta=-contrast_value)

# #     # Update the image in the PySimpleGUI window
# #     window["-IMAGE-"].update(data=cv2.imencode(".png", img_contrast)[1].tobytes())

# # # Close the window and destroy the OpenCV window
# # window.close()
# # cv2.destroyAllWindows()


import cv2
import PySimpleGUI as sg
import numpy as np

# Load an image
img = cv2.imread("test2.jpg")

# Resize the image to half its original size
img = cv2.resize(img, (int(img.shape[1]/3), int(img.shape[0]/4)))

# Create the PySimpleGUI layout
layout = [
    [sg.Image(filename="", key="-IMAGE-")],
    [sg.Text(f"Edit contrast"), sg.Slider(range=(0, 255), key="contrast_slider", default_value=0, size=(20, 20), enable_events=True, orientation="horizontal")],
    [sg.Text(f"Remove noise"), sg.Slider(range=(3, 21, 2), key="filter_slider", default_value=3, size=(20, 20), enable_events=True, orientation="horizontal")],
    [sg.Checkbox('Logarithmic transformation', key='log_checkbox', default=False, enable_events=True), sg.Checkbox('Gamma transformation', key='gamma_checkbox', default=False, enable_events=True)],
    [sg.Checkbox('Morphology operation', key='morph_checkbox', default=False, enable_events=True)]
]

# Create the PySimpleGUI window
window = sg.Window("Image Processing", layout)

# Main event loop
while True:
    event, values = window.read()
    if event == sg.WINDOW_CLOSED:
        break

    # Get the slider values
    contrast_value = values["contrast_slider"]
    filter_size = int(values["filter_slider"]) # Convert filter size to integer

    # Ensure that the filter_size is odd
    if filter_size % 2 == 0:
        filter_size += 1

    # Apply a median filter to remove noise
    img_filtered = cv2.medianBlur(img, filter_size)
    
    

    # Apply logarithmic transformation if checkbox is checked
    if values['log_checkbox']:
        img_filtered = np.log10(img_filtered + 1) * 255 / np.log10(256)
        img_filtered = np.uint8(img_filtered)

    # Apply gamma transformation if checkbox is checked
    if values['gamma_checkbox']:
        gamma = 0.5 # Change this value to adjust gamma
        img_filtered = np.power(img_filtered / 255, gamma) * 255
        img_filtered = np.uint8(img_filtered)

    # Apply morphology operation if checkbox is checked
    if values['morph_checkbox']:
        kernel_size = 5  # Change this value to adjust kernel size
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        img_filtered = cv2.morphologyEx(img_filtered, cv2.MORPH_OPEN, kernel)

    # Enhance the contrast using OpenCV
    img_contrast = cv2.convertScaleAbs(img_filtered, alpha=1 + contrast_value/255, beta=-contrast_value)

    # Update the image in the PySimpleGUI window
    window["-IMAGE-"].update(data=cv2.imencode(".png", img_contrast)[1].tobytes())

# Close the window and destroy the OpenCV window
window.close()
cv2.destroyAllWindows()




# import cv2
# import PySimpleGUI as sg
# import numpy as np

# # Load an image
# img = cv2.imread("test2.jpg")

# # Resize the image to half its original size
# img = cv2.resize(img, (int(img.shape[1]/3), int(img.shape[0]/4)))

# # set up the blob detector
# detector_params = cv2.SimpleBlobDetector_Params()

# detector_params.filterByCircularity = True
# detector_params.minCircularity = 0.5
# detector_params.maxCircularity = 1.0

# detector_params.filterByArea = True
# detector_params.minArea = 100
# detector_params.maxArea = 4000

# detector = cv2.SimpleBlobDetector_create(detector_params)

# # Create the PySimpleGUI layout
# layout = [
#     [sg.Image(filename="", key="-IMAGE-")],
#     [sg.Text(f"Edit contrast"), sg.Slider(range=(0, 255), key="contrast_slider", default_value=0, size=(20, 20), enable_events=True, orientation="horizontal")],
#     [sg.Text(f"Remove noise"), sg.Slider(range=(3, 21, 2), key="filter_slider", default_value=3, size=(20, 20), enable_events=True, orientation="horizontal")],
#     [sg.Checkbox('Logarithmic transformation', key='log_checkbox', default=False, enable_events=True), sg.Checkbox('Gamma transformation', key='gamma_checkbox', default=False, enable_events=True)],
#     [sg.Checkbox('Morphology operation', key='morph_checkbox', default=False, enable_events=True)]
# ]

# # Create the PySimpleGUI window
# window = sg.Window("Image Processing", layout)

# # Main event loop
# while True:
#     event, values = window.read()
#     if event == sg.WINDOW_CLOSED:
#         break

#     # Get the slider values
#     contrast_value = values["contrast_slider"]
#     filter_size = int(values["filter_slider"]) # Convert filter size to integer

#     # Ensure that the filter_size is odd
#     if filter_size % 2 == 0:
#         filter_size += 1

#     # Apply a median filter to remove noise
#     img_filtered = cv2.medianBlur(img, filter_size)
    
    

#     # Apply logarithmic transformation if checkbox is checked
#     if values['log_checkbox']:
#         img_filtered = np.log10(img_filtered + 1) * 255 / np.log10(256)
#         img_filtered = np.uint8(img_filtered)

#     # Apply gamma transformation if checkbox is checked
#     if values['gamma_checkbox']:
#         gamma = 0.5 # Change this value to adjust gamma
#         img_filtered = np.power(img_filtered / 255, gamma) * 255
#         img_filtered = np.uint8(img_filtered)   

#     # Apply morphology operation if checkbox is checked
#     if values['morph_checkbox']:
#         kernel_size = 5  # Change this value to adjust kernel size
#         kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
#         img_filtered = cv2.morphologyEx(img_filtered, cv2.MORPH_OPEN, kernel)

#     # Enhance the contrast using OpenCV
#     img_contrast = cv2.convertScaleAbs(img_filtered, alpha=1 + contrast_value/255, beta=-contrast_value)

#     keypoints = detector.detect(img_contrast)

#         # Loop over the blobs
#     for keypoint in keypoints:
#         # Calculate the blob's position and size
#         x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
#         r = int(keypoint.size / 2)
#         # Draw a circle around the blob
#         cv2.circle(img_contrast, (x, y), r, (0, 255, 0), thickness=2)

#     # Update the image in the PySimpleGUI window
#     window["-IMAGE-"].update(data=cv2.imencode(".png", img_contrast)[1].tobytes())

# # Close the window and destroy the OpenCV window
# window.close()
# cv2.destroyAllWindows()


