import cv2
import PySimpleGUI as sg
import numpy as np
import matplotlib.pyplot as plt

# Load an image
img = cv2.imread("test1.jpg")

# Resize the image to half its original size
img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))

# Create the PySimpleGUI layout
layout = [
    [sg.Image(filename="", key="-IMAGE-")],
    [sg.Text(f"Edit contrast"), sg.Slider(range=(0, 255), key="contrast_slider", default_value=0, size=(20, 20), enable_events=True, orientation="horizontal")],
    [sg.Text(f"Remove noise"), sg.Slider(range=(3, 21, 2), key="filter_slider", default_value=3, size=(20, 20), enable_events=True, orientation="horizontal")],
    [sg.Checkbox('Logarithmic transformation', key='log_checkbox', default=False, enable_events=True), sg.Checkbox('Gamma transformation', key='gamma_checkbox', default=False, enable_events=True)],
    [sg.Checkbox('Morphology operation', key='morph_checkbox', default=False, enable_events=True)],
    [sg.Canvas(key='-CANVAS-')],
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

    # Apply edge detection to highlight cell outlines
    edges = cv2.Canny(img_contrast, threshold1=30, threshold2=100)

    # Find contours in the edge map
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the contours on the original image
    img_contours = img.copy()
    cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 1)

    
    # Compute the areas of the detected cells
    areas = [cv2.contourArea(c) for c in contours]

    # Update the image in the PySimpleGUI window
    window["-IMAGE-"].update(data=cv2.imencode(".png", img_contours)[1].tobytes())

# Display statistics about cell area as a histogram
hist, bins = np.histogram(areas, bins=30)
fig, ax = plt.subplots()
ax.bar(bins[:-1], hist, width=np.diff(bins))
ax.set_xlabel('Area (pixels)')
ax.set_ylabel('Cell count')
ax.set_title('Cell Area Histogram')
plt.show()

# Close the window and destroy the OpenCV window
window.close()
cv2.destroyAllWindows()

