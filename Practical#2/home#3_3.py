import cv2
import numpy as np
from matplotlib import pyplot as plt

from matplotlib.colors import NoNorm

# convenience function that estimates the difference between 2 images
def diffscore(img1, img2):
    return np.sum(abs(img1-img2))

# convenience function to draw 1 image - no typing 4 lines anymore
def display(img, single_layer = False):
    plt.figure()
    plt.axis("off")
    if single_layer:
        plt.imshow(img,cmap='gray',norm=NoNorm())
        # this ensures plt doesn't treat our grayscale array as a messed-up 3-channel image
    else:
        plt.imshow(img)
    plt.show()

src = cv2.cvtColor(cv2.imread("src.jpg"), cv2.COLOR_BGR2RGB)

def alien_filter(img):
    out = img.copy()
    outr = out[:,:,0]
    outg = out[:,:,1]
    outb = out[:,:,2]
    out = np.stack((outg, outb, outr), axis=2)
    # np.stack() stacks arrays in higher dimensions
    # because outx are all 2D slices
    return out

def vintage_filter(img):
    # Logarithmic transform
    log_transformed = (25 * np.log10(1 + img)).astype(np.uint8)

    # Piecewise transform
    piecewise_transformed = img.copy()
    piecewise_transformed = np.where(piecewise_transformed < 128, 0.5 * piecewise_transformed, 2 * piecewise_transformed - 255)

    # Combine the two transformations
    combined = cv2.addWeighted(log_transformed, 0.5, piecewise_transformed, 0.5, 0)

    return combined

#display(pastel_filter(src))
display(vintage_filter(src))
display(alien_filter(src))