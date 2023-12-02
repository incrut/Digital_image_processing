import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

plt.rcParams['figure.figsize'] = [10, 10]
# convenience function that estimates time per execution
def timer_avg20(func, *args):
    start = time.monotonic()
    for i in range(20):
        func(*args)
    end = time.monotonic()
    return (end-start)/20

# convenience function that estimates the difference between 2 images
def diffscore(img1, img2):
    return np.sum(abs(img1-img2))
    
# read original image, converting to RGB because that format is expected by matplotlib
src = cv2.cvtColor(cv2.imread("src.jpg"), cv2.COLOR_BGR2RGB)
plt.axis("off")
plt.imshow(src)
print(f"Original size: {src.shape}")
plt.rcParams['figure.figsize'] = [8, 8]

for size_coeff in [0.1, 0.2, 0.5, 2, 4, 10]:
    near_img = cv2.resize(src, None, fx = size_coeff, fy = size_coeff, interpolation = cv2.INTER_NEAREST)
    bilinear_img = cv2.resize(src, None, fx = size_coeff, fy = size_coeff, interpolation = cv2.INTER_LINEAR)
    bicubic_img = cv2.resize(src, None, fx = size_coeff, fy = size_coeff, interpolation = cv2.INTER_CUBIC)
    
    # Concatenating all interpolation images in a single image
    img_concatenated = np.concatenate((near_img, bilinear_img, bicubic_img), axis=1)
    
    print("-"*50)
    print("Image shape before interpolation: ", src.shape)
    print(f"Image shape after interpolation, factor {size_coeff}: ", near_img.shape)
    print(f"\n{' '*8}INTER_NEAREST{' '*14}INTER_LINEAR{' '*16}INTER_CUBIC")
    # Create figure to show original image
    plt.figure()
    plt.axis("off")
    plt.imshow(img_concatenated)
    plt.show()

def your_nn_func(src, x_coeff, y_coeff):
    (x_ori, y_ori, d_ori) = src.shape # the shape is rows * columns * color channels
    
    ## code here, don't forget to return your glorious resized image ##
    resized = np.full((int(x_ori*x_coeff), int(y_ori*y_coeff), d_ori), 1)
    
    return resized
   
x_coeff, y_coeff = 0.5, 0.5

builtin_nn = cv2.resize(src, None, fx = x_coeff, fy = y_coeff, interpolation = cv2.INTER_NEAREST)
your_nn = your_nn_func(src, x_coeff, y_coeff)

sep_strip = np.full((int(src.shape[0]*x_coeff), 24, 3), 255) # for convenience

result = np.concatenate((builtin_nn, sep_strip, your_nn), axis=1)
plt.axis("off")
plt.imshow(result)

print(f"Value difference between images: {diffscore(builtin_nn, your_nn)}")

cv2_avg = timer_avg20(cv2.resize, src, None, None, 2, 2, cv2.INTER_NEAREST)
your_avg = timer_avg20(your_nn_func, src, 2, 2)
print(f"The average time per image with cv2 is {cv2_avg}, and {your_avg} with your function")

def your_bilin_func(src, x_coeff, y_coeff):
    (x_ori, y_ori, d_ori) = src.shape # the shape is rows * columns * color channels
    
    ## code here, don't forget to return your glorious resized image ##
    resized = np.full((int(x_ori*x_coeff), int(y_ori*y_coeff), d_ori), 1)
    
    return resized

x_coeff, y_coeff = 0.5, 0.5

builtin_bilin = cv2.resize(src, None, fx = x_coeff, fy = y_coeff, interpolation = cv2.INTER_LINEAR)
your_bilin = your_bilin_func(src, x_coeff, y_coeff)

sep_strip = np.full((int(src.shape[0]*x_coeff), 24, 3), 255) # for convenience

result = np.concatenate((builtin_bilin, sep_strip, your_bilin), axis=1)
plt.axis("off")
plt.imshow(result)

print(f"Value difference between images: {diffscore(builtin_bilin, your_bilin)}")

cv2_avg = timer_avg20(cv2.resize, src, None, None, 2, 2, cv2.INTER_LINEAR)
your_avg = timer_avg20(your_bilin_func, src, 2, 2)
print(f"The average time per image with cv2 is {cv2_avg}, and {your_avg} with your function")