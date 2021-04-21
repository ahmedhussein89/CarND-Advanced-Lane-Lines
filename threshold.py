import cv2
import numpy as np
import matplotlib.image as mpimg


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if(orient == 'x'):
        orient_image = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    if(orient == 'y'):
        orient_image = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the derivative or gradient
    absolute_image = np.absolute(orient_image)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_image = np.uint8(255 * absolute_image / np.max(absolute_image))
    # 5) Create a mask of 1's where the scaled gradient magnitude
    binary_output = np.zeros_like(scaled_image)
            # is > thresh_min and < thresh_max
    binary_output[(scaled_image >= thresh[0]) & (scaled_image <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    # 2) Take the gradient in x and y separately
    # 3) Take the absolute value of the x and y gradients
    gradient_x = np.abs(cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    gradient_y = np.abs(cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    direction = np.arctan2(gradient_y, gradient_x)
    # 5) Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(gray_image)
    binary_output[(direction >= thresh[0]) & (direction <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output

def hls_select(img, thresh=(0, 255)):
    # 1) Convert to HLS color space
    HLS_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    S_image = HLS_image[:, :, 2]
    # 2) Apply a threshold to the S channel
    binary_output = np.zeros_like(S_image)
    binary_output[(S_image > thresh[0]) & (S_image <= thresh[1])] = 1
    # 3) Return a binary image of threshold result
    return binary_output

def process_all(image):
    abs_x_image = abs_sobel_thresh(image, orient='x', thresh=(20, 80))
    abs_y_image = abs_sobel_thresh(image, orient='y', thresh=(30, 100))
    dir_binary = dir_threshold(image, sobel_kernel=15, thresh=(0.7, 1.1))
    mag_binary  = mag_thresh(image, sobel_kernel=3, mag_thresh=(50, 150))
    hls_binary  = hls_select(image, thresh=(150, 255))
    combined_binary = np.zeros_like(dir_binary)
    combined_binary[((abs_x_image == 1) & (abs_y_image == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (hls_binary == 1)] = 1
    return combined_binary

