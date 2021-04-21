#!/usr/bin/env python3
import os
import cv2
import sys
import numpy as np
from optparse import OptionParser
import matplotlib.pyplot as plt
from lane_detection import Lane_Manager
from binary_thresholding import *
from calibration import calibration
from utilities import prespective_image


def process_image(image_name, show_image=False):
    image = plt.imread(image_name)

    height, width = image.shape[:2]
    mask = np.zeros(image.shape)
    mask[(height//2):, :] = 1

    HLS = cv2.cvtColor(image, cv2.COLOR_RGB2HLS) * mask

    S_image = HLS[:, :, 2]

    if show_image:
        plt.imshow(S_image, cmap="gray")
        plt.show()

    # Blur image to remove noise
    blured_image = cv2.GaussianBlur(S_image, (5, 5), 0)

    abs_x_image  = abs_sobel_thresh(blured_image, orient='x', thresh=(20, 80))
    abs_y_image  = abs_sobel_thresh(blured_image, orient='y', thresh=(20, 80))

    if show_image:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(abs_x_image, cmap="gray")
        ax1.set_title('abs_x_image', fontsize=50)
        ax2.imshow(abs_y_image, cmap="gray")
        ax2.set_title('abs_y_image', fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()

    dir_binary = dir_threshold(S_image, sobel_kernel=15, thresh=(0.7, 1.1))
    mag_binary = mag_thresh(S_image, sobel_kernel=3, mag_thresh=(50, 150))

    if show_image:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax2.imshow(dir_binary, cmap="gray")
        ax2.set_title('Dir', fontsize=50)
        ax1.imshow(mag_binary, cmap="gray")
        ax1.set_title('Mag', fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()

    # Give more weight for S Channel and less for mag&dir and less for abs
    combined_binary = np.zeros_like(S_image)
    combined_binary[((abs_x_image == 1) & (abs_y_image == 1))] = 1
    combined_binary[((mag_binary == 1) & (dir_binary == 1))]   = 1 # 2
    combined_binary[(S_image > 200)]                           = 1 # 3

    if show_image:
        norm_image = cv2.normalize(combined_binary, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        plt.imshow(norm_image, cmap="gray")
        plt.show()

    dest_image, inv_M = prespective_image(combined_binary, show_image)

    if show_image:
        plt.imshow(dest_image, cmap="gray")
        plt.show()

    lane_manager  = Lane_Manager()
    color_image = lane_manager.process(dest_image)

    new_image = cv2.warpPerspective(color_image, inv_M, (image.shape[1], image.shape[0]))
    result = cv2.addWeighted(image, 1, new_image, 0.3, 0)
    plt.imshow(result)
    plt.show()


def process_video(image, show_image=False):
    height, width = image.shape[:2]
    mask = np.zeros(image.shape)
    mask[(height//2):, :] = 1

    HLS = cv2.cvtColor(image, cv2.COLOR_RGB2HLS) * mask

    S_image = HLS[:, :, 2]

    if show_image:
        plt.imshow(S_image, cmap="gray")
        plt.show()

    # Blur image to remove noise
    blured_image = cv2.GaussianBlur(S_image, (5, 5), 0)

    abs_x_image  = abs_sobel_thresh(blured_image, orient='x', thresh=(20, 80))
    abs_y_image  = abs_sobel_thresh(blured_image, orient='y', thresh=(20, 80))

    if show_image:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(abs_x_image, cmap="gray")
        ax1.set_title('abs_x_image', fontsize=50)
        ax2.imshow(abs_y_image, cmap="gray")
        ax2.set_title('abs_y_image', fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()

    dir_binary = dir_threshold(S_image, sobel_kernel=15, thresh=(0.7, 1.1))
    mag_binary = mag_thresh(S_image, sobel_kernel=3, mag_thresh=(50, 150))

    if show_image:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax2.imshow(dir_binary, cmap="gray")
        ax2.set_title('Dir', fontsize=50)
        ax1.imshow(mag_binary, cmap="gray")
        ax1.set_title('Mag', fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()

    # Give more weight for S Channel and less for mag&dir and less for abs
    combined_binary = np.zeros_like(S_image)
    combined_binary[((abs_x_image == 1) & (abs_y_image == 1))] = 1
    combined_binary[((mag_binary == 1) & (dir_binary == 1))]   = 1 # 2
    combined_binary[(S_image > 200)]                           = 1 # 3

    if show_image:
        norm_image = cv2.normalize(combined_binary, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        plt.imshow(norm_image, cmap="gray")
        plt.show()

    dest_image = prespective_image(combined_binary, show_image)

    if show_image:
        plt.imshow(dest_image, cmap="gray")
        plt.show()

    lane_manager  = Lane_Manager()
    lane_manager.process(dest_image)


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-i", "--image",  dest="image",  help="process image",         default="test_images/test1.jpg")
    parser.add_option("-v", "--video",  dest="video",  help="process video",         default="")
    parser.add_option("-s", "--show",   dest="show",   help="show every step image", default=False)
    (options, args) = parser.parse_args()

    if(not os.path.exists(options.image)):
        print(f"Input image \"{options.image}\" not exists")
        sys.exit(-1)

    process_image(options.image, options.show)

    if(options.video):
        process_image(options.video, options.show)

