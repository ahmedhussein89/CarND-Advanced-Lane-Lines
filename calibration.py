import cv2
import glob
import numpy as np
import matplotlib.image as mpimg


def calibration():
    nx = 9
    ny = 6

    objp = np.zeros((ny*nx,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

    # read all calibartion images
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    calibration_images = glob.glob('camera_cal/*.jpg')
    for image in calibration_images:
        calb_image = mpimg.imread(image)
        gray_image = cv2.cvtColor(calb_image, cv2.COLOR_RGB2GRAY)
        ret, corners = cv2.findChessboardCorners(calb_image, (nx, ny), None)
        if(ret == True):
            objpoints.append(objp)
            imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_image.shape[::-1], None, None)

    return mtx, dist
