## Advanced Lane Finding

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

### Contents

1. [Introdution](#introdution)
2. [Data Samples](#data-samples)
3. [Requirements](#requirements)
4. [Algorithm](#algorithm)
5. [Result](#result)
6. [Challenges](#challenges)
7. [Conclusion](#conclusion)
8. [Refernces](#refernces)

---

### Introdution

In this project, we will write a software pipeline to identify the lane boundaries in a video. The video was created from the centre camera.
It also required to calculate the curvature of the road.

---

### Data Samples

The test images are coloured with a resolution of 1280x720. The data is split into test images and three different videos with gradient difficulty. Blow a sample of images.

| | |
|:-:|:-:|
| ![straight_lines1](test_images/straight_lines1.jpg) | ![straight_lines2](test_images/straight_lines2.jpg) |
| ![test1](test_images/test1.jpg) | ![test5](test_images/test5.jpg) |

---

### Requirements

This project is pure Python with a few depenendies

- OpenCV [1][1]
- Numpy [2][2]
- matplotlib [2][2]

---

### Algorithm

1. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
2. Apply a distortion correction to raw images.
3. Use color transforms, gradients, etc., to create a thresholded binary image.
4. Apply a perspective transform to rectify binary image ("birds-eye view").
5. Detect lane pixels and fit to find the lane boundary.
6. Determine the curvature of the lane and vehicle position with respect to center.
7. Warp the detected lane boundaries back onto the original image.
8. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


#### 1. Compute the camera calibration matrix

In this project, we will use a pinhole camera model[4]. Although the model is simple to use It comes with industrial defects. It can be called camera distortion[5]. Camera distortion is classified into two types.

| Radial distortion | Tangential distortion |
|:-:|:-:|
| ![radial-distortions](https://learnopencv.com/wp-content/uploads/2019/10/radial-distortions-1024x352.png) [6][6] | ![Tangential distortion](https://www.researchgate.net/profile/Athanasia-Louloudi/publication/297737545/figure/fig7/AS:338353800138758@1457681240203/Tangential-Distortion.png) [7][7] |

To overcome these distortions, We need to find a nonlinear equation which maps the distorted images into ideal image. This operation is called camera calibration.
The camera calibration processes are a sequence of steps that try to reverse the camera distortion. The most common camera calibration is capturing several images for a checkerboard. Which will be passed to an optimization algorithm to find the distortion parameters. These parameters can be used to underscore the images.

The code for calibration is located on `calibration.py` file.
We will use `findChessboardCorners` from `OpenCV`. which will get internal corners.
The order of chekerboard corners in camera plane is welldefined. After getting 2d corner and predefined corner in camera plane we can use 
`calibrateCamera` from `OpenCV` too. Which will estimate camera matrix and camera distorition paramters.

```python
ret, corners = cv2.findChessboardCorners(gray_image, grid_size, None)
```

#### 2. Apply a distortion correction

Now can undistor the input camera, I select this two images because the ridal distortion is so clear in left side at `undistort5` and in both sides of `undistort1`.

```python
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_image.shape[::-1],None, None)
```
| | |
|:-:|:-:|
| ![undistort1](output_images/calibration1.webp) | ![undistort5](output_images/calibration5.webp) |

#### 3. Binary thresholding

The code for calibration is located on `binary_thresholding.py` file.
We need to isolate the lanes from the remaining of the image. We will use binary thresholding which means that anything passes the threshold we are set to one We need a combination of thresholding operation to isolate the lane lines.

##### Sobel[9][9]

1. SobelX and SobelY
2. Edge direction and magniture.
3. using Saturation channel from `HLS` color space.

| | | |
|:-:|:-:|:-:|
| ![sobelxy](output_images/sobelxy.webp) | ![sobel_mag_dir](output_images/sobel_mag_dir.webp) | ![HLS](output_images/HLS.webp) |
![combined](output_images/combined.webp)

At the thresholding stage. I used two strategies.
First, I convert the input image from colour to grey and use grey in the remaining stages ie. I use grey for Sobel and combine the result with `S` channel from `HLS` colour space.
Second, I convert the input image from colour to `HLS` colour space and use the `S` channel in the remaining stages.

TThe second approach is better except starting from frame number 1036. That's why I return to the first approach.
I mention that again on the challenges part.

#### 4. Apply a perspective transform

#### 5. Detect lane pixels

#### 6. Determine the curvature

#### 7. Warp the detected lane boundaries

#### 8. Output visual display

---

### Result

![sobelxy](output_images/text_image_output.webp)

---

### Challenges

- I used to convert the colour image to different colour spaces `HLS,HSV,LAB` and understand the difference.

| | | |
|:-:|:-:|:-:|
| ![H_HLS](media/H_HLS_Channel.webp) | ![L_HLS](media/L_HLS_Channel.webp) | ![S_HLS](media/S_HLS_Channel.webp) |
| ![H_HSV](media/H_HSV_Channel.webp) | ![S_HSV](media/S_HSV_Channel.webp) | ![V_HSV](media/V_HSV_Channel.webp) |
| ![L_LAB](media/L_LAB_Channel.webp) | ![A_LAB](media/A_LAB_Channel.webp) | ![B_LAB](media/B_LAB_Channel.webp) |

- Using the `S` channel as a source for binary thresholding. It generates more clear binary output better than grey still It can be affected by noise at the end of the first video.

| | color | gray | s_channel | combined |
|:-:|:-:|:-:|:-:|:-:|
| Normal | ![sobelxy](media/image_10.webp) | ![sobel_mag_dir](media/image_10_gray.webp) | ![HLS](media/image_10_s_channel.webp) | ![HLS](media/image_10_combined.webp)
| Noise | ![sobelxy](media/image_1036.webp) | ![sobel_mag_dir](media/image_1036_gray.webp) | ![HLS](media/image_1036_s_channel.webp) | ![HLS](media/image_1036_combined.webp)

---

### Conclusion

---

### Refernces

1. https://opencv.org/
2. https://numpy.org/
3. https://matplotlib.org/
4. https://en.wikipedia.org/wiki/Pinhole_camera
5. https://en.wikipedia.org/wiki/Distortion_%28optics%29
6. https://learnopencv.com/understanding-lens-distortion/
7. Evaluation of a Bisensor System for 3D Modeling of Indoor Environments
8. https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
9. https://en.wikipedia.org/wiki/Sobel_operator

[1]: https://opencv.org/
[2]: https://numpy.org/
[3]: https://matplotlib.org/
[4]: https://en.wikipedia.org/wiki/Pinhole_camera
[5]: https://en.wikipedia.org/wiki/Distortion_%28optics%29
[6]: https://learnopencv.com/understanding-lens-distortion/
[7]: https://www.researchgate.net/publication/297737545_Evaluation_of_a_Bisensor_System_for_3D_Modeling_of_Indoor_Environments
[8]: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
[9]: https://en.wikipedia.org/wiki/Sobel_operator