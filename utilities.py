import cv2
import numpy as np


def prespective_image(image, show_image):
    height, width = image.shape[:2]

    source_points = np.array([
        [270,  670], # Point 0
        [575,  460], # Point 1
        [740,  460], # Point 2
        [1100, 670]  # Point 3
    ], np.float32)

    dest_points = np.array([
        [270,  670], # Point 0
        [270,  0],   # Point 1
        [1101, 0],   # Point 2
        [1100, 670]  # Point 3
    ], np.float32)

    M = cv2.getPerspectiveTransform(source_points,
                                    dest_points)
    inv_M = cv2.getPerspectiveTransform(dest_points,
                                        source_points)

    if(show_image and False):
        source_pts = source_points.astype(np.int32).reshape((-1, 1, 2))
        dest_pts   = dest_points.astype(np.int32).reshape((-1, 1, 2))

        image2 = cv2.polylines(np.copy(image),
                              [source_pts],
                              True,
                              (255, 0, 0),
                              2)

        image2 = cv2.polylines(image2,
                              [dest_pts],
                              True,
                              (0, 0, 255),
                              2)

        plt.imshow(image2)
        plt.show()

    return cv2.warpPerspective(image, M, (width, height)), inv_M
