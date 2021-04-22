import cv2
import numpy as np
import matplotlib.pyplot as plt


class Lane:
    def __init__(self, windows_count = 9, margin = 100, minpix = 50):
        # HYPERPARAMETERS
        # Choose the number of sliding windows
        self.windows_count = windows_count
        # Set the width of the windows +/- margin
        self.margin = margin
        # Set minimum number of pixels found to recenter window
        self.minpix = minpix
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None

    def find_lane_pixels(self, binary_warped, x_base, out_img):
        # Set height of windows - based on nwindows above and image shape
        window_height = np.int32(binary_warped.shape[0]//self.windows_count)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero  = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        x_current  = x_base

        # Create empty lists to receive left and right lane pixel indices
        lane_inds  = []

        # Step through the windows one by one
        for window in range(self.windows_count):
            # Identify window boundaries in x and y (and right and left)
            win_y_low  = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            ### TO-DO: Find the four below boundaries of the window ###
            win_x_low   = x_current  - self.margin
            win_x_high  = x_current  + self.margin

            # Draw the windows on the visualization image
            if False:
                cv2.rectangle(out_img, (win_x_low,win_y_low), (win_x_high,win_y_high),(0, 255, 0), 2)

            good_inds  = [index for index, value in enumerate(zip(nonzeroy, nonzerox)) if (value[0] < win_y_high and  value[0] >= win_y_low) and (value[1] < win_x_high  and  value[1] >= win_x_low)]

            lane_inds.append(good_inds)

            # Append these indices to the lists
            if(nonzerox[good_inds].shape[0] > self.minpix):
                x_current = int(np.mean(nonzerox[good_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            lane_inds  = np.concatenate(lane_inds).astype(np.int32)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left and right line pixel positions
        self.x  = nonzerox[lane_inds]
        self.y  = nonzeroy[lane_inds]

        if self.allx:
            self.allx.append(self.x)
            self.ally.append(self.y)
        else:
            self.allx = np.copy(self.x)
            self.ally = np.copy(self.y)

    def fit_polynomial(self, out_img, color):
        lane_fit = np.polyfit(self.y, self.x, 2)

        # Generate x and y values for plotting
        self.ploty = np.linspace(0, out_img.shape[0]-1, out_img.shape[0])

        try:
            self.recent_xfitted = lane_fit[0]*self.ploty**2 + lane_fit[1]*self.ploty + lane_fit[2]
            self.detected = True
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            self.recent_xfitted = 1*self.ploty**2 + 1*self.ploty
            self.detected = False

        ## Visualization ##
        # Colors in the left and right lane regions
        out_img[self.y, self.x] = color

        # Plots the left and right polynomials on the lane lines
        if False:
            plt.imshow(out_img)
            plt.plot(self.recent_xfitted, self.ploty, color='yellow')
            plt.show()

    def measure_curvature_real(self):
        '''
        Calculates the curvature of polynomial functions in meters.
        '''
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension

        fit_cr = np.polyfit(ym_per_pix * self.y, xm_per_pix * self.x, 2)

        # Define y-value where we want radius of curvature
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(self.y)

        # Calculation of R_curve (radius of curvature)
        curverad  = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])

        x = fit_cr[0] * y_eval * ym_per_pix ** 2 + fit_cr[1] * y_eval * ym_per_pix + fit_cr[2]

        return x, curverad

class Lane_Manager:
    def __init__(self):
        self.left_lane  = Lane()
        self.right_lane = Lane()

    def process(self, binary_warped):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        out_img = cv2.normalize(out_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint    = np.int32(histogram.shape[0]//2)
        leftx_base  = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        self.left_lane.find_lane_pixels(binary_warped, leftx_base,   out_img)
        self.right_lane.find_lane_pixels(binary_warped, rightx_base, out_img)

        self.left_lane.fit_polynomial(out_img, [255, 0, 0])
        left_x, left_curvered = self.left_lane.measure_curvature_real()

        self.right_lane.fit_polynomial(out_img, [0, 0, 255])
        right_x, right_curvered = self.right_lane.measure_curvature_real()

        self.out_img = out_img
        new_center = (((right_x - left_x) / 2) + left_x)

        warp_zero  = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        pts_left  = np.array([np.transpose(np.vstack([self.left_lane.recent_xfitted,
                                                      self.left_lane.ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.right_lane.recent_xfitted,
                                                     self.right_lane.ploty])))])
        pts = np.hstack((pts_left, pts_right))

        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Compute Car Offset
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        center = out_img.shape[1]//2 * xm_per_pix

        offset = center - new_center
        return color_warp, left_curvered, right_curvered, offset

