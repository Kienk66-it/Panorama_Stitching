import cv2
import matplotlib.pyplot as plt
import numpy as np
import pano
plt.rcParams['figure.figsize'] = [10, 6]

"""
  From two image paths (left & right) create a panorama from them.
    0. Assign two string as images path
    1. Read image from image path
    2. Detect Keypoint and Descriptors by SIFT
    3. Finding potential matches by KNN
    4. Computing Homography matrix, and remove outliers by RANSAC
    5. Warping and Merging and Plotting a complete panorama image
    6. Plot out panorama image
"""

# 0. Images path
left_path = './img/2.jpg'
right_path = './img/1.jpg'

# 1. Read image from image path
left_gray, left_origin, left_rgb = pano.read_image(left_path)
right_gray, right_origin, right_rgb = pano.read_image(right_path)

# 2. Detect Keypoint and Descriptors by SIFT
kp_left, des_left = pano.SIFT(left_gray)
kp_right, des_right = pano.SIFT(right_gray)

# 3. Finding potential matches by KNN
matches = pano.KNNmatcher(kp_left, des_left, left_rgb, kp_right, des_right, right_rgb, 0.5)

# 4. Computing Homography matrix, and remove outliers by RANSAC
inliers, H = pano.ransac(matches, 0.5, 2000)
#pano.plot_keypoints(left_gray, left_rgb, kp_left, right_gray, right_rgb, kp_right, inliers)

# 5. Warping and Merging and Plotting a complete panorama image
lwarp, rwarp, stitch_image = pano.stitch_img(left_rgb, right_rgb, H)
 
plt.imshow(stitch_image)
plt.show()
