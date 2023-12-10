import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm
from scipy import ndimage


####################
#  Basic Functions #
####################

def read_image(img_path):
  """
    Read image from a path and convert them to gray or rgb
    Inputs:
      img_path: str: image path
    Returns:
      img: cv2 image: Original Image
      img_gray: cv2 image: Gray image
      img_rgb: cv2 image: RGB image
  """
  img = cv2.imread(img_path)
  img_gray= cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  return img_gray, img, img_rgb

def SIFT(img_gray):
  """
    Extracting keypoints and descriptors from a gray image!
    Inputs:
      img_gray: cv2 image: Gray Image
    Returns:
      kp: tuple: Keypoint Location (position, scale, orientation)
      des: numpy.ndarray: Keypoint Descriptor - 128 elements vector
  """
  # siftDetector= cv2.xfeatures2d.SIFT_create()
  siftDetector= cv2.SIFT_create()
  kp, des = siftDetector.detectAndCompute(img_gray, None)
  return kp, des

def plot_sift(img_gray, img_rgb, kp):
  """
    Draw on gray image the position of keypoints
      Center position refers to the position of that keypoint
      Circle refers to the scale of that keypoint
      Radius refers to the reference orientation
    Inputs:
      img_gray: cv2 image: Gray Image
      img_rgb: cv2 image: RGB Image
      kp: tuple: Keypoint Location
    Returns:
      img_kp: cv2 image: Gray Image with keypoints
  """
  tmp = img_rgb.copy()
  img_kp = cv2.drawKeypoints(img_gray, kp, tmp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
  plt.imshow(img_kp)
  return img_kp

def plot_matches(matches, total_img):
  """
    Plot lines to show found matches on total_img
      total_img is a concatenated image for two singles [1]+[2] => [12]
      matches is a matrix[N,4] with N is number of matches and
      stores the position of matched keypoint pairs [[kp1.x kp1.y kp2.x kp2.y],[],[],...]
    Inputs:
      matches: numpy.ndarray: A matrix stores matches keypoint from two singles same size image
      total_img: cv2.image: A concatenated image for those two image
    Returns:
      None, return a matched keypoints image
  """
  match_img = total_img.copy()
  # Need to compute offset since total_img is a concatenated image
  offset = total_img.shape[1]/2
  fig, ax = plt.subplots()
  ax.set_aspect('equal')
  ax.imshow(np.array(match_img), cmap = "gray")
  colors = [plt.cm.tab10(i) for i in range(10)]

  # Mark each eypoint position by a red X
  ax.plot(matches[:, 0], matches[:, 1], 'xr')
  ax.plot(matches[:, 2] + offset, matches[:, 3], 'xr')
  
  # Draw a line connect two match keypoint found in matches
  for i in range(matches.shape[0]):
    ax.plot([matches[i, 0], matches[i, 2] + offset], [matches[i, 1], matches[i, 3]],
      colors[np.random.randint(10)], linewidth=1)
  plt.show()

def plot_keypoints(left_gray, left_rgb, kp_left, right_gray, right_rgb, kp_right, inliers):
  """
    Drawing Keypoints on gray images and 
    Plotting lines to show found matches on total_img
    Inputs:
      Inputs:
      left_gray - right_gray: cv2 image: Gray Image
      left_rgb - right_gray: cv2 image: RGB Image
      kp_left - kp_right: tuple: Keypoint Location
      inliers: numpy.ndarray: A matrix [Mx4] of reliable matching pairs derived from matches
    Returns:
      None
  """
  total_img = np.concatenate((left_rgb, right_rgb), axis=1)
  plt.subplot(1,2,1)
  kp_left_img = plot_sift(left_gray, left_rgb, kp_left)
  plt.subplot(1,2,2)
  kp_right_img = plot_sift(right_gray, right_rgb, kp_right)
  plot_matches(inliers[:100], total_img)
  plt.show()


####################
#  KNN Matching    #
####################

def KNNmatcher(kp1, des1, img1, kp2, des2, img2, threshold):
  """
    Compare the descriptor to find matching pairs from two images
    then validate those matchings by a threshold,
    Inputs:
      kp1: tuple: Keypoint Location at the first image (position, scale, orientation)
      des1: numpy.ndarray: Keypoint Descriptor of the first image - 128 elements vector
      img1: cv.image: The first image
      kp2: tuple: Keypoint Location at the second image (position, scale, orientation)
      des2: numpy.ndarray: Keypoint Descriptor of the second image - 128 elements vector
      img2: cv.image: The second image
      threshold: int: a threshold to validate the reliance of matching pairs.
    Returns:
      matches: numpy.ndarray: A matrix of reliable matching pairs [[kp1.x kp1.y kp2.x kp2.y],[],[],...]
  """
  # BFMatcher with default params, then return k-best matches
  bf = cv2.BFMatcher()
  kmatches = bf.knnMatch(des1,des2, k=2)

  # Validate the found matches by knn
  good_matches = []
  for m,n in kmatches:
    if m.distance < threshold*n.distance:
      good_matches.append([m])

  # Store pairs keypoint position on matches
  matches = []
  for pair in good_matches:
    matches.append(list(kp1[pair[0].queryIdx].pt + kp2[pair[0].trainIdx].pt))
  matches = np.array(matches)
  np.random.shuffle(matches)
  return matches

####################
#  Ransac Matching #
####################

def homography(pairs):
  """
    Computing a homography that can project four points to four other points
    Inputs:
      pairs: numpy.ndarray: A matrix [4x4] of 4 matching pairs poistions [[kp1.x kp1.y kp2.x kp2.y],[],[],[]]
    Returns:
      H: numpy.ndarray: A matrix [3x3] that could transform four points into four other points
  """
  rows = []
  for i in range(pairs.shape[0]):
      p1 = np.append(pairs[i][0:2], 1)
      p2 = np.append(pairs[i][2:4], 1)
      row1 = [0, 0, 0, p1[0], p1[1], p1[2], -p2[1]*p1[0], -p2[1]*p1[1], -p2[1]*p1[2]]
      row2 = [p1[0], p1[1], p1[2], 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1], -p2[0]*p1[2]]
      rows.append(row1)
      rows.append(row2)
  rows = np.array(rows)
  U, s, V = np.linalg.svd(rows)
  H = V[-1].reshape(3, 3)
  H = H/H[2, 2] # Normalize by the value of H[2,2]
  return H

def random_point(matches, k=4):
  """
    Selecting randomly k samples in found matches
    Inputs:
      matches: numpy.ndarray: A matrix [Nx4] of found matching pairs' position [[kp1.x kp1.y kp2.x kp2.y],[],[],...]
      k: int: The number of selected samples (default = 4)
    Returns:
      point: numpy.ndarray: A matrix [3x3] that could transform four points into four other points
  """
  idx = random.sample(range(len(matches)), k)
  pairs = [matches[i] for i in idx ]
  pairs = np.array(pairs)
  return pairs

def get_error(points, H):
  """
    Computing least square error of orignal points and estimated points using homography matrix
      all_p1: all selected points position on first image         [[kp1.x kp1.y],[],[],...]
      all_p2: all corressponding of p1 points on second image     [[kp2.x kp2.y],[],[],...]
      estimate_p2: estimated version of all_p2 by all_p1 and homograpy matrix
    Inputs:
      points: numpy.ndarray: A matrix [Nx4] of found matching pairs' position [[kp1.x kp1.y kp2.x kp2.y],[],[],...]
      H: numpy.ndarray: The defined homography matrix
    Returns:
      errors: float: A value refer to least square error between original and estimate version
  """
  # Split keypoint position from matches, padding for matrix multiplication
  num_points = len(points)
  all_p1 = np.concatenate((points[:, 0:2], np.ones((num_points, 1))), axis=1)
  all_p2 = points[:, 2:4]
  # Estimating the position by homography matrix
  estimate_p2 = np.zeros((num_points, 2))
  for i in range(num_points):
    temp = np.dot(H, all_p1[i])
    estimate_p2[i] = (temp/temp[2])[0:2] # set index 2 to 1 and slice the index 0, 1

  # Compute Least Square Error between original and estimated
  errors = np.linalg.norm(all_p2 - estimate_p2 , axis=1) ** 2

  return errors

def ransac(matches, threshold, iters):
  """
    How RANSAC works? (RANdom SAmple Consensus)
      1. Select randomly 4 samples from found matches by knn to compute homography matrix
      2. Remove bad homography matrix by a threshold
      3. Determine the performance of matrix by inliers counting
      4. Repeat 1-3 in a number of iterations, chose the matrix has the largest M number of inliers
    Inputs:
      matches: numpy.ndarray: A matrix [Nx4] of found matching pairs' position [[kp1.x kp1.y kp2.x kp2.y],[],[],...]
      threshold: float [0,1]: A value to remove bad homography by computing L2Norm of original and estimated points
      iters: the iterations of loop to find the desired homography matrix
    Returns:
      best_inliers: numpy.ndarray: A matrix [Mx4] of reliable matching pairs derived from matches
      best_H: numpy.ndarray: The correspoding homography matrix
  """
  num_best_inliers = 0

  for i in range(iters):
    # 1. Select randomly 4 samples then compute homography matrix
    points = random_point(matches)
    H = homography(points)
    #  Safe check, avoid dividing by zero
    if np.linalg.matrix_rank(H) < 3:
        continue

    # 2. Remove bad homography matrix by a threshold
    errors = get_error(matches, H)
    idx = np.where(errors < threshold)[0]

    # 3. Counting the number of inliers
    inliers = matches[idx]
    num_inliers = len(inliers)

    # 4. Updating the better inliers and homography matrix
    if num_inliers > num_best_inliers:
        best_inliers = inliers.copy()
        num_best_inliers = num_inliers
        best_H = H.copy()
  # Return the best inliers and homography matrix using RANSAC
  print("inliers/matches: {}/{}".format(num_best_inliers, len(matches)))
  return best_inliers, best_H

def match_ransac(left_path, right_path, plot_out = False):
  """
    Computing the desired Homography matrix and removed outliers
    Plot good matches determined by RANSAC
      1. Read image from image path
      2. Detect Keypoint and Descriptors by SIFT
      3. Finding potential matches by KNN
      4. Computing Homography matrix, and remove outliers by RANSAC
      5. Plot out 100 inliers matches
    Inputs:
      left_path: str: The left image path
      right_path: str: The right image path
      plot_out: boolean: Allow to plot keypoints images (Default = False)
    Returns:
      left_rgb: cv2 image: The left rgb image
      right_rgb: cv2 image: The right rgb image
      H: numpy.ndarray: The desired homography matrix by ransac
  """
  # 1. Read image from image path
  left_gray, left_origin, left_rgb = read_image(left_path)
  right_gray, right_origin, right_rgb = read_image(right_path)

  # 2. Detect Keypoint and Descriptors by SIFT
  kp_left, des_left = SIFT(left_gray)
  kp_right, des_right = SIFT(right_gray)

  # 3. Finding potential matches by KNN
  matches = KNNmatcher(kp_left, des_left, left_rgb, kp_right, des_right, right_rgb, 0.5)

  # 4. Computing Homography matrix, and remove outliers by RANSAC
  inliers, H = ransac(matches, 0.5, 2000)

  # 5. Plot out 100 inliers matches (optional)
  if plot_out == True:
    plot_keypoints(left_gray, left_rgb, kp_left, right_gray, right_rgb, kp_right, inliers)

  return left_rgb, right_rgb, H

####################
#  Panorama Image  #
####################

def stitch_img(left, right, H):
  """
    Warping two images on the same plane then merge into a panorama image
    Inputs:
      left: cv2 image: The first image
      right: cv2 image: The second image
      H: numpy.ndarray: The desired homography matrix
    Returns:
      lwarp: cv2 image: The warped version of first image
      rwarp: cv2 image: The warped version of second image
      stitch_image: cv2 image: The final image after merged
  """
  print("stiching image ...")

  # Convert to double and normalize to avoid noise.
  left = cv2.normalize(left.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
  right = cv2.normalize(right.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

  # Computing a transformation for both image to merge correctly.
  height_l, width_l, channel_l = left.shape
  height_r, width_r, channel_r = right.shape
  corners = [[0, 0, 1], [width_l, 0, 1], [width_l, height_l, 1], [0, height_l, 1]]
  corners_new = [np.dot(H, corner) for corner in corners]
  corners_new = np.array(corners_new).T
  x_news = corners_new[0] / corners_new[2]
  y_news = corners_new[1] / corners_new[2]
  y_min = min(y_news)
  x_min = min(x_news)
  translation_mat = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
  H = np.dot(translation_mat, H)

  # Executing transformations for both image and padding for a new size.
  # Left Image
  height_new = int(round(abs(y_min) + height_l))
  width_new = int(round(abs(x_min) + width_l))
  size = (width_new, height_new)
  warped_l = cv2.warpPerspective(src=left, M=H, dsize=size)
  warped_l1 = warped_l.copy()
  # Right Image
  height_new = int(round(abs(y_min) + height_r))
  width_new = int(round(abs(x_min) + width_r))
  size = (width_new, height_new)
  warped_r = cv2.warpPerspective(src=right, M=translation_mat, dsize=size)

  black = np.zeros(3)  # Black pixel for rgb image.
  # Stitching procedure, store results in warped_l.
  # Compare with a black pixel
  for i in tqdm(range(warped_r.shape[0])):
    for j in range(warped_r.shape[1]):
      pixel_l = warped_l[i, j, :]
      pixel_r = warped_r[i, j, :]
      if not np.array_equal(pixel_l, black) and np.array_equal(pixel_r, black):
        warped_l[i, j, :] = pixel_l
      elif np.array_equal(pixel_l, black) and not np.array_equal(pixel_r, black):
        warped_l[i, j, :] = pixel_r
      elif not np.array_equal(pixel_l, black) and not np.array_equal(pixel_r, black):
        warped_l[i, j, :] = (pixel_l + pixel_r) / 2
      else:
        pass

  stitch_image = warped_l[:warped_r.shape[0], :warped_r.shape[1], :]
  lwarp, rwarp = warped_l1[:warped_r.shape[0], :warped_r.shape[1], :], warped_r
  return lwarp, rwarp, stitch_image

def pano_stitching(left_path, right_path):
  """
    From two image paths (left & right) create a panorama from them.
      1. Read image from image path
      2. Detect Keypoint and Descriptors by SIFT
      3. Finding potential matches by KNN
      4. Computing Homography matrix, and remove outliers by RANSAC
      5. Warping and Merging and Plotting a complete panorama image
    Inputs:
      left_path: cv2 image: The left image path
      right_path: cv2 image: The right image path
    Returns:
      lwarp: cv2 image: The warped version of first image
      rwarp: cv2 image: The warped version of second image
      stitch_image: cv2 image: The final image after merged (Panorama)
  """
  # 1-4. Computing desired Homograpy matrix
  left_rgb, right_rgb, H = match_ransac(left_path, right_path)

  # 5. Warping and Merging
  lwarp, rwarp, stitch_image = stitch_img(left_rgb, right_rgb, H)
  # plt.imshow(stitch_image)
  return lwarp, rwarp, stitch_image
