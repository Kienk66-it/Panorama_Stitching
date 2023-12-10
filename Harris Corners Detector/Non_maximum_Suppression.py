import numpy as np
import cv2
import math
from google.colab.patches import cv2_imshow
from scipy.ndimage import convolve
from skimage.feature import corner_harris, corner_peaks
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from scipy import ndimage, datasets
from skimage.metrics import structural_similarity as compare_ssim
from Harris_Response import *
from Non_maximum_Suppression import *
from Get_coordinates import *
from Comparation_models import *
from Draw_lines import *


def local_non_maximal_suppression(corner_resp, kernel_size=7):
    """
    To find the best interest point in each local neighborhood
    
    corner_resp: response matrix
    kernel_size: m x m around feature
    """
  # Making kernel_size odd
  if kernel_size % 2 == 0:
    kernel_size = kernel_size + 1
  pad = int(kernel_size/2)

  # Pad for window
  corner_resp = np.pad(corner_resp, ((pad, pad), (pad, pad)), mode="constant")

  nms_corner_resp = np.zeros_like(corner_resp)

  h, w = corner_resp.shape

  # Find max value
  for i in range(pad, h-pad):
      for j in range(pad, w-pad):
          max_val = np.amax(corner_resp[i - pad: i + pad + 1, j - pad: j + pad + 1])
          if corner_resp[i, j] == np.amax(corner_resp[i - pad: i + pad + 1, j - pad: j + pad + 1]):
              nms_corner_resp[i, j] = max_val

  # remove padding
  nms_corner_resp = nms_corner_resp[pad:h-pad, pad:w-pad]

  return nms_corner_resp