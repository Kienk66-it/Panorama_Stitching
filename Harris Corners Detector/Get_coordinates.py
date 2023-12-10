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


def get_max_corners(corner_resp):
    """
    get coorodinates of max corner points
    
    """
  #To get coordinates of each corner points
  corner_resp = np.array(corner_resp)
  r, c = np.where(corner_resp >  0.01 * corner_resp.max())
  val = corner_resp[r, c]
  corners_xy = np.vstack((r, c)).T

  #sort by val - corner response
  combined = list(zip(val, corners_xy))
  combined = sorted(combined, key=lambda x: x[0], reverse=True)
  val, corners_xy = zip(*combined)
  # print(val)
  # print(corners_xy)
  return corners_xy, val
