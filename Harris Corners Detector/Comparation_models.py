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


def extract_window(matrix, center, size):
    """
    Get a window around the corner point to compare
    """
    y, x = center
    h, w = size, size
    window = matrix[y - h // 2:y + h // 2 + 1, x - w // 2:x + w // 2 + 1]
    return window

# patches comparation
"""
3 models for patches comparation
"""
# SSIM
def SSIM(img1, img2, corners1, corners2, win_size):
  corners1 = np.array(corners1)
  corners2 = np.array(corners2)
  len_1 = len(corners1)
  len_2 = len(corners2)
  pad = win_size// 2
  cor_1 = []
  cor_2 = []
  diff = []
  for i in range (len_1):
    for j in range (len_2):
      cor_1.append(corners1[i])
      cor_2.append(corners2[j])
      win_1 = extract_window(img1, corners1[i], win_size)
      win_2 = extract_window(img2, corners2[j], win_size)
      if (win_1.shape == win_2.shape):
        diff.append(compare_ssim(win_1, win_2))
      else:
        diff.append(0)
  res = list(zip(diff, cor_1, cor_2))
  res = sorted(res, key=lambda x: x[0], reverse=True)
  # res.sort(reverse = True)
  return res

#NCC (Normalized Cross Corelation)
def NCC(img1, img2, corners1, corners2, win_size):
  corners1 = np.array(corners1)
  corners2 = np.array(corners2)
  len_1 = len(corners1)
  len_2 = len(corners2)
  pad = win_size// 2
  cor_1 = []
  cor_2 = []
  diff = []
  for i in range (len_1):
    for j in range (len_2):
      cor_1.append(corners1[i])
      cor_2.append(corners2[j])
      win_1 = extract_window(img1, corners1[i], win_size)
      win_2 = extract_window(img2, corners2[j], win_size)
      if (win_1.shape == win_2.shape):

        sub_1 = win_1 - np.mean(win_1)
        sub_2 = win_2 - np.mean(win_2)

        numerator = np.sum(sub_1 * sub_2)
        denominator = np.sqrt(np.sum(sub_1**2)) * np.sqrt(np.sum(sub_2**2))
        diff.append(numerator/ denominator)
      else:
        diff.append(0)
  res = list(zip(diff, cor_1, cor_2))
  res = sorted(res, key=lambda x: x[0], reverse=True)
  # res.sort(reverse = True)
  return res

#SSD - sum of squared differences
def SSD(img1, img2, corners1, corners2, win_size):
  corners1 = np.array(corners1)
  corners2 = np.array(corners2)
  len_1 = len(corners1)
  len_2 = len(corners2)
  pad = win_size// 2
  cor_1 = []
  cor_2 = []
  diff = []
  for i in range (len_1):
    for j in range (len_2):
      cor_1.append(corners1[i])
      cor_2.append(corners2[j])
      win_1 = extract_window(img1, corners1[i], win_size)
      win_2 = extract_window(img2, corners2[j], win_size)
      if (win_1.shape == win_2.shape):
        diff.append(np.sum(np.abs(win_1 - win_2)))
      else:
        diff.append(0)
  res = list(zip(diff, cor_1, cor_2))
  res = sorted(res, key=lambda x: x[0], reverse=False)
  # res.sort(reverse = True)
  return res