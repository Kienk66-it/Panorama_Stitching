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


#Draw lines
def DRAW(img, res, lines):
  temp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  h, w, dim = img.shape
  map_x1 = np.zeros(10000)
  map_x2 = np.zeros(10000)
  map_y1 = np.zeros(10000)
  map_y2 = np.zeros(10000)
  for i in range (lines):
    # print(res[i][1], end = ' ')
    # print(res[i][2])
    (x1, y1) = res[i][1]
    (x2, y2) = res[i][2]
    if((map_x1[x1] == 0 and map_y1[y1] == 0) and (map_x2[x2] == 0 and map_y2[y2] == 0)):
      map_x1[x1]+= 1
      map_y1[y1]+= 1
      map_x2[x2]+= 1
      map_y2[y2]+= 1
      #cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
      cv2.circle(img, (y1, x1), radius=3, color=[0, 255, 0], thickness=-1)
      cv2.circle(img, (y2 + w// 2, x2), radius=3, color=[0, 255, 0], thickness=-1)
      cv2.line(img, (y1, x1), (y2 + w// 2, x2), (0, 255, 0), 2)
  return img