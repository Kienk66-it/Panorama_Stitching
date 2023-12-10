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


def my_cornerHarris(image, blockSize, ksize, k):
    """
    Caculate Harris Response
    
        image: gray image
        blockSize: window size of blurring function
        ksize: window size of sobel function
        k: [0.04-0.06]
    """
    image_float32 = np.float32(image)

    # Gradient
    Ix = cv2.Sobel(image_float32, cv2.CV_64F, 1, 0, ksize=ksize)
    Iy = cv2.Sobel(image_float32, cv2.CV_64F, 0, 1, ksize=ksize)

    # Compute Ixx, Ixy, Iyy
    Ixx = Ix * Ix
    Ixy = Ix * Iy
    Iyy = Iy * Iy

    # Blurring
    Ixx_blurred = cv2.GaussianBlur(Ixx, (blockSize, blockSize), 0)
    Ixy_blurred = cv2.GaussianBlur(Ixy, (blockSize, blockSize), 0)
    Iyy_blurred = cv2.GaussianBlur(Iyy, (blockSize, blockSize), 0)

    # Caculate Harris response
    determinant = Ixx_blurred * Iyy_blurred - Ixy_blurred**2
    trace = Ixx_blurred + Iyy_blurred
    harris_response = determinant - k * trace**2

    return harris_response