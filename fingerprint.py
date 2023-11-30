import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.util import random_noise, invert
from skimage.morphology import skeletonize
from skimage import feature, filters
from skimage.io import imread
import math

def To2DArr(img):
    im = np.zeros((len(img), len(img[0])), dtype=float)

    for r in range(len(img)):
        for c in range(len(img[r])):
            im[r][c] = np.mean(img[r][c])/255
    
    return im

def Arr2DToBW(img, thresh = 0.95):
    im = np.zeros((len(img), len(img[0])), dtype=float)

    for r in range(len(img)):
        for c in range(len(img[r])):
            im[r][c] = np.mean(img[r][c]) > thresh
    
    return im

def BinarizeFingerprint(imageName, method = "Pixel Blocks", blockSize = 9, thresh = 0.01):
    BnW_image = []

    img = cv2.imread(imageName, cv2.IMREAD_ANYCOLOR)

    if method == 'Canny': # USE CANNY FILTER
        im = To2DArr(img)
        edges1 = feature.canny(im)
        BnW_image = invert(edges1)
    elif method == 'Scharr': # USE SCHARR EDGE DETECTION AND THEN BINARIZATION
        edge_scharr = filters.scharr(img)
        img = invert(edge_scharr)
        BnW_image = Arr2DToBW(img, thresh=0.95)
    elif method == 'Pixel Blocks': # BINARIZATION ON EACH BLOCK SIZE PIXEL CHUNK
        im = To2DArr(img)
        BnW_image = np.zeros((len(im), len(im[0])))
        j = 0
        while j < len(im[0]):
            i = 0
            while i < len(im):
                half = math.floor(blockSize/2)
                rangeJ = range(max(j-half, 0), min(j+half, len(im[0])-1))
                rangeI = range(max(i-half, 0), min(i+half, len(im)-1))
                mean = np.mean([np.mean([im[x][y] for y in rangeJ]) for x in rangeI])
                BnW_image[i][j] = im[i][j] > mean - thresh
                i += 1
            j += 1

        # CREDIT TO https://ieeexplore.ieee.org/document/1716119
    elif method == 'None': # USE PIXEL BY PIXEL THRESHOLD
        (thresh, BnW_image) = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
    else:
        raise Exception

    return BnW_image