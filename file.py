import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.util import random_noise, invert
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

def BinarizeFingerprint(imageName, method = "Pixel Blocks", blockSize = 8):
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
        while j+blockSize-1 < len(im):
            i = 0
            while i+blockSize-1 < len(im[0]):
                mean = np.mean([np.mean(im[i+x][j:j+blockSize-1]) for x in range(0,blockSize)])
                BnW_image[i+math.floor(blockSize/2)][j+math.floor(blockSize/2)] = im[i+math.floor(blockSize/2)][j+math.floor(blockSize/2)] >= mean
                i += 1
            j += 1

        # CREDIT TO https://ieeexplore.ieee.org/document/1716119
    elif method == 'Scharr + Pixel Blocks':
        edge_scharr = filters.scharr(img)
        img = invert(edge_scharr)
        im = To2DArr(img)
        BnW_image = np.zeros((len(im), len(im[0])))
        j = 0
        while j+blockSize-1 < len(im):
            i = 0
            while i+blockSize-1 < len(im[0]):
                mean = np.mean([np.mean(im[i+x][j:j+blockSize-1]) for x in range(0,blockSize)])
                BnW_image[i+math.floor(blockSize/2)][j+math.floor(blockSize/2)] = im[i+math.floor(blockSize/2)][j+math.floor(blockSize/2)] >= mean
                i += 1
            j += 1
    elif method == 'None': # USE PIXEL BY PIXEL THRESHOLD
        (thresh, BnW_image) = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
    else:
        raise Exception

    return BnW_image

BnW_image = BinarizeFingerprint("109_5.tif", method='Pixel Blocks', blockSize=15)
plt.imshow(BnW_image, cmap='gray')
plt.show()