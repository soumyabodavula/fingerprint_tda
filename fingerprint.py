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
from gtda.homology import VietorisRipsPersistence
from gtda.images import RadialFiltration

def ToPersistenceDiagram(filtered):
    persistence = VietorisRipsPersistence(
        metric="euclidean",
        homology_dimensions=[0, 1], # Track connected components, loops
        n_jobs=6
    )

    diagrams_basic = persistence.fit_transform(filtered)
    return diagrams_basic

def ToPointCloud(BnW_image):
    points = []
    BnW_image_cpy = np.array([[[0]*len(BnW_image[0])]*len(BnW_image)])

    for x in range(len(BnW_image)):
        for y in range(len(BnW_image[x])):
            if BnW_image[x][y] != 0:
                points.append([x,y,])

    points = np.array([points])
    return points

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

def BinarizeFingerprint(img, method = "Pixel Blocks Optimized", blockSize = 200, thresh = 0.01):
    BnW_image = []

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
    elif method == 'Pixel Blocks Optimized': # BINARIZATION ON EACH BLOCK SIZE PIXEL CHUNK
        im = To2DArr(img)
        BnW_image = np.zeros((len(im), len(im[0])))
        j = 0
        oldLenI = 0
        newLenI = 0
        while j < len(im[0]):
            i = 0
            means = -1
            
            while i < len(im):
                half = math.floor(blockSize/2)
                newLenI = min(i+half, len(im)-1) - max(i-half, 0)
                if means != -1:
                    #sum = sum - np.sum([im[max(i-half-1, 0)][y] for y in rangeJ]) + np.sum([im[min(i+half, len(im)-1)][y] for y in rangeJ]) 
                    valToRemove = 0
                    valToAdd = 0
                    if(i-half-1 >= 0):
                        valToRemove = np.mean([im[max(i-half, 0)][y] for y in rangeJ])
                        a = means.pop(0)

                    else:
                        valToRemove = 0
                    if(i+half <= len(im)-1):
                        valToAdd = np.mean([im[min(i+half, len(im)-1)][y] for y in rangeJ])
                        means.append(valToAdd)
                    else:
                        valToAdd = 0
                    sum = sum + valToAdd - valToRemove

                    #a = means.pop(0)
                    #means.append(valToAdd)
                    #mean = (mean*oldLenI + valToAdd - valToRemove)/newLenI
                    #mean = np.mean(means)
                    mean = sum/newLenI
                else:
                    rangeJ = range(max(j-half, 0), min(j+half, len(im[0])-1))
                    rangeI = range(max(i-half, 0), min(i+half, len(im)-1))
                    means = [np.mean([im[x][y] for y in rangeJ]) for x in rangeI]
                    sum = np.sum(means)
                    mean = np.mean(means)
                BnW_image[i][j] = im[i][j] > mean - thresh
                oldLenI = min(i+half, len(im)-1) - max(i-half, 0)
                i += 1
            j += 1

        # CREDIT TO https://ieeexplore.ieee.org/document/1716119
    elif method == 'None': # USE PIXEL BY PIXEL THRESHOLD
        (thresh, BnW_image) = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
    else:
        print('Unrecognized Method')
        raise Exception

    return BnW_image


# original_img = cv2.imread("images/badprint1.png", cv2.IMREAD_ANYCOLOR)
# binarized_img = BinarizeFingerprint(original_img, method='Pixel Blocks Optimized', blockSize=15)

# plt.imshow(binarized_img, cmap='gray')
# plt.show()