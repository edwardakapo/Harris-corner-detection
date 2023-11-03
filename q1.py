#Author Oluwademilade Edward Akapo
#student No 101095403

from contextlib import suppress
from cv2 import threshold
import numpy as np
import cv2
from matplotlib import pyplot as plt
import math


#user rotines in opencv to demonstrate the operations of the harris corner detector 

#open the image box-in-scene jpeg and show image in a window
img1 = cv2.imread('Harris corner detector\\box_in_scene.png',0) # queryImage


#compute the minimum eigenvalue of that image, using the routine cv2-cornerMineEigenVal
eval = cv2.cornerMinEigenVal(img1, 3, 3)

# threshold the minimum eigenvalue, if it passes this value set it to white, else black

def harrisCorner(thres):
    eval_copy = np.copy(eval)
    threshold = thres/20 * 0.02
    ret , thresh1 = cv2.threshold(eval_copy , threshold , 225 , cv2.THRESH_BINARY)
    img_mag, img_x, img_y = magnitude(thresh1)
    supressed = supression(img_mag, img_x, img_y)

    # find the keypoints and descriptors with SIFT
    cv2.imshow(window,thresh1)
    cv2.imshow("suppresed" , supressed)

    # final result with detected corners
    




# Take pixels that pass the threshold and use non-maxima supression algorithm to thin out the potential corners,
def calcConvolution(image,kernel):
    imgconv= np.zeros((image.shape[0], image.shape[1]), dtype=float)
    kernel_k = kernel.shape[0]
    img_x = image.shape[0]
    img_y = image.shape[1]
    k = kernel_k //2
    for i in range (0,img_x):
        for j in range (0,img_y):
            val = 0
            for u in range (-k , k+1):
                for v in range(-k , k+1):           
                #--implementation of wrapping
                #--conditions for wrapping
                    x = i - u
                    y = j - v
                    if x < 0 :
                        x = img_x - u -1
                    if y < 0 :
                        y = img_y - v -1
                    if x > img_x - 1 :
                        x = x - img_x - 1 
                    if y > img_y-1:
                        y = y - img_y - 1
                    val += kernel[u+k][v+k] * image[x][y]
            imgconv[i][j] = val
    return imgconv

#--calculates the sobel in x and y then gets the magnitude
def magnitude(image):
    sobel_x = np.zeros((3,3), dtype=float)
    sobel_x[0][0] = 1
    sobel_x[0][1] = 0
    sobel_x[0][2] = -1
    sobel_x[1][0] = 2
    sobel_x[1][1] = 0
    sobel_x[1][2] = -2
    sobel_x[2][0] = 1
    sobel_x[2][1] = 0
    sobel_x[2][2] = -1
    sobel_y = np.transpose(sobel_x)
    img_x = calcConvolution(image,sobel_x)
    img_y = calcConvolution(image, sobel_y)
    x = image.shape[0]
    y = image.shape[1]
    img_mag = np.zeros((x,y), dtype=float)
    for i in range(x):
        for j in range(y):
            mag = math.sqrt(img_x[i][j]**2 + img_y[i][j]**2)
            img_mag[i][j] = mag
    return (img_mag,img_x,img_y)



def calcAngleDegrees(y, x):
    a = math.atan2(y, x) * 180 / math.pi
    if a < 0:
        a = a + 180
    if a >= 22.5 and a <= 67.5:
        a = 45
    elif a > 67.5 and a <=112.5:
        a = 90
    elif a > 112.5 and a <= 157.5:
        a = 135
    else :
        a = 0
    return a



def getCoordinates(i,j,sizex,sizey,u,v):
    x = i + u
    y = j + v
    if x < 0 :
        x = -1
    if y < 0 :
        y = -1
    if x > sizex - 1 :
        x = -1
    if y > sizey -1:
        y = -1
    return (x,y)

# uses the image magnitue, sobel x and y to calculate the non maxima supressed image
def supression(img_mag,img_x,img_y):
    k_x = img_mag.shape[0]
    k_y = img_mag.shape[1]
    x1 = y1 = x2 = y2 = 0
    new_img = np.zeros((k_x,k_y), dtype=float)
    for i in range(k_x):
        for j in range(k_y):
            atan = calcAngleDegrees(img_y[i][j], img_x[i][j])
            # 0,45,90,135
            if atan == 0:
                x1 , y1 = getCoordinates(i,j,k_x,k_y,0,1)
                x2 , y2 = getCoordinates(i,j,k_x,k_y,0,-1)
            elif atan == 45:
                x1 , y1 = getCoordinates(i,j,k_x,k_y,-1,1)
                x2 , y2 = getCoordinates(i,j,k_x,k_y,1,-1)
            elif atan == 90:
                x1 , y1 = getCoordinates(i,j,k_x,k_y,-1,0)
                x2 , y2 = getCoordinates(i,j,k_x,k_y,1,0)
            elif atan == 135:
                x1 , y1 = getCoordinates(i,j,k_x,k_y,-1,1)
                x2 , y2 = getCoordinates(i,j,k_x,k_y,1,1)
            
            if x1 != -1 and x2 != -1 and y1 != -1 and y2 != -1:
                if img_mag[i][j] < img_mag[x1][y1] or  img_mag[i][j] < img_mag[x2][y2]:
                     new_img[i][j] = 0
                else :
                    new_img[i][j] = img_mag[i][j]
            else :
                new_img[i][j] = img_mag[i][j]
    return new_img


# draw the corners in another window with small circles



# use a threshold slider
window = "Corner Detector"
thresholdLvl = 10
maxThreshold = 20

cv2.namedWindow(window)
cv2.createTrackbar("Threshold" , window , thresholdLvl , maxThreshold, harrisCorner)
harrisCorner(thresholdLvl)

cv2.waitKey(0)
cv2.destroyAllWindows()

#submission---
#snap of the slider
#before and after suppression images
# final result with detected corners