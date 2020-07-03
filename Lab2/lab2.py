# Task1 Hint: (with sample code for the SIFT detector)
# Initialize SIFT detector, detect keypoints, store and show SIFT keypoints of original image in a Numpy array
# Define parameters for SIFT initializations such that we find only 10% of keypoints
import cv2
import matplotlib.pyplot as plt

class SiftDetector():
    def __init__(self, norm="L2", params=None):
        self.detector=self.get_detector(params)
        self.norm=norm

    def get_detector(self, params):
        if params is None:
            params={}
            params["n_features"]=0
            params["n_octave_layers"]=3
            params["contrast_threshold"]=0.03
            params["edge_threshold"]=10
            params["sigma"]=1.6

        detector = cv2.xfeatures2d.SIFT_create(
                nfeatures=params["n_features"],
                nOctaveLayers=params["n_octave_layers"],
                contrastThreshold=params["contrast_threshold"],
                edgeThreshold=params["edge_threshold"],
                sigma=params["sigma"])

        return detector

fig = plt.figure(figsize = [16, 24])
img = cv2.imread('image.jpg')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

sift1 = SiftDetector().detector
kp1 = sift1.detect(gray, None)
img1 = cv2.drawKeypoints(gray,kp1,img)
cv2.imshow('SIFT', img1)
cv2.imwrite('default.jpg', img1)


params = {"n_features": 623, "contrast_threshold": 0.1, "n_octave_layers": 3, "edge_threshold": 10, "sigma": 1.6}
sift2 = SiftDetector(params=params).detector
kp2 = sift2.detect(gray, None)
img2 = cv2.drawKeypoints(gray,kp2,img)
cv2.imshow('SIFT LESS', img2)
cv2.imwrite('less.jpg', img2)



# Task2 Hint:
# Upscale the image, compute SIFT features for rescaled image
# Apply BFMatcher with defined params and ratio test to obtain good matches, and then select and draw best 5 matches

# resize
img = cv2.imread('image.jpg')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
resized_img = cv2.resize(img, (0, 0), fx=1.15, fy=1.15)
cv2.imshow('115%',resized_img)
cv2.imwrite('resized.jpg', resized_img)

# compute sift features for resized images
resized_gray = cv2.cvtColor(resized_img,cv2.COLOR_BGR2GRAY)
kp = sift2.detect(resized_gray, None)
resized_img_sift = cv2.drawKeypoints(resized_gray,kp,resized_img)
cv2.imshow('resized_img_sift',resized_img_sift)
cv2.imwrite('resized_img_sift.jpg', resized_img_sift)

# Visually the same. Upscale will not affact SIFT key points.

# 5 best-matching descriptors on both the original and the scaled image
kp1, des1 = sift2.detectAndCompute(gray, None)
kp2, des2 = sift2.detectAndCompute(resized_gray, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)

good = []
for m in matches:
    if m[0].distance < 0.75*m[1].distance:
        good.append([m[0]])
        
good = sorted(good, key = lambda x:x[0].distance)

img3 = cv2.drawMatchesKnn(gray,kp1,resized_gray,kp2,good[:5],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imshow('match',img3)
cv2.imwrite('match.jpg',img3)


# Task3 Hint: (with sampe code for the rotation)
# Rotate the image and compute SIFT features for rotated image
# Apply BFMatcher with defined params and ratio test to obtain good matches, and then select and draw best 5 matches
import math
import numpy as np
import sys

# image: image to rotate
# x:     x-coordinate of point we wish to rotate around
# y:     y-coordinate of point we wish to rotate around
# angle: degrees to rotate image by
#
# Returns a rotated copy of the original image
def rotate(image, x, y, angle):
    rot_matrix = cv2.getRotationMatrix2D((x, y), angle, 1.0)
    h, w = image.shape[:2]

    return cv2.warpAffine(image, rot_matrix, (w, h))

# Get coordinates of center point.
#
# image:  Image that will be rotated
# return: (x, y) coordinates of point at center of image
def get_img_center(image):
    height, width = image.shape[:2]
    center = height // 2, width // 2

    return center

# rotate images
x, y = get_img_center(img)
rotated_img = rotate(img, x, y, angle=300)
cv2.imshow('rotated_img.jpg',rotated_img)
cv2.imwrite('rotated_img.jpg',rotated_img)

# Extract the SIFT features and show the keypoints on the rotated image using the same parameter setting as for Task 1
rotated_gray = cv2.cvtColor(rotated_img,cv2.COLOR_BGR2GRAY)
kp = sift2.detect(rotated_gray, None)
rotated_img_sift = cv2.drawKeypoints(rotated_gray,kp,rotated_img)
cv2.imshow('rotated_img_sift',rotated_img_sift)
cv2.imwrite('rotated_img_sift.jpg', rotated_img_sift)

# Visually the same. Rotation will not affact SIFT key points.

kp1, des1 = sift2.detectAndCompute(gray, None)
kp2, des2 = sift2.detectAndCompute(rotated_gray, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)

good = []
for m in matches:
    if m[0].distance < 0.75*m[1].distance:
        good.append([m[0]])
        
good = sorted(good, key = lambda x:x[0].distance)

img4 = cv2.drawMatchesKnn(gray,kp1,rotated_gray,kp2,good[:6],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imshow('rotated_match',img4)
cv2.imwrite('rotated_match.jpg',img4)


cv2.waitKey()
cv2.destroyAllWindows()