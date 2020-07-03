import cv2 
import numpy as np

kernel_size = (5, 5)
sigma = 1.0
constant_factor = 1.25

I = cv2.cvtColor(cv2.imread('cat.png'), cv2.COLOR_BGR2RGB)
pixel_type = type(I[0][0][0])
L = cv2.GaussianBlur(I, kernel_size, sigma);
H = I.astype(np.float64) - L.astype(np.float64)
H = H.astype(np.float64) * constant_factor
O = H.astype(np.float64) + I.astype(np.float64)
O[O > 255] = 255
O = np.round(O)
O = O.astype(np.uint8)
cv2.imshow('4-I', I)
cv2.imwrite('4-I.jpg', I)
cv2.imshow('4-L', L)
cv2.imwrite('4-L.jpg', L)
cv2.imshow('4-H', H)
cv2.imwrite('4-H.jpg', H)
cv2.imshow('4-O', O)
cv2.imwrite('4-O.jpg', O)
cv2.waitKey()
cv2.destroyAllWindows()