import cv2 
import numpy as np

image = cv2.imread("cat.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
pixel_type = type(image[0][0][0])
Sx = np.array([ [-1, 0, 1], [-2,0,2], [-1,0,1] ])
Sy = np.array([ [-1,-2,-1], [ 0,0,0], [ 1,2,1] ]) 
Ox = cv2.filter2D(image.astype(np.float64), -1, Sx)
Oy = cv2.filter2D(image.astype(np.float64), -1, Sy)
O = (np.absolute(Ox/2) + np.absolute(Oy/2)).astype(pixel_type)

OSobelx = cv2.Sobel(image.astype(np.float64), -1, 1, 0)
OSobely = cv2.Sobel(image.astype(np.float64), -1, 0, 1)
OSobel = (np.absolute(OSobelx/2) + np.absolute(OSobely/2)).astype(pixel_type)
cv2.imshow("3-Sobel",OSobel)
cv2.imwrite('3-Sobel.jpg', OSobel)
cv2.imshow("3-filter2D",O)
cv2.imwrite('3-filter2D.jpg', O)
cv2.waitKey()
cv2.destroyAllWindows()

