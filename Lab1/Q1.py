import numpy as np
import cv2
 
I = cv2.imread('cat.png', flags=0)
a = 0
b = 255
c,d = np.amin(I), np.amax(I)
O = (I.astype(np.float64)-c)*((b-a)/(d-c))+a
O[O > 255] = 255
O = np.round(O)
O = O.astype(np.uint8)

print('Max：{}，Min：{}'.format(d, c))
 
cv2.imshow('1-I', I)
cv2.imwrite('1-I.jpg', I)
cv2.imshow('1-O', O)
cv2.imwrite('1-O.jpg', O)
cv2.waitKey()
cv2.destroyAllWindows()

## Yes it improves