import cv2 
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('Particles.png', 0)
img = np.array(img)
ROW = img.shape[0] 
COL = img.shape[1] 

def max_filter(img, N):
    row = img.shape[0]
    col = img.shape[1]
    A = np.zeros(shape=(row, col), dtype=np.float32)
    pad_width = None
    if N>1:
        pad_width = round((N-1)/2+0.1)
        img = np.pad(img, pad_width, mode='edge')
    for i in range(row): 
        for j in range(col): 
            A[i][j] = np.max(img[i:i+N,j:j+N])
    return A

A = max_filter(img, N=13)
plt.imshow(A, cmap='gray')
plt.show()

def min_filter(img, N):
    row = img.shape[0]
    col = img.shape[1]
    B = np.zeros(shape=(row, col), dtype=np.float32)
    if N>1:
        pad_width = round((N-1)/2+0.1)
        img = np.pad(img, pad_width, mode='edge')
    for i in range(row): 
        for j in range(col): 
            B[i][j] = np.min(img[i:i+N,j:j+N])
    return B

B = min_filter(A, N=13)
plt.imshow(B, cmap='gray')
plt.show()

O = img - B
plt.imshow(O, cmap='gray')
plt.show()

def rmv_shading(img, M, N):
    if M == 0:
        A = max_filter(img, N=N)
        B = min_filter(A, N=N)
        O = img - B
    else:
        A = min_filter(img, N=N)
        B = max_filter(A, N=N)
        O = img - B
    return O

img = cv2.imread('Cells.png',0)
img = np.array(img)
plt.imshow(img, cmap='gray')
plt.show()

O = rmv_shading(img, M=1, N=23)
plt.imshow(O, cmap='gray')
plt.show()
