from matplotlib import pyplot as plt
import cv2

image = cv2.imread("cat.png")
cv2.imshow("Original",image)

hist = cv2.calcHist([image],[0],None,[256],[0,256])

plt.figure()
plt.title("Histogram")
plt.xlabel("h(i)")
plt.ylabel("# of Pixels")
plt.plot(hist)
plt.xlim([0,256])
plt.show()

