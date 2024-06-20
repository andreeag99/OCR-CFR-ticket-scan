import numpy as py
import cv2
import numpy as np
from matplotlib import pyplot as plt
from imutils.perspective import four_point_transform

img = cv2.imread('test4.jpg')

#  Convert BGR to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# define range of ticket color in HSV
lower_blue = np.array([10,0,50])
upper_blue = np.array([100,100,250])

# Threshold the HSV image to get only blue colors
mask = cv2.inRange(hsv, lower_blue, upper_blue)
# Bitwise-AND mask and original image
res = cv2.bitwise_and(img,img, mask= mask)

# Find contours of BW mask
contours, hiearchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x))
cnt = cntsSorted[-1]
print(cnt)

# find the perimeter of the first closed contour
perim = cv2.arcLength(cnt, True)
# setting the precision
epsilon = 0.05*perim
# approximating the contour with a polygon
approxCorners = cv2.approxPolyDP(cnt, epsilon, True)
# check how many vertices has the approximate polygon
approxCornersNumber = len(approxCorners)
print("Number of approximated corners: ", approxCornersNumber)
print(approxCorners)

p1 = approxCorners[0]
p2 = approxCorners[1]
p3 = approxCorners[2]
p4 = approxCorners[3]
print(p1[0], p2[0], p3[0], p4[0])
points = np.array([[p1[0][0],p1[0][1]], [p2[0][0],p2[0][1]], [p3[0][0],p3[0][1]], [p4[0][0],p4[0][1]]])
borderless = four_point_transform(img, points)

cv2.drawContours(img, [cnt], 0, (0,255,0), 5)
cv2.drawContours(img, [approxCorners], 0, (0,0,255), 3)

plt.subplot(321), plt.imshow(img, cmap='gray')
plt.title('Original Image w/ contour drawn'), plt.xticks([]), plt.yticks([])
plt.subplot(323), plt.imshow(mask, cmap='gray')
plt.title('Mask'), plt.xticks([]), plt.yticks([])
plt.subplot(325), plt.imshow(borderless, cmap='gray')
plt.title('Borderless'), plt.xticks([]), plt.yticks([])
plt.show()
