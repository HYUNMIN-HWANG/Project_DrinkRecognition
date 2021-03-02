import cv2 
import numpy as np
import matplotlib.pyplot as plt

# img = cv2.imread(r'../Project01_data/normal_data/cocacola/1.jpg')
# img = cv2.imread(r'../Project01_data/normal_data/cocacola/video_cocacola2_52.jpg')
img = cv2.imread(r'../Project01_data/normal_data/cocacola/video_cocacola1_52.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


blur = cv2.GaussianBlur(gray, (5,5), cv2.BORDER_DEFAULT)
cv2.imshow("blur", blur)

canny = cv2.Canny(blur, 125, 175)
cv2.imshow("canny", canny)

dilated = cv2.dilate(canny, (7,7),iterations=2)
cv2.imshow("dilated", dilated)

cv2.waitKey(0)
cv2.destroyAllWindows()