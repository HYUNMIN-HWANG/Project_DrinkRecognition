import cv2 
import numpy as np
import matplotlib.pyplot as plt

# img = cv2.imread(r'../Project01_data/normal_data/cocacola/1.jpg')
img = cv2.imread(r'../Project01_data/normal_data/cocacola/video_cocacola1_52.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
threshold, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
cv2.imshow("thresh", thresh)
threshold, thresh_inv = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("thresh_inv", thresh_inv)


cv2.waitKey(0)
cv2.destroyAllWindows()