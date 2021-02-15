# 배경 제거 > threshold 적용하여 완벽한 로고 값만 뽑아낸다.

import cv2 
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(r'./Project01/cocacolarcan1.jpg')
img = cv2.imread(r'./Project01/sprite1.jpg')
print(img.shape)    # (225, 225, 3)
cv2.imshow("raw", img)

# 배경 제거===============================
# 사각형 좌표: 시작점의 x, y  ,넢이, 너비
rectangle = (50, 50, 256, 150)

# 초기 마스크 생성
mask = np.zeros(img.shape[:2], np.uint8)
print(mask.shape)   # (225, 225)

# grabCut에 사용할 임시 배열 생성
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# grabCut 실행
cv2.grabCut(img, # 원본 이미지
           mask,       # 마스크
           rectangle,  # 사각형
           bgdModel,   # 배경을 위한 임시 배열
           fgdModel,   # 전경을 위한 임시 배열 
           5,          # 반복 횟수
           cv2.GC_INIT_WITH_RECT) # 사각형을 위한 초기화

# 배경인 곳은 0, 그 외에는 1로 설정한 마스크 생성
mask_2 = np.where((mask==2) | (mask==0), 0, 1).astype('uint8')

# 이미지에 새로운 마스크를 곱행 배경을 제외
img_rgb_nobg = img * mask_2[:, :, np.newaxis]

# plot
cv2.imshow("nobg", img_rgb_nobg)
# plt.imshow(img_rgb_nobg)
# plt.show()

# threshold====================================
gray = cv2.cvtColor(img_rgb_nobg, cv2.COLOR_BGR2GRAY)
threshold, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
cv2.imshow("thresh", thresh)
# threshold, thresh_inv = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
# cv2.imshow("thresh_inv", thresh_inv)

cv2.waitKey(0)
cv2.destroyAllWindows()
