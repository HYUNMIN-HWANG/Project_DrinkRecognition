# https://shilan.tistory.com/entry/Python%EC%9D%84-%EC%9D%B4%EC%9A%A9%ED%95%98%EC%97%AC-%EB%8F%99%EC%98%81%EC%83%81%EC%9C%BC%EB%A1%9C%EB%B6%80%ED%84%B0-%EC%9D%B4%EB%AF%B8%EC%A7%80%EC%B6%94%EC%B6%9C-Pythonv27-OpenCV-Windows
# -*- coding: utf-8 -*-
# __author__ = 'Seran'
 

# 음료수 이미지 데이터가 많이 부족하여 직접 영상 촬영한 후 프레임 단위별로 나눔

import cv2

# mp4 = ['video_cocacola1','video_cocacola2','video_letsbee1','video_letsbee2','video_letsbee3','video_letsbee4',\
#     'video_pocari1','video_pocari2','video_sprite1','video_sprite2','video_tejava1','video_tejava2']

# mp4 = ['video_fanta1','video_fanta2']

mp4 = ['video_tejava3','video_tejava4']

for i in mp4 :
    video_file = '../Project01_data/video/' + str(i) + '.mp4'
    # 영상의 의미지를 연속적으로 캡쳐할 수 있게 하는 class
    vidcap = cv2.VideoCapture(video_file)
    count = 0

    while(vidcap.isOpened()):
        ret, image = vidcap.read()
        if not ret :
            break
        # print(vidcap.get(1))    # 프레임번호 
        if(int(vidcap.get(1)) % 3 == 0):    # 총 프레임을 n 만큼 나눈 곳마다 캡쳐한다.
            print('Saved frame number : ' + str(int(vidcap.get(1))))
            save_path = "../Project01_data/video/" + str(i) + "/" + str(i) + "_" + str(count) +'.jpg'
            # cv2.imwrite("../Project01_data/video/cocacolar%d.jpg" % count, image)
            cv2.imwrite(save_path, image)
            print('Saved frame%d.jpg' % count)
            count += 1
            k = cv2.waitKey(10)
            if k == 27 :    # esc를 누르면 종료
                break
    vidcap.release()
