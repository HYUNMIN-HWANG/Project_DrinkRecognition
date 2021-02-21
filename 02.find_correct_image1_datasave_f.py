import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from PIL import Image
import io
import os
from glob import glob
import warnings
warnings.filterwarnings(action='ignore')
import tensorflow as tf
import random
import cv2 as cv

coke_list=glob('../Project01_data/normal_data/cocacola/*.jpg')
print(len(coke_list))   # 212
fanta_list=glob('../Project01_data/normal_data/fanta/*.jpg')
print(len(fanta_list))   # 212
letsbee_list=glob('../Project01_data/normal_data/letsbee/*.jpg')
print(len(letsbee_list))   # 212
pocari_list = glob('../Project01_data/normal_data/pocari/*.jpg')
print(len(pocari_list))    # 212
sprite_list= glob('../Project01_data/normal_data/sprite/*.jpg')
print(len(sprite_list))   # 212
tejava_list = glob('../Project01_data/normal_data/tejava/*.jpg')
print(len(tejava_list))   # 212

import random
import numpy as np
import cv2 as cv
#################### 

name_list = [coke_list, fanta_list, letsbee_list, pocari_list, sprite_list, tejava_list]
name = ['cocacola','fanta','letsbee','pocari','sprite','tejava']

name_list = [pocari_list, fanta_list, letsbee_list, sprite_list, tejava_list]
name = ['pocari','fanta','letsbee','sprite','tejava']

def labeling (image_list, label, data_list, label_list) :
    for i in image_list :
        try :
            img = cv.imread(i)
            img = cv.resize(img,(64,64))
            # cv.imshow("image", img)
            # cv.waitKey(0)
            img=np.array(img)/255.
            data_list.append(img)
            label_list.append(label)
        except :
            pass
    return data_list, label_list


index = 0
for std in name_list :
    print("index : ", index)
    print("standard : ", name[index])
    
    # 기준이 되는 음료수 데이터를 넣는다.
    std_data = list()
    std_label = list()
    
    copy_list = name_list.copy()
    copy_name = name.copy()

    std_data, std_label = labeling (std, 0, std_data, std_label)     # labeling 0 : 기준이 되는 음료수와 동일하다.

    std_data = np.array(std_data)
    std_label = np.array(std_label)


    # 기준 음료수 외 나머지를 하나의 리스트에 넣는다.
    copy_list.remove(std)               
    copy_name.remove(copy_name[index])  

    dif_data = list()
    dif_label = list()

    for dif in copy_list :
        dif = random.sample(dif, len(dif)//5)                        # 랜덤 추출 : 기준이 아닌 음료수에서는 각각 20% 개수씩 가져온다.
        dif_data, dif_label = labeling (dif, 1, dif_data, dif_label) # labeling 1: 기준이 되는 음료수와 다르다.

    dif_data = np.array(dif_data)
    dif_label = np.array(dif_label)

    # 기준 데이터의 자동 변수 생성
    globals()['std_data_{}'.format(name[index])] = std_data
    globals()['std_label_{}'.format(name[index])] = std_label

    # 비교 데이터의 자동 변수 생성
    globals()['dif_data_{}'.format(name[index])] = dif_data
    globals()['dif_label_{}'.format(name[index])] = dif_label

    index += 1

    # print("std data", len(std_data))
    # print("dif data", len(dif_data))

print("==========자동 변수 생성 확인=========")
print(" std_data|std_label|dif_data|dif_label")
print(len(std_data_cocacola),len(std_label_cocacola),len(dif_data_cocacola),len(dif_label_cocacola))
print(len(std_data_fanta),len(std_label_fanta),len(dif_data_fanta),len(dif_label_fanta))
print(len(std_data_letsbee),len(std_label_letsbee),len(dif_data_letsbee),len(dif_label_letsbee))
print(len(std_data_pocari),len(std_label_pocari),len(dif_data_pocari),len(dif_label_pocari))
print(len(std_data_sprite),len(std_label_sprite),len(dif_data_sprite),len(dif_label_sprite))
print(len(std_data_tejava),len(std_label_tejava),len(dif_data_tejava),len(dif_label_tejava))

# 209 209 210 210
# 209 209 208 208
# 212 212 208 208
# 211 211 209 209
# 211 211 207 207
# 212 212 210 210


# np.save =====================================================================

print("np.save start>>>>>>>>>>>>>>>>>>>>>")
np.save('../Project01_data/9.npy/cv_std_data_cocacola', arr=std_data_cocacola)
np.save('../Project01_data/9.npy/cv_std_label_cocacola', arr=std_label_cocacola)
np.save('../Project01_data/9.npy/cv_dif_data_cocacola', arr=dif_data_cocacola)
np.save('../Project01_data/9.npy/cv_dif_label_cocacola', arr=dif_label_cocacola)

np.save('../Project01_data/9.npy/std_data_fanta', arr=std_data_fanta)
np.save('../Project01_data/9.npy/std_label_fanta', arr=std_label_fanta)
np.save('../Project01_data/9.npy/dif_data_fanta', arr=dif_data_fanta)
np.save('../Project01_data/9.npy/dif_label_fanta', arr=dif_label_fanta)

np.save('../Project01_data/9.npy/std_data_letsbee', arr=std_data_letsbee)
np.save('../Project01_data/9.npy/std_label_letsbee', arr=std_label_letsbee)
np.save('../Project01_data/9.npy/dif_data_letsbee', arr=dif_data_letsbee)
np.save('../Project01_data/9.npy/dif_label_letsbee', arr=dif_label_letsbee)

np.save('../Project01_data/9.npy/std_data_pocari', arr=std_data_pocari)
np.save('../Project01_data/9.npy/std_label_pocari', arr=std_label_pocari)
np.save('../Project01_data/9.npy/dif_data_pocari', arr=dif_data_pocari)
np.save('../Project01_data/9.npy/dif_label_pocari', arr=dif_label_pocari)

np.save('../Project01_data/9.npy/std_data_sprite', arr=std_data_sprite)
np.save('../Project01_data/9.npy/std_label_sprite', arr=std_label_sprite)
np.save('../Project01_data/9.npy/dif_data_sprite', arr=dif_data_sprite)
np.save('../Project01_data/9.npy/dif_label_sprite', arr=dif_label_sprite)

np.save('../Project01_data/9.npy/std_data_tejava', arr=std_data_tejava)
np.save('../Project01_data/9.npy/std_label_tejava', arr=std_label_tejava)
np.save('../Project01_data/9.npy/dif_data_tejava', arr=dif_data_tejava)
np.save('../Project01_data/9.npy/dif_label_tejava', arr=dif_label_tejava)
print("np.save end<<<<<<<<<<<<<<<<<<<<<<<")

# np.save =====================================================================


'''
# 잘 저장되어 있는지 확인
# np.load =====================================================================
print("np.load start>>>>>>>>>>>>>>>>>>")
std_data_cocacola = np.load('../Project01_data/9.npy/cv_std_data_cocacola.npy')
std_label_cocacola = np.load('../Project01_data/9.npy/cv_std_label_cocacola.npy')
dif_data_cocacola = np.load('../Project01_data/9.npy/cv_dif_data_cocacola.npy')
dif_label_cocacola = np.load('../Project01_data/9.npy/cv_dif_label_cocacola.npy')

std_data_fanta = np.load('../Project01_data/9.npy/std_data_fanta.npy')
std_label_fanta = np.load('../Project01_data/9.npy/std_label_fanta.npy')
dif_data_fanta = np.load('../Project01_data/9.npy/dif_data_fanta.npy')
dif_label_fanta = np.load('../Project01_data/9.npy/dif_label_fanta.npy')

std_data_letsbee = np.load('../Project01_data/9.npy/std_data_letsbee.npy')
std_label_letsbee = np.load('../Project01_data/9.npy/std_label_letsbee.npy')
dif_data_letsbee = np.load('../Project01_data/9.npy/dif_data_letsbee.npy')
dif_label_letsbee = np.load('../Project01_data/9.npy/dif_label_letsbee.npy')

std_data_pocari = np.load('../Project01_data/9.npy/std_data_pocari.npy')
std_label_pocari = np.load('../Project01_data/9.npy/std_label_pocari.npy')
dif_data_pocari = np.load('../Project01_data/9.npy/dif_data_pocari.npy')
dif_label_pocari = np.load('../Project01_data/9.npy/dif_label_pocari.npy')

std_data_sprite = np.load('../Project01_data/9.npy/std_data_sprite.npy')
std_label_sprite = np.load('../Project01_data/9.npy/std_label_sprite.npy')
dif_data_sprite = np.load('../Project01_data/9.npy/dif_data_sprite.npy')
dif_label_sprite = np.load('../Project01_data/9.npy/dif_label_sprite.npy')

std_data_tejava = np.load('../Project01_data/9.npy/std_data_tejava.npy')
std_label_tejava = np.load('../Project01_data/9.npy/std_label_tejava.npy')
dif_data_tejava = np.load('../Project01_data/9.npy/dif_data_tejava.npy')
dif_label_tejava = np.load('../Project01_data/9.npy/dif_label_tejava.npy')
print("np.load end<<<<<<<<<<<<<<<<<<<<<<<")
# np.load =====================================================================

# print(std_data_cocacola)
print(std_data_cocacola.shape)   # (209, 64, 64, 3)
print(std_label_cocacola.shape)  # (209,)
print(dif_data_cocacola.shape)   # (210, 64, 64, 3)
print(dif_label_cocacola.shape)  # (210,)
'''