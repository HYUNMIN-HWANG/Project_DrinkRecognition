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
import cv2


# 참고사이트 : https://buillee.tistory.com/111
# https://github.com/skanwngud/Project/blob/main/mask/human_nohuman.py


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


#################### 

name_list = [coke_list, fanta_list, letsbee_list, pocari_list, sprite_list, tejava_list]
name = ['cocacola','fanta','letsbee','pocari','sprite','tejava']


# np.load =====================================================================
print("np.load start>>>>>>>>>>>>>>>>>>")
std_data_cocacola = np.load('../Project01_data/9.npy/gray_std_data_cocacola.npy')    # x
std_label_cocacola = np.load('../Project01_data/9.npy/gray_std_label_cocacola.npy')  # y
dif_data_cocacola = np.load('../Project01_data/9.npy/gray_dif_data_cocacola.npy')    # x
dif_label_cocacola = np.load('../Project01_data/9.npy/gray_dif_label_cocacola.npy')  # y

std_data_fanta = np.load('../Project01_data/9.npy/gray_std_data_fanta.npy')
std_label_fanta = np.load('../Project01_data/9.npy/gray_std_label_fanta.npy')
dif_data_fanta = np.load('../Project01_data/9.npy/gray_dif_data_fanta.npy')
dif_label_fanta = np.load('../Project01_data/9.npy/gray_dif_label_fanta.npy')

std_data_letsbee = np.load('../Project01_data/9.npy/gray_std_data_letsbee.npy')
std_label_letsbee = np.load('../Project01_data/9.npy/gray_std_label_letsbee.npy')
dif_data_letsbee = np.load('../Project01_data/9.npy/gray_dif_data_letsbee.npy')
dif_label_letsbee = np.load('../Project01_data/9.npy/gray_dif_label_letsbee.npy')

std_data_pocari = np.load('../Project01_data/9.npy/gray_std_data_pocari.npy')
std_label_pocari = np.load('../Project01_data/9.npy/gray_std_label_pocari.npy')
dif_data_pocari = np.load('../Project01_data/9.npy/gray_dif_data_pocari.npy')
dif_label_pocari = np.load('../Project01_data/9.npy/gray_dif_label_pocari.npy')

std_data_sprite = np.load('../Project01_data/9.npy/gray_std_data_sprite.npy')
std_label_sprite = np.load('../Project01_data/9.npy/gray_std_label_sprite.npy')
dif_data_sprite = np.load('../Project01_data/9.npy/gray_dif_data_sprite.npy')
dif_label_sprite = np.load('../Project01_data/9.npy/gray_dif_label_sprite.npy')

std_data_tejava = np.load('../Project01_data/9.npy/gray_std_data_tejava.npy')
std_label_tejava = np.load('../Project01_data/9.npy/gray_std_label_tejava.npy')
dif_data_tejava = np.load('../Project01_data/9.npy/gray_dif_data_tejava.npy')
dif_label_tejava = np.load('../Project01_data/9.npy/gray_dif_label_tejava.npy')
print("np.load end<<<<<<<<<<<<<<<<<<<<<<<")
# np.load =====================================================================

# print(std_data_cocacola)
print(std_data_cocacola.shape)   # (209, 64, 64, 3)
print(std_label_cocacola.shape)  # (209,)
print(dif_data_cocacola.shape)   # (1055, 64, 64, 3)
print(dif_label_cocacola.shape)  # (1055,)

std_data = [std_data_cocacola, std_data_fanta, std_data_letsbee, std_data_pocari, std_data_sprite, std_data_tejava]
std_label = [std_label_cocacola, std_label_fanta, std_label_letsbee, std_label_pocari, std_label_sprite, std_label_tejava]
dif_data = [dif_data_cocacola, dif_data_fanta, dif_data_letsbee, dif_data_pocari, dif_data_sprite, dif_data_tejava]
dif_label = [dif_label_cocacola, dif_label_fanta, dif_label_letsbee, dif_label_pocari, dif_label_sprite, dif_label_tejava]

####
# Deep Learning Modeling strat

#1 DATA

for i in range(0, 6) :  # i >> 0,1,2,3,4,5
    print(i)

    x = np.concatenate((std_data[i], dif_data[i]),axis=0)    
    y = np.concatenate((std_label[i], dif_label[i]),axis=0) 
    print(std_data[i].shape)
    print(dif_data[i].shape)
    x = x.reshape(-1, x.shape[1], x.shape[2], 1)
    print(x.shape)

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, shuffle=True, random_state=42)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, train_size=0.9, shuffle=True, random_state=42)

    print("x : ", x_train.shape, x_test.shape, x_valid.shape)  
    print("y : ", y_train.shape, y_test.shape, y_valid.shape)  

    # np.save
    path1 = '../Project01_data/9.npy/gray_x_train_' + str(name[i]) + '.npy'
    np.save(path1, arr=x_train)
    path2 = '../Project01_data/9.npy/gray_x_test_' + str(name[i]) + '.npy'
    np.save(path2, arr=x_test)
    path3 = '../Project01_data/9.npy/gray_x_valid_' + str(name[i]) + '.npy'
    np.save(path3, arr=x_valid)
    path4 = '../Project01_data/9.npy/gray_y_train_' + str(name[i]) + '.npy'
    np.save(path4, arr=y_train)
    path5 = '../Project01_data/9.npy/gray_y_test_' + str(name[i]) + '.npy'
    np.save(path5, arr=y_test)
    path6 = '../Project01_data/9.npy/gray_y_valid_' + str(name[i]) + '.npy'
    np.save(path6, arr=y_valid)


"""
<np.save shape 확인>
0
(209, 80, 80)
(208, 80, 80)
(417, 80, 80, 1)
x :  (337, 80, 80, 1) (42, 80, 80, 1) (38, 80, 80, 1)
y :  (337,) (42,) (38,)
1
(209, 80, 80)
(208, 80, 80)
(417, 80, 80, 1)
x :  (337, 80, 80, 1) (42, 80, 80, 1) (38, 80, 80, 1)
y :  (337,) (42,) (38,)
2
(212, 80, 80)
(208, 80, 80)
(420, 80, 80, 1)
x :  (340, 80, 80, 1) (42, 80, 80, 1) (38, 80, 80, 1)
y :  (340,) (42,) (38,)
3
(211, 80, 80)
(208, 80, 80)
(419, 80, 80, 1)
x :  (339, 80, 80, 1) (42, 80, 80, 1) (38, 80, 80, 1)
y :  (339,) (42,) (38,)
4
(211, 80, 80)
(209, 80, 80)
(420, 80, 80, 1)
x :  (340, 80, 80, 1) (42, 80, 80, 1) (38, 80, 80, 1)
y :  (340,) (42,) (38,)
5
(212, 80, 80)
(206, 80, 80)
(418, 80, 80, 1)
x :  (338, 80, 80, 1) (42, 80, 80, 1) (38, 80, 80, 1)
y :  (338,) (42,) (38,)
"""

'''
# <np.load 잘 돌아가는 지 확인> 
# np.load
x_train = np.load('../Project01_data/9.npy/gray_x_train_cocacola.npy')
x_test = np.load('../Project01_data/9.npy/gray_x_test_cocacola.npy')
x_valid = np.load('../Project01_data/9.npy/gray_x_valid_cocacola.npy')
y_train = np.load('../Project01_data/9.npy/gray_y_train_cocacola.npy')
y_test = np.load('../Project01_data/9.npy/gray_y_test_cocacola.npy')
y_valid = np.load('../Project01_data/9.npy/gray_y_valid_cocacola.npy')


print("x : ", x_train.shape, x_test.shape, x_valid.shape)  
print("y : ", y_train.shape, y_test.shape, y_valid.shape)

# x :  (337, 80, 80, 1) (42, 80, 80, 1) (38, 80, 80, 1)
# y :  (337,) (42,) (38,)
'''