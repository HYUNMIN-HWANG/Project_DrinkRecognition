import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten, BatchNormalization, AveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from PIL import Image
import io
import os
from glob import glob
import warnings
warnings.filterwarnings(action='ignore')
import tensorflow as tf
import shutil
import time
import cv2 


coke_list=glob('..//Project01_data//cocacola//*.jpg')
print(len(coke_list))   # 212
fanta_list=glob('..//Project01_data//fanta//*.jpg')
print(len(fanta_list))   # 212
letsbee_list=glob('..//Project01_data//letsbee//*.jpg')
print(len(letsbee_list))   # 212
pocari_list = glob('..//Project01_data//pocari//*.jpg')
print(len(pocari_list))    # 212
sprite_list= glob('..//Project01_data//sprite//*.jpg')
print(len(sprite_list))   # 212
tejava_list = glob('..//Project01_data//tejava//*.jpg')
print(len(tejava_list))   # 212

# total data (color)
# 493
# 509
# 334
# 629
# 481
# 633


#################### 

# name_list = [coke_list, fanta_list, letsbee_list, pocari_list, sprite_list, tejava_list]
# name = ['cocacola','fanta','letsbee','pocari','sprite','tejava']

name_list = [coke_list]
name = ['cocacola']

train_datagen = ImageDataGenerator(
    # rescale=1./255,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rotation_range=5,
    zoom_range=0.5,
    fill_mode='nearest',
)

etc_datagen = ImageDataGenerator()

# # 1. DATA
# np.load
x_train = np.load('../Project01_data/9.npy/color_x_train_cocacola.npy')
x_test = np.load('../Project01_data/9.npy/color_x_test_cocacola.npy')
x_valid = np.load('../Project01_data/9.npy/color_x_valid_cocacola.npy')
y_train = np.load('../Project01_data/9.npy/color_y_train_cocacola.npy')
y_test = np.load('../Project01_data/9.npy/color_y_test_cocacola.npy')
y_valid = np.load('../Project01_data/9.npy/color_y_valid_cocacola.npy')

print("x : ", x_train.shape, x_test.shape, x_valid.shape)  
print("y : ", y_train.shape, y_test.shape, y_valid.shape)

# x :  (339, 64, 64, 3) (42, 64, 64, 3) (38, 64, 64, 3)
# y :  (339,) (42,) (38,)

print(x_train.min(), x_train.max()) # 0.0 ~ 1.0

print(y_train[:30]) # 0 또는 1 (0이 올바른 이미지, 1이 다른 이미지)

batch = 16
train_generator = train_datagen.flow(x_train, y_train, batch_size=batch)
test_generator = etc_datagen.flow(x_test, y_test, batch_size=batch)
valid_generator = etc_datagen.flow(x_valid, y_valid)


model = load_model('../Project01_data/9.cp/find_cocacola_0.0000.hdf5')

#3 Compile, train

#4 Evaluate
loss, acc = model.evaluate(test_generator)
print("loss : ", loss)
print("acc : ", acc)

# loss :  0.0013317769626155496
# acc :  1.0

################

#5 Predict : Find correct image

# 예측할 이미지를 불러와 전처리하고 model.predict한다.
def prepro_n_predict (whole_list) : 
    result_list = list()
    n = 0
    for img in whole_list :
        try :
            # preprocessing
            print("검사할 이미지 ",img)
            copy_img = img
            img1 = tf.keras.preprocessing.image.load_img(copy_img, color_mode='rgb', target_size=(64,64)) 
            im1 = img1.resize((x_train.shape[1], x_train.shape[2]))
            im1 = np.array(im1)/255.
            im1 = im1.reshape(-1, x_train.shape[1], x_train.shape[2],3)
            print(im1.shape)                    # (1, 64, 64, 3)
            pre = etc_datagen.flow(im1)

            # predict
            pred = model.predict(pre)
            pred = np.where(pred[0][0]>0.5,1,0) # 0이면 올바른 이미지, 1이면 다른 이미지
            print(pred)

            # 0이면 다른 폴더에 사진 저장
            if pred == 0 :
                copy_path = '../Project01_data/sorting_correct/cocacola/'+str(n)+'.jpg'
                shutil.copy(img, copy_path)
                time.sleep(10)
                n += 1
            else :
                pass
        except : 
            pass
    # return result_list


print("전체 데이터 개수 ",len(coke_list))
prepro_n_predict(coke_list)
print(">>> copy end >>>")
    