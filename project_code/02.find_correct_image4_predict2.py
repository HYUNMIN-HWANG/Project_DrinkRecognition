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


# 참고사이트 : https://buillee.tistory.com/111
# https://github.com/skanwngud/Project/blob/main/mask/human_nohuman.py


# coke_list=glob('../Project01_data/normal_data/cocacola/*.jpg')
# print(len(coke_list))   # 212
# fanta_list=glob('../Project01_data/normal_data/fanta/*.jpg')
# print(len(fanta_list))   # 212
# letsbee_list=glob('../Project01_data/normal_data/letsbee/*.jpg')
# print(len(letsbee_list))   # 212
# pocari_list = glob('../Project01_data/normal_data/pocari/*.jpg')
# print(len(pocari_list))    # 212
# sprite_list= glob('../Project01_data/normal_data/sprite/*.jpg')
# print(len(sprite_list))   # 212
# tejava_list = glob('../Project01_data/normal_data/tejava/*.jpg')
# print(len(tejava_list))   # 212


# #################### 

# name_list = [coke_list, fanta_list, letsbee_list, pocari_list, sprite_list, tejava_list]
name = ['cocacola','fanta','letsbee','pocari','sprite','tejava']

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
x_train = np.load('../Project01_data/9.npy/gray_x_train_cocacola.npy')
x_test = np.load('../Project01_data/9.npy/gray_x_test_cocacola.npy')
x_valid = np.load('../Project01_data/9.npy/gray_x_valid_cocacola.npy')
y_train = np.load('../Project01_data/9.npy/gray_y_train_cocacola.npy')
y_test = np.load('../Project01_data/9.npy/gray_y_test_cocacola.npy')
y_valid = np.load('../Project01_data/9.npy/gray_y_valid_cocacola.npy')

print("x : ", x_train.shape, x_test.shape, x_valid.shape)  
print("y : ", y_train.shape, y_test.shape, y_valid.shape)

print(x_train.min(), x_train.max()) # 0.0 ~ 1.0

# x :  (1080, 64, 64, 3) (127, 64, 64, 3) (57, 64, 64, 3)
# y :  (1080,) (127,) (57,)

# print(y_train[:30]) # 0 또는 1 (0이 올바른 이미지, 1이 다른 이미지)


batch = 16
train_generator = train_datagen.flow(x_train, y_train, batch_size=batch)
test_generator = etc_datagen.flow(x_test, y_test, batch_size=batch)
valid_generator = etc_datagen.flow(x_valid, y_valid)

# x :  (1080, 64, 64, 3) (127, 64, 64, 3) (57, 64, 64, 3)
# y :  (1080,) (127,) (57,)

# #2 Modeling
# def modeling() :
#     model = Sequential()
#     model.add(Conv2D(32, (2,2), padding='same', activation='relu', input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3])))
#     model.add(BatchNormalization())
#     model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
#     model.add(BatchNormalization())
#     model.add(MaxPool2D(2,2))
#     model.add(Dropout(0.2))

#     model.add(Conv2D(64, (2,2), padding='same', activation='relu'))
#     model.add(BatchNormalization())
#     model.add(Conv2D(64, (2,2), padding='same', activation='relu'))
#     model.add(BatchNormalization())
#     model.add(MaxPool2D(2,2))
#     model.add(Dropout(0.3))

#     model.add(Conv2D(128, (2,2), padding='same', activation='relu'))
#     model.add(BatchNormalization())
#     model.add(Conv2D(128, (2,2), padding='same', activation='relu'))
#     model.add(BatchNormalization())
#     model.add(MaxPool2D(2,2))
#     model.add(Dropout(0.4))

#     model.add(Flatten())
#     model.add(Dense(128, activation='relu'))
#     model.add(Dense(64, activation='relu'))
#     model.add(Dense(1, activation='sigmoid'))
#     return model

# model = modeling()
model = load_model('../Project01_data/9.cp/gray_find_cocacola_0.0633.hdf5')

#3 Compile, train
# es = EarlyStopping(monitor='val_loss', patience=20, mode='min')
# lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=10, mode='min')
# path = '../Project01_data/9.cp/find_cocacola_{val_loss:.4f}.hdf5'
# cp = ModelCheckpoint(path, monitor='val_loss',mode='min', save_best_only=True)

# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
# hist = model.fit_generator(
#     train_generator, steps_per_epoch=len(x_train) // batch, epochs=100, validation_data=valid_generator, callbacks=[es, lr, cp]
# )

#4 Evaluate
loss, acc = model.evaluate(test_generator)
print("loss : ", loss)
print("acc : ", acc)

#5 Predict : Find correct image
# img1 = tf.keras.preprocessing.image.load_img('../Project01_data/cocacola/1.jpg', color_mode='grayscale', target_size=(80, 80)) # 원래 크기 : (936, 936, 3) # true
# img1 = tf.keras.preprocessing.image.load_img('../Project01_data/cocacola/67.jpg', color_mode='grayscale', target_size=(80, 80)) # 원래 크기 : (936, 936, 3) # true
# img1 = tf.keras.preprocessing.image.load_img('../Project01_data/cocacola/video_cocacola2_26.jpg', color_mode='grayscale', target_size=(80, 80)) # 원래 크기 : (936, 936, 3) # true
# img1 = tf.keras.preprocessing.image.load_img('../Project01_data/cocacola/196.jpg', color_mode='grayscale', target_size=(80, 80)) # 원래 크기 : (936, 936, 3) # true
im1 = img1.resize((x_train.shape[1], x_train.shape[2]))
im1 = np.array(im1)/255.
im1 = im1.reshape(-1, x_train.shape[1], x_train.shape[2],1)
print(im1.shape)    # (1, 80, 80, 1)
true = etc_datagen.flow(im1)

# img2 = tf.keras.preprocessing.image.load_img('../Project01_data/cocacola/60.jpg', color_mode='grayscale', target_size=(80, 80))  # false 펩시인데 이걸 코카콜라라고 인식을 하네?
img2 = tf.keras.preprocessing.image.load_img('../Project01_data/cocacola/244.jpg', color_mode='grayscale', target_size=(80, 80))  # false
# img2 = tf.keras.preprocessing.image.load_img('../Project01_data/cocacola/208.jpg', color_mode='grayscale', target_size=(80, 80))  # false
# img2 = tf.keras.preprocessing.image.load_img('../Project01_data/fanta/1.jpg', color_mode='grayscale', target_size=(80, 80))  # false
im2 = img2.resize((x_train.shape[1], x_train.shape[2]))
im2 = np.array(im2)/255.
im2 = im2.reshape(-1, x_train.shape[1], x_train.shape[2],1)
print(im2.shape)    # (1, 80, 80, 1)
false = etc_datagen.flow(im2)

true_pred = model.predict(true)
true_pred = np.where(true_pred[0][0]>0.5,1,0)
print(true_pred)

false_pred = model.predict(false)
false_pred = np.where(false_pred[0][0]>0.5,1,0)
print(false_pred)

# 0.0005312769 
# 0.037665863

# 0.007896089
# 0.99042237

# 0.010765203
# 0.95107865

# 0.09711467
# 0.95107865