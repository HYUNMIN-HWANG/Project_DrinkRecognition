import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten, BatchNormalization, AveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from PIL import Image
import io
import os
from glob import glob
import warnings
warnings.filterwarnings(action='ignore')
import tensorflow as tf


# 참고사이트 : https://buillee.tistory.com/111
# https://github.com/skanwngud/Project/blob/main/mask/human_nohuman.py

# 이 모델로 갈 것임

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

train_datagen = ImageDataGenerator(
    # rescale=1./255,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rotation_range=5,
    zoom_range=0.5,
    fill_mode='nearest',
)

etc_datagen = ImageDataGenerator()

# 1. DATA
# np.load
x_train = np.load('../Project01_data/9.npy/color_x_train_sprite.npy')
x_test = np.load('../Project01_data/9.npy/color_x_test_sprite.npy')
x_valid = np.load('../Project01_data/9.npy/color_x_valid_sprite.npy')
y_train = np.load('../Project01_data/9.npy/color_y_train_sprite.npy')
y_test = np.load('../Project01_data/9.npy/color_y_test_sprite.npy')
y_valid = np.load('../Project01_data/9.npy/color_y_valid_sprite.npy')

print("x : ", x_train.shape, x_test.shape, x_valid.shape)  
print("y : ", y_train.shape, y_test.shape, y_valid.shape)

# x :  (339, 64, 64, 3) (42, 64, 64, 3) (38, 64, 64, 3)
# y :  (339,) (42,) (38,)

batch = 16
train_generator = train_datagen.flow(x_train, y_train, batch_size=batch)
test_generator = etc_datagen.flow(x_test, y_test, batch_size=batch)
valid_generator = etc_datagen.flow(x_valid, y_valid)

#2 Modeling
def modeling() :
    model = Sequential()
    model.add(Conv2D(32, (2,2), padding='same', activation='relu', input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3])))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(2,2))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (2,2), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (2,2), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(2,2))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (2,2), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (2,2), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(2,2))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model

model = modeling()

#3 Compile, train
es = EarlyStopping(monitor='val_loss', patience=20, mode='min')
lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=10, mode='min')
path = '../Project01_data/9.cp/find_sprite_{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(path, monitor='val_loss',mode='min', save_best_only=True)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit_generator(
    train_generator, steps_per_epoch=len(x_train) // batch, epochs=100, validation_data=valid_generator, callbacks=[es, lr, cp]
)

#4 Evaluate, Predict
loss, acc = model.evaluate(test_generator)
print("loss : ", loss)
print("acc : ", acc)
