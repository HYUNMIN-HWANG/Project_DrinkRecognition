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
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from keras import backend as K

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

train_datagen = ImageDataGenerator(
    width_shift_range=0.2,
    height_shift_range=0.2,
    rotation_range=5,
    zoom_range=0.5,
    fill_mode='nearest',
)

etc_datagen = ImageDataGenerator()

# 1. DATA
# np.load
x_train = np.load('../Project01_data/9.npy/cv_color_x_train_cocacola.npy')
x_test = np.load('../Project01_data/9.npy/cv_color_x_test_cocacola.npy')
x_valid = np.load('../Project01_data/9.npy/cv_color_x_valid_cocacola.npy')
y_train = np.load('../Project01_data/9.npy/cv_color_y_train_cocacola.npy')
y_test = np.load('../Project01_data/9.npy/cv_color_y_test_cocacola.npy')
y_valid = np.load('../Project01_data/9.npy/cv_color_y_valid_cocacola.npy')

print("x : ", x_train.shape, x_test.shape, x_valid.shape)  
print("y : ", y_train.shape, y_test.shape, y_valid.shape)

# x :  (341, 64, 64, 3) (43, 64, 64, 3) (38, 64, 64, 3)
# y :  (341,) (43,) (38,)

batch = 16
train_generator = train_datagen.flow(x_train, y_train, batch_size=batch)
test_generator = etc_datagen.flow(x_test, y_test, batch_size=batch)
valid_generator = etc_datagen.flow(x_valid, y_valid)

#2 Modeling
def modeling() :
    model = Sequential()
    model.add(Conv2D(32, 3, padding='same', activation='elu',\
         input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3])))
    model.add(BatchNormalization())
    model.add(Conv2D(32, 3, padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(2))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, 2, padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, 2, padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(3))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, 2, padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, 2, padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(2))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(128, activation='elu'))
    model.add(Dense(32, activation='elu'))
    model.add(Dense(1, activation='sigmoid'))

    return model

model = modeling()

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

#3 Compile, train
es = EarlyStopping(monitor='val_loss', patience=20, mode='min')
lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=10, mode='min')
path = '../Project01_data/9.cp/find_cocacola_{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(path, monitor='val_loss',mode='min', save_best_only=True)

model.compile(optimizer='adam', metrics=['acc', f1_m], loss='binary_crossentropy')
hist = model.fit_generator(
    train_generator, steps_per_epoch=len(x_train) // batch, epochs=100, \
        validation_data=valid_generator, callbacks=[es, lr, cp])

#4 Evaluate, Predict

loss, acc, f1_score = model.evaluate(test_generator)
print("loss : ", loss)
print("acc : ", acc)
print("f1_score : ", f1_score)

# loss :  0.0006527779041789472
# acc :  1.0
# f1_score :  1.0