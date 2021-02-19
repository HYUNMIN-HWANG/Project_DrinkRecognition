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
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

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
def modeling(optimizer='adam', drop1=0.2, drop2=0.3, drop3=0.4, node1=128, node2=64, activation1='relu', activation2='relu') :
    model = Sequential()
    model.add(Conv2D(32, (2,2), padding='same', activation=activation1, input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3])))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3,3), padding='same', activation=activation1))
    model.add(BatchNormalization())
    model.add(MaxPool2D(2,2))
    model.add(Dropout(drop1))

    model.add(Conv2D(64, (2,2), padding='same', activation=activation1))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (2,2), padding='same', activation=activation1))
    model.add(BatchNormalization())
    model.add(MaxPool2D(2,2))
    model.add(Dropout(drop2))

    model.add(Conv2D(128, (2,2), padding='same', activation=activation1))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (2,2), padding='same', activation=activation1))
    model.add(BatchNormalization())
    model.add(MaxPool2D(2,2))
    model.add(Dropout(drop3))

    model.add(Flatten())
    model.add(Dense(node1, activation=activation2))
    model.add(Dense(node2, activation=activation2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=optimizer, metrics=['acc'], loss='binary_crossentropy')
    return model

# model = modeling()

# hyperparameter 확인
model = KerasClassifier(build_fn=modeling, verbose=1)   

def create_hyperparameters() :
    batches = [16, 32, 64]
    optimizers = ['rmsprop', 'adam', 'adadelta','sgd']
    dropout1 = [0.2, 0.3, 0.4]
    dropout2 = [0.2, 0.3, 0.4]
    dropout3 = [0.2, 0.3, 0.4]
    node1 = [32, 64, 128]
    node2 = [32, 64, 128]
    activation1 =['relu','elu','prelu']
    activation2 =['relu','elu','prelu']
    return {"batch_size" : batches, "optimizer" : optimizers, \
        "drop1" : dropout1, "drop2" : dropout2, "drop3" : dropout3, "node1" : node1, "node2" : node2, "activation1" : activation1, "activation2" : activation2}

hyperparameters = create_hyperparameters()

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
search = RandomizedSearchCV(model, hyperparameters, cv=2)

print("best_params : ", search.best_params_) 
print("best_estimator : ", search.best_estimator_)  
print("best_score : ", search.best_score_)    
acc = search.score(x_test, y_test)
print("Score : ", acc)

# 돌려보자!!!!!!!!!!!!!!!