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
# def modeling(drop1=0.2, node1=128, node2=64, activation1='relu', activation2='relu') :
# def modeling(optimizer='adam', drop2=0.3, drop3=0.4, kernel1=2, kernel2=2, kernel3=2) :
# def modeling(pool1=2, pool2=2, pool3=2) :
    model = Sequential()
    model.add(Conv2D(32, kernel1, padding='same', activation=activation1, input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3])))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel1, padding='same', activation=activation1))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool1))
    model.add(Dropout(drop1))

    model.add(Conv2D(64, kernel2, padding='same', activation=activation1))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel2, padding='same', activation=activation1))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool2))
    model.add(Dropout(drop2))

    model.add(Conv2D(128, kernel3, padding='same', activation=activation1))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel3, padding='same', activation=activation1))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool3))
    model.add(Dropout(drop3))

    model.add(Flatten())
    model.add(Dense(node1, activation=activation2))
    model.add(Dense(node2, activation=activation2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=optimizer, metrics=['acc'], loss='binary_crossentropy')
    return model

def modeling() :
    model = Sequential()
    model.add(Conv2D(32, 3, padding='same', activation='elu', input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3])))
    model.add(BatchNormalization())
    model.add(Conv2D(32, 3, padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool1))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, 2, padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, 2, padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool2))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, 2, padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, 2, padding='same', activation='elu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool3))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(128, activation='elu'))
    model.add(Dense(32, activation='elu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', metrics=['acc'], loss='binary_crossentropy')
    return model

# model = modeling()

# hyperparameter 확인

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
model = KerasClassifier(build_fn=modeling, verbose=1)   

# # 1
# def create_hyperparameters() :
#     batches = [16, 32, 64]  # 16
#     dropout1 = [0.2, 0.3, 0.4]    # 0.2
#     node1 = [32, 64, 128]   # 128
#     node2 = [32, 64, 128]   # 32
#     activation1 =['relu','elu','prelu']   # elu
#     activation2 =['relu','elu','prelu']   # elu
#     return {"batche_size" : batches, "drop1" : dropout1, "node1" : node1, "node2" : node2, "activation1" : activation1, "activation2" : activation2}
# print("best_params : ", search.best_params_) 
# # best_params :  {'node2': 32, 'node1': 128, 'drop3': 0.2, 'drop2': 0.3, 'drop1': 0.2, 'batch_size': 16, 'activation2': 'elu', 'activation1': 'elu'}

# # 2
# def create_hyperparameters() :
#     optimizers = ['rmsprop', 'adam', 'adadelta','sgd'] # adam
#     dropout2 = [0.2, 0.3, 0.4]    # 0.3
#     dropout3 = [0.2, 0.3, 0.4]    # 0.2
#     kernel_size1 = [2, 3]   # 2
#     kernel_size2 = [2, 3]   # 3
#     kernel_size3 = [2, 3]   # 3
#     return {"optimizer" : optimizers, "drop2" : dropout2, "drop3" : dropout3, "kernel1" : kernel_size1, "kernel2" : kernel_size2, "kernel3" : kernel_size3}
# print("best_params : ", search.best_params_) 
# # best_params :  {'optimizer': 'adam', 'kernel3': 2, 'kernel2': 2, 'kernel1': 3, 'drop3': 0.3, 'drop2': 0.2}
# # best_score :  0.80619016289711
# # Score :  0.44186046719551086

# 3
def create_hyperparameters() :
    pool_size1 = [2,3]  # 2
    pool_size2 = [2,3]  # 3
    pool_size3 = [2,3]  # 2
    return {"pool1" : pool_size1, "pool2" : pool_size2, "pool3" : pool_size3}
# best_params :  {'pool3': 2, 'pool2': 3, 'pool1': 2}
# best_estimator :  <tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x000001FD51767820>
# best_score :  0.8150820881128311
# Score :  0.8372092843055725

# total
def create_hyperparameters() :
    batches = [16, 32, 64]  
    dropout1 = [0.2, 0.3, 0.4]    
    node1 = [32, 64, 128]   
    node2 = [32, 64, 128]   
    activation1 =['relu','elu','prelu']   
    activation2 =['relu','elu','prelu']  
    return {"batche_size" : batches, "drop1" : dropout1, "node1" : node1, "node2" : node2, "activation1" : activation1, "activation2" : activation2}
print("best_params : ", search.best_params_) 
# best_params :  {'node2': 32, 'node1': 128, 'drop3': 0.2, 'drop2': 0.3, 'drop1': 0.2, 'batch_size': 16, 'activation2': 'elu', 'activation1': 'elu'}

def create_hyperparameters() :
    optimizers = ['rmsprop', 'adam', 'adadelta','sgd'] 
    dropout2 = [0.2, 0.3, 0.4]   
    dropout3 = [0.2, 0.3, 0.4]    
    kernel_size1 = [2, 3]  
    kernel_size2 = [2, 3]  
    kernel_size3 = [2, 3] 
    return {"optimizer" : optimizers, "drop2" : dropout2, "drop3" : dropout3, "kernel1" : kernel_size1, "kernel2" : kernel_size2, "kernel3" : kernel_size3}
print("best_params : ", search.best_params_) 
# best_params :  {'optimizer': 'adam', 'kernel3': 2, 'kernel2': 2, 'kernel1': 3, 'drop3': 0.3, 'drop2': 0.2}

def create_hyperparameters() :
    pool_size1 = [2,3]  
    pool_size2 = [2,3]  
    pool_size3 = [2,3] 
    return {"pool1" : pool_size1, "pool2" : pool_size2, "pool3" : pool_size3}
print("best_params : ", search.best_params_) 
# best_params :  {'pool3': 2, 'pool2': 3, 'pool1': 2}

hyperparameters = create_hyperparameters()

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
search = RandomizedSearchCV(model, hyperparameters, cv=4)
search.fit(x_train, y_train, verbose=1)

print("best_params : ", search.best_params_) 
print("best_estimator : ", search.best_estimator_)  
print("best_score : ", search.best_score_)    
acc = search.score(x_test, y_test)
print("Score : ", acc)

