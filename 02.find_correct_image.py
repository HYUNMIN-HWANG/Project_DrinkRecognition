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


# test_list = ['A','B','C']
# test_list.remove(test_list[0])
# print(test_list)


# 참고사이트 : https://buillee.tistory.com/111
# https://github.com/skanwngud/Project/blob/main/mask/human_nohuman.py


coke_list=glob('../Project01_data/normal_data/cocacola/*.jpg')
print(len(coke_list))   # 212
fanta_list=glob('../Project01_data/normal_data/fanta/*.jpg')
print(len(fanta_list))   # 100
letsbee_list=glob('../Project01_data/normal_data/letsbee/*.jpg')
print(len(letsbee_list))   # 212
pocari_list = glob('../Project01_data/normal_data/pocari/*.jpg')
print(len(pocari_list))    # 212
sprite_list= glob('../Project01_data/normal_data/sprite/*.jpg')
print(len(sprite_list))   # 212
tejava_list = glob('../Project01_data/normal_data/tejava/*.jpg')
print(len(tejava_list))   # 212

'''
name_list = [coke_list, fanta_list, letsbee_list, pocari_list, sprite_list, tejava_list]
name = ['cocacola','fanta','letsbee','pocari','sprite','tejava']

def labeling (image_list, label, data_list, label_list) :
    for i in image_list :
        try :
            # print(i)
            img = tf.keras.preprocessing.image.load_img(i,
            color_mode='rgb',
            target_size=(128, 128))
            img=np.array(img)/255.
            data_list.append(img)
            label_list.append(label)
        except :
            pass
    return data_list, label_list

index = 0
for std in name_list :
    print("index : ", index)
    
    # 기준이 되는 음료수 데이터를 넣는다.
    std_data = list()
    std_label = list()
    
    copy_list = name_list.copy()
    copy_name = name.copy()
    # print(copy_list)
    # print(std)                        # 기준이 되는 음료수
    std_data, std_label = labeling (std, 0, std_data, std_label)     # labeling 0 : 기준이 되는 음료수와 동일하다.
    # print(std_data[0])
    # print(std_data.type())
    # print(std_label[:5])
    # print(std[0], std_label[:5])
    std_data = np.array(std_data)
    std_label = np.array(std_label)

    # 기준 데이터의 변수 자동 생성
    globals()['std_data_{}'.format(name[index])] = std_data
    globals()['std_label_{}'.format(name[index])] = std_label
    # print(std_data_cocacola)

    # 기준이 되는 음료수를 제외한, 나머지 데이터를 하나의 리스트에 넣는다.
    copy_list.remove(std)               
    copy_name.remove(copy_name[index])  
    # print(copy_list)
    # print(copy_name)             
    dif_data = list()
    dif_label = list()
    # copy_index = 0
    for dif in copy_list :
        dif_data, dif_label = labeling (dif, 1, dif_data, dif_label) # labeling 1: 기준이 되는 음료수와 동일하다.
        # print(len(dif_data))
        # print(len(dif_label))
        # copy_index += 1
    # print(dif_data[0], dif_label[:5])
    dif_data = np.array(dif_data)
    dif_label = np.array(dif_label)

    # 비교 데이터의 변수 자동 생성
    globals()['dif_data_{}'.format(name[index])] = dif_data
    globals()['dif_label_{}'.format(name[index])] = dif_label

    print("std data", len(std_data))
    print("dif data", len(dif_data))

    index += 1

print("==========자동 변수 생성 확인=========")
print(" std_data|std_label|dif_data|dif_label")
print(len(std_data_cocacola),len(std_label_cocacola),len(dif_data_cocacola),len(dif_label_cocacola))
# print(len(std_data_fanta),len(std_label_fanta),len(dif_data_fanta),len(dif_label_fanta))
# print(len(std_data_letsbee),len(std_label_letsbee),len(dif_data_letsbee),len(dif_label_letsbee))
# print(len(std_data_pocari),len(std_label_pocari),len(dif_data_pocari),len(dif_label_pocari))
# print(len(std_data_sprite),len(std_label_sprite),len(dif_data_sprite),len(dif_label_sprite))
# print(len(std_data_tejava),len(std_label_tejava),len(dif_data_tejava),len(dif_label_tejava))

# np.save >> for문 만들어봐
np.save('../Project01_data/9.npy/std_data_cocacola.npy',arr=std_data_cocacola)
np.save('../Project01_data/9.npy/std_label_cocacola.npy',arr=std_label_cocacola)
np.save('../Project01_data/9.npy/dif_data_cocacola.npy',arr=dif_data_cocacola)
np.save('../Project01_data/9.npy/dif_label_cocacola.npy',arr=dif_label_cocacola)


# np.load >> for문 만들어봐
std_data_cocacola = np.load('../Project01_data/9.npy/std_data_cocacola.npy')
std_label_cocacola = np.load('../Project01_data/9.npy/std_label_cocacola.npy')
dif_data_cocacola = np.load('../Project01_data/9.npy/dif_data_cocacola.npy')
dif_label_cocacola = np.load('../Project01_data/9.npy/dif_label_cocacola.npy')

# print(std_data_cocacola)
print(std_data_cocacola.shape)   # (209, 128, 128, 3)
print(std_label_cocacola.shape)  # (209, )
print(dif_data_cocacola.shape)   # (943, 128, 128, 3)
print(dif_label_cocacola.shape)  # (943,)

# Deep Learning Modeling strat >> for문 만들어봐

#1 DATA
train_datagen = ImageDataGenerator(
    # rescale=1./255,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rotation_range=5,
    zoom_range=0.5,
    fill_mode='nearest',
)

# test_datagen = ImageDataGenerator(rescale=1./255)
etc_datagen = ImageDataGenerator()

# batch = 2000

# train = train_datagen.flow_from_directory(
#     '../Project01_data/',
#     target_size=(56,56),
#     batch_size=batch,
#     class_mode = 'categorical', # 6 종류
#     subset="training"   
# )
# # Found 1584 images belonging to 6 classes.

# valid = train_datagen.flow_from_directory(
#     '../Project01_data/',
#     target_size=(56,56),
#     batch_size=batch,
#     class_mode = 'categorical', # 6 종류
#     subset="validation"
# )
# Found 392 images belonging to 6 classes.


# print(train)
# <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x00000200C3BC8550>

# print(train[0][0].shape)    # x
# print(train[0][1].shape)    # y

# print(valid[0][0].shape)     # x
# print(valid[0][1].shape)     # y

# 전체 데이터셋
x = np.concatenate((std_data_cocacola,dif_data_cocacola),axis=0)    
y = np.concatenate((std_label_cocacola,dif_label_cocacola),axis=0) 
print(x.shape, y.shape)  # (1152, 128, 128, 3) (1152,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, shuffle=True, random_state=42)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, train_size=0.95, shuffle=True, random_state=42)

train_generator = train_datagen.flow(x_train, y_train, batch_size=16)
test_generator = etc_datagen.flow(x_test, y_test, batch_size=16)
valid_generator = etc_datagen.flow(x_valid, y_valid)

print(x_train.shape, x_test.shape, x_valid.shape)  # (984, 128, 128, 3) (116, 128, 128, 3) (52, 128, 128, 3)
print(y_train.shape, y_test.shape, y_valid.shape)  # (984,) (116,) (52,)

# np.save
np.save('../Project01_data/9.npy/cocacola_x_train.npy',arr=x_train)
np.save('../Project01_data/9.npy/cocacola_x_test.npy',arr=x_test)
np.save('../Project01_data/9.npy/cocacola_x_valid.npy',arr=x_valid)
np.save('../Project01_data/9.npy/cocacola_y_train.npy',arr=y_train)
np.save('../Project01_data/9.npy/cocacola_y_test.npy',arr=y_test)
np.save('../Project01_data/9.npy/cocacola_y_valid.npy',arr=y_valid)
'''

# np.load
x_train = np.load('../Project01_data/9.npy/cocacola_x_train.npy')
x_test = np.load('../Project01_data/9.npy/cocacola_x_test.npy')
x_valid = np.load('../Project01_data/9.npy/cocacola_x_valid.npy')
y_train = np.load('../Project01_data/9.npy/cocacola_y_train.npy')
y_test = np.load('../Project01_data/9.npy/cocacola_y_test.npy')
y_valid = np.load('../Project01_data/9.npy/cocacola_y_valid.npy')

#2 Modeling
def modeling() :
    model = Sequential()
    model.add(Conv2D(32, (2,2), padding='same', activation='relu', input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3])))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (2,2), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (2,2), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (2,2), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (2,2), padding='same', activation='relu'))
    model.add(BatchNormalization())
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
path = '../Project01_data/9.cp/find_cocacola_{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(path, monitor='val_loss',mode='min', save_best_only=True)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(
    x_train, y_train, steps_per_epoch=8, epochs=100, validation_data=(x_valid, y_valid), validation_steps=4, callbacks=[es, lr,cp]
)

#4 Evaluate, Predict
loss, acc = model.evaluate(x_test, y_test)
print("loss : ", loss)
print("acc : ", acc)

'''
y_pred = model.predict(valid, steps = 5)
np.set_printoptions(formatter = {"float":lambda x: "{0:0.3f}".format(x)})
print(valid.class_indices)
print(output)
'''