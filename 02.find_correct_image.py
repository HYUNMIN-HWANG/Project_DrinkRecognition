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
print(len(coke_list))   # 100
fanta_list=glob('../Project01_data/normal_data/fanta/*.jpg')
print(len(fanta_list))   # 100
letsbee_list=glob('../Project01_data/normal_data/letsbee/*.jpg')
print(len(letsbee_list))   # 100
pocari_list = glob('../Project01_data/normal_data/pocari/*.jpg')
print(len(pocari_list))    # 100
sprite_list= glob('../Project01_data/normal_data/sprite/*.jpg')
print(len(sprite_list))   # 100
tejava_list = glob('../Project01_data/normal_data/tejava/*.jpg')
print(len(tejava_list))   # 100

name_list = [coke_list, fanta_list, letsbee_list, pocari_list, sprite_list, tejava_list]
name = ['cocacola','fanta','letsbee',' pocari','sprite','tejava']

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
    
    # 기준이 되는 음료수 데이터를 넣는다.
    std_data = list()
    std_label = list()
    
    copy_list = name_list.copy()
    copy_name = name.copy()
    # print(copy_list)
    # print(std)                        # 기준이 되는 음료수
    std_data, std_label = labeling (std, 0, std_data, std_label)     # 0 : 기준이 되는 음료수와 동일하다.
    print(std_data[0])
    # print(std_data.type())
    # print(std_label[:5])
    print(std[0], std_label[:5])

    with open('../Project01_data/normal_data/' + str(name[index])+'/'+ str(name[index]) + '_normal.txt', 'w') as f :  # 이미지 경로들이 txt파일에 한 줄씩 저장
        f.write('\n'.join(std_data)+'\n')
        f.write('\n'.append(std_data)+'\n')

    # 기준이 되는 음료수를 제외한, 나머지 데이터를 하나의 리스트에 넣는다.
    copy_list.remove(std)               
    copy_name.remove(copy_name[index])  
    # print(copy_list)
    # print(copy_name)             
    dif_data = list()
    dif_label = list()
    copy_index = 0
    for dif in copy_list :
        dif_data, dif_label = labeling (dif, 1, dif_data, dif_label) # 1: 기준이 되는 음료수와 동일하다.
        # print(len(dif_data))
        # print(len(dif_label))
        with open('../Project01_data/normal_data/' + str(name[index])+'/'+ str(copy_name[copy_index]) + '_dif.txt', 'w') as f :  # 이미지 경로들이 txt파일에 한 줄씩 저장
            f.write('\n'.join(dif_data)+'\n')
        copy_index += 1
    # print(dif_data[0], dif_label[:5])
    print("std data", len(std_data))
    print("dif data", len(dif_data))
    index += 1



'''

coke_data, coke_label = labeling (coke_list, 0)
print(coke_data[0])
print(coke_label[:10])
fanta_data, fanta_label = labeling (fanta_list, 1)
print(fanta_data[0])
print(fanta_label[:10])
letsbee_data, letsbee_label = labeling (letsbee_list, 2)
print(letsbee_data[0])
print(letsbee_label[:10])
pocari_data, pocari_label = labeling (pocari_list, 3)
print(pocari_data[0])
print(pocari_label[:10])
sprite_data, sprite_label = labeling (sprite_list, 4)
print(sprite_data[0])
print(sprite_label[:10])
tejava_data, tejava_label = labeling (tejava_list, 5)
print(tejava_data[0])
print(tejava_label[:10])



#1 DATA
train_datagen = ImageDataGenerator(
    rescale=1./255,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rotation_range=5,
    zoom_range=0.5,
    fill_mode='nearest',
   validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

batch = 2000

train = train_datagen.flow_from_directory(
    '../Project01_data/',
    target_size=(56,56),
    batch_size=batch,
    class_mode = 'categorical', # 6 종류
    subset="training"   
)
# Found 1584 images belonging to 6 classes.

valid = train_datagen.flow_from_directory(
    '../Project01_data/',
    target_size=(56,56),
    batch_size=batch,
    class_mode = 'categorical', # 6 종류
    subset="validation"
)
# Found 392 images belonging to 6 classes.


# print(train)
# <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x00000200C3BC8550>

# print(train[0][0].shape)    # x
# print(train[0][1].shape)    # y

# print(valid[0][0].shape)     # x
# print(valid[0][1].shape)     # y



#2 Modeling
model = Sequential()
model.add(Conv2D(32, (2,2), padding='same', activation='relu', input_shape=(56, 56, 3)))
model.add(BatchNormalization())
model.add(Conv2D(32, (2,2), padding='same', activation='relu'))
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
model.add(Dense(6, activation='softmax'))

#3 Compile, train
es = EarlyStopping(monitor='val_loss', patience=20, mode='min')
lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=10, mode='min')
# path = '../'
# cp = ModelCheckpoint(path, monitor='val_loss',mode='min', save_best_only=True)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit_generator(
    train, steps_per_epoch=8, epochs=100, validation_data=valid, validation_steps=4, callbacks=[es, lr]
)

#4 Evaluate, Predict
loss, acc = model.evaluate(valid)
print("loss : ", loss)
print("acc : ", acc)

output = model.predict_generator(valid, steps = 5)
np.set_printoptions(formatter = {"float":lambda x: "{0:0.3f}".format(x)})
print(valid.class_indices)
print(output)
'''