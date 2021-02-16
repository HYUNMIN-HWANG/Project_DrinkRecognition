import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from PIL import Image

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
    class_mode = 'categorical', # # 6 종류
    subset="validation"
)
# Found 392 images belonging to 6 classes.


print(train)

print(train[0].shape)    # x
'''
print(train[0][1].shape)    # y

print(valid[0][0].shape)     # x
print(valid[0][1].shape)     # y


#2 Modeling
model = Sequential()
model.add(Conv2D(32, (2,2), padding='same', activation='relu', input_shape=(80, 80, 3)))
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
model.add(Dense(1, activation='sigmoid'))

#3 Compile, train
es = EarlyStopping(monitor='val_loss', patience=20, mode='min')
lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=10, mode='min')
path = '../'
cp = ModelCheckpoint(path, monitor='val_loss',mode='min', save_best_only=True)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit_generator(
    train, steps_per_epoch=8, epochs=100, validation_data=test, validation_steps=4, callbacks=[es, lr, cp]
)

#4 Evaluate, Predict
loss, acc = model.evaluate(test)
print("loss : ", loss)
print("acc : ", acc)

'''