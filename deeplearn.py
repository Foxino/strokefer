#alternative approach using keras

from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
import os


train_folder = 'dataset/trainK' #training data location
train_folder_test = 'dataset/validateK' #testing data location
epochs=30 #iterations through dataset


train_n = 0
validate_n = 0

for _, dirnames, filenames in os.walk(train_folder): # calculating the n of images to be used
    train_n += len(filenames)

for _, dirnames, filenames in os.walk(train_folder_test):
    validate_n += len(filenames)

n_class = len(next(os.walk(train_folder))[1]) # N of classes, dynamically calcualted from N of folders in train_folder dir
img_r, img_c = 48, 48
batch_size = 32

#validate before training
input("Found "+str(n_class)+" classes. "+str(train_n)+" training images, " + str(validate_n)+" test images, if this is correct press any key to continue...")

dataGen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=30,
                shear_range=0.3,
                zoom_range=0.3,
                width_shift_range=0.4,
                height_shift_range=0.4,
                horizontal_flip=True,
                fill_mode='nearest')

v_dataGen = ImageDataGenerator(rescale=1./255)

t_generator = dataGen.flow_from_directory(
                train_folder,
                color_mode='grayscale',
                target_size=(img_r,img_c),
                batch_size=batch_size,
                class_mode='categorical',
                shuffle=True
            )
v_generator = v_dataGen.flow_from_directory(
                train_folder_test,
                color_mode='grayscale',
                target_size=(img_r,img_c),
                batch_size=batch_size,
                class_mode='categorical',
                shuffle=True
            )

#creating the neural net

model = Sequential()

#first layer
model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal', input_shape=(img_r,img_c,1)))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal', input_shape=(img_r,img_c,1)))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

#second layer
model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

#third layer
model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

#fourth layer
model.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

#fifth layer
model.add(Flatten())
model.add(Dense(64,kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

#sixth layer
model.add(Dense(64,kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

#seventh layer
model.add(Dense(n_class, kernel_initializer='he_normal')) #output to 5 possible classes
model.add(Activation('softmax'))

print(model.summary())

#train model
from keras.optimizers import RMSprop, SGD, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

checkpoint = ModelCheckpoint('SmartAlarm.h5',monitor='val_loss',mode='min',save_best_only=True,verbose=1)

earlystop = EarlyStopping(monitor='val_loss',min_delta=0,patience=9,verbose=1,restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=3,verbose=1,min_delta=0.0001)

callbacks = [earlystop,checkpoint,reduce_lr]

#compile model
model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'])

history=model.fit_generator(t_generator,steps_per_epoch=train_n//batch_size,epochs=epochs,callbacks=callbacks,validation_data=v_generator,validation_steps=validate_n//batch_size)
