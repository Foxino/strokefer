import sys
import os
import numpy as np
import pandas

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import np_utils


#initialize config
batch_size = 64
epochs = 40
w,h=48,48
output_file = "fer2013"
dataset = "fer2013"

#load dataset
print("... loading dataset")
ds = pandas.read_csv(f"{dataset}.csv")
n_classes = (ds['emotion'].max()+1)

print(f"... {n_classes} classes found")

#prep lists
train_data_img, train_data_lab, test_data_img, test_data_lab = [],[],[],[]

#populate lists
print("... populating training data")
for index, row in ds.iterrows():
    val=row['pixels'].split(" ")
    try:
        #training set
        if 'Training' in row['Usage']:
            train_data_img.append(np.array(val, 'float32'))
            train_data_lab.append(row['emotion'])
        #test/validation set
        elif 'PublicTest' in row['Usage']:
            test_data_img.append(np.array(val, 'float32'))
            test_data_lab.append(row['emotion'])
    except:
        print(f"[error] {index} - {row}")

#convert to NP array
train_data_img, train_data_lab = np.array(train_data_img,'float32'), np.array(train_data_lab,'float32')
test_data_img, test_data_lab = np.array(test_data_img,'float32'), np.array(test_data_lab,'float32')

train_data_lab=np_utils.to_categorical(train_data_lab, num_classes=n_classes)
test_data_lab=np_utils.to_categorical(test_data_lab, num_classes=n_classes)

#normalize data between 0 and 1
train_data_img -= np.mean(train_data_img, axis=0)
train_data_img /= np.std(train_data_img, axis=0)

test_data_img -= np.mean(test_data_img, axis=0)
test_data_img /= np.std(test_data_img, axis=0)

test_data_img = test_data_img.reshape(test_data_img.shape[0], w, h, 1)
train_data_img = train_data_img.reshape(train_data_img.shape[0], w, h, 1)

#keras cnn
model = Sequential()

# L1
model.add(Conv2D(64, kernel_size=(3,3), activation='relu', input_shape=(train_data_img.shape[1:])))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Dropout(0.5))

# L2
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Dropout(0.5))

# L3
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(Conv2D(128, (3,3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

# l4
model.add(Flatten())

model.add(Dense(1024,activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(1024,activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(n_classes,activation='softmax'))

model.compile(
    loss=categorical_crossentropy,
    optimizer=Adam(),
    metrics=['accuracy'])

model.fit(train_data_img, train_data_lab,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(test_data_img, test_data_lab),
          shuffle=True)

#save model
model_json=model.to_json()
with open(f"{output_file}.json","w") as json_file:
    json_file.write(model_json)
model.save_weights(f"{output_file}.h5")
