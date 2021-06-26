from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import  Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array
from imutils import paths
import numpy as np
import random
import cv2
import os

##Creating a CNN Model
model = Sequential()
inputShape = (32, 32,3)
##First Convolution Layer
model.add(Conv2D(32, (5, 5), padding="same",input_shape=inputShape))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(rate=0.25))
##Second Convolution Layer
model.add(Conv2D(32, (5, 5), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(rate=0.25))
##Third Convolution Layer
model.add(Conv2D(64, (5, 5), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(rate=0.25))
##flattening the output
model.add(Flatten())
##adding Denser layer of 500 nodes
model.add(Dense(500))
model.add(Activation("relu"))
 ##softmax classifier
model.add(Dense(43))
model.add(Activation("softmax"))
model.summary()

data = []
labels = []
print("[INFO] loading images...")
img_dir=sorted(list(paths.list_images("dataset")))
random.shuffle(img_dir)
for i in img_dir:
        img = cv2.imread(i)
        img=cv2.resize(img, (32,32))
        img = img_to_array(img)
        data.append(img)
        lab=i.split(os.path.sep)[-2]
        labels.append(lab)
print(len(data))
print(len(labels))
print("[INFO] splitting datas for training...")
# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data,labels, test_size=0.25, random_state=42)
# convert the labels from integers to vectors
trainY = to_categorical(trainY, num_classes=43)
testY = to_categorical(testY, num_classes=43)
print("[INFO]  Training Started...")
print(len(trainY))
print(len(trainX))
print(np.array(trainY).shape)
print(np.array(trainX).shape)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
# training the model for 10 epochs
model.fit(np.array(trainX), np.array(trainY), batch_size=32, epochs=20, validation_data=(np.array(testX), np.array(testY)))
# serialize model to JSON
model_json = model.to_json()
with open("ch_model.json", "w") as json_file:
        json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("ch_model.h5")
print("[INFO] Saved model to disk")


