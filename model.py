import csv
import cv2
import numpy as np 
import os
import csv
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

samples = []
lines = []
images = []
measurements = []

#DATA CAPTURING
with open('C:/Users/User/CarND-Behavioral Cloning/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=512):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for batch_sample in batch_samples:
                name = batch_sample[0].strip()
                nameL = batch_sample[1].strip()
                nameR = batch_sample[2].strip()
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                measurements.append(center_angle)

                #Basic Image
                lines.append(name)
                image = cv2.imread(name)
                images.append(image)
                measurement = float(batch_sample[3])
                measurements.append(measurement)
                #Flipped images
                measurement_flipped = -measurement
                image_flipped = np.fliplr(image)
                images.append(image_flipped)
                measurements.append(measurement_flipped)
                #Multiple Cameras
                steering_center = float(batch_sample[3])
                correction = 0.25
                steering_left = steering_center + correction
                steering_right = steering_center - correction
                image_left = cv2.imread(nameL)
                image_right = cv2.imread(nameR)
                measurements.append(steering_left)
                images.append(image_left)
                measurements.append(steering_right)
                images.append(image_right)
                #Flipped for multiple cameras
                measurement_flipped_left = - steering_left
                image_flipped_left = np.fliplr(image_left)
                images.append(image_flipped_left)
                measurements.append(measurement_flipped_left)
                measurement_flipped_right = - steering_right
                image_flipped_right = np.fliplr(image_right)
                images.append(image_flipped_right)
                measurements.append(measurement_flipped_right)
          
            X_train = np.array(images)
            y_train = np.array(measurements)
            yield shuffle(X_train, y_train)


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)        
        
#importing required libraries
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D #not used
from keras.models import Model
import matplotlib.pyplot as plt


#NVIDIA self-driving-cars network https://devblogs.nvidia.com/deep-learning-self-driving-cars/
model = Sequential()
model.add(Lambda( lambda x: x/255.0-0.5, input_shape=(160,320,3))) #normalization
model.add(Cropping2D(cropping=((70,25),(0,0)))) #cropping the top 70 pixels and bottom 25 

#Layer 1, 2 and 3: strided convolution with a 2x2 stride, 5x5 kernel and ReLu activation function 
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))

#Layer 4 and 5: non-strided convolution, 3x3 kernel and a ReLu activation function
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam') #means squared error and adam optimizer were used
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples),nb_epoch=10)
model.save('model.h5')
