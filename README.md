# Behavioral Cloning

## Overview

This project is part of my Self-Driving Car course. It uses neural network framework, Keras, to train a vehicle to drive around the track. It includes:

a. Model.py containing the script to create and train the model 
b. Drive.py for driving the car in autonomous mode 
c. Model.h5 containing a trained convolution neural network 
d. Video.mp4 with a recording of vehicle autonomously driving around the track. 

Model is trained and tested using simulator provided by Udacity. 

## Model Architecture and Training Strategy 

1. An appropriate model architecture has been employed.  My model replicates the network created by the Self-Driving Cars team at Nvidia (more information available here: https://devblogs.nvidia.com/deep-learning-self-driving-cars/). First data was normalized (Lambda), then images were cropped (70 pixels from the top and 25 from the bottom) so factors like the sky, environment and bonnet were not taken into consideration. Subsequently, three 2x2 strided convolution layers were implemented with depths 24-48 and kernel size of 5x5. Then two non-strided convolutional layers with depths of 64 were used. Each convolutional layers used ReLu activation function which introduces non-linearities.  
2. Attempts to reduce overfitting in the model. The model was trained and validated on different data sets to ensure that the model was not overfitting. Validation samples accounted for 0.2 of total gathered data. The rest was used in training.
3. Model Parameter tuning The parameters were tuned using adam optimizer and loss was calculated using means squared error.
4. Appropriate training data The training data kept the vehicle on the road. It involved centre lane driving, recovering from difficult turns, like the one with sand on the left side. The track was driven and recorded in clockwise and anti-clockwise. In more difficult patterns the speed was decreased to record more frames. If the trained model encountered difficulties in some parts of the track, the car was positioned in a respective part of the track with correct wheel angle and was left there ( speed = 0 ). The model was trained by using images and appropriate steering angles. Speed wasn’t involved. The vehicle’s speed in autonomous mode was controlled by ProportionalIntegral Controller available in drive.py. 
 
## Solution Design Approach 

The main strategy used in my model was to record the vehicle driving around the track and use the images to train the network. Each frame was saved and described in csv with a destination of each frame, velocity, steering angle and braking. For each position, three frames were saved. One in the centre, second with an offset a bit to the left and third a bit to the right. During the training mode, appropriate steering offset was introduced for off-centre frames. Moreover, flipped images were introduced to the model as well. This enabled to significantly reduce overfitting of the model. Only steering angle was used in training the model. As the size of data exceeded the capabilities of my laptop, the generator was used with a batch size of 512.  Using keras, the convolutional neural network was implemented (described above) which is used by Nvidia for self-driving cars. To gauge how well the model performs, the dataset was split into the training set (0.8) and validation set (0.2). The model was saved in model.h5 file. Subsequently, the drive.py file was run which fed the model to the Udacity Simulator in autonomous mode. The biggest challenge was the left-turn near the sand part where the vehicle could easily leave the track. In turned out that the reason for it was different color space used in drive.py and model.py. After it was corrected, the model behaved as expected. The final model is capable of safely driving around the track as it is documented in the video file attached to this submission. 


 


