# **Behavioral Cloning** 
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./CarND-T1-P3.jpg "Architecture Diagram"
[image2]: ./writeup_data/IMG/center_2017_09_05_20_29_53_963.jpg "Center Driving"
[image3]: ./writeup_data/IMG/center_2017_09_05_22_15_58_509.jpg "Recovery Image"
[image4]: ./writeup_data/IMG/center_2017_09_05_22_16_16_554.jpg "Recovery Image"
[image5]: ./writeup_data/IMG/center_2017_09_05_22_16_17_187.jpg "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* nw.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.pdf summarizing the results
* process.sh containig commands to process the training images

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The nw.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.
The drive.py file was modify to run the predictions agains processed images (crop & resize, as in section 4).

### Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of three convolution neural network layers with a 4x4 filter and two 2x2 filters and depths of 16, 32 and 64 (model.py lines 108-112). The model includes multiple ELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 107). Also two dropout layers were included in order to reduce overfitting.

####2. Attempts to reduce overfitting in the model

As stated previously, the model contains two dropout layers in order to reduce overfitting (nw.py lines 114 and 117). 

The model was trained and validated on different data sets (80/20) to ensure that the model was not overfitting (code line 123 validation_split=0.2). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

Another way to generalize and avoid overfitting was to introduce random lighting by defining the `loadAndProcess` function (lines 20 to 27) which was applied for all training and validation images (left/center/right, lines 62/63/64)

There were also alot of datapoints with 0 (actually <0.2) angles which would count to more than half of the dataset so I decided to randomly remove about 50% of these points.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (nw.py line 122)

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road.
I used a combination of center lane driving, recovering from the left and right sides of the road by applying a +/- 0.35 degrees to relative to the center image corresponding angle. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with a very simple network until all the systems were working together: generating training data, processing the images, training the network, using the model to actually drive the car.

My first step was to use a very simple convolution neural network model. This was appropriate because I didn't need good results but it was fast enough to see that the whole `pipeline` is working together.

In order to improve the training speed and also to avoid network overfitting I croped the images 60 pixels from top to a new hight of 80 pixels (new size: 320x80). The top ~60 pixels if the image was usually containing sky & trees and the bottom ~20 pixels was mostly filled by the dashboard. The important information (road & curves) was in the middle. This new image was then resized to 80x80.
For the image processing I used imagemagick and generated static resized image files. I think this way it performed better on my personal laptop. I avoided to use Keras image generator because resizing the images every time I decided to retrain the network would have taken too much time.

I then implemented the same image processing (crop&resize with same dimensions) in the driving.py. 

After this, the whole system was working together: record training data, process the images, train & save the model, load the model & autonomous drivin. The results were poor (always getting off track).

####2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes:
* Lambda        - output 80x80x3, data normalization
* Convolution2D - output 20x20x16
* ELU           - output 20x20x16, exponential activation
* Convolution2D - output 10x10x32
* ELU           - output 10x10x32, exponential activation
* Convolution2D - output 5x5x64
* Flatten       - output 1600
* Dropout       - output 1600
* ELU           - output 1600, exponential activation
* Dense         - output 512
* Dropout       - output 512
* ELU           - output 512, exponential activation
* Dense         - output 1

Here is a visualization of the architecture

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded one lap on track one using center lane driving. I used the mouse and tried to turn as smoothly as possible (tried the keyboard but the results were very bad).

After that I recorded another lap on the other way around and merged the two datasets.

Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to apply higher angles to move back to the center of the lane. These images show what a recovery looks like starting from very close to the left side until back to center of the lane :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process locations where the car would go offtrack.

After the collection process, I had 41307 number of data points. I then preprocessed this data by removing half of the <0.2 angles, thus removing about 6605 samples.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 2 as evidenced by training vs validation loss. 
I used an adam optimizer so that manually training the learning rate wasn't necessary.
