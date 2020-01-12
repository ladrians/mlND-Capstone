# Machine Learning Engineer Nanodegree
## Capstone Project
Luciano Silveira  
January, 2020

## I. Proposal

Refer to the proposal [here](../proposal/proposal.md).

### Domain Background

Machine Learning is starting to touch all aspects of our life, in particular it has a lot of potential for automation in robotic applications.

Being part of the [donkeycar2 robocar community](http://www.donkeycar.com/), an opensource do it yourself self-driving platform for small scale cars; there are many possibilities for exploration in that space.

In this context, the exploration and experimentation has been done mainly with `supervised learning` and the community is starting to play around with `reinforcement learning` (RL) algorithms.

My objective is to implement a classification model for the robocar to detect where it is in a specific race track so the model can be used for other high level models. In particular, it would be useful for `reinforcement learning` and to explore how to autonomously navigate a track withing the specified lane.

The objective of the project is to implement a classification system for a robotic application.

It builds an inference engine extracting information from a race track.

The data was collected using a [donkeycar2 robocar](http://www.donkeycar.com/) following a standard track.

### Problem Statement

In the last few year the development of robotics applications using Machine Learning framework has exploded. This project explores the usage of different Machine Learning models to be used as a inference engine for classification; a classifier to detect a robot within a track and the possible actions to do: go `Straight`, turn `Left` or `Right`. That is to say identify 3 possible classes:

 * Straight or Center of the track
 * Left side of the track
 * Right side of the track

## II. Analysis

### Datasets and Inputs

For this phase a robocar was run to follow a standard race track in recording mode. After a few laps, it recorded thousand of 160x120 RGB images. The sample track is as follows:

![Sample Track](./images/sample_track.JPG)

The training and test data was manually classified and organized in the following classes: `Straight`, `Left`, `Right`; the folder structure:

```
data/
├── Left/
│   ├── 47_cam-image_array_.jpg
│   ├── ...
│   └── 48_cam-image_array_.jpg
├── Right/
│   ├── 46_cam-image_array_.jpg
│   ├── ...
│   └── 47_cam-image_array_.jpg
└── Straight/
    ├── 25_cam-image_array_.jpg
    ├── ...
    └── 26_cam-image_array_.jpg
```

The images distribution by class is:

 * Left: 702 images
 * Right: 719 images
 * Straight: 841 images

Some examples:

** Right **

![Right 1](./images/right_sample01.jpg)
![Right 2](./images/right_sample02.jpg)

** Left **

![Left 1](./images/left_sample01.jpg)
![Left 2](./images/left_sample02.jpg)

** Straight **

![Straight 1](./images/center_sample01.jpg)
![Straight 2](./images/center_sample02.jpg)

Classes by definition are not balanced, in general when normally driving there will be more samples from `Straight` than the `Left`, `Right`; special emphasis must be done to get more data from the missing classes. In particular, our track is almost an oval so it is important to note that we needed to run the robocar in two ways to get equal number of `right` and `left` turns.

Some examples of the type of images from the simulator (Left, Straight, Right):

![Left](../data/left_sample.jpg)
![Straight](../data/center_sample.jpg)
![Right](../data/right_sample.jpg)

All data is within the ![data](./data/) folder.

## III. Methodology

The project was divided in the following steps:

* Data gathering and manual classification
* Framework exploration
* Training and validation
* Summarization of the results

### Data Preprocessing

Manually classify the images in 3 classes: `Straight`, `Left`, `Right` and divide the resultset into two groups: training and valudation.

Augument data to get more information for the training process to avoiding overfitting. Some ideas to explore are:

 * grayscale
 * Horizontal flip of the image
 * Augmentation: Generate random changes on images, such as modification of contranst and coloring to have an effect in the lighting conditions.
 * Cropping
 * Normalization

Detail on this exploration can be chacked on the [DataManagement](./DataManagement.ipynb) jupyter notebook.
 
Some links on these steps:

* [How to prepare/augment images for neural network?](https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network)
* [Introduction to Dataset Augmentation and Expansion](https://algorithmia.com/blog/introduction-to-dataset-augmentation-and-expansion)
* [AutoML for Data Augmentation](https://blog.insightdatascience.com/automl-for-data-augmentation-e87cf692c366)

### Data Exploration

The following analysis was done on the images.

In general, for the problem we are trying to solve only the bottom part of the image is relevant for classification, so we select the region of interest

![ROI](./images/region_of_interest.png)

and cropped the image to get only the neccesary bits.

![ROI cropped](./images/crop_region_of_interest.png)

An alternative analyisis is to change the image to grayscale to remove the 3 RGB channels and only keep one channel:

![ROI](./images/grayscale.png)

Plus some analysis on edge detection using [Canny Edge detection](https://en.wikipedia.org/wiki/Canny_edge_detector)

![ROI](./images/canny_threshold.png)

and thesholding it to find the continuous relevant edges. Finally the selection is to keep the 3 RGB channels but just center on the selected region of interest.

This section was valiadted with the [Data Management](DataManagement.ipynb) jupyter notebook.

#### Input Image Structure

The image provided by the simulator and in runtime is a RGB encoded height:120px and width:160.

As the camera is in a fixed position and the higher part of it is useless for the classification task, it is cropped the upper part of it and just feed the network with a smaller Image containing only lane information. For example (Straight cropped image):

![Center_crop](../data/center_crop_sample.jpg)

### Framework Exploration

Based on the [Build an Image Dataset in TensorFlow](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/5_DataManagement/build_an_image_dataset.py) article, the following code was developed to evaluate different training alternatives.

For more information on the exploration, check the [Classification And Evaluation](ClassificationAndEvaluation.ipynb) jupyter notebook.

### Refinement

The following modes were created

 * Version 1
 * Version 2
 * Version 3

#### Version #1

This is the initial version for a classifier, the objective was to create a working pipeline, reading the correct images, train a classifier and get initial evaluation results.

The result of the training is detailed as follows:

![model1_loss](./images/model1_loss.png)

And validation of information was created to check the associated result

![model1_sample](./images/model1_sample.png)

The image details that out of 4 samples; 2 were correctly classified.

#### Version #2

For the following iteration, the objective was to use a similar CNN from the [robocars project](https://github.com/autorope/donkeycar/blob/dev/donkeycar/parts/keras.py), change the output to be a classifier.

```python
#Class output
class_out = Dense(class_count, activation='softmax', name='class_out')(x)        # 3 categories and find best one based off percentage 0.0-1.0

model = Model(inputs=[img_in], outputs=[class_out])
model.compile(optimizer='adam',
			  loss={'class_out': 'categorical_crossentropy'},
			  loss_weights={'class_out': 0.1},
			  metrics=['accuracy'])
return model
```

The result of the training is detailed as follows:

![model2_loss](./images/model2_loss.png)

And better validation of results:

![model2_sample](./images/model2_sample.png)

#### Version #3

For the next iteration real experimentation took place. The following changes where applied to prevent overfitting the CNN:

 * Reduce the capacity of the network
 * Add weight regularization
 * Add Batch Normalization
 * Modify the dropout

Ideas taken from [Overfit and Underfit](https://github.com/tensorflow/docs/blob/master/site/en/r1/tutorials/keras/overfit_and_underfit.ipynb)

##### Dropout

A technique to reduce overfitting is to apply dropout to the CNN. It is a form of regularization that forces the weights in the network to take only small values, which makes the distribution of weight values more regular and the network can reduce overfitting on small training examples. Dropout is one of the regularization technique applied. The following values were tested, 0.3 was selected.

```python
drop = 0.3 #0.1 # 0.2 # 0.4
```

When appling 0.1 dropout to a certain layer, it randomly kills 10% of the output units in each training epoch. This means dropping out 10%, 20% or 40% of the output units randomly from the applied layer, at the ends makes a more robust network. The case was applied to different convolutions and fully-connected layers.

References on this section:

* [Build an Image Dataset in TensorFlow](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/5_DataManagement/build_an_image_dataset.py)
* [Image classification](https://www.tensorflow.org/tutorials/images/classification)

##### Region Of Interest

Based on the data exploration done in [DataManagement](./DataManagement.ipynb); it was trimmed the higher part of the image:

```python
x = Cropping2D(cropping=((40,0), (0,0)))(x)
```

##### Batch Normalization

Applied the ideas from [here](https://stackoverflow.com/questions/41847376/keras-model-to-json-error-rawunicodeescape-codec-cant-decode-bytes-in-posi)


```python
x = BatchNormalization(input_shape=default_shape)(x)
```

More information [here](https://github.com/autorope/notebooks/blob/master/notebooks/train%20on%20all%20data.ipynb)

##### Keras Improvements

Besides several improvements related to the Keras framework

 * Callbacks
 * Early stopping

The result of the training is detailed as follows:

![model3_loss](./images/model3_loss.png)

And better validation of results:

![model3_sample](./images/model3_sample.png)

## IV. Results

### Model Evaluation and Validation

### Justification

## V. Conclusion

### Reflection

### Improvement

-----------

### Links

 * [Donkeycar Github repository](https://github.com/autorope/donkeycar)
 * [Train autopilot docs](http://docs.donkeycar.com/guide/train_autopilot/)
 * [Reinforcement Learning](https://pathmind.com/wiki/deep-reinforcement-learning)
 * [watermark](https://www.watermarquee.com/watermark)
 * [Save and Restore Models](https://github.com/tensorflow/docs/blob/master/site/en/r1/tutorials/keras/save_and_restore_models.ipynb)
