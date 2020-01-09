# Machine Learning Engineer Nanodegree
## Capstone Project
Luciano Silveira  
January, 2020

## Proposal

Refer to the proposal [here](../proposal/proposal.md).

### Domain Background

Machine Learning is starting to touch all aspects of our life, in particular it has a lot of potential for automation in robotic applications.

Being part of the [donkeycar2 robocar community](http://www.donkeycar.com/), an opensource do it yourself self-driving platform for small scale cars; there are many possibilities for exploration in that space.

In this context, the exploration and experimentation has been done mainly with `supervised learning` and the community is starting to play around with `reinforcement learning` (RL) algorithms.

My objective is to implement a classification model for the robocar to detect where it is in a specific race track so the model can be used for other high level models. In particular, it would be useful for `reinforcement learning` and to explore how to autonomously navigate a track withing the specified lane.

The objective of the project is to implement a classification system for a robotic application.

It builds an inference engine extracting information from a race truck.

The data was collected using a [donkeycar2 robocar](http://www.donkeycar.com/) following a standard track.

### Problem Statement

In the last few year the development of robotics applications using Machine Learning framework has exploded. This project explores the usage of different Machine Learning models to be used as a inference engine for classification; a classifier to detect a robot within a track and the possible actions to do: go `Straight`, turn `Left` or `Right`. That is to say identify 3 possible classes:

 * Center of the track
 * Left side of the track
 * Right side of the track
 
### Datasets and Inputs

For this phase a robocar was run to follow a standard race track in recording mode. After a few laps, it recorded thousand of 160x120 RGB images. The sample track is as follows:

![Sample Track](./images/sample_track.JPG)

The training and test data was manually classified and organized in the following classes: `Center`, `Left`, `Right`.

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
└── Center/
    ├── 25_cam-image_array_.jpg
    ├── ...
    └── 26_cam-image_array_.jpg
```

The images distribution by class is:

 * Left: 702 images
 * Right: 719 images
 * Center: 841 images

The validation set including a [imagelist.txt](data/imagelist.txt) validation file.

Some examples:

** Right **

![Right 1](./images/right_sample01.jpg)
![Right 2](./images/right_sample02.jpg)

** Left **

![Left 1](./images/left_sample01.jpg)
![Left 2](./images/left_sample02.jpg)

** Center **

![Center 1](./images/center_sample01.jpg)
![Center 2](./images/center_sample02.jpg)

Classes by definition are not balanced, in general when normally driving there will be more samples from `Center` than the `Left`, `Right`; special emphasis must be done to get more data from the missing classes.

Some examples of the type of images from the simulator (Left, Center, Right):

![Left](../data/left_sample.jpg)
![Center](../data/center_sample.jpg)
![Right](../data/right_sample.jpg)

### Project Design

The project was divided in the following steps:

* Data gathering and manual classification
* Framework exploration
* Training and validation
* Summarization of the results

#### Data Management

Manually classify the images in 3 classes: `Center`, `Left`, `Right` and divide the resultset into two groups: training (70%) and testing (30%).

Augument data to get more information for the training process to avoiding overfitting. Some ideas to explore are:

 * grayscale
 * Horizontal flip of the image
 * Augmentation: Generate random changes on images, such as modification of contranst and coloring to have an effect in the lighting conditions.
 * Cropping
 * Normalization

Some links on these steps:

* [How to prepare/augment images for neural network?](https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network)
* [Introduction to Dataset Augmentation and Expansion](https://algorithmia.com/blog/introduction-to-dataset-augmentation-and-expansion)
* [AutoML for Data Augmentation](https://blog.insightdatascience.com/automl-for-data-augmentation-e87cf692c366)


##### Data Exploration

The following analysis was done on the images.

In general, for the problem we are trying to solve only the bottom part of the image is relevant for classification, so we select the region of interest

![ROI](./images/region_of_interest.png)

and cropped the image to get only the neccesary bits.

![ROI cropped](./images/crop_region_of_interest.png)

An alternative analyisis is to change the image to grayscale to remove the 3 RGB channels and only keep one channel:

![ROI](./images/grayscale.png)

Plus some analysis on edge detection using [Canny Edge detection](https://en.wikipedia.org/wiki/Canny_edge_detector)

![ROI](./images/canny_threshold.png)

and thesholding it to find the continuous relevant edges

![Sample track](./images/sample_track.JPG)

Finally the selection is to keep the 3 RGB channels but just center on the selected region of interest.

##### Input Image Structure

The image provided by the simulator and in runtime is a RGB encoded height:120px and width:160.

As the camera is in a fixed position and the higher part of it is useless for the classification task, it is cropped the upper part of it and just feed the network with a smaller Image containing only lane information. For example (Center cropped image):

![Center_crop](../data/center_crop_sample.jpg)

Only Image data will be feed to the CNN, some research is needed to check if RGB or just grayscale data is enough (3 vs 1 input dimension).

#### Framework Exploration

TODO

#### Conclusion

TODO

-----------

### Links

 * [Donkeycar Github repository](https://github.com/autorope/donkeycar)
 * [Train autopilot docs](http://docs.donkeycar.com/guide/train_autopilot/)
 * [Reinforcement Learning](https://pathmind.com/wiki/deep-reinforcement-learning)
 * [watermark](https://www.watermarquee.com/watermark)
