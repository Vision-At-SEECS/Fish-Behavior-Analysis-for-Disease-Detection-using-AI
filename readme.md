## Computer Vision based monitoring of fish behavior 
**Enhancing Fish Welfare in aquaculture**
# Introduction

<div align="justify">
We present a computer vision tracking-by-detection technique for monitoring fish behavior in aquaculture. The initial step involves generating a segmentation mask of the fish using Mask-RCNN in Detectron2. Subsequently, frames are associated for tracking using the state-of-the-art DeepSORT technique. The third step involves post-analysis, utilizing various image processing techniques to calculate parameters related to different behaviors, such as depth level, swimming speed, density, etc.
</div> 

# Abstract
<div align="justify">
Observing fish behavior holds significant importance in fisheries research and aquaculture production. A deep understanding of fish behavior plays a critical role in evaluating their well-being, differentiating typical behavior from irregular patterns, and implementing best husbandry practices. While previous studies have explored fish tracking, there remains a deficiency in post-analysis methods capable of offering in-depth insights into fish health and welfare through the assessment of their behavior. This research embarks on the integration of tracking-by-detection using computer vision, coupled with subsequent image processing techniques, all with a dedicated emphasis on behavioral analysis. Our examination predominantly centers on five critical parameters: spacing, speed, direction, angle, and depth. These parameters collectively furnish a comprehensive understanding of fish behavior. Previous research has predominantly focused on employing zebrafish as experimental subjects. However, there are certain limitations associated with this species. It exhibits a smaller width in comparison to the typical body length of other fish, along with reduced activity levels and a transparent body. In consideration of these constraints, the present study has opted for rainbow trout for behavior analysis experiments,which are closer to normal fish size and exhibit a diverse range of behaviors. Due to the collaborative efforts of our industry partner Urbanblue company Switzerland and research partner ZHAW University we implemented a comprehensive system that involved data collection and the deployment of our software in diverse fish farms. This study actively contributes to the advancement of a comprehensive solution aimed at monitoring and enhancing the welfare of fish in aquaculture, offering invaluable insights into their behavior.
</div>

# Trained Models and Datasets

<div align="justify">
Trained models for fish segmentation can be obtained on permission from the authors by filling the form below:

https://docs.google.com/forms/d/e/1FAIpQLScHzbEzj97v6YZn3EdU8Pt4aMXj5cGPe4qJ05mQrM6df54tJg/viewform?usp=sf_link

Dataset used in this project can also be requested using same form.

Dataset comprises of  various real-time video footages of rainbow trout available that can be utilized for training deep learning models for fish detection, tracking, or behavior analysis.

The code required to train the models from scratch can be found in the supporting materials, specifically under the "colab training - file" folder. By using this code, you can obtain the weights and set up the models as per your needs.

Alternatively, if you prefer to use pre-trained models, you can directly request them from the authors. Once obtained, simply place the trained models in the designated "trained_models" folder for seamless integration into your project.


## Step1: Fish segmentation using detectron2 

There are many variants of YOLO that can be used for fish detection followed by tracking, instead of Mask-RCNN. However, since our main focus is on the post-analysis of fish behavior, segmentation has the advantage of providing a precise shape of the fish over bounding box detection.
</div>

Note: An alternative option is to use YOLO if you prefer not to have a precise mask. For reference, please find the link to the YOLO project at: https://github.com/theAIGuysCode/yolov4-deepsort.


## Requirements
-   Python > 3.7
-   Pytorch
-   opencv-python 4.8.0.74
-   Django==3.2.8
-   djangorestframework==3.12.4
-   imutil==0.3.4
-   imutils==0.5.4
-   tensorflow==2.13.0
-   scipy==1.11.1

# Installation Instructions
Create a new conda environment uisng command 

```bash 
conda create --name your_env_name python
```
After running this command, you can activate the virtual environment using:
```bash 
conda activate your_env_name
```
Once the virtual env is created install pytorch using command (For windows and CPU only):  
```bash 
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```
Otherwise for GPU enabled environment or for linux OS or both, various installation commands can be found here: https://pytorch.org/get-started/locally/ <br />

Install OpenCV using command. 

```bash
 pip install opencv-python 
```
Note: Pytorch and OpenCV are the prerequisite to install detectron2

Install the detectron2 repository on local PC git clone 


```bash
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

Install the libraries and modules listed in  requirement.txt file  

```bash 
pip install -r requirements.txt
```


## Step2: Tracking fish using deepSort 

<div align="justify">

DeepSORT is state of art tracking algorithm which tracks object not only based on the velocity and motion of the object but also based on the appearance of the object. It is extension of the original [SORT](https://github.com/abewley/sort) algorithm. 

Note: If the tracking scenario is not complex, an alternative option is the centroid tracker. The code is avaiable in centroid tracker directory. 

## Dependencies

The code is compatible with Python 2.7 and 3. The following dependencies are
needed to run the tracker:

* NumPy
* sklearn
* OpenCV

Additionally, feature generation requires TensorFlow (>= 1.0).

## Installation

First, clone the repository:
```
git clone https://github.com/nwojke/deep_sort.git
```


## Overview of source files

In the top-level directory are executable scripts to execute, evaluate, and
visualize the tracker. The main entry point is in `deep_sort_app.py`.
This file runs the tracker on a MOTChallenge sequence.

In package `deep_sort` is the main tracking code:

* `detection.py`: Detection base class.
* `kalman_filter.py`: A Kalman filter implementation and concrete
   parametrization for image space filtering.
* `linear_assignment.py`: This module contains code for min cost matching and
   the matching cascade.
* `iou_matching.py`: This module contains the IOU matching metric.
* `nn_matching.py`: A module for a nearest neighbor matching metric.
* `track.py`: The track class contains single-target track data such as Kalman
  state, number of hits, misses, hit streak, associated feature vectors, etc.
* `tracker.py`: This is the multi-target tracker class.

The `deep_sort_app.py` expects detections in a custom format, stored in .npy
files. These can be computed from MOTChallenge detections using
`generate_detections.py`. We also provide
[pre-generated detections](https://drive.google.com/open?id=1VVqtL0klSUvLnmBKS89il1EKC3IxUBVK).


## Step3: Post analysis step calculating different parameters 


![Alt ](/Fig/five_parameters.png)

## 1. Fish Density 
Our algorithm utilizes machine learning to calculate the spacing between individual fish, whether measured from head-to-head or center-to-center. And to compute this distance Euclidean Distance is used that is Distance between the centroid of detected bounding box and track. d = √[(x2 — x1)2 + (y2 — y1)2]. 

In Python, there are several libraries that provide functions for calculating the Euclidean distance. Some of the commonly used ones include:

NumPy: NumPy is a powerful library for numerical computing in Python and includes a function for calculating the Euclidean distance.

```bash 
import numpy as np

point1 = np.array([x1, y1])
point2 = np.array([x2, y2])

euclidean_distance = np.linalg.norm(point1 - point2)
```

SciPy: SciPy builds on NumPy and provides additional functionality, including a spatial distance module that includes a function for calculating Euclidean distance.

```bash 
from scipy.spatial import distance

point1 = (x1, y1)
point2 = (x2, y2)

euclidean_distance = distance.euclidean(point1, point2)
```

Scikit-learn: Scikit-learn is a machine learning library for Python, and it includes a pairwise_distances function that can be used to calculate the Euclidean distance.


```bash
from sklearn.metrics import pairwise_distances

points = [[x1, y1], [x2, y2]]

euclidean_distance = pairwise_distances(points, metric='euclidean')[0, 1]
```


![Alt ](/Fig/Density.png)

For more information read this blog article: 
https://towardsdatascience.com/euclidean-distance-numpy-1b2784e966fc

## 2. Speed
Speed is the distance covered per frame. After object detection using detectron2 second step is using object associate or tracking techniques and we have used deep sort to track the fish and compute the speed of each individual fish. 

## 3. Direction

For directional analysis the difference is in the third parameter calculation.First, we detect the fish within the frame and then
track their movements over time. Once we have gathered the tracking information, we analyze the fish’s direction for a
specific range of frames, such as the first 50 frames.We examine whether the fish have moved in negative x, positive xnegative y, or positive y directions.By determining the majority direction among the tracked fish, we are able to identify the outlier fish that deviate from the swarm’s general movement pattern.

![Alt ](/Fig/Direction.png)

## 4. Turn Angle 

For angle, we use OpenCV to calculate the angle and compare the difference of each angle in the present frame to
the angle in the previous frame. If the difference is greater than 90 degrees, then that is an outlier fish.

For more information please refere to this blog: 
https://medium.com/@elvenkim1/easy-calculation-of-angle-using-opencv-contours-and-coordinates-4f5e29851ea3

![Alt ](/Fig/turnangle.png)

## 5. Fish depth level classification 

We use cartesian coordinate system to classify the detected fish into different levels along the vertical y-axis. Our primary objective is to detect outliers, which refer to fish that isolate themselves and are located either at the very bottom of the tank. Initially, we segmented the fish and then employed bounding box coordinates to classify them into upper quartiles, middle quartiles, and lower quartiles.

## Integrating the Three Steps for Comprehensive Behavior Analysis 

We leverage various matplotlib visualization techniques to gain insights into the movement patterns of fish.

For more information please refer to this blog:
https://www.geeksforgeeks.org/data-visualization-with-python/

## Acknowlegdement 

We would like to express our gratitude to the creators of the following projects and GitHub repositories. The materials used belong to the original authors, and we have adapted them for our specific case study on fish behavior in aquaculture.

https://github.com/facebookresearch/detectron2

https://github.com/computervisioneng/deep_sort

</div>

# Authors 
Kanwal Aftab - National University of Sciences and Technology (NUST)

Linda Tschirren - Zurich University of Applied Sciences (ZHAW)

Boris Pasini - Zurich University of Applied Sciences (ZHAW)

Peter Zeller - UrbanBlue GmbH

Bostan Khan - National University of Sciences and Technology (NUST)

Muhammad Moazam Fraz - National University of Sciences and Technology (NUST)

