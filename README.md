# ALAF - Automatic Labeled And Fine-Tuning Library

# Improved object detection in traffic videos by fine-tuning deep convolutional neural networks with automatically labeled training data

In this repository, a new proposal has been developed. It is focused on using a fine adjustment of the network (it does not increase the execution time) that allows it to be automatically adapted to the traffic scene without human intervention. Firstly, we propose to apply a super resolution algorithm to detect objects in the scene that would otherwise go unnoticed by the DCNN object detection method. These detected objects or labelled data are used to generate a training dataset that is used to tune the network. All this process is done offline and only once per scene. Once the training is finished, the network has been adapted to better detect objects with the given camera distance and perspective.


A demo is provided to run our proposal on the following jupyter notebook:

* https://github.com/IvanGarcia7/ALAF/blob/main/ALAF/DEMO.ipynb

Within the jupyter notebook, all the necessary steps are established to initialize the work environment, download the pre-trained super-resolution and object detection models and carry out the proposal. For this purpose, a function has been developed that automatically performs the pre-processing phase and subsequent element detection, denoted as make_inference_SR. This function takes as input the image on which you want to detect the elements, the name under which the image with the generated detections will be stored and finally the directory in which you want to store it.


# Workflow of the proposed technique:

![WORKFLOW](https://github.com/IvanGarcia7/ALAF/blob/main/Images/Diagram.png?raw=true)


The Offline part is composed of the SR application, in addition to the generation of the dataset for Fine-Tuning. The Online part is composed of the detection performed by the retrained model.


# Execution Test:

## Input Image:

After downloading and loading the necessary models, we want to improve the detections on a given image as input, e.g. the Figure below:

![INPUT IMAGE](https://github.com/IvanGarcia7/ALAF/blob/main/Images/RAW.jpg?raw=true)


After performing the successive detections, an output image will be generated. Using the values determined by the function, an XML file is subsequently generated, which is necessary to generate the Tensorflow 2 training records.

## Output:

![OUTPUT IMAGE](https://github.com/IvanGarcia7/ALAF/blob/main/Images/OURS.jpg?raw=true)


The scripts mentioned in the Fine-Tuning phase can be found in the following path:

* https://github.com/IvanGarcia7/ALAF/tree/main/Scripts


# Evaluation:

To evaluate the mAP(Mean Average Precision) of the model, a function is included in the notebook to generate a json with the appropriate format. In the following path, we have loaded the annotations of the five video sequences (4 sequences corresponding to the NGSIM Dataset) and one extra sequence (GRAM Dataset M30-HD) used in the article, ideal for performing tests on the models obtained after Fine-Tuning.

* https://github.com/IvanGarcia7/NGSIM-Dataset-Annotations


# Test Our Proposal Without Fine-Tuning the Model:


You can execute our proposal as indicated in the demo by executing the following function:

``` 
make_inference_SRSR(‘path_image_to_infer’, ‘image_name’, ‘path_to_save_images‘, classes_coco_to_detect, 0.5, 0, min_score_to_detect, path_SR_model)
```

Example:

```
make_inference_SR(‘/Data/small_objects/Evaluation/1.jpg’, ‘1_output.jpg’, ‘/Data/small_objects/SR_OUTPUT/‘, [3], 0.5, 0, 0.35, ‘/Data/small_objects/SR_MODEL/‘)
```

# Docker Image:

A Docker image is included in order to install all the libraries required to run our proposal:

* https://github.com/IvanGarcia7/ALAF/blob/main/Dockerfile

# REQUIREMENTS:

* Tensorflow 2
* Tensorflow Object Detection
* OpenCV
* Numpy



