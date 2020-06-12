
# Parrot 2000
Parrot 2000 is a project meant to explore Image detection technologies. It detects parrots in images and is based on [Detectron2](https://github.com/facebookresearch/detectron2)
## Description
This project is a template for retraining a Detectron2 model with a new image class for object detection.
## Visuals
![](parrot.gif)
## Installation
Install packages listed in `requirements.txt`, be aware that the pytorch version listed in the document is the specific one that I needed for my particular CUDA version, you may require a different one.
I personally prefer to use a virtualenv to install the packages, but that is up to you. Other options would be to install this packages in the global Python configuration of your machine or to use a Docker instance.

## Usage
Optional: based on Daniel Bourke tutorial (mentioned in the acknowledgment part of this README), I am using [Weights & Biases](https://www.wandb.com/)
for testing different models from Detectron2 zoo (I also took the list of models to try from Daniel work, limiting it to models that my GPU was able to load in memory). In order to make use of this great tool
you should create an account in Weights & Biases page, and create a project. If you want to skip this part you will need
to remove all references to the wandb package in the file `utils/trainer_config.py`.

1) First you will need to download several files. This project was built around images obtained from [Open Images Dataset](https://storage.googleapis.com/openimages/web/index.html).
From there, you will need to download:
* Boxes annotation Train csv
* Boxes annotation Validate csv  
* Image IDs Train
* Image IDs Validate
* Metadata class names

These files should work from any given version of Open Images Dataset, as long as these csv files have the same column names (which I checked with a few). But this 
project was only tested with V6. After downloading these files, place them inside the dataset folder.

2) You will need to create a `config.py` file, to be placed in the root of the project. In that file you should specify the following keys (referencing the name of the files you choose and Weights&Biases project name):
```
ANNOTATION_TRAIN_CSV = 'oidv6-train-annotations-bbox.csv'
ANNOTATION_VALIDATE_CSV = 'test-annotations-bbox.csv'
BOXABLE_TRAIN_CSV = 'train-images-boxable.csv'
BOXABLE_TEST_CSV = 'test-images.csv'
CLASS_DESCRIPTION_CSV = 'class-descriptions-boxable.csv'
WANDB_PROJECT_NAME = 'parrot-2000'
```
3) Logic to download images is placed on data_exploration.ipynb Jupyter Notebook. I considered moving this to a normal Python 
script but I think it is still useful to visualize the data before generating the dataset and it is practical to see the images  from there. After running all the cells, images will be downloaded inside the “dataset” “train” and “validate” folders.

4) Run `python dataset/obtain_dataset.py` to create the final csv files that will be used by detectron.

5) Run `python train_detectron.py` to train several models. You can visualize in Weights&Biases which one gives the best results. 
Be aware that Detectron2 uses different keys for the configuration of different models, for example   MODEL.ROI_HEADS.NUM_CLASSES 
is the key to set the number of classes to predict for a R-CNN model but a Retinanet need to use the key MODEL.RETINANET.NUM_CLASSES.
More info in their [documentation](https://detectron2.readthedocs.io/tutorials/datasets.html)

6) You can run `python test_detectron.py` with custom pictures located in `dataset/custom_pictures_test`
7) You can run `python test_detectron_video.py` with a custom pictures located in `dataset/custom_video_test`

## Support
Please, if you have any problems just raise an issue on GitHub
## Roadmap
This is a project I created with the purpose of learning some functionalities around object detection. At this point in time, I am not sure if I will keep
working on it.

## Authors and acknowledgment
Created by Matías Aiskovich.

This project is basically a merge between two tutorials:
* [Faster R-CNN (object detection) implemented by Keras for custom data from Google’s Open Images Dataset V4](https://towardsdatascience.com/faster-r-cnn-object-detection-implemented-by-keras-for-custom-data-from-googles-open-images-125f62b9141a), from here I took a lot of the logic from filtering Open 
Images dataset to only our desired class (in this case, Parrot)
* [Replicating Airbnb's Amenity Detection with Detectron2](https://www.mrdbourke.com/airbnb-amenity-detection/) From this great tutorial by Daniel Bourke, I took most of the logic to convert the dataset to Detectron2 required format, and also most of the logic on training the model and using Weights & Biases.

## Recommended reads
Besides the two tutorials mentioned above, if you are unfamiliar with object detection and the metrics that are commonly used I recommend this resources:
* [mAP (mean Average Precision) for Object Detection](https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173)
* [Intersection over Union (IoU) for object detection](https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/)

## License
[GNU General Public License v3.0](https://choosealicense.com/licenses/gpl-3.0/)


