import detectron2
from detectron2.utils.logger import setup_logger
setup_logger() # this logs Detectron2 information such as what the model is doing when it's training
# import some common libraries
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
import random
from detectron2.data import DatasetCatalog, MetadataCatalog
# import some common detectron2 utilities
from detectron2.utils.visualizer import Visualizer # a class to help visualize Detectron2 predictions on an image
from detectron2.data import MetadataCatalog # stores information about the model such as what the training/test data is, what the class names are
from detectron2.structures import BoxMode
import json
import pprint
import os
# configuration file
import config


def get_image_ids(image_folder=None):
    """
    Explores a folder of images and gets their ID from their file name.
    Returns a list of all image ID's in image_folder.
    E.g. image_folder/608fda8c976e0ac.jpg -> ["608fda8c976e0ac"]

    Params
    ------
    image_folder (str): path to folder of images, e.g. "../validation/"
    """
    return [os.path.splitext(img_name)[0] for img_name in os.listdir(image_folder) if img_name.endswith(".jpg")]


def format_annotations(image_folder, annotation_file, target_classes=None):
    """
    Formats annotation_file based on images contained in image_folder.
    Will get all unique image IDs and make sure annotation_file
    only contains those (the target images).
    Adds meta-data to annotation_file such as class names and categories.
    If target_classes isn't None, the returned annotations will be filtered by this list.
    Note: image_folder and annotation_file should both be validation if working on
    validation set or both be training if working on training set.

    Params
    ------
    image_folder (str): path to folder of target images.
    annotation_file (str): path to annotation file of target images.
    target_classes (list), optional: a list of target classes you'd like to filter labels.
    """
    # Get all image ids from target directory
    image_ids = get_image_ids(image_folder)

    # Setup annotation file and classnames
    annot_file = pd.read_csv(annotation_file)
    classes = pd.read_csv("dataset/"+config.CLASS_DESCRIPTION_CSV,
                          names=["LabelName", "ClassName"])

    # Create classname column on annotations which converts label codes to string labels
    annot_file["ClassName"] = annot_file["LabelName"].map(classes.set_index("LabelName")["ClassName"])

    # Sort annot_file by "ClassName" for alphabetical labels (used with target_classes)
    annot_file.sort_values(by=["ClassName"], inplace=True)

    # Make sure we only get the images we're concerned about
    if target_classes:
        annot_file = annot_file[annot_file["ImageID"].isin(image_ids) & annot_file["ClassName"].isin(target_classes)]
    else:
        annot_file = annot_file[annot_file["ImageID"].isin(image_ids)]

    assert len(annot_file.ImageID.unique()) == len(image_ids), "Label unique ImageIDs doesn't match target folder."

    # Add ClassID column, e.g. "Bathtub, Toilet" -> 1, 2
    annot_file["ClassName"] = pd.Categorical(annot_file["ClassName"])
    annot_file["ClassID"] = annot_file["ClassName"].cat.codes

    return annot_file


def rel_to_absolute(bbox, height, width):
    """
    Converts bounding box dimensions from relative to absolute pixel values (Detectron2 style).
    See: https://detectron2.readthedocs.io/modules/structures.html#detectron2.structures.BoxMode

    Params
    ------
    bbox (array): relative dimensions of bounding box in format (x0, y0, x1, y1 or Xmin, Ymin, Xmax, Ymax)
    height (int): height of image
    width (int): width of image
    """
    bbox[0] = np.round(np.multiply(bbox[0], width))  # x0
    bbox[1] = np.round(np.multiply(bbox[1], height))  # y0
    bbox[2] = np.round(np.multiply(bbox[2], width))  # x1
    bbox[3] = np.round(np.multiply(bbox[3], height))  # y1
    return [i.astype("object") for i in bbox]  # convert all to objects for JSON saving


def get_image_dicts(image_folder, annotation_file, target_classes=None):
    """
    Create Dectectron2 style labels in the form of a list of dictionaries.

    Params
    ------
    image_folder (str): target folder containing images
    annotations (DataFrame): DataFrame of image label data
    target_classes (list): names of target Open Images classes

    Note: image_folder and annotation_file should both relate to the same dataset (this could be improved).
    E.g.
    image_folder = valid_images & annotation_file = valid_annotations (both are for the validation set)
    """
    # Get name of dataset from image_folder
    dataset_name = str(image_folder)

    # Create dataset specific annotations.
    annotations = format_annotations(image_folder=image_folder,
                                     annotation_file=annotation_file,
                                     target_classes=target_classes)

    # Get all unique image ids from target folder
    img_ids = get_image_ids(image_folder)
    # Add some verbosity
    print(f"\nUsing {annotation_file} for annotations...")
    print(f"On dataset: {dataset_name}")
    print("Classes we're using:\n{}".format(annotations["ClassName"].value_counts()))
    print(f"Total number of images: {len(img_ids)}")

    # Start creating image dictionaries (Detectron2 style labelling)
    img_dicts = []
    for idx, img in tqdm(enumerate(img_ids)):
        record = {}

        # Get image metadata
        file_name = image_folder + img + ".jpg"
        height, width = cv2.imread(file_name).shape[:2]
        img_data = annotations[annotations["ImageID"] == img].reset_index()  # reset_index(): important for images
        # with multiple objects
        # Verbosity for image label troubleshooting
        # print(f"On image: {img}")
        # print(f"Image category: {img_data.ClassID.values}")
        # print(f"Image label: {img_data.ClassName.values}")

        # Update record dictionary
        record["file_name"] = file_name
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        # Create list of image annotations (labels)
        img_annotations = []
        for i in range(len(img_data)):  # this is where we loop through examples with multiple objects in an image
            category_id = img_data.loc[i]["ClassID"].astype(
                "object")  # JSON (for evalution) can't take int8 (NumPy type) must be native Python type
            # Get bounding box coordinates in Detectron2 style (x0, y0, x1, y1)
            bbox = np.float32(img_data.loc[i][["XMin", "YMin", "XMax", "YMax"]].values)
            # Convert bbox from relative to absolute pixel dimensions
            bbox = rel_to_absolute(bbox=bbox, height=height, width=width)
            # Setup annot (1 annot = 1 label, there might be more) dictionary
            annot = {
                "bbox": bbox,
                "bbox_mode": BoxMode.XYXY_ABS,
                # See: https://detectron2.readthedocs.io/modules/structures.html#detectron2.structures.BoxMode.XYXY_ABS
                "category_id": category_id
            }
            img_annotations.append(annot)

        # Update record dictionary with annotations
        record["annotations"] = img_annotations

        # Add record dictionary with image annotations to img_dicts list
        img_dicts.append(record)

    # Save img_dicts to JSON for use later
    prefix = "validate" if "validate" in image_folder else "train"
    json_file = os.path.join(image_folder, prefix + "_labels.json")
    print(f"\nSaving labels to: {json_file}...")
    with open(json_file, "w") as f:
        json.dump(img_dicts, f)

    print("Showing an example:")
    pprint.pprint(random.sample(img_dicts, 1))

    # return img labels dictionary
    return img_dicts

def load_json_labels(image_folder):
    """
    Returns Detectron2 style labels of images (list of dictionaries) in image_folder based on JSON label file in image_folder.

    Note: Requires JSON label to be in image_folder. See get_image_dicts().

    Params
    ------
    image_folder (str): target folder containing images
    """
    # Get absolute path of JSON label file
    for file in os.listdir(image_folder):
        if file.endswith(".json"):
            json_file = os.path.join(image_folder, file)

    # Check to see if json_file exists
    assert json_file, "No .json label file found, please make one with get_image_dicts()"

    with open(json_file, "r") as f:
        img_dicts = json.load(f)

    # Convert bbox_mode to Enum of BoxMode.XYXY_ABS (doesn't work loading normal from JSON)
    for img_dict in img_dicts:
        for annot in img_dict["annotations"]:
            annot["bbox_mode"] = BoxMode.XYXY_ABS

    return img_dicts

def register_parrot_dataset(folder_called=None):
    # Loop through different datasets
    for dataset in ["train", "validate"]:
        # Create dataset name strings
        dataset_name = "parrot_" + dataset
        print(f"Registering {dataset_name}")
        # Register the datasets with Detectron2's DatasetCatalog, which has space for a lambda function to preprocess it
        if folder_called:
            dataset = folder_called+dataset
            DatasetCatalog.register(dataset_name, lambda dataset=dataset: load_json_labels(dataset))
        else:
            DatasetCatalog.register(dataset_name, lambda dataset=dataset: load_json_labels(dataset))
    MetadataCatalog.get('parrot_validate').set(thing_classes=["Parrot", "Dummy class"])
    # Create the metadata for our dataset (the main thing being the classnames we're using)
    return MetadataCatalog.get('parrot_train').set(thing_classes=["Parrot","Dummy class"])






def create_dataset_files():
    img_dicts_train = get_image_dicts(image_folder='dataset/train/',
                                    annotation_file="dataset/"+config.ANNOTATION_TRAIN_CSV,
                                    target_classes=["Parrot"])

    img_dicts_validate = get_image_dicts(image_folder='dataset/validate/',
                                    annotation_file="dataset/"+config.ANNOTATION_VALIDATE_CSV,
                                    target_classes=["Parrot"])
    loaded_train_img_dicts = load_json_labels('dataset/train')
    register_parrot_dataset()
    # Setup metadata variable
    parrot_metadata = MetadataCatalog.get("parrot_train")
    for label in random.sample(img_dicts_train, 3):
        img = cv2.imread(label["file_name"])
        visualizer = Visualizer(img[:, :, ::-1],  # revervse the image pixel order (BGR -> RGB)
                                metadata=parrot_metadata,  # use the metadata variable we created about
                                scale=0.5)
        vis = visualizer.draw_dataset_dict(label)
        cv2.imshow('predicted', vis.get_image()[:, :, ::-1])
        cv2.waitKey(0)

if __name__ == "__main__" :
    create_dataset_files()