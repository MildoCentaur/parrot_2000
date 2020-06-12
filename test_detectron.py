from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import cv2
import random
from detectron2.utils.visualizer import Visualizer # a class to help visualize Detectron2 predictions on an image
from detectron2 import model_zoo
import os
from dataset.obtain_dataset import register_parrot_dataset, load_json_labels



cfg=get_cfg()


# Add some basic instructions for the Detectron2 model from the model_zoo: https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

# We're only dealing with 2 classes (coffeemaker and fireplace)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1


# Get the final model weights from the outputs directory
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

# Set the testing threshold (a value between 0 and 1, higher makes it more difficult for a prediction to be made)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

parrot_metadata = register_parrot_dataset()

# Tell the config what the test dataset is (we've already done this)
cfg.DATASETS.TEST = ("parrot_validate",)

# Setup a default predictor from Detectron2: https://detectron2.readthedocs.io/modules/engine.html#detectron2.engine.defaults.DefaultPredictor
predictor = DefaultPredictor(cfg)
img_dicts_validate = load_json_labels('dataset/validate')


for img_name in os.listdir('dataset/custom_pictures_test'): # select random samples from the validation set
  img = cv2.imread('dataset/custom_pictures_test/'+img_name)
  outputs = predictor(img)
  visualizer = Visualizer(img[:, :, ::-1],
                          metadata=parrot_metadata,
                          scale=0.7)
  visualizer = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))
  cv2.imshow('predict',visualizer.get_image()[:, :, ::-1])
  cv2.waitKey(0)