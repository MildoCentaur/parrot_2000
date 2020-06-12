from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo # a series of pre-trained Detectron2 models: https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md
from dataset.obtain_dataset import register_parrot_dataset, load_json_labels,get_image_dicts
from detectron2.engine import DefaultPredictor
import random
from detectron2.utils.visualizer import Visualizer # a class to help visualize Detectron2 predictions on an image
import cv2
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from utils.trainer_config import ParrotTrainer
import os
import torch

# models_to_try = {
#     # model alias : model setup instructions
#     "R50-FPN-3x": "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
#     "R101-FPN-3x": "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml",
#     "RN-R50-1x": "COCO-Detection/retinanet_R_50_FPN_1x.yaml",
#     "RN-R50-3x": "COCO-Detection/retinanet_R_50_FPN_3x.yaml",
#     "RN-R101-3x": "COCO-Detection/retinanet_R_101_FPN_3x.yaml"
# }
models_to_try = {
    # model alias : model setup instructions
    "R50-FPN-3x": "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
}

# Call function in order to register parrot dataset
parrot_metadata = register_parrot_dataset('dataset/')
for i, model in models_to_try.items():
    print('training model {0}'.format(i))
    torch.cuda.empty_cache()
    # Setup a model config (recipe for training a Detectron2 model)
    cfg = get_cfg()

    cfg.model_name = model
    # Add some basic instructions for the Detectron2 model from the model_zoo: https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md
    cfg.merge_from_file(model_zoo.get_config_file(model))

    # Add some pretrained model weights from an object detection model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)

    # Setup datasets to train/validate on (this will only work if the datasets are registered with DatasetCatalog)
    cfg.DATASETS.TRAIN = ("parrot_train",)
    cfg.DATASETS.TEST = ("parrot_validate",)

    # How many dataloaders to use? This is the number of CPUs to load the data into Detectron2, Colab has 2, so we'll use 2
    cfg.DATALOADER.NUM_WORKERS = 2

    # How many images per batch? The original models were trained on 8 GPUs with 16 images per batch, since we have 1 GPU: 16/8 = 2.
    cfg.SOLVER.IMS_PER_BATCH = 2

    # We do the same calculation with the learning rate as the GPUs, the original model used 0.01, so we'll divide by 8: 0.01/8 = 0.00125.
    cfg.SOLVER.BASE_LR = 0.00125

    # How many iterations are we going for? (300 is okay for our small model, increase for larger datasets)
    cfg.SOLVER.MAX_ITER = 1200

    # ROI = region of interest, as in, how many parts of an image are interesting, how many of these are we going to find?
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128

    # We're only dealing with 1 class (parrot)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    # Setup output directory, all the model artefacts will get stored here in a folder called "outputs"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Setup the default Detectron2 trainer, see: https://detectron2.readthedocs.io/modules/engine.html#detectron2.engine.defaults.DefaultTrainer
    trainer = ParrotTrainer(cfg)







