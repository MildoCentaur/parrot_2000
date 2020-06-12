import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import cv2
# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
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

cam = cv2.VideoCapture('dataset/custom_video_test/parrot.mp4')
#fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#out = cv2.VideoWriter('output.mp4',fourcc, 20.0, (1280,720))

while True:
    ret_val, img = cam.read()
    if cv2.waitKey(1) == 27:
        break  # esc to quit
    im = img#cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    outputs = predictor(im)
    visualizer = Visualizer(im[:, :, ::-1],
                            metadata=parrot_metadata,
                            scale=1)
    visualizer = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow('predicted', visualizer.get_image()[:, :, ::-1])
    #out.write(visualizer.get_image()[:, :, ::-1])
#out.release()
cam.release()
cv2.destroyAllWindows()




