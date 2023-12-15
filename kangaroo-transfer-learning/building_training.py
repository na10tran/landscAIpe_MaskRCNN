# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 15:18:02 2023

@author: nathant
"""
import os
import sys
import re
import random
import numpy as np
#import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

#Libraries' order is important
import geopandas as gpd
# Geoio dan önce geopandası yüklemelisin.
import geoio
import json
import tensorflow as tf

"""gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
"""
RGB_PanSharpen_NEW_Path = "building/RGB-PanSharpen-NEW/"
Geojson_Renamed_Path = "building/geojson/building-renamed/"
# Root directory of the project
ROOT_DIR = os.path.abspath("C:/Users/nathant/Downloads/Mask-RCNN-TF2/kangaroo-transfer-learning/")  #CHANGE FILE PATH
DATASET_DIR = os.path.abspath("C:/Users/nathant/Downloads/Mask-RCNN-TF2/kangaroo-transfer-learning/building/")  #CHANGE FILE PATH

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_spacenet_0151.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
    

class SpaceNetConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "SpaceNet"
    BACKBONE = "resnet50"
    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 building

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 640
    IMAGE_MAX_DIM = 640

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels
    RPN_ANCHOR_RATIOS = [0.25, 1, 4]

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    USE_MINI_MASK = True

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 4

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5

    MAX_GT_INSTANCES=250
    DETECTION_MAX_INSTANCES=350

config = SpaceNetConfig()
config.display()

def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

def fill_between(polygon):
    """
    Returns: a bool array
    """
    img = Image.new('1', (650, 650), False)
    ImageDraw.Draw(img).polygon(polygon, outline=True, fill=True)
    mask = np.array(img)
    return mask

class SpaceNetDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """

    def load_dataset(self, dataset_dir, start=1, end=400):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("SpaceNetDataset", 1, "building")

        # define data locations for images and annotations
        images_dir = os.path.join(dataset_dir, "RGB-PanSharpen-NEW/")
        print("PATH IS: "+ str(images_dir))
        annotations_dir = os.path.join(dataset_dir, "/geojson/building-renamed/")

        # Iterate through all files in the folder to
        #add class, images and annotaions
        for filename in os.listdir(images_dir)[start:end]:
            if str(filename[-4:]) == '.png':
              image_dir = os.path.join(images_dir,str(filename))
              tif_number_match = re.search(r'(\d+)\.png$', str(filename))
              tif_number_match = int(tif_number_match.group(1))
              #print(tif_number_match)
              ann_path  = os.path.join(annotations_dir,str(tif_number_match)+".geojson")
              self.add_image('SpaceNetDataset', image_id=tif_number_match, path=image_dir, annotation=ann_path)

    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        if image_id == 0:
            image_id = 1
        image_dir = os.path.join(DATASET_DIR, "RGB-PanSharpen-NEW/"+str(image_id)+".png")
        im = Image.open(image_dir)
        return np.asarray(im)

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        if image_id == 0:
            image_id = 1
        info = self.image_info[image_id]
        if info["source"] == "shapes":
            return info["shapes"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        if image_id == 0:
            image_id = 1
        masks = np.zeros((650,650))
        ResimPATH = 'C:/Users/nathant/Downloads/Mask-RCNN-TF2/kangaroo-transfer-learning/building/RGB-PanSharpen-Renamed/'+str(image_id)+'.tif' #CHANGE THIS FILE PATH!!!!!!!!
        RGBTIFResmi = geoio.GeoImage(ResimPATH)

        with open(DATASET_DIR+"/geojson/building-renamed/"+str(image_id)+".geojson") as f:
            data = json.load(f)
            allBuildings = data['features']

            for building in allBuildings:
                veri = building['geometry']['coordinates'][0]

                tip = str(building['geometry']['type'])
                coordinates = list()
                if tip == ('Point'):
                    continue

                elif tip == ('MultiPolygon'):

                    if isinstance(veri,float): continue

                    kucukBinalar = (building['geometry']['coordinates'])
                    for b in range(len(kucukBinalar)):
                        veri = kucukBinalar[b][0]
                        for i in veri:
                            xPixel, yPixel = RGBTIFResmi.proj_to_raster(i[0], i[1])
                            xPixel = 649 if xPixel > 649 else xPixel
                            yPixel = 649 if yPixel > 649 else yPixel
                            coordinates.append((xPixel,yPixel))
                else:
                    if isinstance(veri,float): continue

                    for i in veri:
                        xPixel, yPixel = RGBTIFResmi.proj_to_raster(i[0], i[1])
                        xPixel = 649 if xPixel > 649 else xPixel
                        yPixel = 649 if yPixel > 649 else yPixel
                        coordinates.append((xPixel,yPixel))

                maske = fill_between(coordinates)
                masks = np.dstack((masks,maske))

        if masks.shape != (650,650):
            masks = masks[:,:,1:]
            class_ids = np.asarray([1]*masks.shape[2])
        else:
            class_ids=np.ones((1))
            masks = masks.reshape((650,650,1))
        return masks.astype(bool), class_ids.astype(np.int32)
"""
# Training dataset
dataset_train = SpaceNetDataset()
dataset_train.load_dataset(DATASET_DIR,1,1300)
dataset_train.prepare()

# Validation dataset

# Load and display random samples
non_zero_image_ids = [image_id for image_id in dataset_train.image_ids if image_id != 0]
#print(non_zero_image_ids)
image_ids = np.random.choice(non_zero_image_ids,4)

for image_id in image_ids:
    #print(image_id)
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)
    print("DEBUGGING THIS SHIT",dataset_train.class_names)
    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)


# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
model.load_weights(COCO_MODEL_PATH, by_name=True,
                    exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                            "mrcnn_bbox", "mrcnn_mask"])

# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=1,
            layers='all')

model_path = 'TRAINED_BUILDING_FILE.h5'
model.keras_model.save_weights(model_path)
"""
#########################################################DETECTION

class InferenceConfig(SpaceNetConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

#Recreate the model in inference mode
#model = modellib.MaskRCNN(mode="inference",
#                          config=inference_config,
#                          model_dir=MODEL_DIR)
#
# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
#model_path = model.find_last()

# Load trained weights
#print("Loading weights from ", model_path)
#model.load_weights(model_path, by_name=True)

# Test on a random image
#image_id = random.choice(dataset_val.image_ids)
#original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
#    modellib.load_image_gt(dataset_val, inference_config,
#                           image_id, use_mini_mask=False)
#
#log("original_image", original_image)
#log("image_meta", image_meta)
#log("gt_class_id", gt_class_id)
#log("gt_bbox", gt_bbox)
#log("gt_mask", gt_mask)
#
#visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
#                            dataset_train.class_names, figsize=(8, 8))
#
#results = model.detect([original_image], verbose=1)
#r = results[0]
#visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
#                            dataset_val.class_names, r['scores'], ax=get_ax())


