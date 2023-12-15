# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 02:43:23 2023

@author: nathant
"""

import building_helper
import mrcnn
import mrcnn.config
import mrcnn.model
import mrcnn.visualize as visualize
import cv2
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from numpy import expand_dims
from numpy import mean
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.utils import Dataset
from mrcnn.utils import compute_ap, compute_recall
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image
# Compute VOC-Style mAP @ IoU=0.5
# Running on 10 images. Increase for better accuracy.
dataset_val = SpaceNetDataset()
dataset_val.load_dataset(DATASET_DIR,1310,2000)
dataset_val.prepare()
model = mrcnn.model.MaskRCNN(mode="inference",
                             config=config,
                             model_dir=os.getcwd())

# Load the weights into the model.
#Kangaro_mask_rcnn_trained
#mask_rcnn_spacenet_0151
model.load_weights("WEIGHTS/crowd_ai_weights.h5", by_name=True)
image_ids = np.random.choice(dataset_val.image_ids, 6)
APs = []
for image_id in image_ids:
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset_val, inference_config,
                               image_id, use_mini_mask=False)
    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]
    # Compute AP
    AP, precisions, recalls, overlaps =\
        utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                         r["rois"], r["class_ids"], r["scores"], r['masks'])
    APs.append(AP)

print("mAP: ", np.mean(APs))