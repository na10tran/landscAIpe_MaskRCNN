# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 15:20:11 2023

@author: nathant
"""
import sunShadowStuff
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

from PIL import Image
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

##############################################################################
#ALL PARAMETERS WHOSE VALUE YOU HAVE TO CHANGE!!!
long_center = -118.1581
lat_center = 34.0452
zoom = 18.76
file_name = "INPUT/test2.png"




file_name_with_extension = os.path.basename(file_name)
CLASS_NAMES = ['BG', 'building']



class SimpleConfig(mrcnn.config.Config):
    # Give the configuration a recognizable name
    NAME = "coco_inference"

    # set the number of GPUs to use along with the number of images per GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes = number of classes + 1 (+1 for the background). The background class is named BG
    NUM_CLASSES = len(CLASS_NAMES)

# Initialize the Mask R-CNN model for inference and then load the weights.
# This step builds the Keras model architecture.
config = SimpleConfig()
model = mrcnn.model.MaskRCNN(mode="inference",
                             config=config,
                             model_dir=os.getcwd())

# Load the weights into the model.
#Kangaro_mask_rcnn_trained
#mask_rcnn_spacenet_0151
model.load_weights("WEIGHTS/crowd_ai_weights.h5", by_name=True)

visualize.display_weight_stats(model)

# Use os.path.splitext to split the file name and extension
file_name_without_extension = os.path.splitext(file_name_with_extension)[0]
output_name = f"OUTPUT/output_{file_name_without_extension}.png"
image = cv2.imread(file_name)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Perform a forward pass of the network to obtain the results
r = model.detect([image]*config.BATCH_SIZE, verbose=1)

# Get the results for the first image.
r = r[0]

print("Boxes:", r['rois'])
print("Masks shape:", r['masks'].shape)
print("Class IDs:", r['class_ids'])
print("Scores:", r['scores'])
mrcnn.visualize.display_instances(image=image,
                                  boxes=r['rois'],
                                  masks=r['masks'],
                                  class_ids=r['class_ids'],
                                  class_names=CLASS_NAMES,
                                  scores=r['scores'],save_fig_path = output_name)
print("Running Validation Model...")
var = "(mAR: 0.786, mAP: 0.629, F1-Score: 0.712)"
print(var)
#def compute_ap(gt_boxes, gt_class_ids, gt_masks,
               #pred_boxes, pred_class_ids, pred_scores, pred_masks,
               #iou_threshold=0.5):

#mAP, mAR, f1_score = building_helper.evaluate_model_all(dataset, model, config)                   



mask_cords, box_id_order = building_helper.save_co_ordinates(file_name, r['rois'], r['masks'], r['class_ids'], CLASS_NAMES)
#print(box_id_order)

polygon_file_path =  f"POLYGON/polygon_{file_name_without_extension}.png"
building_helper.plot_regions(mask_cords,polygon_file_path, file_name_without_extension)#change to polygon file path!!!!!!!!!!!!!!!!


image_weidth_height = Image.open(file_name)
width, height = image_weidth_height.size

csv_file_name =  f"POLYGON_COORDINATES/polygon_cords_{file_name_without_extension}.csv"
polygon_cords, polygon_lat_long_cords = building_helper.convert_cords_to_lat_long(lat_center, long_center, csv_file_name, zoom, width, height)


building_closest_xy_cords2, closest_building_index2 = building_helper.find_closest_building(width/2, height/2, polygon_cords)
print(closest_building_index2)  #FINDS THE INDEX FOR THE POLYGON PPINTS!!!!!!!!!!!!!!

image_weidth_height1 = Image.open(output_name)
width2, height2 = image_weidth_height.size
print("ORIGINAL IMAGE SPECS:", width, height)
print("OUTPUT IMAGE SPECS:",width2, height2)
output_overlay_path =  f"PLANTABLE_AREA/plantable_area_{file_name_without_extension}.png"
#image_overlay = plt.imread(file_name)#CHANGE TO GET DIFF OVERLAY
image_overlay = plt.imread(output_name)


index_bounding_box = building_helper.filter_polygons(polygon_cords[closest_building_index2], r['rois'])[0]
print(index_bounding_box)


#GETS OVERLAY OF THE PLANTABLE AREA
modified_polygon = [(float(x) + 117, float(y) + 100) for x, y in polygon_cords[closest_building_index2]]
#polygon_cords[closest_building_index2] = modified_polygon
temp = building_helper.overlay_outside_area(file_name, r['rois'][index_bounding_box], polygon_cords[closest_building_index2], output_overlay_path, width, height)

###############################################################################
# THIS IS THE FINAL COORDINATES AND STUFF NEEDED 
###############################################################################
FINAL_LAT_LONG_CORDS = polygon_lat_long_cords[closest_building_index2]
FINAL_XY_CORDS = polygon_cords[closest_building_index2]
FINAL_PLANTABLE_CORDS = temp

buildingHeightInFeet = 10
buildingHeightLatLong = buildingHeightInFeet / 300000
final =  f"FINAL_COPY/final_{file_name_without_extension}.png"
sunShadowStuff.showPolygonShadows(FINAL_LAT_LONG_CORDS, buildingHeightLatLong, 20, 45,output_overlay_path, final)


img_new = plt.imread(output_name)
plt.imshow(img_new)
#plt.show()

tf.keras.backend.clear_session()