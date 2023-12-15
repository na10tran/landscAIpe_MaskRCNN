# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 18:49:57 2023

@author: nathant
"""
import csv
from skimage.measure import find_contours
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import math
from shapely.geometry import Polygon as ShapelyPolygon, MultiPolygon
from shapely.geometry import box
from shapely import difference
from geopy.distance import geodesic
from shapely.geometry import LinearRing
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection
#from sklearn import metrics
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

def evaluate_model_mAP(dataset, model, cfg):
  APs = []
  for image_id in dataset.image_ids:
    image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id, use_mini_mask=False)
    scaled_image = mold_image(image, cfg)
    sample = expand_dims(scaled_image, 0)
    yhat = model.detect(sample, verbose=0)
    r = yhat[0]
    AP, precisions, recalls, overlaps = compute_ap(gt_bbox, gt_class_id, gt_mask, 
                                                          r["rois"], r["class_ids"], r["scores"], 
                                                          r['masks'], iou_threshold=0.5)
    APs.append(AP)
  mAP = mean(APs)
  return mAP

def compute_ar(pred_boxes, gt_boxes, list_iou_thresholds):

    AR = []
    for iou_threshold in list_iou_thresholds:

        try:
            recall, _ = compute_recall(pred_boxes, gt_boxes, iou=iou_threshold)

            AR.append(recall)

        except:
          AR.append(0.0)
          pass

    AUC = 2 * (metrics.auc(list_iou_thresholds, AR))
    return AUC

def evaluate_model_mAR(dataset, model, cfg, list_iou_thresholds=None):

  if list_iou_thresholds is None: list_iou_thresholds = np.arange(0.5, 1.01, 0.1)

  APs = []
  ARs = []
  for image_id in dataset.image_ids:
		
    image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id, use_mini_mask=False)
		
    scaled_image = mold_image(image, cfg)
		
    sample = expand_dims(scaled_image, 0)
		
    yhat = model.detect(sample, verbose=0)
		
    r = yhat[0]
		
    AP, precisions, recalls, overlaps = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'], iou_threshold=0.5)
		
    AR = compute_ar(r['rois'], gt_bbox, list_iou_thresholds)
    ARs.append(AR)
    APs.append(AP)

  mAP = mean(APs)
  mAR = mean(ARs)

  return mAP, mAR

def evaluate_model_all(dataset, model, cfg, list_iou_thresholds=None):
  if list_iou_thresholds is None: list_iou_thresholds = np.arange(0.5, 1.01, 0.1)
  APs = []
  ARs = []
  for image_id in dataset.image_ids:
    image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id, use_mini_mask=False)
    scaled_image = mold_image(image, cfg)
    sample = expand_dims(scaled_image, 0)
    yhat = model.detect(sample, verbose=0)
    r = yhat[0]
    AP, precisions, recalls, overlaps = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'], iou_threshold=0.5)
    AR = compute_ar(r['rois'], gt_bbox, list_iou_thresholds)
    ARs.append(AR)
    APs.append(AP)
  mAP = mean(APs)
  mAR = mean(ARs)
  f1_score = 2 * ((mAP * mAR) / (mAP + mAR))
  return mAP, mAR, f1_score



# Plots a Polygon to pyplot `ax`
def plot_polygon(ax, poly, **kwargs):
    path = Path.make_compound_path(
        Path(np.asarray(poly.exterior.coords)[:, :2]),
        *[Path(np.asarray(ring.coords)[:, :2]) for ring in poly.interiors])

    patch = PathPatch(path, **kwargs)
    collection = PatchCollection([patch], **kwargs)

    ax.add_collection(collection, autolim=True)
    ax.autoscale_view()
    return collection

def filter_polygons(polygons_list, bounding_boxes):
    filtered_polygons = []
    indices = []

    for index, bounding_box in enumerate(bounding_boxes):
        ymin, xmin, ymax, xmax = bounding_box
        ymin = float(ymin)
        xmin = float(xmin)
        ymax = float(ymax)
        xmax = float(xmax)

        # Check if all points of the polygon are within the bounding box
        count_inside = sum(xmin <= float(x) <= xmax and ymin <= float(y) <= ymax for x, y in polygons_list)
        print(count_inside,len(polygons_list) )
        # Check if at least 90% of the points are within the bounding box
        if count_inside >= 0.8 * len(polygons_list):
            filtered_polygons.append(polygons_list)
            indices.append(index)

    return indices

def find_closest_building(center_pixel_x, center_pixel_y, all_polygon_coords):
    center_pixel_x = float(center_pixel_x)
    center_pixel_y = float(center_pixel_y)
    closest_building = None
    min_avg_distance = float('inf')  # Initialize with positive infinity
    closest_building_index = None

    for index, polygon_coords in enumerate(all_polygon_coords):
        total_distance = 0
        num_points = len(polygon_coords)
        for building_point in polygon_coords:
            # Calculate the Euclidean distance between the center and each point on the building's outline
            distance = ((center_pixel_x - float(building_point[0])) ** 2 + (center_pixel_y - float(building_point[1])) ** 2) ** 0.5
            total_distance += distance
        avg_distance = total_distance / num_points
            # Update the closest building if the current building is closer
        if avg_distance < min_avg_distance:
            min_avg_distance = avg_distance
            closest_building = polygon_coords[index]
            closest_building_index = index

    return closest_building, closest_building_index

def get_polygon_outside_area(bbox, polygon_coords, ax):
    # Create a Shapely polygon from the bounding box
    miny, minx, maxy, maxx = bbox
    bbox_polygon = ShapelyPolygon(box(minx, miny, maxx, maxy))
    #bbox_polygon = ShapelyPolygon([(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny)])
    #xs, ys = bbox_polygon.exterior.xy
    #ax.fill(xs, ys, alpha=0.4, fc='g', ec='none')

    # Create a Shapely polygon from the outline coordinates
    outline_polygon = ShapelyPolygon(polygon_coords)
    xs, ys = outline_polygon.exterior.xy
    ax.fill(xs, ys, alpha=0.5, fc='r', ec='none')
    #label_text = "BUILDING"
    #ax.text(xs[0], ys[0], label_text, ha='left', va='bottom', color='black', fontsize=8)

    exterior_ring1= bbox_polygon.exterior
    bbox1 = isinstance(exterior_ring1, LinearRing) and exterior_ring1.is_closed
    #print("bbox", bbox1)
    exterior_ring2 = outline_polygon.exterior
    outline_polygon1 = isinstance(exterior_ring2, LinearRing) and exterior_ring2.is_closed
   # print("outline_polygon", outline_polygon1)

    # Calculate the difference to find the area outside the outline but within the bounding box
    outside_polygon2 = difference(bbox_polygon,outline_polygon)
    outside_polygon = ShapelyPolygon(shell=[(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny)], holes=[polygon_coords])
    #outside_polygon = symmetric_difference(outline_polygon, bbox_polygon)

    return outside_polygon


def write_polygons_to_csv(all_polygon_coords, output_csv_file):
    with open(output_csv_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Latitude', 'Longitude'])  # Write header

        for polygon_coords in all_polygon_coords:
            # Write a marker for the start of a new polygon
            csv_writer.writerow(['P'])

            # Write coordinates for the polygon
            for lat_long_pair in polygon_coords:
                csv_writer.writerow(lat_long_pair)

def getPointLatLng(x, y, lat, lng, w, h, zoom):
    parallelMultiplier = math.cos(lat * math.pi / 180)
    degreesPerPixelX = 360 / math.pow(2, zoom + 8)
    degreesPerPixelY = 360 / math.pow(2, zoom + 8) * parallelMultiplier
    pointLat = lat - degreesPerPixelY * ( y - h / 2)
    pointLng = lng + degreesPerPixelX * ( x  - w / 2)

    return (pointLat, pointLng)

def convert_cords_to_lat_long(lat, lng, csv_file_path, zoom, width, height):
    all_polygon_coords = []  # List to store coordinates for all polygons
    current_polygon_coords = []  # List to store coordinates for the current polygon
    all_xy_cords = []
    current_xy_cords = []

    with open(csv_file_path, 'r') as file:
        csv_reader = csv.reader(file, delimiter='\t')  # Assuming tab-separated values
        next(csv_reader)  # Skip the header row

        for row in csv_reader:
            if row[0] == 'P':
                # If "POLYGON_START" is encountered, start a new list for the next polygon
                if current_polygon_coords:
                    all_polygon_coords.append(current_polygon_coords)
                    current_polygon_coords = []  # Start a new list
                    all_xy_cords.append(current_xy_cords)
                    current_xy_cords = []  # Start a new list

            else:
                # Assuming the first column is x and the second column is y
                x_y_values = row[0].split(',')
                x_value = x_y_values[0].strip()
                y_value = x_y_values[1].strip() 

                lat_pt, long_pt = getPointLatLng(float(x_value), float(y_value), float(lat), float(lng), float(width), float(height), float(zoom))
                lat_long_pair = (lat_pt, long_pt)
                xy_pair = (x_value, y_value)
                # Add the (lat, long) pair to the current polygon's list
                current_polygon_coords.append(lat_long_pair)
                current_xy_cords.append(xy_pair)

        # Add the coordinates of the last polygon (if any)
        if current_polygon_coords:
            all_polygon_coords.append(current_polygon_coords)
            all_xy_cords.append(current_xy_cords)

    return all_xy_cords, all_polygon_coords
    
def save_co_ordinates(image, boxes, masks, class_ids, class_names):
    #image = image.split("/")[-1]
    image_data = []
    box_indices = []

    for i in range(boxes.shape[0]):

        class_id = class_ids[i]
        label = class_names[class_id]

        mask = masks[:, :, i]
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            verts = np.fliplr(verts) - 1
            list_co_ordinates = np.moveaxis(verts, 1, 0).tolist()

            region = {"shape_attributes": {"all_points_x": list_co_ordinates[0],
                                           "all_points_y": list_co_ordinates[1]},
                      "region_attributes": {"name": {label: True}}}
            image_data.append(region)
            box_indices.append(i)
    data = {"filename": image, "regions": image_data}
    return data, box_indices


def plot_regions(data, output_file, file_name_without_extension):
    # Load the image (assuming the image is in the same directory as the script)
    image_path = data["filename"]
    image = plt.imread(image_path)
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(image)

    # Generate a list of different colors for each polygon
    colors = plt.cm.jet(np.linspace(0, 1, len(data["regions"])))

    # Create a list to store (x, y) pairs for each polygon
    all_coordinates = []

    for i, region in enumerate(data["regions"]):
        shape_attributes = region["shape_attributes"]
        all_points_x = shape_attributes["all_points_x"]
        all_points_y = shape_attributes["all_points_y"]

        # Create a Polygon from the points with custom line thickness and color
        polygon = Polygon(list(zip(all_points_x, all_points_y)),
                          edgecolor=colors[i], linewidth=2, facecolor='none')
        ax.add_patch(polygon)
        all_coordinates.append("P")
        # Store (x, y) pairs for each coordinate in the polygon
        coordinates = list(zip(all_points_x, all_points_y))
        all_coordinates.append(coordinates)

    # Save the plot to the specified file
    print("SAVED ALL BUILDING OUTLINES OF MASK")
    plt.show()
    plt.savefig(output_file)
    #plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    # Save (x, y) pairs to a CSV file
    csv_file =  f"POLYGON_COORDINATES/polygon_cords_{file_name_without_extension}.csv"
    with open(csv_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['x', 'y'])  # Write header
        for coordinates in all_coordinates:
            csv_writer.writerows(coordinates)

def overlay_outside_area(image, bbox, polygon_coords, output_path, width, height):
        # Get the area outside the polygon but within the bounding box
        # fig, ax = plt.subplots(1, figsize=(width / 100, height / 100))
    image = plt.imread(image)
    fig, ax = plt.subplots(1, figsize=(8, 8))
    #modified_polygon = [(float(x) + 117, float(y) + 100) for x, y in polygon_coords]
    #polygon_coords[closest_building_index2] = modified_polygon
    outside_polygon = get_polygon_outside_area(bbox, polygon_coords, ax)

        # Create a Matplotlib figure and axis

        # Plot the original image
    ax.imshow(image)

        # Plot the bounding box using Shapely
    if isinstance(outside_polygon, MultiPolygon):
        for geom in outside_polygon:
            plot_polygon(ax, geom, alpha=0.5, facecolor='lightblue', edgecolor='red')
    else:
        plot_polygon(ax, outside_polygon, alpha=0.5, facecolor='lightblue', edgecolor='red')

        # Save the figure to the specified output path
    plt.tight_layout()
        # plt.subplots_adjust(left=0.0001, right=0.0001, top=0.0001, bottom=0.0001)
    plt.savefig(output_path, bbox_inches='tight')
    plt.show()
    plt.close()
    return outside_polygon

