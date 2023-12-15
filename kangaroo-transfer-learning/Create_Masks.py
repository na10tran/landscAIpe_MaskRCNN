# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 16:13:41 2023

@author: nathant
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import json
import geoio
RGB_PanSharpen_Renamed_Path = "building/RGB-PanSharpen-Renamed/"
RGB_PanSharpen_NEW_Path = "building/RGB-PanSharpen-NEW/"
Geojson_Renamed_Path = "building/geojson/building-renamed/"
MASKS_Path = "building/MASKS/"

# All the pictures from the given path
ResimAdlari = os.listdir(RGB_PanSharpen_Renamed_Path)
# All the geojson-files from the given path
GeoAdlari = os.listdir(Geojson_Renamed_Path)

# DosyaNumarasi takes the number from 0 to how big your training dataset
#for DosyaNumarasi in range(len(GeoAdlari)):
for DosyaNumarasi in range(3563, len(GeoAdlari)):
    # Your code here
    # DosyaNumarasi will take values from 2777 to len(GeoAdlari) - 1
    # You can use DosyaNumarasi as an index to access elements in GeoAdlari
    # Example: GeoAdlari[DosyaNumarasi

    # data holds the info for the each GeoJSON-file
    with open(Geojson_Renamed_Path + f'{DosyaNumarasi+1}.geojson') as f:
        data = json.load(f)

    # RGBTIFResmi holds the info for each TIF-file
    RGBTIFResmi = geoio.GeoImage(RGB_PanSharpen_Renamed_Path + f'{DosyaNumarasi+1}.tif')

    cokgenler = [] # Hold the coordinates for each building in the picture.
                   # (Outside for loop is for each picture, and here, cokgenler
                   # will hold the coordinates for each building in one picture.)
    types = [] # Holds the type of the buldings (MultiPolygon - Partial Building - Point)
               # We are not interested in the points.

    # Create the pane size of 650x650 to put the figures from geojson-file. Otherwise,
    # the buildings may be flipped or they may saved one by one
    arkaPlan = np.zeros([650,650])
    plt.imshow(arkaPlan)

    try:
        # We do not know how many buildings the picture includes.
        # So, we just give very big number to make sure that we utilized
        # all the buildings in one picture.
        # In short, bina keeps what order the building is.
        for bina in range(2000):
            tip = str(data['features'][bina]['geometry']['type'])
            types.append(tip) # Append all the type of the buildings

            # If type is point, do not do anything
            if tip == ('Point'):
                pass

            # If type is MultiPolygon, cokgenler will hold the coordinates
            elif tip == ('MultiPolygon'):
                kucukBinalar = (data['features'][bina]['geometry']['coordinates'])
                for b in range(len(kucukBinalar)):
                    cokgenler.append(kucukBinalar[b])

            # For the rest of the types, cokgenler will hold the coordinates again
            else:
                cokgenler.append(data['features'][bina]['geometry']['coordinates'])

    except IndexError:
        # If we utilized all the buildings in the given picture,
        # lest create mask for each one.

        # cokgenBina holds the each building's coordinates
        for cokgenBina in cokgenler:

            # binaNoktalari holds the individual edge coordinates for each building.
            for binaNoktalari in cokgenBina:

                # To hold the edge coordinates (in pixel form)
                doldurX = []
                doldurY = []

                # noktas holds x and y for each edge coordinate
                for noktas in binaNoktalari:

                    # Convert Latitude&Longitude to the pixels
                    xPixel, yPixel = RGBTIFResmi.proj_to_raster(noktas[0], noktas[1])

                    # The pixels may be 650 which defaces the masks.
                    xPixel = 649 if xPixel > 649 else xPixel
                    yPixel = 649 if yPixel > 649 else yPixel

                    # Keep x and y in pixel form
                    doldurX.append(xPixel)
                    doldurY.append(yPixel)

                # To paint between given pixel values
                plt.fill_between(doldurX, doldurY, facecolor='red')

                # To remove white area around matplotlib figure
                fig = plt.figure(1)
                extent = plt.gca().get_window_extent().transformed(fig.dpi_scale_trans.inverted())

                # Adjust the DPI for 650x650
                # and save the figure
                # While saving, you should put them in order; 0 to ...
                fig.savefig(MASKS_Path + str(DosyaNumarasi+1)+'.png', bbox_inches=extent, dpi=215.24)

        # Close the figure after an image is done.
        plt.close()