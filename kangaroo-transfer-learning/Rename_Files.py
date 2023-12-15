# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 19:43:04 2023

@author: nathant
"""

import os
import re

RGB_PanSharpen_Path = "building/RGB-PanSharpen"
RGB_PanSharpen_Renamed_Path = "building/RGB-PanSharpen-Renamed"
Geojson_Path = "building/geojson/building"
Geojson_Renamed_Path = "building/geojson/building-renamed"

def extract_numeric_part(filename):
    # Extracts numeric part from the filename
    return re.search(r'\d+', filename).group()

tif_files = os.listdir(RGB_PanSharpen_Path)
geojson_files = os.listdir(Geojson_Path)

# Make sure both directories have the same number of files
if len(tif_files) == len(geojson_files):
    # Sort the files based on their numeric part
    tif_files.sort(key=lambda x: int(extract_numeric_part(x)))
    geojson_files.sort(key=lambda x: int(extract_numeric_part(x)))

    for idx, (tif_file, geojson_file) in enumerate(zip(tif_files, geojson_files), start=1):
        # Create new filenames with sequential numbers
        new_tif_name = f"{idx}.tif"
        new_geojson_name = f"{idx}.geojson"

        # Rename and move files to the renamed directories
        os.rename(os.path.join(RGB_PanSharpen_Path, tif_file), os.path.join(RGB_PanSharpen_Renamed_Path, new_tif_name))
        os.rename(os.path.join(Geojson_Path, geojson_file), os.path.join(Geojson_Renamed_Path, new_geojson_name))

    print("Files successfully renamed and moved to the renamed directories.")
else:
    print("Directories have a different number of files.")
