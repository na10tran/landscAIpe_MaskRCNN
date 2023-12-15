# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 16:12:04 2023

@author: nathant
"""

# -*- coding: utf-8 -*-

from osgeo import gdal
import numpy as np
import os
import subprocess
RGB_PanSharpen_Renamed_Path = "building/RGB-PanSharpen-Renamed/"
RGB_PanSharpen_NEW_Path = "building/RGB-PanSharpen-NEW/"

def bit16_to_8Bit(inputRaster, outputRaster, outputPixType='Byte', outputFormat='png', percentiles=[2, 98]):
    '''
    Convert 16bit image to 8bit
    Source: Medium.com, 'Creating Training Datasets for the SpaceNet Road Detection and Routing Challenge' by Adam Van Etten and Jake Shermeyer
    '''

    srcRaster = gdal.Open(inputRaster)
    cmd = ['gdal_translate', '-ot', outputPixType, '-of',
           outputFormat]

    # iterate through bands
    for bandId in range(srcRaster.RasterCount):
        bandId = bandId+1
        band = srcRaster.GetRasterBand(bandId)

        bmin = band.GetMinimum()
        bmax = band.GetMaximum()
        # if not exist minimum and maximum values
        if bmin is None or bmax is None:
            (bmin, bmax) = band.ComputeRasterMinMax(1)
        # else, rescale
        band_arr_tmp = band.ReadAsArray()
        bmin = np.percentile(band_arr_tmp.flatten(),
                             percentiles[0])
        bmax= np.percentile(band_arr_tmp.flatten(),
                            percentiles[1])

        cmd.append('-scale_{}'.format(bandId))
        cmd.append('{}'.format(bmin))
        cmd.append('{}'.format(bmax))
        cmd.append('{}'.format(0))
        cmd.append('{}'.format(255))
    cmd.append(inputRaster)
    cmd.append(outputRaster)
    print("Conversin command:", cmd)
    subprocess.call(cmd)

files = os.listdir(RGB_PanSharpen_Renamed_Path)
files = sorted(files, key=lambda x: int(x.split('.')[0]))

for file in files:
    if file.endswith('.tif'):
        resimPath = os.path.join(RGB_PanSharpen_Renamed_Path, file)
        dstPath = RGB_PanSharpen_NEW_Path
        filename, _ = os.path.splitext(file)
        dstFile = filename + ".png"
        dstPath = os.path.join(dstPath, dstFile)
        bit16_to_8Bit(resimPath, dstPath)
