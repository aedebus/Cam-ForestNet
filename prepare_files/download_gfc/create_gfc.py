# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 14:28:13 2022

@author: Amandine Debus

This script creates a shapefile from the Global Forest Change tiff file by selecting the forest loss in 2020  

"""

#Create shape files for forest loss in 2020: http://geospatialpython.com/2013/11/extracting-features-from-images.html

import imageio
import numpy as np
from osgeo import gdalnumeric
from osgeo import gdal
from osgeo import ogr, osr
import fire
import os


def create_gfc(year, img_input):
    """
    Parameters
    ----------
    year : int
        to select the year of the forest loss event we want to separate
    img_input : str
        name of the .tiff GFC image to use in the input file

    Returns
    -------
    shapefile with all the shapes of forest loss in a given year
    
    """
    
#STEP 1: Create black and white picture with white for zones where forest loss in 2020 
    img = imageio.imread(os.path.join(os.getcwd(), 'input', img_input))
    img_arr=np.array(img, dtype=np.uint8)
    path_out_year=os.path.join(os.getcwd(), 'output', str(year))
    path_out=os.path.join(os.getcwd(), 'output', str(year), img_input[30:-4])
    
    if os.path.exists(path_out_year) == False:
        os.mkdir(path_out_year)
    if os.path.exists(path_out) == False:
        os.mkdir(path_out)
                 
    img_new=os.path.join(path_out, 'GFC.tiff') #output

    rgb = np.zeros((3, img_arr.shape[0], img_arr.shape[1],), np.uint8)
    suff=str(year)[2:]
    for i in range (len(img_arr)):
        for j in range (len(img_arr)):
            if img_arr[i][j]==int(suff):
                for k in range (3):
                    rgb[k][i][j]=np.uint8(255)
                

    gdalnumeric.SaveArray(rgb.astype(gdalnumeric.numpy.uint8), img_new, format="GTIFF", prototype=os.path.join(os.getcwd(), 'input', img_input))

#STEP 2: Select isolated pixels and create shapefile

    img_new_sh=os.path.join(path_out, "extract.shp") #output file
    img_new_layer=os.path.join(path_out, "extract") #OGR layer
    img_new_DS=gdal.Open(img_new) #open input raster
    band=img_new_DS.GetRasterBand(1) #first band
    mask=band #gdal uses band as mask
    driver= ogr.GetDriverByName("ESRI Shapefile")
    shp = driver.CreateDataSource(img_new_sh) #set up output

    #copy spatial reference
    img_sr=osr.SpatialReference() 
    img_sr.ImportFromWkt(img_new_DS.GetProjectionRef())
    layer=shp.CreateLayer(img_new_layer, img_sr)

    #set up dbf file
    fd=ogr.FieldDefn("DN", ogr.OFTInteger)
    layer.CreateField(fd)
    dst_field = 0

    #Automatically extract features from an image

    extract=gdal.Polygonize(band, mask, layer, dst_field, [], None)


def main():
    fire.Fire(create_gfc)

if __name__ == "__main__":
    main()
