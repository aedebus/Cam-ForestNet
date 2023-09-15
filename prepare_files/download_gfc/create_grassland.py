# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 14:28:13 2022

@author: Amandine Debus

This script creates a shapefile from the ESA WorldCover 2020 map for grassland

"""


import imageio
import numpy as np
from osgeo import gdalnumeric
from osgeo import gdal
from osgeo import ogr, osr
import fire
import os


def create_grassland():
    """

    Returns
    -------
    shapefile with all shapes of grassland
    
    """
    path_in = os.path.join(os.getcwd(), 'input', 'WorldCover')
    list_files = os.listdir(path_in)
    for file in list_files:
        img = imageio.imread(os.path.join(path_in, file))
        img_arr=np.array(img, dtype=np.uint8)
        path_out = os.path.join(os.getcwd(), 'input', 'WorldCover', file[:-4])

        if os.path.exists(path_out) == False:
            os.mkdir(path_out)
                 
        img_new = os.path.join(path_out, 'WorldCover.tiff') #output

        rgb = np.zeros((3, img_arr.shape[0], img_arr.shape[1],), np.uint8)
        suff1 = 20 #shrubland on QGIS
        suff2 = 30 #grassland on QGIS
        for i in range (len(img_arr)):
            for j in range (len(img_arr)):
                if img_arr[i][j]==int(suff1) or img_arr[i][j]==int(suff2):
                    for k in range (3):
                         rgb[k][i][j]=np.uint8(255)


        gdalnumeric.SaveArray(rgb.astype(gdalnumeric.numpy.uint8), img_new, format="GTIFF", prototype=os.path.join(os.getcwd(), 'input', 'WorldCover', file))

        img_new_sh=os.path.join(path_out, "WorldCover.shp") #output file
        img_new_layer=os.path.join(path_out, "WorldCover") #OGR layer
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

        WorldCover=gdal.Polygonize(band, mask, layer, dst_field, [], None)


def create_grassland_file(file):
    """

    Returns
    -------
    shapefile with all shapes of grassland

    """
    path_in = os.path.join(os.getcwd(), 'input', 'WorldCover')
    img = imageio.imread(os.path.join(path_in, file))
    img_arr = np.array(img, dtype=np.uint8)
    path_out = os.path.join(os.getcwd(), 'input', 'WorldCover', file[:-4])

    if os.path.exists(path_out) == False:
        os.mkdir(path_out)

    img_new = os.path.join(path_out, 'WorldCover.tiff')  # output

    rgb = np.zeros((3, img_arr.shape[0], img_arr.shape[1],), np.uint8)
    suff1 = 20  # shrubland on QGIS
    suff2 = 30  # grassland on QGIS
    for i in range(len(img_arr)):
        for j in range(len(img_arr)):
            if img_arr[i][j] == int(suff1) or img_arr[i][j] == int(suff2):
                for k in range(3):
                    rgb[k][i][j] = np.uint8(255)

    gdalnumeric.SaveArray(rgb.astype(gdalnumeric.numpy.uint8), img_new, format="GTIFF",
                          prototype=os.path.join(os.getcwd(), 'input', 'WorldCover', file))

    img_new_sh = os.path.join(path_out, "WorldCover.shp")  # output file
    img_new_layer = os.path.join(path_out, "WorldCover")  # OGR layer
    img_new_DS = gdal.Open(img_new)  # open input raster
    band = img_new_DS.GetRasterBand(1)  # first band
    mask = band  # gdal uses band as mask
    driver = ogr.GetDriverByName("ESRI Shapefile")
    shp = driver.CreateDataSource(img_new_sh)  # set up output

    # copy spatial reference
    img_sr = osr.SpatialReference()
    img_sr.ImportFromWkt(img_new_DS.GetProjectionRef())
    layer = shp.CreateLayer(img_new_layer, img_sr)

    # set up dbf file
    fd = ogr.FieldDefn("DN", ogr.OFTInteger)
    layer.CreateField(fd)
    dst_field = 0

    # Automatically extract features from an image

    WorldCover = gdal.Polygonize(band, mask, layer, dst_field, [], None)


def main():
    fire.Fire(create_grassland_file)

if __name__ == "__main__":
    main()
