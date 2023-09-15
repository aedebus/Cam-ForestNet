# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 14:28:13 2022

@author: Amandine Debus

This script creates a shapefile from the ESA WorldCover 2020 map for grassland

"""


import shapefile
import numpy as np
import fire
import os
from shapely.geometry import shape as shapely_shape, Polygon, MultiPolygon, Point
from osgeo import gdalnumeric
from osgeo import gdal
from osgeo import ogr, osr


def create_grassland_small(file):
    """

    Returns
    -------
    shapefile with all shapes of grassland

    """
    shape = shapefile.Reader(os.path.join(os.getcwd(), 'input', 'WorldCover', file, 'WorldCover.shp'))
    all_shapes= shape.shapes()
    if len(all_shapes)>1:
        index = int(input("Enter an integer between 0 and " + str (len(all_shapes)-1)+ ':'))
    else:
        index=0

    for i in range(index):
        s=all_shapes[i]
        polygon=Polygon(s.points)


        driver = ogr.GetDriverByName("ESRI Shapefile")
        path_out = os.path.join(os.getcwd(), 'input', 'WorldCover', file, 'shapes')
        if os.path.exists(path_out) == False:
            os.mkdir(path_out)
        path_out_index = os.path.join(os.getcwd(), 'input', 'WorldCover', file, 'WorldCover_shapes', str(i))
        if os.path.exists(path_out_index) == False:
            os.mkdir(path_out_index)
        ds = driver.CreateDataSource(os.path.join(path_out_index, "WorldCover_small.shp"))  # set up output
        layer = ds.CreateLayer('', None, ogr.wkbPolygon)
        layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))
        defn = layer.GetLayerDefn()
        fd = ogr.FieldDefn("DN", ogr.OFTInteger)
        layer.CreateField(fd)
        feat = ogr.Feature(defn)
        feat.SetField('id', 123)
        geom = ogr.CreateGeometryFromWkb(polygon.wkb)
        feat.SetGeometry(geom)
        layer.CreateFeature(feat)
        feat = geom = None
        ds = layer = feat = geom = None


def main():
    fire.Fire(create_grassland_small)

if __name__ == "__main__":
    main()
