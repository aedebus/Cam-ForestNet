# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 17:54:27 2022

@author: Amandine Debus

This script extracts individual polygon from GFC_2020 and generates pkl files following the format of the ForestNet dataset
"""
import shapefile
from shapely.geometry import shape as shapely_shape, Polygon, MultiPolygon, Point
import pickle
from osgeo import gdal
from osgeo import ogr, osr
import os
import random
import fire


def polygon_from_shapes(typ):
    """
        
    Returns
    -------
    pickle file forest_loss and shapefile with polygon

    """

    year = 2020
    if typ == 'maize':
        r1= shapefile.Reader(os.path.join(os.getcwd(), 'input', 'WorldCereal', 'worldcereal_maize_intersection.shp'))
    elif typ == 'winter cereal':
        r1= shapefile.Reader(os.path.join(os.getcwd(), 'input', 'WorldCereal', 'worldcereal_winter_cereal_intersection.shp'))
    all_shapes_1=r1.shapes()
    #maximum = int(input("Enter an integer between 0 and " + str (len(all_shapes_1)-1)+ ':')) #otherwise too many shapes
    

    for index in range (0, len(all_shapes_1)-1): #to extract all shapes
        shape=all_shapes_1[index]
        polygon=Polygon(shape.points)
        if typ == 'maize':
            path_shapes=os.path.join(os.getcwd(), 'output', str(year), 'maize.shp')
        elif typ == 'winter cereal':
            path_shapes=os.path.join(os.getcwd(), 'output', str(year), 'winter_cereal.shp')
        if os.path.exists(path_shapes) == False:
            os.mkdir(path_shapes)
                
        path_out=os.path.join(path_shapes, str(index))
        if os.path.exists(path_out) == False:
            os.mkdir(path_out)
            
        outfile=open(os.path.join(path_out, 'forest_loss_region.pkl'), 'wb')
        pickle.dump(polygon, outfile)
        outfile.close()
        
        # STEP 5: Save polygon in shapefile : https://gis.stackexchange.com/questions/52705/how-to-write-shapely-geometries-to-shapefiles
        driver= ogr.GetDriverByName("ESRI Shapefile")
        ds = driver.CreateDataSource(os.path.join(path_out,"extract_polygon.shp")) #set up output
        layer=ds.CreateLayer('', None, ogr.wkbPolygon)
        layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))
        defn = layer.GetLayerDefn()
        fd=ogr.FieldDefn("DN", ogr.OFTInteger)
        layer.CreateField(fd)
        feat = ogr.Feature(defn)
        feat.SetField('id', 123)
        geom = ogr.CreateGeometryFromWkb(polygon.wkb)
        feat.SetGeometry(geom)
        layer.CreateFeature(feat)
        feat = geom = None
        ds = layer = feat = geom = None
        
            
        
def main():
    fire.Fire(polygon_from_shapes)

if __name__ == "__main__":
    main()
