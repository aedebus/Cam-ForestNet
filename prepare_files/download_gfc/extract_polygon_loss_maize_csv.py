# -*- coding: utf-8 -*-
"""
Created on 7 July 2022

@author: Amandine Debus

"""
import shapefile
from shapely.geometry import shape as shapely_shape, Polygon, MultiPolygon, Point
import pickle
from osgeo import gdal
from osgeo import ogr, osr
import os
import random
import fire
import csv
import numpy as np


def polygon_from_csv(csv_file):
    """

    :param csv_file: csv file from WorldCereal
    :return: pickle file forest_loss and shapefile with polygon
    """
    #STEP 1: Extract coordinates for land use type chosen
    list_coord=[]
    with open(os.path.join(os.getcwd(), 'input', 'WorldCereal', csv_file), mode = 'r') as file:
        csvFile = csv.reader(file)
        for lines in csvFile:
                if 'longitude' not in lines:
                    lon = float(lines[4])
                    lat = float(lines[3])
                    point = Point(lon, lat)
                    list_coord.append((lon,lat))

    for i in range(0, len(list_coord)):
        img_input=[]
        x=list_coord[i][0]
        y=list_coord[i][1]

        if x<=10:
            img_input.append('10N_000E')
        if x>=10:
            img_input.append('10N_010E')
        if y>=10:
            img_input.append('20N_010E')

        point = Point(x,y)
        #STEP 2: Extract one polygon from GFC
        r2 = shapefile.Reader(os.path.join(os.getcwd(), 'output', '2017', img_input[0], 'extract.shp'))
        all_shapes_2=r2.shapes()
        cont=True
        count=0
        while cont and count<len(all_shapes_2):
            polygon_shape=Polygon(all_shapes_2[count].points)
            if point.within(polygon_shape):
                polygon = polygon_shape
                cont = False
            else:
                count+=1

        if cont==True and len(img_input)>1: #if not found in img_input[0], look in other img_input
            r2 = shapefile.Reader(os.path.join(os.getcwd(), 'output', '2017', img_input[1], 'extract.shp'))
            all_shapes_2=r2.shapes()
            count=0
            while cont and count<len(all_shapes_2):
                polygon_shape=Polygon(all_shapes_2[count].points)
                if point.within(polygon_shape):
                    polygon = polygon_shape
                    cont = False
                else:
                    count+=1

            if cont==True and len(img_input)>2: #if not found in img_input[1], look in other img_input
                r2 = shapefile.Reader(os.path.join(os.getcwd(), 'output', '2017', img_input[2], 'extract.shp'))
                all_shapes_2=r2.shapes()
                count=0
                while cont and count<len(all_shapes_2):
                    polygon_shape=Polygon(all_shapes_2[count].points)
                    if point.within(polygon_shape):
                        polygon = polygon_shape
                        cont = False
                    else:
                        count += 1
        if cont==True:
            print(str(i)+':'+'Change year')

        else:
    
    #STEP 3: Build pkl file from polygon
            path_shapes=os.path.join(os.getcwd(), 'output', '2017', 'maize.shp')
            if os.path.exists(path_shapes) == False:
                os.mkdir(path_shapes)

            path_out=os.path.join(path_shapes, str(i))
            if os.path.exists(path_out) == False:
                os.mkdir(path_out)

            outfile=open(os.path.join(path_out, 'forest_loss_region.pkl'), 'wb')
            pickle.dump(polygon, outfile)
            outfile.close()

        # STEP 4: Save polygon in shapefile : https://gis.stackexchange.com/questions/52705/how-to-write-shapely-geometries-to-shapefiles
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
    fire.Fire(polygon_from_csv)

if __name__ == "__main__":
    main()
