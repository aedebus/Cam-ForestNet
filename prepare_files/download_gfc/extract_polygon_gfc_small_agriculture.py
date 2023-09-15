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


def polygon_from_shapes(shapes, year):
    """
    Parameters
    ----------
    shapes : shp
        shapefile where we know the land use
    year : int
        to select the year of the forest loss events
        
    Returns
    -------
    pickle file forest_loss and shapefile with polygon

    """
    #STEP 1: Extract one shape from shapes
    r1= shapefile.Reader(os.path.join(os.getcwd(), 'input', 'Shapes', shapes))
    all_shapes_1=r1.shapes()
    #index=random.randint(0, len(all_shapes_1)) #NB: not using random selection of the shapes to maximize the number of images that can be obtained from one known shapefile
    # if len(all_shapes_1)>1:
    #     index = int(input("Enter an integer between 0 and " + str (len(all_shapes_1)-1)+ ':'))
    # else: 
    #     index=0 #NB: too many shapes sometimes 
    for index in range (0, len(all_shapes_1)): #to extract all shapes
        shape=all_shapes_1[index]
  
    #STEP 2: Depending on the shape, choose img_input ('10N_010E' or '20N_010E' or '10N_000E' = 3 zones covering Cameroon)
        img_input=[]
        x_min=shape.points[0][0]
        y_min=shape.points[0][1]
        x_max=shape.points[0][0]
        y_max=shape.points[0][1]
    
        if x_min<=10:
            img_input.append('10N_000E')
        if x_max>=10:
            img_input.append('10N_010E')
        if y_max>=10:
            img_input.append('20N_010E')

        point = Point(shape.points[0][0],shape.points[0][1])
    #STEP 3: Extract one polygon from GFC
        r2 = shapefile.Reader(os.path.join(os.getcwd(), 'output', str(year), img_input[0], 'extract.shp'))
        all_shapes_2=r2.shapes()
        cont=True
        count=0
        while cont and count<len(all_shapes_2):
            polygon_shape=Polygon(all_shapes_2[count].points)
            if polygon_shape.intersects(point):
                polygon = polygon_shape
                cont = False
            else:
                count+=1
    
        if cont==True and len(img_input)>1: #if not found in img_input[0], look in other img_input
            r2 = shapefile.Reader(os.path.join(os.getcwd(), 'output', str(year), img_input[1], 'extract.shp'))
            all_shapes_2=r2.shapes()
            count=0
            while cont and count<len(all_shapes_2):
                polygon_shape=Polygon(all_shapes_2[count].points)
                if polygon_shape.intersects(point):
                    polygon = polygon_shape
                    cont = False
                else:
                    count+=1
                
            if cont==True and len(img_input)>2: #if not found in img_input[1], look in other img_input
                r2 = shapefile.Reader(os.path.join(os.getcwd(), 'output', str(year), img_input[2], 'extract.shp'))
                all_shapes_2=r2.shapes()
                count=0
                while cont and count<len(all_shapes_2):
                    polygon_shape=Polygon(all_shapes_2[count].points)
                    if polygon_shape.intersects(point):
                        polygon = polygon_shape
                        cont = False
                    else:
                        count += 1
        if cont==True:
            print(str(index)+':'+'Change year')
    
        else: 
    #STEP 4: Build pkl file from polygon 
            path_shapes=os.path.join(os.getcwd(), 'output', str(year), shapes)
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
