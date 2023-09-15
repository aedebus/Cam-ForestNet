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

#TO GENERATE MORE FOREST LOSS PATCHES WITHIN THE KNOWN SHAPEFILES

    
    
def polygon_from_shapes_x4():
    """
    
    """
    #STEP 1: Extract one shape from shapes
    for year in ['2015', '2016', '2017', '2019', '2020']:
        r1= shapefile.Reader(os.path.join(os.getcwd(), 'input', 'Shapes', 'agro_industrial_plantations_' + year + '.shp'))
        all_shapes_1=r1.shapes()
        #index=random.randint(0, len(all_shapes_1)) #NB: not using random selection of the shapes to maximize the number of images that can be obtained from one known shapefile
        # if len(all_shapes_1)>1:
        #     index = int(input("Enter an integer between 0 and " + str (len(all_shapes_1)-1)+ ':'))
        # else: 
        #     index=0 #NB: too many shapes sometimes 
        if year != '2020':
            list_indices = [11, 12, 13, 15]
        else:
            list_indices = [11, 12 ,14 , 83]

        for index in list_indices: #to extract all shapes
            shape=all_shapes_1[index]
            shape_polygon=Polygon(shape.points)
    
        #STEP 2: Depending on the shape, choose img_input ('10N_010E' or '20N_010E' or '10N_000E' = 3 zones covering Cameroon)
            img_input=[]
            x_min=shape_polygon.bounds[0]
            y_min=shape_polygon.bounds[1]
            x_max=shape_polygon.bounds[2]
            y_max=shape_polygon.bounds[3]
        
            if x_min<=10:
                img_input.append('10N_000E')
            if x_max>=10:
                img_input.append('10N_010E')
            if y_max>=10:
                img_input.append('20N_010E')
        
        #STEP 3: Extract one polygon from GFC
            r2 = shapefile.Reader(os.path.join(os.getcwd(), 'output', str(year), img_input[0], 'extract.shp'))
            all_shapes_2=r2.shapes()
            cont=True
            count=0
            while cont and count<len(all_shapes_2):
                polygon_shape=all_shapes_2[count]
                inside = True 
                for p in polygon_shape.points:
                    point=Point(p[0], p[1])  #https://stackoverflow.com/questions/52149075/how-can-i-check-if-a-polygon-contains-a-point
                    if shape_polygon.intersects(point):
                        inside= inside and True
                    else:
                        inside= inside and False
                    
                if inside==True:
                    polygon_used=Polygon(polygon_shape.points) #Already the one that we have generated previously
                    cont=False
                    cont2 = True
                    count+=1
                    while cont2 and count<len(all_shapes_2):
                        polygon_shape=all_shapes_2[count]
                        inside = True 
                        for p in polygon_shape.points:
                            point=Point(p[0], p[1])  #https://stackoverflow.com/questions/52149075/how-can-i-check-if-a-polygon-contains-a-point
                            if shape_polygon.intersects(point):
                                inside= inside and True
                            else:
                                inside= inside and False
                        
                        if inside==True:
                            polygon_used2=Polygon(polygon_shape.points)
                            cont2=False
                            cont3=True
                            count+=1
                            while cont3 and count<len(all_shapes_2):
                                polygon_shape=all_shapes_2[count]
                                inside = True 
                                for p in polygon_shape.points:
                                    point=Point(p[0], p[1])  #https://stackoverflow.com/questions/52149075/how-can-i-check-if-a-polygon-contains-a-point
                                    if shape_polygon.intersects(point):
                                        inside= inside and True
                                    else:
                                        inside= inside and False
                                if inside==True:
                                    polygon_used3=Polygon(polygon_shape.points)
                                    cont3=False
                                    cont4 = True
                                    count+=1
                                    while cont4 and count<len(all_shapes_2):
                                        polygon_shape=all_shapes_2[count]
                                        inside = True 
                                        for p in polygon_shape.points:
                                            point=Point(p[0], p[1])  #https://stackoverflow.com/questions/52149075/how-can-i-check-if-a-polygon-contains-a-point
                                            if shape_polygon.intersects(point):
                                                inside= inside and True
                                            else:
                                                inside= inside and False
                                        if inside==True:
                                            polygon = Polygon(polygon_shape.points)
                                            cont4 = False
                                        else:
                                            count+=1
                                else:
                                     count+=1
                        else:
                            count+=1

                else:
                    count+=1
                
                
            if cont4==True:
                print(str(index)+':'+'Change year')
            
            else: 
            #STEP 4: Build pkl file from polygon 
                path_shapes=os.path.join(os.getcwd(), 'output', str(year), 'agro_industrial_plantations_' + year + '.shp_additional_fruit')
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



def polygon_from_shapes():
    polygon_from_shapes_x4()

        
def main():
    fire.Fire(polygon_from_shapes)

if __name__ == "__main__":
    main()
