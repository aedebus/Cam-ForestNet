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


def polygon_from_csv(csv_file, type):
    """

    :param csv_file: csv file from Geowiki with labels
    :param type: LU type to extract
    :return: pickle file forest_loss and shapefile with polygon
    """
    #STEP 1: Extract coordinates for land use type chosen
    list_coord=[]
    #Shape Cameroon ref: Hijmans, Robert J. and University of California, Berkeley. Museum of Vertebrate Zoology (https://geodata.lib.utexas.edu/catalog/stanford-nn132pn3168)
    #This polygon shapefile contains the boundary of Cameroon (adm0). This layer is part of the Global Administrative Areas 2015 (v2.8) dataset. Hijmans, R. and University of California, Berkeley, Museum of Vertebrate Zoology. (2015).
    # Boundary, Cameroon, 2015. UC Berkeley, Museum of Vertebrate Zoology. Available at: http://purl.stanford.edu/nn132pn3168 This layer is presented in the WGS84 coordinate system for web display purposes.
    # Downloadable data are provided in native coordinate system or projection.
    cameroon=shapefile.Reader(os.path.join(os.getcwd(), 'input', 'GeoWiki', 'cameroon.shp'))
    shape_cameroon=cameroon.shapes()[0]
    polygon_cameroon=Polygon(shape_cameroon.points)
    with open(os.path.join(os.getcwd(), 'input', 'GeoWiki', csv_file), mode = 'r') as file:
        csvFile = csv.reader(file)
        for lines in csvFile:
            if 'controls' in csv_file:
                if lines[4]==str(type):
                    lon = float(lines[5])
                    lat = float(lines[6])
                    point = Point(lon,lat)
                    if polygon_cameroon.intersects(point): #check if coordinates in Cameroon
                        if 'buildings' in type:
                            if 'predominant' in lines[3]:
                                list_coord.append((lon,lat))
                        else:
                            list_coord.append((lon,lat))
            elif 'campaign' in csv_file:
                if lines[12] == str(type):
                    lon = float(lines[22])
                    lat = float(lines[23])
                    point = Point(lon, lat)
                    if polygon_cameroon.intersects(point):  # check if coordinates in Cameroon
                        if 'buildings' in type:
                            if 'predominant' in lines[11]:
                                list_coord.append((lon,lat))
                        else:
                            list_coord.append((lon,lat))

    for i in range(0, len(list_coord)):
    #STEP 2: Create polygon = box 1 km square with lon/lat centroid
    #Haversine formula: https://stackoverflow.com/questions/639695/how-to-convert-latitude-or-longitude-to-meters --> not needed for small distances
        lon = list_coord[i][0]
        lat = list_coord[i][1]
        #Approximate short distances: https://gis.stackexchange.com/questions/75528/understanding-terms-in-length-of-degree-formula
        m_per_deg_lat = 111132.954 - 559.822 * np.cos(2 * lat) + 1.175 * np.cos(4 * lat) - 0.0023 * np.cos(6 * lat)
        m_per_deg_lon = 111132.954 * np.cos(lat) - 93.5 * np.cos(3 * lat) + 0.118 * np.cos(5 * lat)
        deg_lat = 500 / m_per_deg_lat
        deg_lon = 500 / m_per_deg_lon

        polygon = Polygon ([[lon - deg_lon, lat - deg_lat], [lon - deg_lon, lat + deg_lat], [lon + deg_lon, lat + deg_lat], [lon + deg_lon, lat - deg_lat]])
    

    #STEP 3: Build pkl file from polygon
        year = 2020
        if '/' in type:
            type = type.replace('/', '_')
        path_shapes=os.path.join(os.getcwd(), 'output', str(year), 'GeoWiki_notGFC_' + str(type) +'.shp')
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
