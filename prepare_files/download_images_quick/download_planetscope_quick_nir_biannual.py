# -*- coding: utf-8 -*-
"""

@author: Amandine Debus
"""

import os
import fire
import ee
import shapefile
import pickle
import multiprocessing
import geemap
from retry import retry
import requests
import logging
import shutil


# def getRequests(year, shapes, path):
#   """Generates a list of work items to be downloaded.
#   """
#   list_dir = os.listdir(os.path.join(os.path.dirname(os.getcwd()), 'download_gfc', 'output', str(year), shapes))
#   list_points = []
#   for index in list_dir:
#       path_in = os.path.join(os.path.dirname(os.getcwd()), 'download_gfc', 'output', str(year), shapes, str(index))
#       if os.path.exists(path_in) == True:
#           r = shapefile.Reader(os.path.join(path_in, 'extract_polygon.shp'))
#           sh = r.shapes()[0]
#           loss = ee.Geometry.Polygon(sh.points)
#           loss_region = pickle.load(open(os.path.join(path_in, 'forest_loss_region.pkl'), 'rb'))
#           centroid_x = loss_region.centroid.xy[0][0]
#           centroid_y = loss_region.centroid.xy[1][0]
#           list_points.append((index, ee.Geometry.Point([centroid_x, centroid_y]), year, path))
#
#   return list_points

def getRequests(year):
  """Generates a list of work items to be downloaded.
  """
  list_dir = os.listdir(os.path.join(os.path.dirname(os.getcwd()), 'download_gfc', 'output', str(year)))
  list_points = []
  for shapes in list_dir:
    if (shapes != '10N_000E') and (shapes != '10N_010E') and (shapes != '20N_010E'):
        list_index = os.listdir(os.path.join(os.path.dirname(os.getcwd()), 'download_gfc', 'output', str(year), shapes))
        path_out_year = os.path.join(os.getcwd(), 'output', str(year))
        if os.path.exists(path_out_year) == False:
            os.mkdir(path_out_year)

        path_out_shapes = os.path.join(path_out_year, shapes)
        if os.path.exists(path_out_shapes) == False:
            os.mkdir(path_out_shapes)

        for index in list_index:
            path_in = os.path.join(os.path.dirname(os.getcwd()), 'download_gfc', 'output', str(year), shapes, str(index))
            if os.path.exists(path_in) == True:
                r = shapefile.Reader(os.path.join(path_in, 'extract_polygon.shp'))
                sh = r.shapes()[0]
                loss = ee.Geometry.Polygon(sh.points)
                loss_region = pickle.load(open(os.path.join(path_in, 'forest_loss_region.pkl'), 'rb'))
                centroid_x = loss_region.centroid.xy[0][0]
                centroid_y = loss_region.centroid.xy[1][0]
                list_points.append((index, ee.Geometry.Point([centroid_x, centroid_y]), year, path_out_shapes))

  return list_points

@retry(tries=10, delay=1, backoff=2)
def getResult(index, point, year, path):
  """Handle the HTTP requests to download an image."""

  # Generate the desired image from the given point.
  x = point.getInfo()["coordinates"][0]
  y = point.getInfo()["coordinates"][1]
  point = ee.Geometry.Point(point['coordinates']) #point as x and y coordinates
  region = point.buffer(166*4.77).bounds()
  

  if (year != 2020):  
    image = (ee.ImageCollection("projects/planet-nicfi/assets/basemaps/africa")
            .filterBounds(region)
            .filter(ee.Filter.eq('cadence', 'biannual'))
            .filterDate(str(year+1)+'-01-01', str(year+5)+'-12-31')
            .first()
            .clip(region)
            .select('R', 'G', 'B', 'N')
            .visualize(min=64, max=5454, gamma=1.8))
  else: 
    image = (ee.ImageCollection("projects/planet-nicfi/assets/basemaps/africa")
            .filterBounds(region)
            .filter(ee.Filter.eq('cadence', 'biannual'))
            .filterDate(str(year)+'-01-01', str(year)+'-12-31')
            .first()
            .clip(region)
            .select('R', 'G', 'B', 'N')
            .visualize(min=64, max=5454, gamma=1.8))  

    # Fetch the URL from which to download the image.
  url = image.getThumbURL({
        'region': region,
        'dimensions': '332x332',
        'format': 'png'})

    # Handle downloading the actual pixels.
  r = requests.get(url, stream=True)
  if r.status_code != 200:
    raise r.raise_for_status()

  if os.path.exists(os.path.join(path, str(index))) == False:
    os.mkdir(os.path.join(path, str(index)))

  if os.path.exists(os.path.join(path, str(index), 'planet_nir_biannual')) == False:
    os.mkdir(os.path.join(path, str(index), 'planet_nir_biannual'))

  filename = os.path.join(path, str(index), 'planet_nir_biannual', str(year)+'_'+str(x)+'_'+ str(y)+'_'+str(index)+'.png') #remove: str(index)
  with open(filename, 'wb') as out_file:
    shutil.copyfileobj(r.raw, out_file)
  print("Done: ", index)

# def planetscope_from_gfc(year, shapes):
#     """
#
#
#     Parameters
#     ----------
#     year : int
#         year of the forest loss event.
#     shapes : str
#         name of the shapefile where we know land-use.
#
#     Returns
#     -------
#     planetscope png image with size 332 x 332 pixels centered on the gfc shape
#
#     """
#     #STEP 1: Access Google Earth Engine
#     #ee.Authenticate() #Done once in generate_all_landsat
#     ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
#     print('init ok')
#     path_out_year = os.path.join(os.getcwd(), 'output', str(year))
#     if os.path.exists(path_out_year) == False:
#         os.mkdir(path_out_year)
#
#     path_out_shapes = os.path.join(path_out_year, shapes)
#     if os.path.exists(path_out_shapes) == False:
#         os.mkdir(path_out_shapes)
#
#     items = getRequests(year, shapes, path_out_shapes)
#     pool = multiprocessing.Pool(25)
#     pool.starmap(getResult, items)
#     pool.close()
#     pool.join()

def planetscope_from_gfc(year):
    """
    without shapes

    Parameters
    ----------
    year : int
        year of the forest loss event.

    Returns
    -------
    planetscope png image with size 332 x 332 pixels centered on the gfc shape

    """
    # STEP 1: Access Google Earth Engine
    # ee.Authenticate() #Done once in generate_all_landsat
    ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
    print('init ok')

    items = getRequests(year)
    pool = multiprocessing.Pool(25)
    pool.starmap(getResult, items)
    pool.close()
    pool.join()

def main():
    fire.Fire(planetscope_from_gfc)

if __name__ == "__main__":
    main()
