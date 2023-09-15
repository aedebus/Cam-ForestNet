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



def panSharpenL8(image): #https://springboardstag.co.uk/gis.stackexchange.com/questions/296615/pansharpen-landsat-mosaic-in-google-earth-engine
    rgb = image.select('B4', 'B3', 'B2')
    pan = image.select('B8')
    huesat = rgb.rgbToHsv().select('hue', 'saturation')
    upres = ee.Image.cat(huesat, pan).hsvToRgb()
    return image.addBands(upres)

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
  """Generates a list of work items to be downloaded - without shapes
  """
  list_dir = os.listdir(os.path.join(os.path.dirname(os.getcwd()), 'download_gfc', 'output', str(year)))
  list_points = []
  for shapes in list_dir:
      if 'additional' in shapes:
        if (shapes != '10N_000E') and (shapes != '10N_010E') and (shapes != '20N_010E'):
            list_index= os.listdir(os.path.join(os.path.dirname(os.getcwd()), 'download_gfc', 'output', str(year), shapes))
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
                    list_points.append((index, ee.Geometry.Point([centroid_x, centroid_y]), year, os.path.join(path_out_shapes)))

  return list_points

@retry(tries=10, delay=1, backoff=2)
def getResult(index, point, year, path):
  """Handle the HTTP requests to download an image."""

  # Generate the desired image from the given point.
  x = point.getInfo()["coordinates"][0]
  y = point.getInfo()["coordinates"][1]
  point = ee.Geometry.Point(point['coordinates'])
  region = point.buffer(166*15).bounds()
  #No reflectance and no composites
  # #Check image exists:
  filtered_images = ee.ImageCollection("LANDSAT/LC08/C02/T1_TOA")\
                    .filterBounds(region)\
                    .filterDate(str(year+1)+'-01-01', str(year+5)+'-12-31')\
                    .filterMetadata('CLOUD_COVER', 'less_than', 20)\
                    .filterMetadata('CLOUD_COVER_LAND', 'less_than', 20)

  if (len(filtered_images.getInfo()['features'])>0):
      image = (ee.ImageCollection("LANDSAT/LC08/C02/T1_TOA")
               .filterBounds(region)
               .filterDate(str(year+1)+'-01-01', str(year+5)+'-12-31')
               .map(panSharpenL8)
               .filterMetadata('CLOUD_COVER', 'less_than', 20)
               .filterMetadata('CLOUD_COVER_LAND', 'less_than', 20)
               .sort('CLOUD_COVER')
               .first()
               .clip(region)
               .select('B4', 'B3', 'B2')).visualize(min=0.0, max=0.4)

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

      if os.path.exists(os.path.join(path, str(index), 'landsat')) == False:
          os.mkdir(os.path.join(path, str(index), 'landsat'))

      filename = os.path.join(path, str(index), 'landsat', str(year)+'_'+str(x)+'_' + str(y)+'.png')
      with open(filename, 'wb') as out_file:
          shutil.copyfileobj(r.raw, out_file)
      print("Done: ", index)
  else:
      print('No image for index:', index)



# def landsat_from_gfc(year, shapes):
#     """
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
#     landsat png image with size 332 x 332 pixels centered on the gfc shape
#
#     """
#     #ee.Authenticate()
#     ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
#     print('init ok')
#     path_out_year=os.path.join(os.getcwd(), 'output', str(year))
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

def landsat_from_gfc(year):
    """
    without shapes
    Parameters
    ----------
    year : int
        year of the forest loss event

    Returns
    -------
    landsat png image with size 332 x 332 pixels centered on the gfc shape

    """
    #ee.Authenticate()
    ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
    print('init ok')

    items = getRequests(year)
    pool = multiprocessing.Pool(25)
    pool.starmap(getResult, items)
    pool.close()
    pool.join()


def main():
    fire.Fire(landsat_from_gfc)


if __name__ == "__main__":
    main()
