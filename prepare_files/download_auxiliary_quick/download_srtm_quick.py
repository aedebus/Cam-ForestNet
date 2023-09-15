'''Adapted from Stanford ML Group (Irvin et al., 2020) by Amandine Debus '''

'''REFERENCES FOR DATASET USED
Citation:  NASA JPL (2020). NASADEM Merged DEM Global 1 arc second V001 [Data set]. NASA EOSDIS Land Processes DAAC. Accessed 2020-12-30 from doi:10.5067/MEaSUREs/NASADEM/NASADEM_HGT.001 
Link GEE: https://developers.google.com/earth-engine/datasets/catalog/NASA_NASADEM_HGT_001#citations '''

import fire
import numpy as np
import os
import time
from PIL import Image
import pickle
import multiprocessing
#import geemap
from retry import retry
import requests
import logging
import shutil

def getRequests(year):
  """Generates a list of work items to be downloaded - without shapes
  """
  import ee
  list_dir = os.listdir(os.path.join(os.path.dirname(os.getcwd()), 'download_images_quick', 'output', str(year)))
  list_points = []
  for shapes in list_dir:
            list_index= os.listdir(os.path.join(os.path.dirname(os.getcwd()), 'download_images_quick', 'output', str(year), shapes))
            path_out_year = os.path.join(os.getcwd(), 'output', str(year))
            if os.path.exists(path_out_year) == False:
                os.mkdir(path_out_year)

            path_out_shapes = os.path.join(path_out_year, shapes)
            if os.path.exists(path_out_shapes) == False:
                os.mkdir(path_out_shapes)

            for index in list_index:
                list_sensors=os.listdir(os.path.join(os.path.dirname(os.getcwd()), 'download_images_quick', 'output', str(year), shapes, index))
                for sensor in list_sensors:
                    list_images = os.listdir(os.path.join(os.path.dirname(os.getcwd()), 'download_images_quick', 'output', str(year), shapes, index, sensor))
                    image_name = list_images[0]
                    centroid_x = float(image_name.split('_')[1])
                    if 'png' in (image_name.split('_')[2]):
                        centroid_y = float((image_name.split('_')[2])[:-4])
                    else:
                        centroid_y = float(image_name.split('_')[2])
                    if sensor == 'planet_fixed': 
                        if os.path.exists(os.path.join(path_out_shapes, str(index), 'planet', str(centroid_x) + '_' + str(centroid_y) + '_' + 'altitude.tif')) == False:
                            list_points.append((index, ee.Geometry.Point([centroid_x, centroid_y]), year, os.path.join(path_out_shapes), 'planet'))
                    #else: 
                    #    if os.path.exists(os.path.join(path_out_shapes, str(index), sensor, str(centroid_x) + '_' + str(centroid_y) + '_' + 'altitude.tif')) == False:
                    #        list_points.append((index, ee.Geometry.Point([centroid_x, centroid_y]), year, os.path.join(path_out_shapes), sensor))

  return list_points

@retry(tries=10, delay=1, backoff=2)
def getResult(index, point, year, path, sensor):
  """Handle the HTTP requests to download an image."""
  import ee
  # Generate the desired image from the given point.
  lon = point.getInfo()["coordinates"][0]
  lat = point.getInfo()["coordinates"][1]
  if sensor == 'landsat':
      param = (0.05 * ((332 * 15) / 30)) / 372  # same area as image: resolution 30 m SRTM
      region = ee.Geometry.BBox(lon - param, lat - param, lon + param, lat + param)

  elif sensor == 'planet':
      param = (0.05 * ((332 * 4.77) / 30)) / 372
      region = ee.Geometry.BBox(lon - param, lat - param, lon + param, lat + param)

  else:
      print('Sensor not available/known')

  # We use NasaDEM instead of SRTM, i.e. reprocessing of SRTM with improved accuracy
  image = (ee.Image('NASA/NASADEM_HGT/001')).select('elevation').clip(region)

  # Fetch the URL from which to download the image.
  url = image.getThumbURL({
      'region': region,
      'dimensions': '332x332',
      'format': 'GEO_TIFF'})

  # Handle downloading the actual pixels.
  r = requests.get(url, stream=True)
  if r.status_code != 200:
      raise r.raise_for_status()

  if os.path.exists(os.path.join(path, str(index))) == False:
      os.mkdir(os.path.join(path, str(index)))

  if os.path.exists(os.path.join(path, str(index), sensor)) == False:
      os.mkdir(os.path.join(path, str(index), sensor))

  filename = os.path.join(path, str(index), sensor, str(lon) + '_' + str(lat) + '_' + 'altitude.tif')
  with open(filename, 'wb') as out_file:
      shutil.copyfileobj(r.raw, out_file)
  print("Done: ", index)


def convert_srtm(year):
    import richdem as rd
    list_to_delete = []
    path_in = os.path.join(os.getcwd(), 'output', str(year))
    list_shapes = os.listdir(path_in)
    for shapes in list_shapes:
        list_index = os.listdir(os.path.join(path_in, shapes))
        for index in list_index:
            list_sensors = os.listdir(os.path.join(path_in, shapes, index))
            for sensor in  list_sensors:
                if sensor == 'landsat':
                    divider = 2

                elif sensor == 'planet':
                    divider = 6
                path = os.path.join(path_in, shapes, index, sensor)
                if os.path.exists(os.path.join(path_in, shapes, index, sensor, 'slope.npy')) == False:
                    for n in range(len(os.listdir(path))):
                        if '_altitude' in os.listdir(path)[n]:
                            alt_name = os.listdir(path)[n]
                    file_name_alt = os.path.join(path, str(alt_name))
                    alt_tif = rd.LoadGDAL(file_name_alt, no_data = 999999)  # https://www.earthdatascience.org/tutorials/get-slope-aspect-from-digital-elevation-model/
                    slope_tif = rd.TerrainAttribute(alt_tif, attrib='slope_degrees')
                    aspect_tif = rd.TerrainAttribute(alt_tif, attrib='aspect')

                    altitude_path = os.path.join(path, 'altitude.npy') #From Irvin et al. (adapted)
                    slope_path = os.path.join(path, 'slope.npy')
                    aspect_path = os.path.join(path, 'aspect.npy')

                    altitude_arr = np.array(Image.open(file_name_alt), dtype = 'float64')
                    slope_arr = np.array(slope_tif, dtype = 'float64')
                    aspect_arr = np.array(aspect_tif, dtype = 'float64')

                    altitude_arr_format = np.zeros((332,332))
                    slope_arr_format = np.zeros((332,332))
                    aspect_arr_format = np.zeros((332,332))

                    for i in range(332):
                        for j in range(332):
                            if (i//divider) >= altitude_arr.shape[0]:
                                x = altitude_arr.shape[0] - 1
                            else:
                                x = i//divider
                            if (j//divider) >= altitude_arr.shape[1]:
                                y = altitude_arr.shape[1] - 1
                            else:
                                y = j//divider

                            altitude_arr_format[i][j] = altitude_arr[x][y]
                            slope_arr_format[i][j] = 100*slope_arr[x][y]
                            aspect_arr_format[i][j] = 100*aspect_arr[x][y]


                    np.save(altitude_path, altitude_arr_format) #From Irvin et al. (adapted)
                    np.save(slope_path, slope_arr_format)  # From Irvin et al. (adapted)
                    np.save(aspect_path, aspect_arr_format)  # From Irvin et al. (adapted)
                    list_to_delete.append(file_name_alt)
    return list_to_delete


def download_srtm(year):
    choice = int(input("1-Download data, 2- Convert to the right format: "))
    if choice ==1:
        import ee
        ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
        print('init ok')

        items = getRequests(year)
        pool = multiprocessing.Pool(25)
        pool.starmap(getResult, items)
        pool.close()
        pool.join()

    if choice == 2:
        import richdem as rd
        list_to_delete = convert_srtm(year)
        time.sleep(120)
        for file in list_to_delete:
            os.remove(file)


if __name__ == "__main__":
    fire.Fire(download_srtm)
