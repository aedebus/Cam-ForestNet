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
        path_out_year = os.path.join(os.getcwd(), 'output_timeseries', str(year))
        if os.path.exists(path_out_year) == False:
            os.mkdir(path_out_year)

        path_out_shapes = os.path.join(path_out_year, shapes)
        if os.path.exists(path_out_shapes) == False:
            os.mkdir(path_out_shapes)

        for index in list_index:
            path_in = os.path.join(os.path.dirname(os.getcwd()), 'download_gfc', 'output', str(year), shapes, str(index))
            loss_region = pickle.load(open(os.path.join(path_in, 'forest_loss_region.pkl'), 'rb'))
            if os.path.exists(path_in) == True and os.path.exists(os.path.join(path_out_shapes, str(index), 'planet', str(year)+'_'+str(loss_region.centroid.xy[0][0])+'_' + str(loss_region.centroid.xy[1][0])+'_'+'timeseries1.png')) == False:
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
  if len((ee.ImageCollection("projects/planet-nicfi/assets/basemaps/africa")
           .filterBounds(region)
           .filter(ee.Filter.eq('cadence', 'monthly')) #monthly
           .filterDate(str(year+1)+'-01-01', str(year+5)+'-12-31')).getInfo()['features'])>=2:
    
    image1 = (ee.ImageCollection("projects/planet-nicfi/assets/basemaps/africa")
            .filterBounds(region)
            .filter(ee.Filter.eq('cadence', 'monthly')) #monthly
            .filterDate(str(year+1)+'-01-01', str(year+5)+'-12-31')
            .first()
            .clip(region)
            .select('R', 'G', 'B')
            .visualize(min=64, max=5454, gamma=1.8))
    im1_nonvs = ee.ImageCollection("projects/planet-nicfi/assets/basemaps/africa").filterBounds(region).filter(ee.Filter.eq('cadence', 'monthly')).filterDate(str(year+1)+'-01-01', str(year+5)+'-12-31').first()

    date1=ee.Date(im1_nonvs.getInfo()['properties']['system:time_start'])
    # Fetch the URL from which to download the image.
    
    url = image1.getThumbURL({
        'region': region,
        'dimensions': '332x332',
        'format': 'png'})

    # Handle downloading the actual pixels.
    r = requests.get(url, stream=True)
    if r.status_code != 200:
        raise r.raise_for_status()

    if os.path.exists(os.path.join(path, str(index))) == False:
        os.mkdir(os.path.join(path, str(index)))

    if os.path.exists(os.path.join(path, str(index), 'planet')) == False:
        os.mkdir(os.path.join(path, str(index), 'planet'))

    filename = os.path.join(path, str(index), 'planet', str(year)+'_'+str(x)+'_'+ str(y)+'_'+'timeseries1.png') #remove: str(index)
    with open(filename, 'wb') as out_file:
        shutil.copyfileobj(r.raw, out_file)
    print('ok:', filename) 
    
    '''
    id_remove = im1_nonvs.getInfo()['properties']['system:index']
    
    new_coll = ee.ImageCollection("projects/planet-nicfi/assets/basemaps/africa").filterBounds(region).filter(ee.Filter.eq('cadence', 'monthly')).filterDate(str(year+1)+'-01-01', str(year+5)+'-12-31').filter(ee.Filter.neq('system:index', str(id_remove)))

    image2 = (new_coll
            .first()
            .clip(region)
            .select('R', 'G', 'B')
            .visualize(min=64, max=5454, gamma=1.8))
    
    im2_nonvs = new_coll.first()

    date2=ee.Date(im2_nonvs.getInfo()['properties']['system:time_start'])
    diff = (date1.difference(date2, "months")).abs().getInfo()

    n = len(new_coll.getInfo()['features'])
    k=1
    

    while diff < 2 and n>k and int((new_coll.toList(n-k, k)).length().getInfo())> 0 :
          image2 = ee.Image(new_coll.toList(n-k, k).get(0)).clip(region).select('R', 'G', 'B').visualize(min=64, max=5454, gamma=1.8)
          im2_nonvs = ee.Image(new_coll.toList(n-k, k).get(0))
          date2=ee.Date(im2_nonvs.getInfo()['properties']['system:time_start'])
          diff = (date1.difference(date2, "months")).abs().getInfo()
          k+=1

    if  diff>=2:   
    # Fetch the URL from which to download the image.
        
        url = image2.getThumbURL({
            'region': region,
            'dimensions': '332x332',
            'format': 'png'})

        # Handle downloading the actual pixels.
        r = requests.get(url, stream=True)
        if r.status_code != 200:
            raise r.raise_for_status()

        if os.path.exists(os.path.join(path, str(index))) == False:
            os.mkdir(os.path.join(path, str(index)))

        if os.path.exists(os.path.join(path, str(index), 'planet')) == False:
            os.mkdir(os.path.join(path, str(index), 'planet'))

        filename = os.path.join(path, str(index), 'planet', str(year)+'_'+str(x)+'_'+ str(y)+'_'+'timeseries2.png') #remove: str(index)
        with open(filename, 'wb') as out_file:
            shutil.copyfileobj(r.raw, out_file)
        print('ok:', filename) 

        
        id_remove = im2_nonvs.getInfo()['properties']['system:index']
    
        new_coll = ee.ImageCollection("projects/planet-nicfi/assets/basemaps/africa").filterBounds(region).filter(ee.Filter.eq('cadence', 'monthly')).filterDate(str(year+1)+'-01-01', str(year+5)+'-12-31').filter(ee.Filter.neq('system:index', str(id_remove)))

        image3 = (new_coll
                .first()
                .clip(region)
                .select('R', 'G', 'B')
                .visualize(min=64, max=5454, gamma=1.8))
        
        im3_nonvs = new_coll.first()

        date3=ee.Date(im3_nonvs.getInfo()['properties']['system:time_start'])
        diff = (date2.difference(date3, "months")).abs().getInfo()

        while diff < 2 and n>k and int((new_coll.toList(n-k, k)).length().getInfo())> 0 :
            image3 = ee.Image(new_coll.toList(n-k, k).get(0)).clip(region).select('R', 'G', 'B').visualize(min=64, max=5454, gamma=1.8)
            im3_nonvs = ee.Image(new_coll.toList(n-k, k).get(0))

            date3=ee.Date(im3_nonvs.getInfo()['properties']['system:time_start'])
            diff = (date2.difference(date3, "months")).abs().getInfo()
            k+=1

        if diff>=2:
            # Fetch the URL from which to download the image.
            
            url = image3.getThumbURL({
                'region': region,
                'dimensions': '332x332',
                'format': 'png'})

            # Handle downloading the actual pixels.
            r = requests.get(url, stream=True)
            if r.status_code != 200:
                raise r.raise_for_status()

            if os.path.exists(os.path.join(path, str(index))) == False:
                os.mkdir(os.path.join(path, str(index)))

            if os.path.exists(os.path.join(path, str(index), 'planet')) == False:
                os.mkdir(os.path.join(path, str(index), 'planet'))

            filename = os.path.join(path, str(index), 'planet', str(year)+'_'+str(x)+'_'+ str(y)+'_'+'timeseries3.png') #remove: str(index)
            with open(filename, 'wb') as out_file:
                shutil.copyfileobj(r.raw, out_file)
            print('ok:', filename) 
            
            
            id_remove = im3_nonvs.getInfo()['properties']['system:index']
    
            new_coll = ee.ImageCollection("projects/planet-nicfi/assets/basemaps/africa").filterBounds(region).filter(ee.Filter.eq('cadence', 'monthly')).filterDate(str(year+1)+'-01-01', str(year+5)+'-12-31').filter(ee.Filter.neq('system:index', str(id_remove)))

            image4 = (new_coll
                    .first()
                    .clip(region)
                    .select('R', 'G', 'B')
                    .visualize(min=64, max=5454, gamma=1.8))
            
            im4_nonvs = new_coll.first()

            date4=ee.Date(im4_nonvs.getInfo()['properties']['system:time_start'])
            diff = (date3.difference(date4, "months")).abs().getInfo()

            while diff < 2 and n>k and int((new_coll.toList(n-k, k)).length().getInfo())> 0 :
                image4 = ee.Image(new_coll.toList(n-k, k).get(0)).clip(region).select('R', 'G', 'B').visualize(min=64, max=5454, gamma=1.8)
                im4_nonvs = ee.Image(new_coll.toList(n-k, k).get(0))
                date4=ee.Date(im4_nonvs.getInfo()['properties']['system:time_start'])
                diff = (date3.difference(date4, "months")).abs().getInfo()
                k+=1
            
        
            if diff>=2:
                # Fetch the URL from which to download the image.
                
                url = image4.getThumbURL({
                    'region': region,
                    'dimensions': '332x332',
                    'format': 'png'})

                # Handle downloading the actual pixels.
                r = requests.get(url, stream=True)
                if r.status_code != 200:
                    raise r.raise_for_status()

                if os.path.exists(os.path.join(path, str(index))) == False:
                    os.mkdir(os.path.join(path, str(index)))

                if os.path.exists(os.path.join(path, str(index), 'planet')) == False:
                    os.mkdir(os.path.join(path, str(index), 'planet'))

                filename = os.path.join(path, str(index), 'planet', str(year)+'_'+str(x)+'_'+ str(y)+'_'+'timeseries4.png') #remove: str(index)
                with open(filename, 'wb') as out_file:
                    shutil.copyfileobj(r.raw, out_file)
                print('ok:', filename) 
                
                
                id_remove = im4_nonvs.getInfo()['properties']['system:index']
    
                new_coll = ee.ImageCollection("projects/planet-nicfi/assets/basemaps/africa").filterBounds(region).filter(ee.Filter.eq('cadence', 'monthly')).filterDate(str(year+1)+'-01-01', str(year+5)+'-12-31').filter(ee.Filter.neq('system:index', str(id_remove)))

                image5 = (new_coll
                        .first()
                        .clip(region)
                        .select('R', 'G', 'B')
                        .visualize(min=64, max=5454, gamma=1.8))
                
                im5_nonvs = new_coll.first()

                date5=ee.Date(im5_nonvs.getInfo()['properties']['system:time_start'])
                diff = (date4.difference(date5, "months")).abs().getInfo()

                while diff < 2 and n>k and int((new_coll.toList(n-k, k)).length().getInfo())> 0 :
                    image5 = ee.Image(new_coll.toList(n-k, k).get(0)).clip(region).select('R', 'G', 'B').visualize(min=64, max=5454, gamma=1.8)
                    im5_nonvs = ee.Image(new_coll.toList(n-k, k).get(0))
                    date5=ee.Date(im5_nonvs.getInfo()['properties']['system:time_start'])
                    diff = (date5.difference(date4, "months")).abs().getInfo()
                    k+=1
                

                if diff>=2:

                # Fetch the URL from which to download the image.
                    url = image5.getThumbURL({
                        'region': region,
                        'dimensions': '332x332',
                        'format': 'png'})

                    # Handle downloading the actual pixels.
                    r = requests.get(url, stream=True)
                    if r.status_code != 200:
                        raise r.raise_for_status()

                    if os.path.exists(os.path.join(path, str(index))) == False:
                        os.mkdir(os.path.join(path, str(index)))

                    if os.path.exists(os.path.join(path, str(index), 'planet')) == False:
                        os.mkdir(os.path.join(path, str(index), 'planet'))

                    filename = os.path.join(path, str(index), 'planet', str(year)+'_'+str(x)+'_'+ str(y)+'_'+'timeseries5.png') #remove: str(index)
                    with open(filename, 'wb') as out_file:
                        shutil.copyfileobj(r.raw, out_file)
                    print('ok:', filename) 
                
                else:
                    print('4 images for index:', index)
                    
            else:
                print('3 images for index:', index)
        else:
            print('2 images for index:', index)
    else:
        print('No enough images for index:', index)'''

    
  
  

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
    #getResult(items[0][0], items[0][1], items[0][2], items[0][3])

def main():
    fire.Fire(planetscope_from_gfc)

if __name__ == "__main__":
    main()
