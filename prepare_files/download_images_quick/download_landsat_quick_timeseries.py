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
    if (shapes != '10N_000E') and (shapes != '10N_010E') and (shapes != '20N_010E'):
        list_index= os.listdir(os.path.join(os.path.dirname(os.getcwd()), 'download_gfc', 'output', str(year), shapes))
        path_out_timeseries = os.path.join(os.getcwd(), 'output_timeseries')
        if os.path.exists(path_out_timeseries) == False:
            os.mkdir(path_out_timeseries)
        path_out_year = os.path.join(os.getcwd(), 'output_timeseries', str(year))
        if os.path.exists(path_out_year) == False:
            os.mkdir(path_out_year)

        path_out_shapes = os.path.join(path_out_year, shapes)
        if os.path.exists(path_out_shapes) == False:
            os.mkdir(path_out_shapes)

        for index in list_index:
            path_in = os.path.join(os.path.dirname(os.getcwd()), 'download_gfc', 'output', str(year), shapes, str(index))
            loss_region = pickle.load(open(os.path.join(path_in, 'forest_loss_region.pkl'), 'rb'))
            if os.path.exists(path_in) == True and (os.path.exists(os.path.join(path_out_shapes, str(index), 'landsat', str(year)+'_'+str(loss_region.centroid.xy[0][0])+'_' + str(loss_region.centroid.xy[1][0])+'_'+'timeseries5.png')) == False 
                                                    or os.path.getsize(os.path.join(path_out_shapes, str(index), 'landsat', str(year)+'_'+str(loss_region.centroid.xy[0][0])+'_' + str(loss_region.centroid.xy[1][0])+'_'+'timeseries5.png')) < 10000):
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
                    
  if (len(filtered_images.getInfo()['features'])>=2): #at least two images
      image1 = (filtered_images
               .map(panSharpenL8)
               .sort('CLOUD_COVER')
               .first()
               .clip(region)
               .select('B4', 'B3', 'B2')).visualize(min=0.0, max=0.4)
      im1_nonps = (filtered_images
               .sort('CLOUD_COVER')
               .first()
               .clip(region)
               .select('B4', 'B3', 'B2'))
      
      date1=ee.Date(im1_nonps.getInfo()['properties']['DATE_ACQUIRED'])
      #print(im1_nonps.getInfo()['properties']['DATE_ACQUIRED'])

      # Fetch the URL from which to download the image.
      '''
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

      if os.path.exists(os.path.join(path, str(index), 'landsat')) == False:
          os.mkdir(os.path.join(path, str(index), 'landsat'))

      filename = os.path.join(path, str(index), 'landsat', str(year)+'_'+str(x)+'_' + str(y)+'_'+'timeseries1.png')
      with open(filename, 'wb') as out_file:
          shutil.copyfileobj(r.raw, out_file)
      print('ok:', filename)
      '''
      id_remove = im1_nonps.getInfo()['properties']['system:index']
      #print('all', index, len(filtered_images.getInfo()['features']))
      new_coll = ((filtered_images
               .sort('CLOUD_COVER'))
               .filter(ee.Filter.neq('system:index', str(id_remove))))
      
      #print('removed', index, len(new_coll.getInfo()['features']))
      #print('list', index, len(filtered_images.toList(len(filtered_images.getInfo()['features']), 1)))
      
      image2 = (new_coll
                .map(panSharpenL8)
                .first()
                .clip(region)
                .select('B4', 'B3', 'B2')).visualize(min=0.0, max=0.4)
    
      im2_nonps = (new_coll
                   .first()
                   .clip(region)
                   .select('B4', 'B3', 'B2'))
      
      date2=ee.Date(im2_nonps.getInfo()['properties']['DATE_ACQUIRED'])
      diff = (date1.difference(date2, "months")).abs().getInfo()
      #print('diff:', diff)
      n = len(new_coll.getInfo()['features'])
      k=1  
      while diff < 2 and n>k and int((new_coll.toList(n-k, k)).length().getInfo())> 0 :
          #print('enter while')
          #print(len(new_coll.getInfo()['features']))
          #id_to_remove = im2_nonps.getInfo()['properties']['system:index']
          #print(index, id_to_remove)
          #new_coll = new_coll.filter(ee.Filter.neq('system:index', str(id_remove)))
          #print(index, len(new_coll.getInfo()['features']))
          image2 = (panSharpenL8(ee.Image(new_coll.toList(n-k, k).get(0)))
                    .clip(region)
                    .select('B4', 'B3', 'B2')).visualize(min=0.0, max=0.4)
          print((new_coll.toList(n-k, k)).length().getInfo())
          im2_nonps = (ee.Image(new_coll.toList(n-k, k)
                       .get(0))
                       .clip(region)
                       .select('B4', 'B3', 'B2'))
          
          date2=ee.Date(im2_nonps.getInfo()['properties']['DATE_ACQUIRED'])
          diff = (date1.difference(date2, "months")).abs().getInfo()
          print(index,diff)
          k+=1

      #print('exit while')
      if  diff>=2:   
      #print(im2_nonps.getInfo()['properties']['DATE_ACQUIRED'])
      # Fetch the URL from which to download the image.
        '''
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

        if os.path.exists(os.path.join(path, str(index), 'landsat')) == False:
            os.mkdir(os.path.join(path, str(index), 'landsat'))

        filename = os.path.join(path, str(index), 'landsat', str(year)+'_'+str(x)+'_' + str(y)+'_'+'timeseries2.png')
        with open(filename, 'wb') as out_file:
            shutil.copyfileobj(r.raw, out_file)
        print('ok:', filename)
        '''
        #id_to_remove = im2_nonps.getInfo()['properties']['system:index']
        new_coll = new_coll.filter(ee.Filter.neq('system:index', str(id_remove)))
        image3 = (new_coll
                .map(panSharpenL8)
                .first()
                .clip(region)
                .select('B4', 'B3', 'B2')).visualize(min=0.0, max=0.4)
        
        im3_nonps = (new_coll
                    .first()
                    .clip(region)
                    .select('B4', 'B3', 'B2'))
        
        date3=ee.Date(im3_nonps.getInfo()['properties']['DATE_ACQUIRED'])
        diff = (date2.difference(date3, "months")).abs().getInfo()

        while diff < 2 and n>k and int((new_coll.toList(n-k, k)).length().getInfo())>0:
            #id_to_remove = im3_nonps.getInfo()['properties']['system:index']
            #new_coll = new_coll.filter(ee.Filter.neq('system:index', str(id_remove)))
            image3 = (panSharpenL8(ee.Image(new_coll.toList(n-k, k).get(0)))
                    .clip(region)
                    .select('B4', 'B3', 'B2')).visualize(min=0.0, max=0.4)
          
            im3_nonps = (ee.Image(new_coll.toList(n-k, k)
                       .get(0))
                       .clip(region)
                       .select('B4', 'B3', 'B2'))
            date3=ee.Date(im3_nonps.getInfo()['properties']['DATE_ACQUIRED'])
            diff = (date2.difference(date3, "months")).abs().getInfo()
            k+=1
        
        if  diff>=2:      
            #print(im3_nonps.getInfo()['properties']['DATE_ACQUIRED'])
            # Fetch the URL from which to download the image.
            '''
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

            if os.path.exists(os.path.join(path, str(index), 'landsat')) == False:
                os.mkdir(os.path.join(path, str(index), 'landsat'))

            filename = os.path.join(path, str(index), 'landsat', str(year)+'_'+str(x)+'_' + str(y)+'_'+'timeseries3.png')
            with open(filename, 'wb') as out_file:
                shutil.copyfileobj(r.raw, out_file)
            print('ok:', filename) 
            '''
            id_to_remove = im3_nonps.getInfo()['properties']['system:index']
            new_coll = new_coll.filter(ee.Filter.neq('system:index', str(id_remove)))

            image4 = (new_coll
                    .map(panSharpenL8)
                    .first()
                    .clip(region)
                    .select('B4', 'B3', 'B2')).visualize(min=0.0, max=0.4)
            
            im4_nonps = (new_coll
                        .first()
                        .clip(region)
                        .select('B4', 'B3', 'B2'))
            
            date4=ee.Date(im4_nonps.getInfo()['properties']['DATE_ACQUIRED'])
            diff = (date3.difference(date4, "months")).abs().getInfo()

            while diff < 2 and n>k and int((new_coll.toList(n-k, k)).length().getInfo())>0:
                #id_to_remove = im4_nonps.getInfo()['properties']['system:index']
                #new_coll = new_coll.filter(ee.Filter.neq('system:index', str(id_remove)))
                image4 = (panSharpenL8(ee.Image(new_coll.toList(n-k, k).get(0))).clip(region).select('B4', 'B3', 'B2')).visualize(min=0.0, max=0.4)
          
                im4_nonps = (ee.Image(new_coll.toList(n-k, k).get(0)).clip(region).select('B4', 'B3', 'B2'))
                
                date4=ee.Date(im4_nonps.getInfo()['properties']['DATE_ACQUIRED'])
                diff = (date3.difference(date4, "months")).abs().getInfo()
                k+=1
            
            if  diff>=2:     
                #print(im4_nonps.getInfo()['properties']['DATE_ACQUIRED'])
                # Fetch the URL from which to download the image.
                '''
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

                if os.path.exists(os.path.join(path, str(index), 'landsat')) == False:
                    os.mkdir(os.path.join(path, str(index), 'landsat'))

                filename = os.path.join(path, str(index), 'landsat', str(year)+'_'+str(x)+'_' + str(y)+'_'+'timeseries4.png')
                with open(filename, 'wb') as out_file:
                    shutil.copyfileobj(r.raw, out_file)
                print('ok:', filename) 
                '''
                id_to_remove = im4_nonps.getInfo()['properties']['system:index']
                new_coll = new_coll.filter(ee.Filter.neq('system:index', str(id_remove)))
                image5 = (new_coll
                        .map(panSharpenL8)
                        .first()
                        .clip(region)
                        .select('B4', 'B3', 'B2')).visualize(min=0.0, max=0.4)
                
                im5_nonps = (new_coll
                        .first()
                        .clip(region)
                        .select('B4', 'B3', 'B2'))
                
                date5=ee.Date(im5_nonps.getInfo()['properties']['DATE_ACQUIRED'])
                diff = (date4.difference(date5, "months")).abs().getInfo()

                while diff < 2 and n>k and int((new_coll.toList(n-k, k)).length().getInfo())>0:
                    #id_to_remove = im5_nonps.getInfo()['properties']['system:index']
                    #new_coll = new_coll.filter(ee.Filter.neq('system:index', str(id_remove)))
                    image5 = (panSharpenL8(ee.Image(new_coll.toList(n-k, k).get(0))).clip(region).select('B4', 'B3', 'B2')).visualize(min=0.0, max=0.4)
          
                    im5_nonps = (ee.Image(new_coll.toList(n-k, k).get(0)).clip(region).select('B4', 'B3', 'B2'))
                    
                    date5=ee.Date(im5_nonps.getInfo()['properties']['DATE_ACQUIRED'])
                    diff = (date4.difference(date5, "months")).abs().getInfo()
                    k+=1
                
                if  diff>=2:       
                    #print(im5_nonps.getInfo()['properties']['DATE_ACQUIRED'])
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

                    if os.path.exists(os.path.join(path, str(index), 'landsat')) == False:
                        os.mkdir(os.path.join(path, str(index), 'landsat'))

                    filename = os.path.join(path, str(index), 'landsat', str(year)+'_'+str(x)+'_' + str(y)+'_'+'timeseries5.png')
                    with open(filename, 'wb') as out_file:
                        shutil.copyfileobj(r.raw, out_file)
                    print('ok:', filename) 
                else:
                    print('4 images for index:', index)
            else:
                print('3 enough images for index:', index)
        else:
            print('2 images for index:', index)
      else:
          print('1 image for index:', index)
  else:
      print('No enough images for index:', index)

  
      



# def landsat_from_gfc(year, shapes):
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
    #getResult(items[0][0], items[0][1], items[0][2], items[0][3])


def main():
    fire.Fire(landsat_from_gfc)


if __name__ == "__main__":
    main()
