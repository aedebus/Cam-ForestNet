'''Adapted from Stanford ML Group (Irvin et al., 2020) by Amandine Debus '''

'''REFERENCES FOR DATASET USED: 
Citation: Saha, S., et al. 2011, updated daily. NCEP Climate Forecast System Version 2 (CFSv2) 6-hourly Products. Research Data Archive at the National Center for Atmospheric Research, Computational and Information Systems Laboratory. https://doi.org/10.5065/D61C1TXF. Accessed dd mmm yyyy.
DOI: https://doi.org/10.5065/D61C1TXF
Link GEE: https://developers.google.com/earth-engine/datasets/catalog/NOAA_CFSV2_FOR6H?hl=en#description

NB: CFSR not chosen because only since 2018 
'''

import fire
import numpy as np
import os
import ee
import pickle
import multiprocessing
import geemap
from retry import retry
import requests
import logging
import shutil


class NCEPDownloader(): #Structure of the class from Irvin et al.

    def get_ncep(self):
        surface_level_albedo = 'forecast_albedo' #NOT IN NCEP CFSv2, From ECMWF ERA5-Land Hourly : https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_HOURLY?hl=en

        clear_sky_longwave_flux_downward = 'Clear_Sky_Downward_Long_Wave_Flux_surface' #NOT IN NCEP CFSv2
        clear_sky_longwave_flux_upward = 'Clear_Sky_Upward_Long_Wave_Flux_surface' ##NOT IN NCEP CFSv2
        clear_sky_solar_flux_downward = 'Clear_Sky_Downward_Solar_Flux_surface' ##NOT IN NCEP CFSv2E
        clear_sky_solar_flux_upward = 'Clear_Sky_Upward_Solar_Flux_atmosphere_top' ##NOT IN NCEP CFSv2
        #ASSUMING MAXIMUM RADIATION OVER A YEAR --> SEE DOWNLOAD_NCEP

        direct_evaporation_from_bare_soil = 'ESoil_tavg' #NOT IN NCEP CFSv2, FROM GLDAS: https://developers.google.com/earth-engine/datasets/catalog/NASA_GLDAS_V021_NOAH_G025_T3H?hl=en#bands
        longwave_radiation_flux_downward = 'Downward_Long-Wave_Radp_Flux_surface_6_Hour_Average'
        shortwave_radiation_flux_downward = 'Downward_Short-Wave_Radiation_Flux_surface_6_Hour_Average'
        ground_heat_net_flux = 'Ground_Heat_Flux_surface' ##NOT IN NCEP CFSv2, in CFSR BUT ONLY SINCE 2018: https://developers.google.com/earth-engine/datasets/catalog/NOAA_CFSR?hl=en
        latent_heat_net_flux = 'Latent_heat_net_flux_surface_6_Hour_Average'
        specific_humidity = 'Specific_humidity_height_above_ground'
        max_specific_humidity = 'Maximum_specific_humidity_at_2m_height_above_ground_6_Hour_Interval'
        min_specific_humidity = 'Minimum_temperature_height_above_ground_6_Hour_Interval'
        potential_evaporation_rate = 'Potential_Evaporation_Rate_surface_6_Hour_Average'
        # ground_level_precipitation = 'Precipitation_rate_surface_6_Hour_Average' ##IN NCEP CFSv2 BUT NOT IN METERS
        ground_level_precipitation = 'total_precipitation' #ECMWF
        sensible_heat_net_flux = 'Sensible_heat_net_flux_surface_6_Hour_Average'
        volumetric_soil_moisture_content1 = 'Volumetric_Soil_Moisture_Content_depth_below_surface_layer_5_cm'
        volumetric_soil_moisture_content2 = 'Volumetric_Soil_Moisture_Content_depth_below_surface_layer_25_cm'
        volumetric_soil_moisture_content3 = 'Volumetric_Soil_Moisture_Content_depth_below_surface_layer_70_cm'
        volumetric_soil_moisture_content4 = 'Volumetric_Soil_Moisture_Content_depth_below_surface_layer_150_cm'
        air_pressure_at_surface_level = 'Pressure_surface'
        wind_component_u = 'u-component_of_wind_height_above_ground'
        wind_component_v = 'v-component_of_wind_height_above_ground'
        water_runoff_at_surface_level = 'Qs_acc' #NOT IN NCEP CFSv2, FROM GLDAS

        sublimation = 'snow_evaporation' ##NOT IN NCEP CFSv2, FROM ECWF BUT IN M WATER EQUIVALENT (NOT W/m^2)

        temperature = 'Temperature_height_above_ground' #other height or depth?
        longwave_radiation_flux_upward = 'Upward_Long-Wave_Radp_Flux_surface_6_Hour_Average'
        shortwave_radiation_flux_upward = 'Upward_Short-Wave_Radiation_Flux_surface_6_Hour_Average'


        return surface_level_albedo, clear_sky_longwave_flux_downward, clear_sky_longwave_flux_upward, clear_sky_solar_flux_downward, clear_sky_solar_flux_upward, \
               direct_evaporation_from_bare_soil, longwave_radiation_flux_downward, shortwave_radiation_flux_downward, ground_heat_net_flux, latent_heat_net_flux, \
               specific_humidity, max_specific_humidity, min_specific_humidity, potential_evaporation_rate, ground_level_precipitation, sensible_heat_net_flux, \
               volumetric_soil_moisture_content1, volumetric_soil_moisture_content2, volumetric_soil_moisture_content3, volumetric_soil_moisture_content4, \
               air_pressure_at_surface_level, wind_component_u, wind_component_v, water_runoff_at_surface_level, sublimation, temperature, longwave_radiation_flux_upward, shortwave_radiation_flux_upward

    def names(self):
        names = {
            'forecast_albedo': 'surface_level_albedo',
            'Clear_Sky_Downward_Long_Wave_Flux_surface': 'clear_sky_longwave_flux_downward',
            'Clear_Sky_Upward_Long_Wave_Flux_surface':  'clear_sky_longwave_flux_upward',
            'Clear_Sky_Downward_Solar_Flux_surface': 'clear_sky_solar_flux_downward',
            'Clear_Sky_Upward_Solar_Flux_atmosphere_top':'clear_sky_solar_flux_upward',
            'ESoil_tavg': 'direct_evaporation_from_bare_soil',
            'Downward_Long-Wave_Radp_Flux_surface_6_Hour_Average':'longwave_radiation_flux_downward',
            'Downward_Short-Wave_Radiation_Flux_surface_6_Hour_Average': 'shortwave_radiation_flux_downward',
            'Ground_Heat_Flux_surface': 'ground_heat_net_flux',
            'Latent_heat_net_flux_surface_6_Hour_Average': 'latent_heat_net_flux',
            'Specific_humidity_height_above_ground' : 'specific_humidity',
            'Maximum_specific_humidity_at_2m_height_above_ground_6_Hour_Interval': 'max_specific_humidity',
            'Minimum_temperature_height_above_ground_6_Hour_Interval':'min_specific_humidity',
            'Potential_Evaporation_Rate_surface_6_Hour_Average':'potential_evaporation_rate',
            'total_precipitation':'ground_level_precipitation',
            'Sensible_heat_net_flux_surface_6_Hour_Average': 'sensible_heat_net_flux',
            'Volumetric_Soil_Moisture_Content_depth_below_surface_layer_5_cm':'volumetric_soil_moisture_content1',
            'Volumetric_Soil_Moisture_Content_depth_below_surface_layer_25_cm':'volumetric_soil_moisture_content2',
            'Volumetric_Soil_Moisture_Content_depth_below_surface_layer_70_cm':'volumetric_soil_moisture_content3',
            'Volumetric_Soil_Moisture_Content_depth_below_surface_layer_150_cm':'volumetric_soil_moisture_content4',
            'Pressure_surface':'air_pressure_at_surface_level',
            'u-component_of_wind_height_above_ground': 'wind_component_u',
            'v-component_of_wind_height_above_ground':'wind_component_v',
            'Qs_acc' : 'water_runoff_at_surface_level',
             'snow_evaporation':'sublimation',
            'Temperature_height_above_ground': 'temperature',
            'Upward_Long-Wave_Radp_Flux_surface_6_Hour_Average':'longwave_radiation_flux_upward',
            'Upward_Short-Wave_Radiation_Flux_surface_6_Hour_Average':'shortwave_radiation_flux_upward',
        }
        return names



def getRequests(year):
  """Generates a list of work items to be downloaded.
  """
  list_dir = os.listdir(os.path.join(os.path.dirname(os.getcwd()), 'download_images_quick', 'output', str(year)))
  list_points = []
  for shapes in list_dir:
      list_index = os.listdir(os.path.join(os.path.dirname(os.getcwd()), 'download_images_quick', 'output', str(year), shapes))
      path_out_year = os.path.join(os.getcwd(), 'output', str(year))
      if os.path.exists(path_out_year) == False:
          os.mkdir(path_out_year)

      path_out_shapes = os.path.join(path_out_year, shapes)
      if os.path.exists(path_out_shapes) == False:
          os.mkdir(path_out_shapes)

      for index in list_index:
          path_out = os.path.join(path_out_shapes, index)
          if os.path.exists(path_out) == False:
              os.mkdir(path_out)
          path_in = os.path.join(os.path.dirname(os.getcwd()), 'download_images_quick', 'output', str(year), shapes, str(index))
          if os.path.exists(path_in) == True:
              list_sensors = os.listdir(os.path.join(os.path.dirname(os.getcwd()), 'download_images_quick', 'output', str(year), shapes, index))
              for sensor in list_sensors:
                  list_images = os.listdir(os.path.join(os.path.dirname(os.getcwd()), 'download_images_quick', 'output', str(year), shapes, index, sensor))
                  image_name = list_images[0]
                  centroid_x = float(image_name.split('_')[1])
                  if sensor == 'landsat':
                      centroid_y = float((image_name.split('_')[2])[:-4]) #no index in landsat_quick
                  elif sensor == 'planet':
                      centroid_y = float((image_name.split('_')[2]))
                  list_points.append((index, ee.Geometry.Point([centroid_x, centroid_y]), path_out, year, sensor))

  return list_points

@retry(tries=10, delay=1, backoff=2)
def getResult(index, point, path, year, sensor):

  # Generate the desired image from the given point.
  #NB: Python does not have an evaluate function
  lon = point.getInfo()["coordinates"][0]
  lat = point.getInfo()["coordinates"][1]
  td = NCEPDownloader()
  ncep_parameters = td.get_ncep()
  names = td.names()
  ncep_path = os.path.join(path, sensor, 'ncep')  # From Irvin et al.
  os.makedirs(ncep_path, exist_ok=True)  # From Irvin et al.

  # We look for parameters up to 5 years before loss event:
  reducer_mean = ee.Reducer.mean()
  reducer_max = ee.Reducer.max()
  reducer_min = ee.Reducer.min()

  for param in ncep_parameters:
      if param == 'forecast_albedo' or param == 'total_precipitation' or param == 'snow_evaporation':
          res = 11132
          if sensor == 'landsat':
              param1 = (0.00005 * ((332 * 15) / res)) / 3600
              param2 = (0.00005 * ((332 * 15) / res)) / 1800
              geometry = ee.Geometry.BBox(lon - param1, lat - param2, lon + param1, lat + param2)

          elif sensor == 'planet':
              param1 = (0.00005 * ((332 * 4.77) / res)) / 3600
              param2 = (0.00005 * ((332 * 4.77) / res)) / 1800
              geometry = ee.Geometry.BBox(lon - param1, lat - param2, lon + param1, lat + param2)
          else:
              print('Sensor not available/known')

          dataset_ecmwf = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY").filter(
              ee.Filter.date(str(year - 5) + '-01-01', str(year - 1) + '-12-31')).filterBounds(geometry)
          dataset_param = dataset_ecmwf.select(param)

      elif param == 'ESoil_tavg' or param == 'Qs_acc':
          res = 27830
          if sensor == 'landsat':
              param1 = (0.00005 * ((332 * 15) / res)) / 1400
              param2 = (0.00005 * ((332 * 15) / res)) / 720
              geometry = ee.Geometry.BBox(lon - param1, lat - param2, lon + param1, lat + param2)

          elif sensor == 'planet':
              param1 = (0.00005 * ((332 * 4.77) / res)) / 1400
              param2 = (0.00005 * ((332 * 4.77) / res)) / 720
              geometry = ee.Geometry.BBox(lon - param1, lat - param2, lon + param1, lat + param2)
          else:
              print('Sensor not available/known')

          dataset_gldas = ee.ImageCollection("NASA/GLDAS/V021/NOAH/G025/T3H").filter(
              ee.Filter.date(str(year - 5) + '-01-01', str(year - 1) + '-12-31')).filterBounds(geometry)
          dataset_param = dataset_gldas.select(param)

      elif param == 'Ground_Heat_Flux_surface':
          res = 55660
          if sensor == 'landsat':
              param1 = (0.00005 * ((332 * 15) / res)) / 720
              param2 = (0.00005 * ((332 * 15) / res)) / 360
              geometry = ee.Geometry.BBox(lon - param1, lat - param2, lon + param1, lat + param2)

          elif sensor == 'planet':
              param1 = (0.00005 * ((332 * 4.77) / res)) / 720
              param2 = (0.00005 * ((332 * 4.77) / res)) / 360
              geometry = ee.Geometry.BBox(lon - param1, lat - param2, lon + param1, lat + param2)
          else:
              print('Sensor not available/known')
          if year > 2018:
              dataset_cfsr = ee.ImageCollection("NOAA/CFSR").filter(
                  ee.Filter.date('2018-12-13', str(year - 1) + '-12-31')).filterBounds(
                  geometry)  # Available only since 13-12-2018
          else:
              dataset_cfsr = ee.ImageCollection("NOAA/CFSR").filter(
                  ee.Filter.date('2018-12-13', '2021-12-31')).filterBounds(geometry)
          dataset_param = dataset_cfsr.select(param)

      else:
          res = 22264
          if sensor == 'landsat':
              param1 = (0.00005 * ((332 * 15) / res)) / 1800
              param2 = (0.00005 * ((332 * 15) / res)) / 900
              geometry = ee.Geometry.BBox(lon - param1, lat - param2, lon + param1, lat + param2)

          elif sensor == 'planet':
              param1 = (0.00005 * ((332 * 4.77) / res)) / 1800
              param2 = (0.00005 * ((332 * 4.77) / res)) / 900
              geometry = ee.Geometry.BBox(lon - param1, lat - param2, lon + param1, lat + param2)
          else:
              print('Sensor not available/known')

          if 'Clear_Sky' in param:  # ASSUMING MAXIMUM RADIATION OVER A YEAR
              list_images = []

              if 'long' in param:
                  if 'downward' in param:
                      param_clear_sky = 'Downward_Long-Wave_Radp_Flux_surface'
                  else:
                      param_clear_sky = 'Upward_Long-Wave_Radp_Flux_surface_6_Hour_Average'
              else:
                  if 'downward' in param:
                      param_clear_sky = 'Downward_Short-Wave_Radiation_Flux_surface_6_Hour_Average'
                  else:
                      param_clear_sky = 'Upward_Short-Wave_Radiation_Flux_surface_6_Hour_Average'

              for i in range(1, 6):
                  dataset_clear_sky = ee.ImageCollection("NOAA/CFSV2/FOR6H").filter(
                      ee.Filter.date(str(year - i) + '-01-01', str(year - i + 1) + '-12-31')).filterBounds(
                      geometry)
                  image_mean = dataset_clear_sky.select(param_clear_sky).reduce(reducer_max)
                  list_images.append(image_mean)

              dataset_param = ee.ImageCollection(list_images)

          else:
              dataset_ncep = ee.ImageCollection("NOAA/CFSV2/FOR6H").filter(
                  ee.Filter.date(str(year - 5) + '-01-01', str(year - 1) + '-12-31')).filterBounds(geometry)
              dataset_param = dataset_ncep.select(param)

      if sensor == 'landsat':
          scale = 30
      elif sensor == 'planet':
          scale = 4.77

      image_mean = dataset_param.reduce(reducer_mean)
      p = ee.Geometry.Point(lon, lat)
      mean_dictionary = image_mean.reduceRegion(ee.Reducer.first(), p, scale)

      image_max = dataset_param.reduce(reducer_max)
      max_dictionary = image_max.reduceRegion(ee.Reducer.first(), p, scale)

      image_min = dataset_param.reduce(reducer_min)
      min_dictionary = image_min.reduceRegion(ee.Reducer.first(), p, scale)


      if param == 'Specific_humidity_height_above_ground' or param == 'Maximum_specific_humidity_at_2m_height_above_ground_6_Hour_Interval' or param == 'Minimum_temperature_height_above_ground_6_Hour_Interval' or param == 'total_precipitation' or 'Volumetric_Soil_Moisture_Content' in param:
          if np.array(mean_dictionary.getInfo()[param + '_mean']) == None:
              np.save(os.path.join(ncep_path, names[param] + '_avg.npy'), 0)
          else:
              np.save(os.path.join(ncep_path, names[param] + '_avg.npy'),
                      10000 * np.array(mean_dictionary.getInfo()[param + '_mean']))  # From Irvin et al. (adapted)
          if np.array(max_dictionary.getInfo()[param + '_max']) == None:
              np.save(os.path.join(ncep_path, names[param] + '_max.npy'), 0)
          else:
              np.save(os.path.join(ncep_path, names[param] + '_max.npy'),
                      10000 * np.array(max_dictionary.getInfo()[param + '_max']))  # From Irvin et al. (adapted)
          if np.array(min_dictionary.getInfo()[param + '_min']) == None:
              np.save(os.path.join(ncep_path, names[param] + '_min.npy'), 0)
          else:
              np.save(os.path.join(ncep_path, names[param] + '_min.npy'),
                      10000 * np.array(min_dictionary.getInfo()[param + '_min']))  # From Irvin et al. (adapted)

      elif param == 'Pressure_surface':
          if np.array(mean_dictionary.getInfo()[param + '_mean']) == None:
              np.save(os.path.join(ncep_path, names[param] + '_avg.npy'), 0)
          else:
              np.save(os.path.join(ncep_path, names[param] + '_avg.npy'),
                      0.1 * np.array(mean_dictionary.getInfo()[param + '_mean']))  # From Irvin et al. (adapted)
          if np.array(max_dictionary.getInfo()[param + '_max']) == None:
              np.save(os.path.join(ncep_path, names[param] + '_max.npy'), 0)
          else:
              np.save(os.path.join(ncep_path, names[param] + '_max.npy'),
                      0.1 * np.array(max_dictionary.getInfo()[param + '_max']))  # From Irvin et al. (adapted)
          if np.array(min_dictionary.getInfo()[param + '_min']) == None:
              np.save(os.path.join(ncep_path, names[param] + '_min.npy'), 0)
          else:
              np.save(os.path.join(ncep_path, names[param] + '_min.npy'),
                      0.1 * np.array(min_dictionary.getInfo()[param + '_min']))  # From Irvin et al. (adapted)

      elif 'component_of_wind_height_above_ground' in param or param == 'Qs_acc':
          if np.array(mean_dictionary.getInfo()[param + '_mean']) == None:
              np.save(os.path.join(ncep_path, names[param] + '_avg.npy'), 0)
          else:
              np.save(os.path.join(ncep_path, names[param] + '_avg.npy'),
                      100 * np.array(mean_dictionary.getInfo()[param + '_mean']))  # From Irvin et al. (adapted)
          if np.array(max_dictionary.getInfo()[param + '_max']) == None:
              np.save(os.path.join(ncep_path, names[param] + '_max.npy'), 0)
          else:
              np.save(os.path.join(ncep_path, names[param] + '_max.npy'),
                      100 * np.array(max_dictionary.getInfo()[param + '_max']))  # From Irvin et al. (adapted)
          if np.array(min_dictionary.getInfo()[param + '_min']) == None:
              np.save(os.path.join(ncep_path, names[param] + '_min.npy'), 0)
          else:
              np.save(os.path.join(ncep_path, names[param] + '_min.npy'),
                      100 * np.array(min_dictionary.getInfo()[param + '_min']))  # From Irvin et al. (adapted)

      elif 'Clear_Sky' in param:
          if 'long' in param:
              if 'downward' in param:
                  param_clear_sky = 'Downward_Long-Wave_Radp_Flux_surface_6_Hour_Average_max'
              else:
                  param_clear_sky = 'Upward_Long-Wave_Radp_Flux_surface_6_Hour_Average_max'
          else:
              if 'downward' in param:
                  param_clear_sky = 'Downward_Short-Wave_Radiation_Flux_surface_6_Hour_Average_max'
              else:
                  param_clear_sky = 'Upward_Short-Wave_Radiation_Flux_surface_6_Hour_Average_max'

          if np.array(mean_dictionary.getInfo()[param_clear_sky + '_mean']) == None:
              np.save(os.path.join(ncep_path, names[param] + '_avg.npy'), 0)
          else:
              np.save(os.path.join(ncep_path, names[param] + '_avg.npy'),
                      np.array(mean_dictionary.getInfo()[param_clear_sky + '_mean']))  # From Irvin et al. (adapted)
          if np.array(max_dictionary.getInfo()[param_clear_sky + '_max']) == None:
              np.save(os.path.join(ncep_path, names[param] + '_max.npy'), 0)
          else:
              np.save(os.path.join(ncep_path, names[param] + '_max.npy'),
                      np.array(max_dictionary.getInfo()[param_clear_sky + '_max']))  # From Irvin et al. (adapted)
          if np.array(min_dictionary.getInfo()[param_clear_sky + '_min']) == None:
              np.save(os.path.join(ncep_path, names[param] + '_min.npy'), 0)
          else:
              np.save(os.path.join(ncep_path, names[param] + '_min.npy'),
                      np.array(min_dictionary.getInfo()[param_clear_sky + '_min']))  # From Irvin et al. (adapted)


      else:
          # Need to convert to np.array, otherwise array of object cannot be used
          if np.array(mean_dictionary.getInfo()[param + '_mean']) == None:
              np.save(os.path.join(ncep_path, names[param] + '_avg.npy'), 0)
          else:
              np.save(os.path.join(ncep_path, names[param] + '_avg.npy'),
                      np.array(mean_dictionary.getInfo()[param + '_mean']))  # From Irvin et al. (adapted)
          if np.array(max_dictionary.getInfo()[param + '_max']) == None:
              np.save(os.path.join(ncep_path, names[param] + '_max.npy'), 0)
          else:
              np.save(os.path.join(ncep_path, names[param] + '_max.npy'),
                      np.array(max_dictionary.getInfo()[param + '_max']))  # From Irvin et al. (adapted)
          if np.array(min_dictionary.getInfo()[param + '_min']) == None:
              np.save(os.path.join(ncep_path, names[param] + '_min.npy'), 0)
          else:
              np.save(os.path.join(ncep_path, names[param] + '_min.npy'),
                      np.array(min_dictionary.getInfo()[param + '_min']))  # From Irvin et al. (adapted)

  print("Done", index)

def download_ncep(year):
    logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
    logging.debug('This will get logged to a file')
    ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
    print('init ok')

    items = getRequests(year)
    pool = multiprocessing.Pool(25)
    pool.starmap(getResult, items)
    pool.close()
    pool.join()



if __name__ == "__main__":
    fire.Fire(download_ncep)
