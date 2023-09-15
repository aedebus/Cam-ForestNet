'''Adapted from Stanford ML Group (Irvin et al., 2020) by Amandine Debus '''

import fire
import json
import os
import pickle
import geopy
import time
import random 

# Constants for zoom values from:
# https://nominatim.org/release-docs/develop/api/Reverse/
MAJOR_MINOR_STREETS_ZOOM = 17
CITY_ZOOM = 10

class OpenStreetMapDownloader(): #IRVIN
    def __init__(self):
        self.osm = geopy.geocoders.Nominatim(user_agent='deforestation_drivers_quick'+str(random.random())) #AD
        
    def download_osm(self, lat, lon, path):
        closest_street = self.osm.reverse((lat, lon), MAJOR_MINOR_STREETS_ZOOM) #AD
        closest_city = self.osm.reverse((lat, lon), CITY_ZOOM)
        street_file = open(os.path.join(path, 'closest_street.json'), 'w')
        city_file = open(os.path.join(path, 'closest_city.json'), 'w')
        json.dump(closest_street[0].raw,street_file)
        json.dump(closest_city[0].raw, city_file)
        time.sleep(1)
        print('Done:', path)

 
def download_all(): #AD
    years = [2018]
    for year in years:
        td = OpenStreetMapDownloader()
        list_dir = os.listdir(os.path.join(os.path.dirname(os.getcwd()), 'download_images_quick', 'output', str(year)))
        for shapes in ['new_timber_plantations.shp']:
            list_index = os.listdir(
                os.path.join(os.path.dirname(os.getcwd()), 'download_images_quick', 'output', str(year), shapes))
            path_out_year = os.path.join(os.getcwd(), 'output', str(year))
            if os.path.exists(path_out_year) == False:
                os.mkdir(path_out_year)

            path_out_shapes = os.path.join(path_out_year, shapes)
            if os.path.exists(path_out_shapes) == False:
                os.mkdir(path_out_shapes)

            for index in ['46']:
                path_out = os.path.join(path_out_shapes, index)
                if os.path.exists(path_out) == False:
                    os.mkdir(path_out)

                path_in = os.path.join(os.path.dirname(os.getcwd()), 'download_images_quick', 'output', str(year),
                                    shapes, str(index))
                if os.path.exists(path_in) == True:
                    list_sensors = os.listdir(
                        os.path.join(os.path.dirname(os.getcwd()), 'download_images_quick', 'output', str(year), shapes,
                                    index))
                    for sensor in list_sensors:
                        list_images = os.listdir(
                            os.path.join(os.path.dirname(os.getcwd()), 'download_images_quick', 'output', str(year),
                                        shapes, index, sensor))
                        
                        image_name = list_images[0]
                        lon = float(image_name.split('_')[1])
                        if 'png' in (image_name.split('_')[2]):
                            lat = float((image_name.split('_')[2])[:-4])
                        else:
                            lat = float(image_name.split('_')[2])
                            
                        if sensor == 'planet_fixed':
                            if (os.path.exists(os.path.join(path_out, 'planet', 'closest_street.json')) == False) or (os.path.exists(os.path.join(path_out, 'planet', 'closest_city.json')) == False):

                                #td = OpenStreetMapDownloader()
                                td.download_osm(lat, lon, os.path.join(path_out, 'planet'))
                        
                        else:
                            if (os.path.exists(os.path.join(path_out, sensor, 'closest_street.json')) == False) or (os.path.exists(os.path.join(path_out, sensor, 'closest_city.json')) == False) or (os.path.getsize(os.path.join(path_out, sensor, 'closest_street.json')) == 0) or (os.path.getsize(os.path.join(path_out, sensor, 'closest_city.json')) == 0):

                                #td = OpenStreetMapDownloader()
                                td.download_osm(lat, lon, os.path.join(path_out, sensor))
                        
                            



if __name__ == "__main__":
    fire.Fire(download_all)