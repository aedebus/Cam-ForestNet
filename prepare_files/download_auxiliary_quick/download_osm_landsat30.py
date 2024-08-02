import fire
import numpy as np
import os
import xarray as xr
import pandas as pd
import shutil


def download_osm_landsat30():
    """
    :return: osm parameters for all images created
    """
    years = ['2015', '2016', '2017', '2018', '2019', '2020']
    for year in years:
        path_origin = os.path.join(os.path.dirname(os.getcwd()), 'download_images_quick', 'output', str(year))
        list_dir = os.listdir(path_origin)
        for d in list_dir:
            if ('production_forest' not in d) and ('small_scale_agriculture' not in d) and ('timber_sales' not in d) and ('mining_permit' not in d) and ('small_scale_agriculture' not in d) and ('Subsistence agriculture' not in d) and ('sar' not in d):
                shape = d
                path_sh = os.path.join(path_origin, shape)
                list_dir_sh = os.listdir(path_sh)
                for dn in list_dir_sh:
                    index = dn
                    if os.path.exists(os.path.join(path_sh, str(index), 'landsat_30')) == True and os.path.exists(os.path.join(path_sh, str(index), 'landsat')) == True:
                        first_image = os.listdir(os.path.join(path_sh, str(index), 'landsat_30'))[0]
                        lon = float(first_image.split('_')[1])
                        if 'png' in (first_image.split('_')[2]):
                            lat = float((first_image.split('_')[2])[:-4])
                        else:
                            lat = float(first_image.split('_')[2])
                        path_out_year = os.path.join(os.getcwd(), 'output', str(year))
                        if os.path.exists(path_out_year) == False:
                            os.mkdir(path_out_year)

                        path_out_shapes = os.path.join(path_out_year, shape)
                        if os.path.exists(path_out_shapes) == False:
                            os.mkdir(path_out_shapes)

                        path_out_index = os.path.join(path_out_shapes, str(index))
                        if os.path.exists(path_out_index) == False:
                            os.mkdir(path_out_index)

                        path_out_index_sensor = os.path.join(path_out_index, 'landsat_30')
                        if os.path.exists(path_out_index_sensor) == False:
                            os.mkdir(path_out_index_sensor)
                        
                        source1= os.path.join(path_out_index,'landsat', 'closest_street.json')
                        shutil.copyfile(os.path.join(source1), os.path.join(path_out_index_sensor, 'closest_street.json'))

                        source2= os.path.join(path_out_index,'landsat', 'closest_city.json')
                        shutil.copyfile(os.path.join(source2), os.path.join(path_out_index_sensor, 'closest_city.json'))

                        print(path_out_index_sensor)
                    

    

if __name__ == "__main__":
    fire.Fire(download_osm_landsat30)
