'''Adapted from Stanford ML Group (Irvin et al., 2020) by Amandine Debus '''

import json
import fire
import os
import numpy as np
from shapely.geometry import Polygon, Point
import shapefile



def get_peat_polygons():
    file = shapefile.Reader(os.path.join(os.getcwd(), 'input', 'peat_lands.shp')) #AD
    shapes = file.shapes() #AD
    polygons = []

    for feat in shapes: #AD
        coords1 = np.array(feat.points, dtype='object').squeeze() #AD

        if len(coords1.shape) == 3:
            for i in range(len(coords1)):
                coords2 = np.array(coords1[i], dtype='object').squeeze()
                assert len(coords2.shape) == 2
                coords2 = np.flip(coords2[:, :2], axis=1)
                polygons.append(Polygon(coords2))

        elif len(coords1.shape) == 1:
            for i in range(len(coords1)):
                coords2 = np.array(coords1[i], dtype='object').squeeze()
                if len(coords2.shape) == 1:
                    for j in range(len(coords2)):
                        coords3 = np.array(coords2[j]).squeeze()
                        assert len(coords3.shape) == 2
                        coords3 = np.flip(coords3[:, :2], axis=1)
                        polygons.append(Polygon(coords3))
                else:
                    assert len(coords2.shape) == 2
                    coords2 = np.flip(coords2[:, :2], axis=1)
                    polygons.append(Polygon(coords2))

        else:
            assert len(coords1.shape) == 2
            coords2 = np.flip(coords1[:, :2], axis=1)
            polygons.append(Polygon(coords2))
    
    return polygons 

def save_peat_result(polygons, lat, lon, path):
    result = {'peat': False}
    point = Point(lon, lat)
    for p in polygons:
        if point.within(p):
            coords = list(zip(*p.exterior.coords.xy))
            result['peat'] = True 
            result['polygon'] = coords

    json.dump(result, 
              open(os.path.join(path, 'peat.json'), 'w'))

    
def get_peat(sensor):
    polygons = get_peat_polygons()
    #AD
    years = ['2015', '2016', '2016', '2017', '2018', '2019', '2020']
    for year in years:
        path_origin = os.path.join(os.path.dirname(os.getcwd()), 'download_images_quick', 'output', str(year))
        list_dir = os.listdir(path_origin)
        for d in list_dir:
            shape = d
            path_sh = os.path.join(path_origin, shape)
            list_dir_sh = os.listdir(path_sh)
            for dn in list_dir_sh:
                index = dn
                if os.path.exists(os.path.join(path_sh, index, sensor)):
                    first_image = os.listdir(os.path.join(path_sh, index, sensor))[0]
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

                    path_out_index_sensor = os.path.join(path_out_index, sensor)
                    if os.path.exists(path_out_index_sensor) == False:
                        os.mkdir(path_out_index_sensor)

                    save_peat_result(polygons, lat, lon, path_out_index_sensor)
                
                elif os.path.exists(os.path.join(path_sh, index, sensor)) == False and os.path.exists(os.path.join(path_sh, index, sensor + '_fixed')) == True:
                    first_image = os.listdir(os.path.join(path_sh, index, sensor + '_fixed'))[0]
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

                    path_out_index_sensor = os.path.join(path_out_index, sensor)
                    if os.path.exists(path_out_index_sensor) == False:
                        os.mkdir(path_out_index_sensor)

                    save_peat_result(polygons, lat, lon, path_out_index_sensor)


def get_peat_shape_year(sensor, shape, year):
    polygons = get_peat_polygons()
    path_origin = os.path.join(os.path.dirname(os.getcwd()), 'download_images_quick', 'output', str(year))
    path_sh = os.path.join(path_origin, shape)
    list_dir_sh = os.listdir(path_sh)
    for dn in list_dir_sh:
        index = dn
        if os.path.exists(os.path.join(path_sh, index, sensor)):
            first_image = os.listdir(os.path.join(path_sh, index, sensor))[0]
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

            path_out_index_sensor = os.path.join(path_out_index, sensor)
            if os.path.exists(path_out_index_sensor) == False:
                os.mkdir(path_out_index_sensor)

            save_peat_result(polygons, lat, lon, path_out_index_sensor)


if __name__ == "__main__":
    fire.Fire(get_peat)
    #fire.Fire(get_peat_shape_year)
