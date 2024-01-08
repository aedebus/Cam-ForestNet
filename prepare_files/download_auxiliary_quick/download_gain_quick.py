import os
import fire
import numpy as np
import ee

def download_gain_from_images(sensor):
    """

    :param sensor: landsat or planetscope
    :return: gains for all images generated
    """
    #ee.Authenticate()
    ee.Initialize()
    print('init ok')
    years = [2015, 2016, 2017, 2018, 2019, 2020]
    for year in years:
        path_origin = os.path.join(os.path.dirname(os.getcwd()), 'download_images_quick', 'output', str(year))
        list_dir = os.listdir(path_origin)
        for d in list_dir:
            shape = d
            path_sh = os.path.join(path_origin, shape)
            list_dir_sh = os.listdir(path_sh)
            for dn in list_dir_sh:
                index = dn
                if os.path.exists(os.path.join(path_sh, str(index), sensor)):
                    first_image = os.listdir(os.path.join(path_sh, str(index), sensor))[0]
                    lon = float(first_image.split('_')[1])
                    if 'png' in (first_image.split('_')[2]):
                        lat = float((first_image.split('_')[2])[:-4])
                    else:
                        lat = float(first_image.split('_')[2])

                    if sensor == 'landsat':
                        param = (0.05 * ((332 * 15) / 30.92)) / 372  # same area as image: resolution 30.92 m
                        geometry = ee.Geometry.BBox(lon - param, lat - param, lon + param, lat + param)
                        divider = 2

                    elif sensor == 'planet':
                        param = (0.05 * ((332 * 4.77) / 30.92)) / 372
                        geometry = ee.Geometry.BBox(lon - param, lat - param, lon + param, lat + param)
                        divider = 6

                    else:
                        print('Sensor not available/known')

                    dataset = ee.Image('UMD/hansen/global_forest_change_2021_v1_9').select('gain').sampleRectangle(region=geometry)
                    gain = dataset.get('gain')  # Forest gain during the study period

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

                    gain_arr = np.array(gain.getInfo(), dtype='uint8')
                    gain_arr_format = np.zeros((332, 332))

                    for i in range(332):
                        for j in range(332):
                            if (i // divider) >= gain_arr.shape[0]:
                                x = gain_arr.shape[0] - 1
                            else:
                                x = i // divider

                            if (j // divider) >= gain_arr.shape[1]:
                                y = gain_arr.shape[1] - 1
                            else:
                                y = j // divider

                            gain_arr_format[i][j] = gain_arr[x][y]

                    gain_fn = np.array([gain_arr_format], dtype='uint8')

                    np.save(os.path.join(path_out_index_sensor, 'gain.npy'), gain_fn)
                    print('Ok:', path_out_index_sensor)

                elif os.path.exists(os.path.join(path_sh, str(index), sensor)) == False and os.path.exists(os.path.join(path_sh, str(index), sensor + '_fixed')) == True:
                    first_image = os.listdir(os.path.join(path_sh, str(index), sensor + '_fixed'))[0]
                    lon = float(first_image.split('_')[1])
                    if 'png' in (first_image.split('_')[2]):
                        lat = float((first_image.split('_')[2])[:-4])
                    else:
                        lat = float(first_image.split('_')[2])

                    if sensor == 'landsat':
                        param = (0.05 * ((332 * 15) / 30.92)) / 372  # same area as image: resolution 30.92 m
                        geometry = ee.Geometry.BBox(lon - param, lat - param, lon + param, lat + param)
                        divider = 2

                    elif sensor == 'planet':
                        param = (0.05 * ((332 * 4.77) / 30.92)) / 372
                        geometry = ee.Geometry.BBox(lon - param, lat - param, lon + param, lat + param)
                        divider = 6

                    else:
                        print('Sensor not available/known')

                    dataset = ee.Image('UMD/hansen/global_forest_change_2021_v1_9').select('gain').sampleRectangle(region=geometry)
                    gain = dataset.get('gain')  # Forest gain during the study period

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

                    gain_arr = np.array(gain.getInfo(), dtype='uint8')
                    gain_arr_format = np.zeros((332, 332))

                    for i in range(332):
                        for j in range(332):
                            if (i // divider) >= gain_arr.shape[0]:
                                x = gain_arr.shape[0] - 1
                            else:
                                x = i // divider

                            if (j // divider) >= gain_arr.shape[1]:
                                y = gain_arr.shape[1] - 1
                            else:
                                y = j // divider

                            gain_arr_format[i][j] = gain_arr[x][y]

                    gain_fn = np.array([gain_arr_format], dtype='uint8')

                    np.save(os.path.join(path_out_index_sensor, 'gain.npy'), gain_fn)
                    print('Ok:', path_out_index_sensor)


def download_gain_from_shape(sensor, shape):
    """

    :param sensor: landsat or planetscope
    :param shape: .shp file
    :return: gains for all images for the shape selected (all indexes, all years)
    """
    #ee.Authenticate()
    ee.Initialize()


    years = [2015, 2016, 2017, 2018, 2019, 2020]
    for year in years:
        path_origin = os.path.join(os.path.dirname(os.getcwd()), 'download_images_quick', 'output', str(year), shape)
        list_dir = os.listdir(path_origin)

        if len(list_dir) > 0:
            for i in list_dir:
                index = int(i)
                if os.path.exists(os.path.join(path_origin, str(index), sensor)) == True:
                    first_image = os.listdir(os.path.join(path_origin, str(index), sensor))[0]
                    lon = float(first_image.split('_')[1])
                    lat = float(first_image.split('_')[2])
    
                    if sensor == 'landsat':
                        param = (0.05 * ((332 * 15) / 30.92)) / 372  # same area as image: resolution 30.92 m
                        geometry = ee.Geometry.BBox(lon - param, lat - param, lon + param, lat + param)
                        divider = 2
    
                    elif sensor == 'planet':
                        param = (0.05 * ((332 * 4.77) / 30.92)) / 372
                        geometry = ee.Geometry.BBox(lon - param, lat - param, lon + param, lat + param)
                        divider = 6
    
                    else:
                        print('Sensor not available/known')
    
                    dataset = (ee.Image('UMD/hansen/global_forest_change_2021_v1_9')).select('gain').sampleRectangle(region=geometry)
                    gain = dataset.get('gain')  # Forest gain during the study period
    
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
    
                    gain_arr = np.array(gain.getInfo(), dtype='uint8')
                    gain_arr_format = np.zeros((332, 332))

                    for i in range(332):
                        for j in range(332):
                            if (i // divider) >= gain_arr.shape[0]:
                                x = gain_arr.shape[0] - 1
                            else:
                                x = i // divider

                            if (j // divider) >= gain_arr.shape[1]:
                                y = gain_arr.shape[1] - 1
                            else:
                                y = j // divider

                            gain_arr_format[i][j] = gain_arr[x][y]
    
                    gain_fn = np.array([gain_arr_format], dtype='uint8')
    
                    np.save(os.path.join(path_out_index_sensor, 'gain.npy'), gain_fn)
    
        else:
            print('no files')


def download_gain_from_year(sensor, shape, year):
    """

    :param sensor: landsat or planetscope
    :param shape: .shp file
    :param year: year forest gain event
    :return: gains for all images for the shape and year selected (all indexes)
    """
    #ee.Authenticate()
    ee.Initialize()

    path_origin = os.path.join(os.path.dirname(os.getcwd()), 'download_images_quick', 'output', str(year), shape)
    list_dir = os.listdir(path_origin)

    if len(list_dir) > 0:
        for i in list_dir:
            index = int(i)
            if os.path.exists(os.path.join(path_origin, str(index), sensor)) == True:
                first_image = os.listdir(os.path.join(path_origin, str(index), sensor))[0]
                lon = float(first_image.split('_')[1])
                lat = float(first_image.split('_')[2])
    
                if sensor == 'landsat':
                    param = (0.05 * ((332 * 15) / 30.92)) / 372  # same area as image: resolution 30.92 m
                    geometry = ee.Geometry.BBox(lon - param, lat - param, lon + param, lat + param)
                    divider = 2
    
                elif sensor == 'planet':
                    param = (0.05 * ((332 * 4.77) / 30.92)) / 372
                    geometry = ee.Geometry.BBox(lon - param, lat - param, lon + param, lat + param)
                    divider = 6
    
                else:
                    print('Sensor not available/known')
    
                dataset = (ee.Image('UMD/hansen/global_forest_change_2021_v1_9')).select('gain').sampleRectangle(region=geometry)
                gain = dataset.get('gain')  # Forest gain during the study period
    
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
    
                gain_arr = np.array(gain.getInfo(), dtype = 'uint8')
                gain_arr_format = np.zeros((332, 332))

                for i in range(332):
                    for j in range(332):
                        if (i // divider) >= gain_arr.shape[0]:
                            x = gain_arr.shape[0] - 1
                        else:
                            x = i // divider

                        if (j // divider) >= gain_arr.shape[1]:
                            y = gain_arr.shape[1] - 1
                        else:
                            y = j // divider

                        gain_arr_format[i][j] = gain_arr[x][y]
    
                gain_fn = np.array([gain_arr_format], dtype='uint8')
                np.save(os.path.join(path_out_index_sensor, 'gain.npy'), gain_fn)
    
    else:
        print('no files')



def download_gain_from_index(sensor, shape, year):
    """

    :param sensor: landsat or planetscope
    :param shape: .shp file
    :param year: forest gain event
    :return: gains for images with the selected shape, year and index
    """
    #ee.Authenticate()
    ee.Initialize()

    path_origin = os.path.join(os.path.dirname(os.getcwd()), 'download_images_quick', 'output', str(year), shape)
    list_dir = os.listdir(path_origin)
    if len(list_dir) > 0:
        list_index = []
        for i in list_dir:
            list_index.append(i)
        index = int(input("Enter an integer among:" + str(list_index)))
        if os.path.exists(os.path.join(path_origin, str(index), sensor)) == True:
            first_image = os.listdir(os.path.join(path_origin, str(index), sensor))[0]
            lon = float(first_image.split('_')[1])
            lat = float(first_image.split('_')[2])
    
            if sensor == 'landsat':
                param = (0.05 * ((332 * 15) / 30.92)) / 372  # same area as image: resolution 30.92 m
                geometry = ee.Geometry.BBox(lon - param, lat - param, lon + param, lat + param)
                divider = 2
    
            elif sensor == 'planet':
                param = (0.05 * ((332 * 4.77) / 30.92)) / 372
                geometry = ee.Geometry.BBox(lon - param, lat - param, lon + param, lat + param)
                divider = 6
    
            else:
                print('Sensor not available/known')
    
            dataset = ee.Image('UMD/hansen/global_forest_change_2021_v1_9').select('gain').sampleRectangle(region=geometry)
            gain = dataset.get('gain')  # Forest gain during the study period
    
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
    
            gain_arr = np.array(gain.getInfo(), dtype='uint8')
            gain_arr_format = np.zeros((332, 332))
    
            for i in range(332):
                for j in range(332):
                    if (i // divider) >= gain_arr.shape[0]:
                        x = gain_arr.shape[0] - 1
                    else:
                        x = i // divider

                    if (j // divider) >= gain_arr.shape[1]:
                        y = gain_arr.shape[1] - 1
                    else:
                        y = j // divider

                    gain_arr_format[i][j] = gain_arr[x][y]
    
            gain_fn = np.array([gain_arr_format], dtype='uint8')
    
            np.save(os.path.join(path_out_index_sensor, 'gain.npy'), gain_fn)
            print('Ok:', index)

    else:
        print('no files')


if __name__ == "__main__":
    fire.Fire(download_gain_from_images)
    #fire.Fire(download_gain_from_shape)
    #fire.Fire(download_gain_from_year)
    #fire.Fire(download_gain_from_index)

