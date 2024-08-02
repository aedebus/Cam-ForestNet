import fire
import numpy as np
import os
import ee

def troubleshoot_planet(lat,lon,path,year):
    param = (0.05 * ((332 * 4.77) / 30)) / 720
    geometry = ee.Geometry.BBox(lon - param, lat - param, lon + param, lat + param)
    dataset = ee.ImageCollection("LANDSAT/LC08/C01/T1").filterDate(str(year + 1) + '-01-01', str(year + 5) + '-12-31').filterBounds(geometry)  # Use landsat for IR
    filtered_dataset = dataset.sort('CLOUD_COVER')
    if int(filtered_dataset.size().getInfo()) > 0:
        list_dataset = filtered_dataset.toList(filtered_dataset.size())
        for idx in range(list_dataset.length().getInfo()):
            image = ee.Image(list_dataset.get(0))
            nir = np.array((image.select('B5').sampleRectangle(region=geometry, defaultValue=0)).get('B5').getInfo(),
                           dtype='uint16')
            swir1 = np.array((image.select('B6').sampleRectangle(region=geometry, defaultValue=0)).get('B6').getInfo(),
                             dtype='uint16')
            swir2 = np.array((image.select('B7').sampleRectangle(region=geometry, defaultValue=0)).get('B7').getInfo(),
                             dtype='uint16')
            nir_arr = np.zeros((332, 332))
            swir1_arr = np.zeros((332, 332))
            swir2_arr = np.zeros((332, 332))

            for i in range(332):
                for j in range(332):
                    if (i // 12) >= nir.shape[0]:
                        x = nir.shape[0] - 1
                    else:
                        x = i // 12

                    if (j // 12) >= nir.shape[1]:
                        y = nir.shape[1] - 1
                    else:
                        y = j // 12

                    nir_arr[i][j] = nir[x][y]
                    swir1_arr[i][j] = swir1[x][y]
                    swir2_arr[i][j] = swir2[x][y]

            ir = np.array([nir_arr, swir1_arr, swir2_arr], dtype='uint16').transpose()
            name = str(year) + '_' + 'ir' + '_' + str(idx)

            np.save(os.path.join(path, name + '.npy'), ir)
            print('Ok:', path)

def download_ir(lat, lon, path, sensor, year):
    #print(path)
    if sensor == 'landsat':
        param = (0.05 * ((332*15)/30))/747
        geometry = ee.Geometry.BBox(lon - param, lat - param, lon + param, lat + param)
        dataset = ee.ImageCollection("LANDSAT/LC08/C02/T1_TOA").filterDate(str(year+1)+'-01-01', str(year+5)+'-12-31').filterBounds(geometry) #Only after 14/07/2024 (for landsat_nir_ps)
        filtered_dataset = dataset.filterMetadata('CLOUD_COVER', 'less_than', 20)
        filtered_dataset = filtered_dataset .filterMetadata('CLOUD_COVER_LAND', 'less_than', 20)
        filtered_dataset = filtered_dataset.sort('CLOUD_COVER')
        if int(filtered_dataset.size().getInfo())>0:
            list_dataset = filtered_dataset.toList(filtered_dataset.size())
            for idx in range(list_dataset.length().getInfo()):
                image =ee.Image(list_dataset.get(idx))
                nir = np.array((image.select('B5').sampleRectangle(region=geometry, defaultValue = 0)).get('B5').getInfo(), dtype = 'uint16')
                swir1 = np. array((image.select('B6').sampleRectangle(region=geometry, defaultValue = 0)).get('B6').getInfo(), dtype = 'uint16')
                swir2 = np. array((image.select('B7').sampleRectangle(region=geometry, defaultValue = 0)).get('B7').getInfo(), dtype = 'uint16')

                nir_arr = np.zeros((332, 332))
                swir1_arr = np.zeros((332, 332))
                swir2_arr = np.zeros((332, 332))

                for i in range(332):
                    for j in range(332):
                        if (i // 4) >= nir.shape[0]:
                            x = nir.shape[0] - 1
                        else:
                            x = i // 4

                        if (j // 4) >= nir.shape[1]:
                            y = nir.shape[1] - 1
                        else:
                            y = j // 4

                        nir_arr[i][j] = nir[x][y]
                        swir1_arr[i][j] = swir1[x][y]
                        swir2_arr[i][j] = swir2[x][y]

                ir = np.array([nir_arr, swir1_arr, swir2_arr], dtype = 'uint16').transpose()
                name = str(year) + '_'+ 'ir' + '_' + str(idx)

                np.save(os.path.join(path, name + '.npy'), ir)
        else:
            print('no data:', path)
            #troubleshoot_planet(lat, lon, path, year)
        # if (list_dataset.length().getInfo()) > 0:
        #     if filtered_dataset.size().getInfo() >= 5:
        #         list_new_col = []
        #         for idx in range(5):
        #             img_idx = ee.Image(list_dataset.get(idx))
        #             list_new_col.append(img_idx)
        #
        #         collectionFromConstructor = ee.ImageCollection(list_new_col)
        #         composite = ee.Algorithms.Landsat.simpleComposite(collectionFromConstructor)
        #         composite_img = composite.sampleRectangle(region=geometry)
        #         nir = np.array(composite_img.get('B5').getInfo(), dtype='uint16')
        #         swir1 = np.array(composite_img.get('B6').getInfo(), dtype='uint16')
        #         swir2 = np.array(composite_img.get('B7').getInfo(), dtype='uint16')
        #
        #         nir_arr = np.zeros((332, 332))
        #         swir1_arr = np.zeros((332, 332))
        #         swir2_arr = np.zeros((332, 332))
        #
        #         for i in range(332):
        #             for j in range(332):
        #                 nir_arr[i][j] = nir[i // 4][j // 4]
        #                 swir1_arr[i][j] = swir1[i // 4][j // 4]
        #                 swir2_arr[i][j] = swir2[i // 4][j // 4]
        #
        #         ir = np.array([nir_arr, swir1_arr, swir2_arr]).transpose()
        #         np.save(os.path.join(path, 'composite.npy'),ir)
        #
        #     else:
        #         list_new_col = []
        #         for idx in range(list_dataset.length().getInfo()):
        #             img_idx = ee.Image(list_dataset.get(idx))
        #             list_new_col.append(img_idx)
        #             collectionFromConstructor = ee.ImageCollection(list_new_col)
        #             composite = ee.Algorithms.Landsat.simpleComposite(collectionFromConstructor)
        #             composite_img=composite.sampleRectangle(region=geometry)
        #             nir = np.array(composite_img.get('B5').getInfo(), dtype='uint16')
        #             swir1 = np.array(composite_img.get('B6').getInfo(), dtype='uint16')
        #             swir2 = np.array(composite_img.get('B7').getInfo(), dtype='uint16')
        #
        #             nir_arr = np.zeros((332, 332))
        #             swir1_arr = np.zeros((332, 332))
        #             swir2_arr = np.zeros((332, 332))
        #
        #             for i in range(332):
        #                 for j in range(332):
        #                     nir_arr[i][j] = nir[i // 4][j // 4]
        #                     swir1_arr[i][j] = swir1[i // 4][j // 4]
        #                     swir2_arr[i][j] = swir2[i // 4][j // 4]
        #
        #             ir = np.array([nir_arr, swir1_arr, swir2_arr]).transpose()
        #
        #             np.save(os.path.join(path, 'composite.npy'), ir)


    elif sensor == 'planet':
        param = (0.05 * ((332*4.77)/30))/720
        geometry = ee.Geometry.BBox(lon - param, lat - param, lon + param, lat + param)
        dataset = ee.ImageCollection("LANDSAT/LC08/C01/T1").filterDate(str(year + 1) + '-01-01', str(year + 5) + '-12-31').filterBounds(geometry) #Use landsat for IR
        filtered_dataset = dataset.filterMetadata('CLOUD_COVER', 'less_than', 20)
        filtered_dataset = filtered_dataset.filterMetadata('CLOUD_COVER_LAND', 'less_than', 20)
        filtered_dataset = filtered_dataset.sort('CLOUD_COVER')
        if int(filtered_dataset.size().getInfo())>0:
            list_dataset = filtered_dataset.toList(filtered_dataset.size())
            for idx in range(list_dataset.length().getInfo()):
                image = ee.Image(list_dataset.get(idx))
                nir = np.array((image.select('B5').sampleRectangle(region=geometry, defaultValue = 0)).get('B5').getInfo(), dtype='uint16')
                swir1 = np.array((image.select('B6').sampleRectangle(region=geometry, defaultValue = 0)).get('B6').getInfo(), dtype='uint16')
                swir2 = np.array((image.select('B7').sampleRectangle(region=geometry, defaultValue = 0)).get('B7').getInfo(), dtype='uint16')
                nir_arr = np.zeros((332, 332))
                swir1_arr = np.zeros((332, 332))
                swir2_arr = np.zeros((332, 332))

                for i in range(332):
                    for j in range(332):
                        if (i // 12) >= nir.shape[0]:
                            x = nir.shape[0] - 1
                        else:
                            x = i // 12

                        if (j // 12) >= nir.shape[1]:
                            y = nir.shape[1] - 1
                        else:
                            y = j // 12

                        nir_arr[i][j] = nir[x][y]
                        swir1_arr[i][j] = swir1[x][y]
                        swir2_arr[i][j] = swir2[x][y]

                ir = np.array([nir_arr, swir1_arr, swir2_arr], dtype = 'uint16').transpose()
                name = str(year) + '_' + 'ir' + '_' + str(idx)

                np.save(os.path.join(path, name + '.npy'), ir)

        else:
            print('no data:', path)
            troubleshoot_planet(lat, lon, path, year)
    
    elif (sensor == 'sentinel1') or (sensor == 'sentinel2_10m'):
        param = (0.05 * ((332*10)/30))/720
        geometry = ee.Geometry.BBox(lon - param, lat - param, lon + param, lat + param)
        dataset = ee.ImageCollection("LANDSAT/LC08/C01/T1").filterDate(str(year + 1) + '-01-01', str(year + 5) + '-12-31').filterBounds(geometry) #Use landsat for IR
        filtered_dataset = dataset.filterMetadata('CLOUD_COVER', 'less_than', 20)
        filtered_dataset = filtered_dataset.filterMetadata('CLOUD_COVER_LAND', 'less_than', 20)
        filtered_dataset = filtered_dataset.sort('CLOUD_COVER')
        if int(filtered_dataset.size().getInfo())>0:
            list_dataset = filtered_dataset.toList(filtered_dataset.size())
            for idx in range(list_dataset.length().getInfo()):
                image = ee.Image(list_dataset.get(idx))
                nir = np.array((image.select('B5').sampleRectangle(region=geometry, defaultValue = 0)).get('B5').getInfo(), dtype='uint16')
                swir1 = np.array((image.select('B6').sampleRectangle(region=geometry, defaultValue = 0)).get('B6').getInfo(), dtype='uint16')
                swir2 = np.array((image.select('B7').sampleRectangle(region=geometry, defaultValue = 0)).get('B7').getInfo(), dtype='uint16')
                nir_arr = np.zeros((332, 332))
                swir1_arr = np.zeros((332, 332))
                swir2_arr = np.zeros((332, 332))

                for i in range(332):
                    for j in range(332):
                        if (i // 6) >= nir.shape[0]:
                            x = nir.shape[0] - 1
                        else:
                            x = i // 6

                        if (j // 6) >= nir.shape[1]:
                            y = nir.shape[1] - 1
                        else:
                            y = j // 6

                        nir_arr[i][j] = nir[x][y]
                        swir1_arr[i][j] = swir1[x][y]
                        swir2_arr[i][j] = swir2[x][y]

                ir = np.array([nir_arr, swir1_arr, swir2_arr], dtype = 'uint16').transpose()
                name = str(year) + '_' + 'ir' + '_' + str(idx)
                np.save(os.path.join(path, name + '.npy'), ir)

    elif sensor == 'sentinel2_20m':
        param = (0.05 * ((332*20)/30))/747
        geometry = ee.Geometry.BBox(lon - param, lat - param, lon + param, lat + param)
        dataset = ee.ImageCollection("LANDSAT/LC08/C01/T1").filterDate(str(year+1)+'-01-01', str(year+5)+'-12-31').filterBounds(geometry)
        filtered_dataset = dataset.filterMetadata('CLOUD_COVER', 'less_than', 20)
        filtered_dataset = filtered_dataset .filterMetadata('CLOUD_COVER_LAND', 'less_than', 20)
        filtered_dataset = filtered_dataset.sort('CLOUD_COVER')
        if (len(filtered_dataset.getInfo()['features'])>0):
            list_dataset = filtered_dataset.toList(filtered_dataset.size())
            idx = 0
            #for idx in range(list_dataset.length().getInfo()):
            image =ee.Image(list_dataset.get(idx))
            nir = np.array((image.select('B5').sampleRectangle(region=geometry, defaultValue = 0)).get('B5').getInfo(), dtype = 'uint16')
            swir1 = np. array((image.select('B6').sampleRectangle(region=geometry, defaultValue = 0)).get('B6').getInfo(), dtype = 'uint16')
            swir2 = np. array((image.select('B7').sampleRectangle(region=geometry, defaultValue = 0)).get('B7').getInfo(), dtype = 'uint16')

            nir_arr = np.zeros((332, 332))
            swir1_arr = np.zeros((332, 332))
            swir2_arr = np.zeros((332, 332))

            for i in range(332):
                for j in range(332):
                    if (i // 1.5) >= nir.shape[0]:
                        x = nir.shape[0] - 1
                    else:
                        x = int(i // 1.5)

                    if (j // 1.5) >= nir.shape[1]:
                        y = nir.shape[1] - 1
                    else:
                        y = int(j // 1.5)

                    nir_arr[i][j] = nir[x][y]
                    swir1_arr[i][j] = swir1[x][y]
                    swir2_arr[i][j] = swir2[x][y]

            ir = np.array([nir_arr, swir1_arr, swir2_arr], dtype = 'uint16').transpose()
            name = str(year) + '_'+ 'ir' + '_' + str(idx)

            np.save(os.path.join(path, name + '.npy'), ir)
        else:
            print('no image')

        
    elif sensor == 'landsat_30':
        param = (0.05 * ((332*30)/30))/747
        geometry = ee.Geometry.BBox(lon - param, lat - param, lon + param, lat + param)
        dataset = ee.ImageCollection("LANDSAT/LC08/C01/T1").filterDate(str(year+1)+'-01-01', str(year+5)+'-12-31').filterBounds(geometry)
        filtered_dataset = dataset.filterMetadata('CLOUD_COVER', 'less_than', 20)
        filtered_dataset = filtered_dataset .filterMetadata('CLOUD_COVER_LAND', 'less_than', 20)
        filtered_dataset = filtered_dataset.sort('CLOUD_COVER')
        if (len(filtered_dataset.getInfo()['features'])>0):
            list_dataset = filtered_dataset.toList(filtered_dataset.size())
            idx = 0
            #for idx in range(list_dataset.length().getInfo()):
            image =ee.Image(list_dataset.get(idx))
            nir = np.array((image.select('B5').sampleRectangle(region=geometry, defaultValue = 0)).get('B5').getInfo(), dtype = 'uint16')
            swir1 = np. array((image.select('B6').sampleRectangle(region=geometry, defaultValue = 0)).get('B6').getInfo(), dtype = 'uint16')
            swir2 = np. array((image.select('B7').sampleRectangle(region=geometry, defaultValue = 0)).get('B7').getInfo(), dtype = 'uint16')

            nir_arr = np.zeros((332, 332))
            swir1_arr = np.zeros((332, 332))
            swir2_arr = np.zeros((332, 332))

            for i in range(332):
                for j in range(332):
                    if (i // 2) >= nir.shape[0]:
                        x = nir.shape[0] - 1
                    else:
                        x = i // 2

                    if (j // 2) >= nir.shape[1]:
                        y = nir.shape[1] - 1
                    else:
                        y = j // 2

                    nir_arr[i][j] = nir[x][y]
                    swir1_arr[i][j] = swir1[x][y]
                    swir2_arr[i][j] = swir2[x][y]

            ir = np.array([nir_arr, swir1_arr, swir2_arr], dtype = 'uint16').transpose()
            name = str(year) + '_'+ 'ir' + '_' + str(idx)

            np.save(os.path.join(path, name + '.npy'), ir)
        
    elif sensor == 'landsat_nir_ps':
        param = (0.05 * ((332*15)/30))/747
        geometry = ee.Geometry.BBox(lon - param, lat - param, lon + param, lat + param)
        dataset = ee.ImageCollection("LANDSAT/LC08/C02/T1_TOA").filterDate(str(year+1)+'-01-01', str(year+5)+'-12-31').filterBounds(geometry) #Only after 14/07/2024 (for landsat_nir_ps)
        filtered_dataset = dataset.filterMetadata('CLOUD_COVER', 'less_than', 20)
        filtered_dataset = filtered_dataset .filterMetadata('CLOUD_COVER_LAND', 'less_than', 20)
        filtered_dataset = filtered_dataset.sort('CLOUD_COVER')
        print(filtered_dataset.size().getInfo())
        if int(filtered_dataset.size().getInfo())>0:
            list_dataset = filtered_dataset.toList(filtered_dataset.size())
            for idx in range(list_dataset.length().getInfo()):
                image =ee.Image(list_dataset.get(idx))
                nir = np.array((image.select('B5').sampleRectangle(region=geometry, defaultValue = 0)).get('B5').getInfo(), dtype = 'uint16')
                swir1 = np. array((image.select('B6').sampleRectangle(region=geometry, defaultValue = 0)).get('B6').getInfo(), dtype = 'uint16')
                swir2 = np. array((image.select('B7').sampleRectangle(region=geometry, defaultValue = 0)).get('B7').getInfo(), dtype = 'uint16')

                nir_arr = np.zeros((332, 332))
                swir1_arr = np.zeros((332, 332))
                swir2_arr = np.zeros((332, 332))

                for i in range(332):
                    for j in range(332):
                        if (i // 4) >= nir.shape[0]:
                            x = nir.shape[0] - 1
                        else:
                            x = i // 4

                        if (j // 4) >= nir.shape[1]:
                            y = nir.shape[1] - 1
                        else:
                            y = j // 4

                        nir_arr[i][j] = nir[x][y]
                        swir1_arr[i][j] = swir1[x][y]
                        swir2_arr[i][j] = swir2[x][y]

                ir = np.array([nir_arr, swir1_arr, swir2_arr], dtype = 'uint16').transpose()
                name = str(year) + '_'+ 'ir' + '_' + str(idx)

                np.save(os.path.join(path, name + '.npy'), ir)
        else:
            print('no data:', path)


    else:
        print('Sensor not available/known')


def download_ir_from_images(sensor):
    """

    :param sensor: landsat or planetscope
    :return: altitude, slope, aspect for all images created
    """
    #ee.Authenticate()
    ee.Initialize()
    print('ok init')

    years = [2015, 2016, 2017, 2018, 2019, 2020]
    for year in years:
        path_origin = os.path.join(os.path.dirname(os.getcwd()), 'download_images_quick', 'output', str(year))
        list_dir = os.listdir(path_origin)
        for d in list_dir:
            if ('production_forest' not in d) and ('small_scale_agriculture' not in d) and ('timber_sales' not in d) and ('mining_permits' not in d):
                shape = d
                path_sh = os.path.join(path_origin, shape)
                list_dir_sh = os.listdir(path_sh)
                for dn in list_dir_sh:
                    index = dn
                    if os.path.exists(os.path.join(path_sh, index, sensor + '_fixed')) == True and os.path.exists(os.path.join(path_sh, index, sensor)) == False:
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

                        if os.path.exists(os.path.join(path_out_index_sensor, str(year) + '_' + 'ir' + '_0' + '.npy')) == False:
                            download_ir(lat, lon, path_out_index_sensor, sensor, year)
                            print('Ok:', path_out_index_sensor)

                    elif os.path.exists(os.path.join(path_sh, index, sensor)) == True:
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

                        if os.path.exists(os.path.join(path_out_index_sensor, str(year) + '_' + 'ir' + '_0' + '.npy')) == False:
                            download_ir(lat, lon, path_out_index_sensor, sensor, year)
                            print('Ok:', path_out_index_sensor)
                    
                    elif os.path.exists(os.path.join(path_sh, index, sensor +'_nir_ps')) == True and os.path.exists(os.path.join(path_sh, index, sensor)) == False:
                        first_image = os.listdir(os.path.join(path_sh, index, sensor+'_nir_ps'))[0]
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

                        if os.path.exists(os.path.join(path_out_index_sensor, str(year) + '_' + 'ir' + '_0' + '.npy')) == False:
                            download_ir(lat, lon, path_out_index_sensor, sensor+'_nir_ps', year)
                            print('Ok:', path_out_index_sensor)


def download_ir_from_shape(sensor, shape):
    """
    :param sensor: landsat or planetscope
    :param shape: .shp file
    :return: ir/nir .npy file
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

                    download_ir(lat, lon, path_out_index_sensor, sensor, year)



        else:
            print('no files')

def download_ir_from_year(sensor, shape, year):
    """
    :param sensor: landsat or planetscope
    :param shape: .shp file
    :param year: forest loss event
    :return: ir/nir .npy file
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

                download_ir(lat, lon, path_out_index_sensor, sensor, year)

    else:
        print('no files')


def download_ir_from_index(sensor, shape, year):
    """

    :param sensor: landsat or planetscope
    :param shape: .shp file
    :param year: forest loss event
    :return: ir/nir .npy
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

            download_ir(lat, lon, path_out_index_sensor, sensor, year)

    else:
        print('no files')
        
if __name__ == "__main__":
    fire.Fire(download_ir_from_images)
    #fire.Fire(download_ir_from_shape)
    #fire.Fire(download_ir_from_year)
    #fire.Fire(download_ir_from_index)
