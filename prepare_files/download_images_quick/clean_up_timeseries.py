# -*- coding: utf-8 -*-
"""
@author: Amandine Debus
"""

import os
import shutil
import pickle
import ee

years = [2015,2016,2017,2018,2019,2020]
nb_deleted = 0
list_to_redownload =[]
for year in years:
    list_shapes = os.listdir(os.path.join(os.getcwd(),'output_timeseries', str(year)))
    for s in list_shapes:
        list_index = os.listdir(os.path.join(os.getcwd(),'output_timeseries', str(year), s))
        for i in list_index:
            if os.path.exists(os.path.join(os.getcwd(),'output_timeseries', str(year), s, i, 'landsat')) == True:
                image_landsat = os.listdir(os.path.join(os.getcwd(), 'output_timeseries', str(year), s, i, 'landsat'))
                if os.path.getsize(os.path.join(os.getcwd(),'output_timeseries', str(year), s, i, 'landsat', image_landsat[0])) == 0:
                    shutil.rmtree(os.path.join(os.getcwd(),'output_timeseries', str(year), s, i, 'landsat'))
                    nb_deleted += 1
            '''if os.path.exists(os.path.join(os.getcwd(),'output_timeseries', str(year), s, i, 'planet')) == True:
                image_planet = os.listdir(os.path.join(os.getcwd(), 'output_timeseries', str(year), s, i, 'planet'))
                if os.path.getsize(os.path.join(os.getcwd(),'output_timeseries', str(year), s, i, 'planet', image_planet[0])) < 10000:
                    shutil.rmtree(os.path.join(os.getcwd(),'output_timeseries', str(year), s, i, 'planet'))
                    nb_deleted += 1
            
            if os.path.exists(os.path.join(os.getcwd(),'output_timeseries', str(year), s, i, 'planet_fixed')) == True:
                image_planet = os.listdir(os.path.join(os.getcwd(), 'output_timeseries', str(year), s, i, 'planet_fixed'))
                if os.path.getsize(os.path.join(os.getcwd(),'output_timeseries', str(year), s, i, 'planet_fixed', image_planet[0])) < 10000:
                    shutil.rmtree(os.path.join(os.getcwd(),'output_timeseries', str(year), s, i, 'planet_fixed'))
                    list_to_redownload.append([i, ([image_planet[0].split('_')[1], image_planet[0].split('_')[2]]), year, os.path.join(os.getcwd(),'output_timeseries', str(year), s)])
                    nb_deleted += 1'''

#with open("to_redownload", "wb") as fp:   
    #pickle.dump(list_to_redownload, fp)

print('Number of images deleted:', nb_deleted)

#Test

# list_index = os.path.join(os.getcwd(), 'output', '2015', 'agro_industrial_plantations_2015.shp', '0')
# image_landsat = os.listdir(os.path.join(list_index, 'landsat'))
# if os.path.exists(os.path.join(list_index, 'landsat')) == True:
#     if os.path.getsize(os.path.join(list_index, 'landsat', image_landsat[0])) < 10000:
#         shutil.rmtree(os.path.dirname(os.path.join(list_index, 'landsat', image_landsat[0])))


'''Change names
for year in years:
    list_shapes = os.listdir(os.path.join(os.getcwd(),'output_timeseries', str(year)))
    for s in list_shapes:
        list_index = os.listdir(os.path.join(os.getcwd(),'output_timeseries', str(year), s))
        for i in list_index:
            if os.path.exists(os.path.join(os.getcwd(),'output_timeseries', str(year), s, i, 'planet_timeseries')) == True:
                os.rename(os.path.join(os.getcwd(),'output_timeseries', str(year), s, i, 'planet_timeseries'), os.path.join(os.getcwd(),'output_timeseries', str(year), s, i, 'planet'))
                
    
for year in years:
    list_shapes = os.listdir(os.path.join(os.getcwd(),'output_timeseries', str(year)))
    for s in list_shapes:
        list_index = os.listdir(os.path.join(os.getcwd(),'output_timeseries', str(year), s))
        for i in list_index:
            if os.path.exists(os.path.join(os.getcwd(),'output_timeseries', str(year), s, i, 'planet')) == True:
                image_planet = os.listdir(os.path.join(os.getcwd(), 'output_timeseries', str(year), s, i, 'planet'))
                for image in image_planet:
                    if 'timeseries1' in image:
                        spl = image.split('_')
                        if spl[3] != 'timeseries1.png':
                            os.rename(os.path.join(os.getcwd(), 'output_timeseries', str(year), s, i, 'planet', spl[0] +'_' + spl[1] +'_' + spl[2] +'_' + str(i)+'timeseries1.png'), 
                                      os.path.join(os.getcwd(), 'output_timeseries', str(year), s, i, 'planet', spl[0] +'_' + spl[1] +'_' + spl[2] +'_' + 'timeseries1.png'))
                    if 'timeseries2' in image:
                        spl = image.split('_')
                        if spl[3] != 'timeseries2.png':
                            os.rename(os.path.join(os.getcwd(), 'output_timeseries', str(year), s, i, 'planet', spl[0] +'_' + spl[1] +'_' + spl[2] +'_' + str(i)+'timeseries2.png'), 
                                      os.path.join(os.getcwd(), 'output_timeseries', str(year), s, i, 'planet', spl[0] +'_' + spl[1] +'_' + spl[2] +'_' + 'timeseries2.png'))
                    if 'timeseries3' in image:
                        spl = image.split('_')
                        if spl[3] != 'timeseries3.png':
                            os.rename(os.path.join(os.getcwd(), 'output_timeseries', str(year), s, i, 'planet',  spl[0] +'_' + spl[1] +'_' + spl[2] +'_' + str(i)+'timeseries3.png'), 
                                      os.path.join(os.getcwd(), 'output_timeseries', str(year), s, i, 'planet', spl[0] +'_' + spl[1] +'_' + spl[2] +'_' + 'timeseries3.png'))
                    if 'timeseries4' in image:
                        spl = image.split('_')
                        if spl[3] != 'timeseries4.png':
                            os.rename(os.path.join(os.getcwd(), 'output_timeseries', str(year), s, i, 'planet',  spl[0] +'_' + spl[1] +'_' + spl[2] +'_' + str(i)+'timeseries4.png'), 
                                      os.path.join(os.getcwd(), 'output_timeseries', str(year), s, i, 'planet', spl[0] +'_' + spl[1] +'_' + spl[2] +'_' + 'timeseries4.png'))
                    if 'timeseries5' in image:
                        spl = image.split('_')
                        if spl[3] != 'timeseries5.png':
                            os.rename(os.path.join(os.getcwd(), 'output_timeseries', str(year), s, i, 'planet',  spl[0] +'_' + spl[1] +'_' + spl[2] +'_' + str(i)+'timeseries5.png'), 
                                      os.path.join(os.getcwd(), 'output_timeseries', str(year), s, i, 'planet', spl[0] +'_' + spl[1] +'_' + spl[2] +'_' + 'timeseries5.png'))
                    '''