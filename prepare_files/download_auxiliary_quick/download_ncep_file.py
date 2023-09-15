import fire
import numpy as np
import os
import xarray as xr
import pandas as pd

#DATA SOURCE: NCEP Climate Forecast System Version 2 (CFSv2) Monthly Products: https://rda.ucar.edu/#dsrqst/DEBUS587528/
#(CFSV2)

class NCEPDownloader(): #Structure of the class from Irvin et al.

    def get_ncep(self):
        #BELOW: names for GRIB format
        # surface_level_albedo = 'ALBDO_P8_L1'
        # clear_sky_longwave_flux_downward = 'CSDLF_P8_L1'
        # clear_sky_longwave_flux_upward = 'CSULF_P8_L1' #Ground or water surface level
        # clear_sky_solar_flux_downward = 'CSDSF_P8_L1'
        # clear_sky_solar_flux_upward = 'CSUSF_P8_L1' #Ground of water surface level
        #
        # direct_evaporation_from_bare_soil = 'EVBS_P8_L1'
        # longwave_radiation_flux_downward = 'DLWRF_P8_L1'
        # #type of statistical approach: average of forecast averages, each of an equal duration (6h intervals between forecasts: beginning at forecast time'
        # #and NOT: average of N forecasts (or initialized analyses (Intervals of 6 hours)
        # shortwave_radiation_flux_downward = 'DSWRF_P8_L1' #Ground or water surface, average of forecast averages
        # ground_heat_net_flux = 'GFLUX_P8_L1' #average of forecast averages
        # latent_heat_net_flux = 'LHTFL_P8_L1' #average of forecast averages
        # specific_humidity = 'SPFH_P8_L103'
        # max_specific_humidity = 'QMAX_P8_L103' #at 2 m
        # min_specific_humidity = 'QMIN_P8_L103' #at 2 m
        # potential_evaporation_rate = 'PEVPR_P8_L1' #average of forecast averages
        # ground_level_precipitation = 'APCP_P8_L1'
        # sensible_heat_net_flux = 'SHTFL_P8_L1' #average of forecast averages
        # volumetric_soil_moisture_content1 = 'SOILW_P8_2L106_1' #different depths with lv_DBLL0_l0/l1: 0.1 m
        # volumetric_soil_moisture_content2 = 'SOILW_P8_2L106_2' #0.4 m
        # volumetric_soil_moisture_content3 = 'SOILW_P8_2L106_3' #1.0 m
        # volumetric_soil_moisture_content4 = 'SOILW_P8_2L106_4' #2.0 m
        # air_pressure_at_surface_level = 'PRES_P8_L1'
        # wind_component_u = 'UGRD_P8_L103' #specified height level above ground (and not hybrid level)
        # wind_component_v = 'VGRD_P8_L103'
        # water_runoff_at_surface_level = 'WATR_P8_L1'
        #
        # sublimation = 'SBSNO_P8_L1'
        #
        # temperature = 'TMP_P8_L1' #at surface or water level
        # longwave_radiation_flux_upward = 'ULWRF_P8_L1'  #Ground or water surface, average of forecast averages
        # shortwave_radiation_flux_upward = 'USWRF_P8_L1'  #Ground or water surface, average of forecast averages

        surface_level_albedo = 'ALBDO_L1_FcstAvg6hr'
        clear_sky_longwave_flux_downward = 'CSDLF_L1_FcstAvg6hr'
        clear_sky_longwave_flux_upward = 'CSULF_L1_FcstAvg6hr'  # Ground or water surface level
        clear_sky_solar_flux_downward = 'CSDSF_L1_FcstAvg6hr'
        clear_sky_solar_flux_upward = 'CSUSF_L1_FcstAvg6hr' # Ground of water surface level

        direct_evaporation_from_bare_soil = 'EVBS_L1_FcstAvg6hr'
        longwave_radiation_flux_downward = 'DLWRF_L1_FcstAvg6hr'
        # type of statistical approach: average of forecast averages, each of an equal duration (6h intervals between forecasts: beginning at forecast time'
        # and NOT: average of N forecasts (or initialized analyses (Intervals of 6 hours)
        shortwave_radiation_flux_downward = 'DSWRF_L1_FcstAvg6hr'  # Ground or water surface, average of forecast averages
        ground_heat_net_flux = 'GFLUX_L1_FcstAvg6hr'  # average of forecast averages
        latent_heat_net_flux = 'LHTFL_L1_FcstAvg6hr'  # average of forecast averages
        specific_humidity = 'SPF_H_L103_Avg'
        max_specific_humidity = 'QMAX_L103_Pd'  # at 2 m
        min_specific_humidity = 'QMIN_L103_Pd'  # at 2 m
        potential_evaporation_rate = 'PEVPR_L1_FcstAvg6hr'  # average of forecast averages
        ground_level_precipitation = 'A_PCP_L1_AccumAvg'
        sensible_heat_net_flux = 'SHTFL_L1_FcstAvg6hr'  # average of forecast averages
        volumetric_soil_moisture_content1 = 'SOILW_Y106_Avg_1'  # different depths with lv_DBLL0_l0/l1: 0.1 m
        volumetric_soil_moisture_content2 = 'SOILW_Y106_Avg_2'  # 0.4 m
        volumetric_soil_moisture_content3 = 'SOILW_Y106_Avg_3'  # 1.0 m
        volumetric_soil_moisture_content4 = 'SOILW_Y106_Avg_4'  # 2.0 m
        air_pressure_at_surface_level = 'PRES_L1_Avg'
        wind_component_u = 'U_GRD_L6_Avg'  # maximum wind level
        wind_component_v = 'V_GRD_L6_Avg' #maximum wind level
        water_runoff_at_surface_level = 'WATR_L1_AccumAvg'

        sublimation = 'SBSNO_L1_FcstAvg6hr'

        temperature = 'TMP_L1_Avg'  # at surface or water level
        longwave_radiation_flux_upward = 'ULWRF_L1_FcstAvg6hr'  # Ground or water surface, average of forecast averages
        shortwave_radiation_flux_upward = 'USWRF_L1_FcstAvg6hr'  # Ground or water surface, average of forecast averages

        return surface_level_albedo, clear_sky_longwave_flux_downward, clear_sky_longwave_flux_upward, clear_sky_solar_flux_downward, clear_sky_solar_flux_upward, \
               direct_evaporation_from_bare_soil, longwave_radiation_flux_downward, shortwave_radiation_flux_downward, ground_heat_net_flux, latent_heat_net_flux, \
               specific_humidity, max_specific_humidity, min_specific_humidity, potential_evaporation_rate, ground_level_precipitation, sensible_heat_net_flux, \
               volumetric_soil_moisture_content1, volumetric_soil_moisture_content2, volumetric_soil_moisture_content3, volumetric_soil_moisture_content4, \
               air_pressure_at_surface_level, wind_component_u, wind_component_v, water_runoff_at_surface_level, sublimation, temperature, longwave_radiation_flux_upward, shortwave_radiation_flux_upward

    def names(self):
        names = {
            'ALBDO_L1_FcstAvg6hr': 'surface_level_albedo',
            'CSDLF_L1_FcstAvg6hr': 'clear_sky_longwave_flux_downward',
            'CSULF_L1_FcstAvg6hr':  'clear_sky_longwave_flux_upward',
            'CSDSF_L1_FcstAvg6hr': 'clear_sky_solar_flux_downward',
            'CSUSF_L1_FcstAvg6hr':'clear_sky_solar_flux_upward',
            'EVBS_L1_FcstAvg6hr': 'direct_evaporation_from_bare_soil',
            'DLWRF_L1_FcstAvg6hr':'longwave_radiation_flux_downward',
            'DSWRF_L1_FcstAvg6hr': 'shortwave_radiation_flux_downward',
            'GFLUX_L1_FcstAvg6hr': 'ground_heat_net_flux',
            'LHTFL_L1_FcstAvg6hr': 'latent_heat_net_flux',
            'SPF_H_L103_Avg' : 'specific_humidity',
            'QMAX_L103_Pd' : 'max_specific_humidity',
            'QMIN_L103_Pd':'min_specific_humidity',
            'PEVPR_L1_FcstAvg6hr':'potential_evaporation_rate',
            'A_PCP_L1_AccumAvg':'ground_level_precipitation',
            'SHTFL_L1_FcstAvg6hr': 'sensible_heat_net_flux',
            'SOILW_Y106_Avg_1':'volumetric_soil_moisture_content1',
            'SOILW_Y106_Avg_2':'volumetric_soil_moisture_content2',
            'SOILW_Y106_Avg_3':'volumetric_soil_moisture_content3',
            'SOILW_Y106_Avg_4':'volumetric_soil_moisture_content4',
            'PRES_L1_Avg':'air_pressure_at_surface_level',
            'U_GRD_L6_Avg': 'wind_component_u',
            'V_GRD_L6_Avg':'wind_component_v',
            'WATR_L1_AccumAvg' : 'water_runoff_at_surface_level',
            'SBSNO_L1_FcstAvg6hr':'sublimation',
            'TMP_L1_Avg': 'temperature',
            'ULWRF_L1_FcstAvg6hr':'longwave_radiation_flux_upward',
            'USWRF_L1_FcstAvg6hr':'shortwave_radiation_flux_upward',
        }
        return names

    def filelist_name(self):
        filelist = ['flxl.gdas.201101.grb2.nc',
                    'pgbl.gdas.201101.grb2.nc',
                    'pgbh.gdas.201101.grb2.nc',
                    'flxf.gdas.201101.grb2.nc',
                    'flxl.gdas.201102.grb2.nc',
                    'pgbl.gdas.201102.grb2.nc',
                    'pgbh.gdas.201102.grb2.nc',
                    'flxf.gdas.201102.grb2.nc',
                    'flxl.gdas.201103.grb2.nc',
                    'pgbl.gdas.201103.grb2.nc',
                    'pgbh.gdas.201103.grb2.nc',
                    'flxf.gdas.201103.grb2.nc',
                    'flxl.gdas.201104.grb2.nc',
                    'pgbl.gdas.201104.grb2.nc',
                    'pgbh.gdas.201104.grb2.nc',
                    'flxf.gdas.201104.grb2.nc',
                    'flxl.gdas.201105.grb2.nc',
                    'pgbl.gdas.201105.grb2.nc',
                    'pgbh.gdas.201105.grb2.nc',
                    'flxf.gdas.201105.grb2.nc',
                    'flxl.gdas.201106.grb2.nc',
                    'pgbl.gdas.201106.grb2.nc',
                    'pgbh.gdas.201106.grb2.nc',
                    'flxf.gdas.201106.grb2.nc',
                    'flxl.gdas.201107.grb2.nc',
                    'pgbl.gdas.201107.grb2.nc',
                    'pgbh.gdas.201107.grb2.nc',
                    'flxf.gdas.201107.grb2.nc',
                    'flxl.gdas.201108.grb2.nc',
                    'pgbl.gdas.201108.grb2.nc',
                    'pgbh.gdas.201108.grb2.nc',
                    'flxf.gdas.201108.grb2.nc',
                    'flxl.gdas.201109.grb2.nc',
                    'pgbl.gdas.201109.grb2.nc',
                    'pgbh.gdas.201109.grb2.nc',
                    'flxf.gdas.201109.grb2.nc',
                    'flxl.gdas.201110.grb2.nc',
                    'pgbl.gdas.201110.grb2.nc',
                    'pgbh.gdas.201110.grb2.nc',
                    'flxf.gdas.201110.grb2.nc',
                    'flxl.gdas.201111.grb2.nc',
                    'pgbl.gdas.201111.grb2.nc',
                    'pgbh.gdas.201111.grb2.nc',
                    'flxf.gdas.201111.grb2.nc',
                    'flxl.gdas.201112.grb2.nc',
                    'pgbl.gdas.201112.grb2.nc',
                    'pgbh.gdas.201112.grb2.nc',
                    'flxf.gdas.201112.grb2.nc',
                    'flxl.gdas.201201.grb2.nc',
                    'pgbl.gdas.201201.grb2.nc',
                    'pgbh.gdas.201201.grb2.nc',
                    'flxf.gdas.201201.grb2.nc',
                    'flxl.gdas.201202.grb2.nc',
                    'pgbl.gdas.201202.grb2.nc',
                    'pgbh.gdas.201202.grb2.nc',
                    'flxf.gdas.201202.grb2.nc',
                    'flxl.gdas.201203.grb2.nc',
                    'pgbl.gdas.201203.grb2.nc',
                    'pgbh.gdas.201203.grb2.nc',
                    'flxf.gdas.201203.grb2.nc',
                    'flxl.gdas.201204.grb2.nc',
                    'pgbl.gdas.201204.grb2.nc',
                    'pgbh.gdas.201204.grb2.nc',
                    'flxf.gdas.201204.grb2.nc',
                    'flxl.gdas.201205.grb2.nc',
                    'pgbl.gdas.201205.grb2.nc',
                    'pgbh.gdas.201205.grb2.nc',
                    'flxf.gdas.201205.grb2.nc',
                    'flxl.gdas.201206.grb2.nc',
                    'pgbl.gdas.201206.grb2.nc',
                    'pgbh.gdas.201206.grb2.nc',
                    'flxf.gdas.201206.grb2.nc',
                    'flxl.gdas.201207.grb2.nc',
                    'pgbl.gdas.201207.grb2.nc',
                    'pgbh.gdas.201207.grb2.nc',
                    'flxf.gdas.201207.grb2.nc',
                    'flxl.gdas.201208.grb2.nc',
                    'pgbl.gdas.201208.grb2.nc',
                    'pgbh.gdas.201208.grb2.nc',
                    'flxf.gdas.201208.grb2.nc',
                    'flxl.gdas.201209.grb2.nc',
                    'pgbl.gdas.201209.grb2.nc',
                    'pgbh.gdas.201209.grb2.nc',
                    'flxf.gdas.201209.grb2.nc',
                    'flxl.gdas.201210.grb2.nc',
                    'pgbl.gdas.201210.grb2.nc',
                    'pgbh.gdas.201210.grb2.nc',
                    'flxf.gdas.201210.grb2.nc',
                    'flxl.gdas.201211.grb2.nc',
                    'pgbl.gdas.201211.grb2.nc',
                    'pgbh.gdas.201211.grb2.nc',
                    'flxf.gdas.201211.grb2.nc',
                    'flxl.gdas.201212.grb2.nc',
                    'pgbl.gdas.201212.grb2.nc',
                    'pgbh.gdas.201212.grb2.nc',
                    'flxf.gdas.201212.grb2.nc',
                    'flxl.gdas.201301.grb2.nc',
                    'pgbl.gdas.201301.grb2.nc',
                    'pgbh.gdas.201301.grb2.nc',
                    'flxf.gdas.201301.grb2.nc',
                    'flxl.gdas.201302.grb2.nc',
                    'pgbl.gdas.201302.grb2.nc',
                    'pgbh.gdas.201302.grb2.nc',
                    'flxf.gdas.201302.grb2.nc',
                    'flxl.gdas.201303.grb2.nc',
                    'pgbl.gdas.201303.grb2.nc',
                    'pgbh.gdas.201303.grb2.nc',
                    'flxf.gdas.201303.grb2.nc',
                    'flxl.gdas.201304.grb2.nc',
                    'pgbl.gdas.201304.grb2.nc',
                    'pgbh.gdas.201304.grb2.nc',
                    'flxf.gdas.201304.grb2.nc',
                    'flxl.gdas.201305.grb2.nc',
                    'pgbl.gdas.201305.grb2.nc',
                    'pgbh.gdas.201305.grb2.nc',
                    'flxf.gdas.201305.grb2.nc',
                    'flxl.gdas.201306.grb2.nc',
                    'pgbl.gdas.201306.grb2.nc',
                    'pgbh.gdas.201306.grb2.nc',
                    'flxf.gdas.201306.grb2.nc',
                    'flxl.gdas.201307.grb2.nc',
                    'pgbl.gdas.201307.grb2.nc',
                    'pgbh.gdas.201307.grb2.nc',
                    'flxf.gdas.201307.grb2.nc',
                    'flxl.gdas.201308.grb2.nc',
                    'pgbl.gdas.201308.grb2.nc',
                    'pgbh.gdas.201308.grb2.nc',
                    'flxf.gdas.201308.grb2.nc',
                    'flxl.gdas.201309.grb2.nc',
                    'pgbl.gdas.201309.grb2.nc',
                    'pgbh.gdas.201309.grb2.nc',
                    'flxf.gdas.201309.grb2.nc',
                    'flxl.gdas.201310.grb2.nc',
                    'pgbl.gdas.201310.grb2.nc',
                    'pgbh.gdas.201310.grb2.nc',
                    'flxf.gdas.201310.grb2.nc',
                    'flxl.gdas.201311.grb2.nc',
                    'pgbl.gdas.201311.grb2.nc',
                    'pgbh.gdas.201311.grb2.nc',
                    'flxf.gdas.201311.grb2.nc',
                    'flxl.gdas.201312.grb2.nc',
                    'pgbl.gdas.201312.grb2.nc',
                    'pgbh.gdas.201312.grb2.nc',
                    'flxf.gdas.201312.grb2.nc',
                    'flxl.gdas.201401.grb2.nc',
                    'pgbl.gdas.201401.grb2.nc',
                    'pgbh.gdas.201401.grb2.nc',
                    'flxf.gdas.201401.grb2.nc',
                    'flxl.gdas.201402.grb2.nc',
                    'pgbl.gdas.201402.grb2.nc',
                    'pgbh.gdas.201402.grb2.nc',
                    'flxf.gdas.201402.grb2.nc',
                    'flxl.gdas.201403.grb2.nc',
                    'pgbl.gdas.201403.grb2.nc',
                    'pgbh.gdas.201403.grb2.nc',
                    'flxf.gdas.201403.grb2.nc',
                    'flxl.gdas.201404.grb2.nc',
                    'pgbl.gdas.201404.grb2.nc',
                    'pgbh.gdas.201404.grb2.nc',
                    'flxf.gdas.201404.grb2.nc',
                    'flxl.gdas.201405.grb2.nc',
                    'pgbl.gdas.201405.grb2.nc',
                    'pgbh.gdas.201405.grb2.nc',
                    'flxf.gdas.201405.grb2.nc',
                    'flxl.gdas.201406.grb2.nc',
                    'pgbl.gdas.201406.grb2.nc',
                    'pgbh.gdas.201406.grb2.nc',
                    'flxf.gdas.201406.grb2.nc',
                    'flxl.gdas.201407.grb2.nc',
                    'pgbl.gdas.201407.grb2.nc',
                    'pgbh.gdas.201407.grb2.nc',
                    'flxf.gdas.201407.grb2.nc',
                    'flxl.gdas.201408.grb2.nc',
                    'pgbl.gdas.201408.grb2.nc',
                    'pgbh.gdas.201408.grb2.nc',
                    'flxf.gdas.201408.grb2.nc',
                    'flxl.gdas.201409.grb2.nc',
                    'pgbl.gdas.201409.grb2.nc',
                    'pgbh.gdas.201409.grb2.nc',
                    'flxf.gdas.201409.grb2.nc',
                    'flxl.gdas.201410.grb2.nc',
                    'pgbl.gdas.201410.grb2.nc',
                    'pgbh.gdas.201410.grb2.nc',
                    'flxf.gdas.201410.grb2.nc',
                    'flxl.gdas.201411.grb2.nc',
                    'pgbl.gdas.201411.grb2.nc',
                    'pgbh.gdas.201411.grb2.nc',
                    'flxf.gdas.201411.grb2.nc',
                    'flxl.gdas.201412.grb2.nc',
                    'pgbl.gdas.201412.grb2.nc',
                    'pgbh.gdas.201412.grb2.nc',
                    'flxf.gdas.201412.grb2.nc',
                    'flxl.gdas.201501.grb2.nc',
                    'pgbl.gdas.201501.grb2.nc',
                    'pgbh.gdas.201501.grb2.nc',
                    'flxf.gdas.201501.grb2.nc',
                    'flxl.gdas.201502.grb2.nc',
                    'pgbl.gdas.201502.grb2.nc',
                    'pgbh.gdas.201502.grb2.nc',
                    'flxf.gdas.201502.grb2.nc',
                    'flxl.gdas.201503.grb2.nc',
                    'pgbl.gdas.201503.grb2.nc',
                    'pgbh.gdas.201503.grb2.nc',
                    'flxf.gdas.201503.grb2.nc',
                    'flxl.gdas.201504.grb2.nc',
                    'pgbl.gdas.201504.grb2.nc',
                    'pgbh.gdas.201504.grb2.nc',
                    'flxf.gdas.201504.grb2.nc',
                    'flxl.gdas.201505.grb2.nc',
                    'pgbl.gdas.201505.grb2.nc',
                    'pgbh.gdas.201505.grb2.nc',
                    'flxf.gdas.201505.grb2.nc',
                    'flxl.gdas.201506.grb2.nc',
                    'pgbl.gdas.201506.grb2.nc',
                    'pgbh.gdas.201506.grb2.nc',
                    'flxf.gdas.201506.grb2.nc',
                    'flxl.gdas.201507.grb2.nc',
                    'pgbl.gdas.201507.grb2.nc',
                    'pgbh.gdas.201507.grb2.nc',
                    'flxf.gdas.201507.grb2.nc',
                    'flxl.gdas.201508.grb2.nc',
                    'pgbl.gdas.201508.grb2.nc',
                    'pgbh.gdas.201508.grb2.nc',
                    'flxf.gdas.201508.grb2.nc',
                    'flxl.gdas.201509.grb2.nc',
                    'pgbl.gdas.201509.grb2.nc',
                    'pgbh.gdas.201509.grb2.nc',
                    'flxf.gdas.201509.grb2.nc',
                    'flxl.gdas.201510.grb2.nc',
                    'pgbl.gdas.201510.grb2.nc',
                    'pgbh.gdas.201510.grb2.nc',
                    'flxf.gdas.201510.grb2.nc',
                    'flxl.gdas.201511.grb2.nc',
                    'pgbl.gdas.201511.grb2.nc',
                    'pgbh.gdas.201511.grb2.nc',
                    'flxf.gdas.201511.grb2.nc',
                    'flxl.gdas.201512.grb2.nc',
                    'pgbl.gdas.201512.grb2.nc',
                    'pgbh.gdas.201512.grb2.nc',
                    'flxf.gdas.201512.grb2.nc',
                    'flxl.gdas.201601.grb2.nc',
                    'pgbl.gdas.201601.grb2.nc',
                    'pgbh.gdas.201601.grb2.nc',
                    'flxf.gdas.201601.grb2.nc',
                    'flxl.gdas.201602.grb2.nc',
                    'pgbl.gdas.201602.grb2.nc',
                    'pgbh.gdas.201602.grb2.nc',
                    'flxf.gdas.201602.grb2.nc',
                    'flxl.gdas.201603.grb2.nc',
                    'pgbl.gdas.201603.grb2.nc',
                    'pgbh.gdas.201603.grb2.nc',
                    'flxf.gdas.201603.grb2.nc',
                    'flxl.gdas.201604.grb2.nc',
                    'pgbl.gdas.201604.grb2.nc',
                    'pgbh.gdas.201604.grb2.nc',
                    'flxf.gdas.201604.grb2.nc',
                    'flxl.gdas.201605.grb2.nc',
                    'pgbl.gdas.201605.grb2.nc',
                    'pgbh.gdas.201605.grb2.nc',
                    'flxf.gdas.201605.grb2.nc',
                    'flxl.gdas.201606.grb2.nc',
                    'pgbl.gdas.201606.grb2.nc',
                    'pgbh.gdas.201606.grb2.nc',
                    'flxf.gdas.201606.grb2.nc',
                    'flxl.gdas.201607.grb2.nc',
                    'pgbl.gdas.201607.grb2.nc',
                    'pgbh.gdas.201607.grb2.nc',
                    'flxf.gdas.201607.grb2.nc',
                    'flxl.gdas.201608.grb2.nc',
                    'pgbl.gdas.201608.grb2.nc',
                    'pgbh.gdas.201608.grb2.nc',
                    'flxf.gdas.201608.grb2.nc',
                    'flxl.gdas.201609.grb2.nc',
                    'pgbl.gdas.201609.grb2.nc',
                    'pgbh.gdas.201609.grb2.nc',
                    'flxf.gdas.201609.grb2.nc',
                    'flxl.gdas.201610.grb2.nc',
                    'pgbl.gdas.201610.grb2.nc',
                    'pgbh.gdas.201610.grb2.nc',
                    'flxf.gdas.201610.grb2.nc',
                    'flxl.gdas.201611.grb2.nc',
                    'pgbl.gdas.201611.grb2.nc',
                    'pgbh.gdas.201611.grb2.nc',
                    'flxf.gdas.201611.grb2.nc',
                    'flxl.gdas.201612.grb2.nc',
                    'pgbl.gdas.201612.grb2.nc',
                    'pgbh.gdas.201612.grb2.nc',
                    'flxf.gdas.201612.grb2.nc',
                    'flxl.gdas.201701.grb2.nc',
                    'pgbl.gdas.201701.grb2.nc',
                    'pgbh.gdas.201701.grb2.nc',
                    'flxf.gdas.201701.grb2.nc',
                    'flxl.gdas.201702.grb2.nc',
                    'pgbl.gdas.201702.grb2.nc',
                    'pgbh.gdas.201702.grb2.nc',
                    'flxf.gdas.201702.grb2.nc',
                    'flxl.gdas.201703.grb2.nc',
                    'pgbl.gdas.201703.grb2.nc',
                    'pgbh.gdas.201703.grb2.nc',
                    'flxf.gdas.201703.grb2.nc',
                    'flxl.gdas.201704.grb2.nc',
                    'pgbl.gdas.201704.grb2.nc',
                    'pgbh.gdas.201704.grb2.nc',
                    'flxf.gdas.201704.grb2.nc',
                    'flxl.gdas.201705.grb2.nc',
                    'pgbl.gdas.201705.grb2.nc',
                    'pgbh.gdas.201705.grb2.nc',
                    'flxf.gdas.201705.grb2.nc',
                    'flxl.gdas.201706.grb2.nc',
                    'pgbl.gdas.201706.grb2.nc',
                    'pgbh.gdas.201706.grb2.nc',
                    'flxf.gdas.201706.grb2.nc',
                    'flxl.gdas.201707.grb2.nc',
                    'pgbl.gdas.201707.grb2.nc',
                    'pgbh.gdas.201707.grb2.nc',
                    'flxf.gdas.201707.grb2.nc',
                    'flxl.gdas.201708.grb2.nc',
                    'pgbl.gdas.201708.grb2.nc',
                    'pgbh.gdas.201708.grb2.nc',
                    'flxf.gdas.201708.grb2.nc',
                    'flxl.gdas.201709.grb2.nc',
                    'pgbl.gdas.201709.grb2.nc',
                    'pgbh.gdas.201709.grb2.nc',
                    'flxf.gdas.201709.grb2.nc',
                    'flxl.gdas.201710.grb2.nc',
                    'pgbl.gdas.201710.grb2.nc',
                    'pgbh.gdas.201710.grb2.nc',
                    'flxf.gdas.201710.grb2.nc',
                    'flxl.gdas.201711.grb2.nc',
                    'pgbl.gdas.201711.grb2.nc',
                    'pgbh.gdas.201711.grb2.nc',
                    'flxf.gdas.201711.grb2.nc',
                    'flxl.gdas.201712.grb2.nc',
                    'pgbl.gdas.201712.grb2.nc',
                    'pgbh.gdas.201712.grb2.nc',
                    'flxf.gdas.201712.grb2.nc',
                    'flxl.gdas.201801.grb2.nc',
                    'pgbl.gdas.201801.grb2.nc',
                    'pgbh.gdas.201801.grb2.nc',
                    'flxf.gdas.201801.grb2.nc',
                    'flxl.gdas.201802.grb2.nc',
                    'pgbl.gdas.201802.grb2.nc',
                    'pgbh.gdas.201802.grb2.nc',
                    'flxf.gdas.201802.grb2.nc',
                    'flxl.gdas.201803.grb2.nc',
                    'pgbl.gdas.201803.grb2.nc',
                    'pgbh.gdas.201803.grb2.nc',
                    'flxf.gdas.201803.grb2.nc',
                    'flxl.gdas.201804.grb2.nc',
                    'pgbl.gdas.201804.grb2.nc',
                    'pgbh.gdas.201804.grb2.nc',
                    'flxf.gdas.201804.grb2.nc',
                    'flxl.gdas.201805.grb2.nc',
                    'pgbl.gdas.201805.grb2.nc',
                    'pgbh.gdas.201805.grb2.nc',
                    'flxf.gdas.201805.grb2.nc',
                    'flxl.gdas.201806.grb2.nc',
                    'pgbl.gdas.201806.grb2.nc',
                    'pgbh.gdas.201806.grb2.nc',
                    'flxf.gdas.201806.grb2.nc',
                    'flxl.gdas.201807.grb2.nc',
                    'pgbl.gdas.201807.grb2.nc',
                    'pgbh.gdas.201807.grb2.nc',
                    'flxf.gdas.201807.grb2.nc',
                    'flxl.gdas.201808.grb2.nc',
                    'pgbl.gdas.201808.grb2.nc',
                    'pgbh.gdas.201808.grb2.nc',
                    'flxf.gdas.201808.grb2.nc',
                    'flxl.gdas.201809.grb2.nc',
                    'pgbl.gdas.201809.grb2.nc',
                    'pgbh.gdas.201809.grb2.nc',
                    'flxf.gdas.201809.grb2.nc',
                    'flxl.gdas.201810.grb2.nc',
                    'pgbl.gdas.201810.grb2.nc',
                    'pgbh.gdas.201810.grb2.nc',
                    'flxf.gdas.201810.grb2.nc',
                    'flxl.gdas.201811.grb2.nc',
                    'pgbl.gdas.201811.grb2.nc',
                    'pgbh.gdas.201811.grb2.nc',
                    'flxf.gdas.201811.grb2.nc',
                    'flxl.gdas.201812.grb2.nc',
                    'pgbl.gdas.201812.grb2.nc',
                    'pgbh.gdas.201812.grb2.nc',
                    'flxf.gdas.201812.grb2.nc',
                    'flxl.gdas.201901.grb2.nc',
                    'pgbl.gdas.201901.grb2.nc',
                    'pgbh.gdas.201901.grb2.nc',
                    'flxf.gdas.201901.grb2.nc',
                    'flxl.gdas.201902.grb2.nc',
                    'pgbl.gdas.201902.grb2.nc',
                    'pgbh.gdas.201902.grb2.nc',
                    'flxf.gdas.201902.grb2.nc',
                    'flxl.gdas.201903.grb2.nc',
                    'pgbl.gdas.201903.grb2.nc',
                    'pgbh.gdas.201903.grb2.nc',
                    'flxf.gdas.201903.grb2.nc',
                    'flxl.gdas.201904.grb2.nc',
                    'pgbl.gdas.201904.grb2.nc',
                    'pgbh.gdas.201904.grb2.nc',
                    'flxf.gdas.201904.grb2.nc',
                    'flxl.gdas.201905.grb2.nc',
                    'pgbl.gdas.201905.grb2.nc',
                    'pgbh.gdas.201905.grb2.nc',
                    'flxf.gdas.201905.grb2.nc',
                    'flxl.gdas.201906.grb2.nc',
                    'pgbl.gdas.201906.grb2.nc',
                    'pgbh.gdas.201906.grb2.nc',
                    'flxf.gdas.201906.grb2.nc',
                    'flxl.gdas.201907.grb2.nc',
                    'pgbl.gdas.201907.grb2.nc',
                    'pgbh.gdas.201907.grb2.nc',
                    'flxf.gdas.201907.grb2.nc',
                    'flxl.gdas.201908.grb2.nc',
                    'pgbl.gdas.201908.grb2.nc',
                    'pgbh.gdas.201908.grb2.nc',
                    'flxf.gdas.201908.grb2.nc',
                    'flxl.gdas.201909.grb2.nc',
                    'pgbl.gdas.201909.grb2.nc',
                    'pgbh.gdas.201909.grb2.nc',
                    'flxf.gdas.201909.grb2.nc',
                    'flxl.gdas.201910.grb2.nc',
                    'pgbl.gdas.201910.grb2.nc',
                    'pgbh.gdas.201910.grb2.nc',
                    'flxf.gdas.201910.grb2.nc',
                    'flxl.gdas.201911.grb2.nc',
                    'pgbl.gdas.201911.grb2.nc',
                    'pgbh.gdas.201911.grb2.nc',
                    'flxf.gdas.201911.grb2.nc',
                    'flxl.gdas.201912.grb2.nc',
                    'pgbl.gdas.201912.grb2.nc',
                    'pgbh.gdas.201912.grb2.nc',
                    'flxf.gdas.201912.grb2.nc',
                    'ipvl.gdas.201101.grb2.nc',
                    'ipvh.gdas.201101.grb2.nc',
                    'ipvl.gdas.201102.grb2.nc',
                    'ipvh.gdas.201102.grb2.nc',
                    'ipvl.gdas.201103.grb2.nc',
                    'ipvh.gdas.201103.grb2.nc',
                    'ipvl.gdas.201104.grb2.nc',
                    'ipvh.gdas.201104.grb2.nc',
                    'ipvl.gdas.201105.grb2.nc',
                    'ipvh.gdas.201105.grb2.nc',
                    'ipvl.gdas.201106.grb2.nc',
                    'ipvh.gdas.201106.grb2.nc',
                    'ipvl.gdas.201107.grb2.nc',
                    'ipvh.gdas.201107.grb2.nc',
                    'ipvl.gdas.201108.grb2.nc',
                    'ipvh.gdas.201108.grb2.nc',
                    'ipvl.gdas.201109.grb2.nc',
                    'ipvh.gdas.201109.grb2.nc',
                    'ipvl.gdas.201110.grb2.nc',
                    'ipvh.gdas.201110.grb2.nc',
                    'ipvl.gdas.201111.grb2.nc',
                    'ipvh.gdas.201111.grb2.nc',
                    'ipvl.gdas.201112.grb2.nc',
                    'ipvh.gdas.201112.grb2.nc',
                    'ipvl.gdas.201201.grb2.nc',
                    'ipvh.gdas.201201.grb2.nc',
                    'ipvl.gdas.201202.grb2.nc',
                    'ipvh.gdas.201202.grb2.nc',
                    'ipvl.gdas.201203.grb2.nc',
                    'ipvh.gdas.201203.grb2.nc',
                    'ipvl.gdas.201204.grb2.nc',
                    'ipvh.gdas.201204.grb2.nc',
                    'ipvl.gdas.201205.grb2.nc',
                    'ipvh.gdas.201205.grb2.nc',
                    'ipvl.gdas.201206.grb2.nc',
                    'ipvh.gdas.201206.grb2.nc',
                    'ipvl.gdas.201207.grb2.nc',
                    'ipvh.gdas.201207.grb2.nc',
                    'ipvl.gdas.201208.grb2.nc',
                    'ipvh.gdas.201208.grb2.nc',
                    'ipvl.gdas.201209.grb2.nc',
                    'ipvh.gdas.201209.grb2.nc',
                    'ipvl.gdas.201210.grb2.nc',
                    'ipvh.gdas.201210.grb2.nc',
                    'ipvl.gdas.201211.grb2.nc',
                    'ipvh.gdas.201211.grb2.nc',
                    'ipvl.gdas.201212.grb2.nc',
                    'ipvh.gdas.201212.grb2.nc',
                    'ipvl.gdas.201301.grb2.nc',
                    'ipvh.gdas.201301.grb2.nc',
                    'ipvl.gdas.201302.grb2.nc',
                    'ipvh.gdas.201302.grb2.nc',
                    'ipvl.gdas.201303.grb2.nc',
                    'ipvh.gdas.201303.grb2.nc',
                    'ipvl.gdas.201304.grb2.nc',
                    'ipvh.gdas.201304.grb2.nc',
                    'ipvl.gdas.201305.grb2.nc',
                    'ipvh.gdas.201305.grb2.nc',
                    'ipvl.gdas.201306.grb2.nc',
                    'ipvh.gdas.201306.grb2.nc',
                    'ipvl.gdas.201307.grb2.nc',
                    'ipvh.gdas.201307.grb2.nc',
                    'ipvl.gdas.201308.grb2.nc',
                    'ipvh.gdas.201308.grb2.nc',
                    'ipvl.gdas.201309.grb2.nc',
                    'ipvh.gdas.201309.grb2.nc',
                    'ipvl.gdas.201310.grb2.nc',
                    'ipvh.gdas.201310.grb2.nc',
                    'ipvl.gdas.201311.grb2.nc',
                    'ipvh.gdas.201311.grb2.nc',
                    'ipvl.gdas.201312.grb2.nc',
                    'ipvh.gdas.201312.grb2.nc',
                    'ipvl.gdas.201401.grb2.nc',
                    'ipvh.gdas.201401.grb2.nc',
                    'ipvl.gdas.201402.grb2.nc',
                    'ipvh.gdas.201402.grb2.nc',
                    'ipvl.gdas.201403.grb2.nc',
                    'ipvh.gdas.201403.grb2.nc',
                    'ipvl.gdas.201404.grb2.nc',
                    'ipvh.gdas.201404.grb2.nc',
                    'ipvl.gdas.201405.grb2.nc',
                    'ipvh.gdas.201405.grb2.nc',
                    'ipvl.gdas.201406.grb2.nc',
                    'ipvh.gdas.201406.grb2.nc',
                    'ipvl.gdas.201407.grb2.nc',
                    'ipvh.gdas.201407.grb2.nc',
                    'ipvl.gdas.201408.grb2.nc',
                    'ipvh.gdas.201408.grb2.nc',
                    'ipvl.gdas.201409.grb2.nc',
                    'ipvh.gdas.201409.grb2.nc',
                    'ipvl.gdas.201410.grb2.nc',
                    'ipvh.gdas.201410.grb2.nc',
                    'ipvl.gdas.201411.grb2.nc',
                    'ipvh.gdas.201411.grb2.nc',
                    'ipvl.gdas.201412.grb2.nc',
                    'ipvh.gdas.201412.grb2.nc',
                    'ipvl.gdas.201501.grb2.nc',
                    'ipvh.gdas.201501.grb2.nc',
                    'ipvl.gdas.201502.grb2.nc',
                    'ipvh.gdas.201502.grb2.nc',
                    'ipvl.gdas.201503.grb2.nc',
                    'ipvh.gdas.201503.grb2.nc',
                    'ipvl.gdas.201504.grb2.nc',
                    'ipvh.gdas.201504.grb2.nc',
                    'ipvl.gdas.201505.grb2.nc',
                    'ipvh.gdas.201505.grb2.nc',
                    'ipvl.gdas.201506.grb2.nc',
                    'ipvh.gdas.201506.grb2.nc',
                    'ipvl.gdas.201507.grb2.nc',
                    'ipvh.gdas.201507.grb2.nc',
                    'ipvl.gdas.201508.grb2.nc',
                    'ipvh.gdas.201508.grb2.nc',
                    'ipvl.gdas.201509.grb2.nc',
                    'ipvh.gdas.201509.grb2.nc',
                    'ipvl.gdas.201510.grb2.nc',
                    'ipvh.gdas.201510.grb2.nc',
                    'ipvl.gdas.201511.grb2.nc',
                    'ipvh.gdas.201511.grb2.nc',
                    'ipvl.gdas.201512.grb2.nc',
                    'ipvh.gdas.201512.grb2.nc',
                    'ipvl.gdas.201601.grb2.nc',
                    'ipvh.gdas.201601.grb2.nc',
                    'ipvl.gdas.201602.grb2.nc',
                    'ipvh.gdas.201602.grb2.nc',
                    'ipvl.gdas.201603.grb2.nc',
                    'ipvh.gdas.201603.grb2.nc',
                    'ipvl.gdas.201604.grb2.nc',
                    'ipvh.gdas.201604.grb2.nc',
                    'ipvl.gdas.201605.grb2.nc',
                    'ipvh.gdas.201605.grb2.nc',
                    'ipvl.gdas.201606.grb2.nc',
                    'ipvh.gdas.201606.grb2.nc',
                    'ipvl.gdas.201607.grb2.nc',
                    'ipvh.gdas.201607.grb2.nc',
                    'ipvl.gdas.201608.grb2.nc',
                    'ipvh.gdas.201608.grb2.nc',
                    'ipvl.gdas.201609.grb2.nc',
                    'ipvh.gdas.201609.grb2.nc',
                    'ipvl.gdas.201610.grb2.nc',
                    'ipvh.gdas.201610.grb2.nc',
                    'ipvl.gdas.201611.grb2.nc',
                    'ipvh.gdas.201611.grb2.nc',
                    'ipvl.gdas.201612.grb2.nc',
                    'ipvh.gdas.201612.grb2.nc',
                    'ipvl.gdas.201701.grb2.nc',
                    'ipvh.gdas.201701.grb2.nc',
                    'ipvl.gdas.201702.grb2.nc',
                    'ipvh.gdas.201702.grb2.nc',
                    'ipvl.gdas.201703.grb2.nc',
                    'ipvh.gdas.201703.grb2.nc',
                    'ipvl.gdas.201704.grb2.nc',
                    'ipvh.gdas.201704.grb2.nc',
                    'ipvl.gdas.201705.grb2.nc',
                    'ipvh.gdas.201705.grb2.nc',
                    'ipvl.gdas.201706.grb2.nc',
                    'ipvh.gdas.201706.grb2.nc',
                    'ipvl.gdas.201707.grb2.nc',
                    'ipvh.gdas.201707.grb2.nc',
                    'ipvl.gdas.201708.grb2.nc',
                    'ipvh.gdas.201708.grb2.nc',
                    'ipvl.gdas.201709.grb2.nc',
                    'ipvh.gdas.201709.grb2.nc',
                    'ipvl.gdas.201710.grb2.nc',
                    'ipvh.gdas.201710.grb2.nc',
                    'ipvl.gdas.201711.grb2.nc',
                    'ipvh.gdas.201711.grb2.nc',
                    'ipvl.gdas.201712.grb2.nc',
                    'ipvh.gdas.201712.grb2.nc',
                    'ipvl.gdas.201801.grb2.nc',
                    'ipvh.gdas.201801.grb2.nc',
                    'ipvl.gdas.201802.grb2.nc',
                    'ipvh.gdas.201802.grb2.nc',
                    'ipvl.gdas.201803.grb2.nc',
                    'ipvh.gdas.201803.grb2.nc',
                    'ipvl.gdas.201804.grb2.nc',
                    'ipvh.gdas.201804.grb2.nc',
                    'ipvl.gdas.201805.grb2.nc',
                    'ipvh.gdas.201805.grb2.nc',
                    'ipvl.gdas.201806.grb2.nc',
                    'ipvh.gdas.201806.grb2.nc',
                    'ipvl.gdas.201807.grb2.nc',
                    'ipvh.gdas.201807.grb2.nc',
                    'ipvl.gdas.201808.grb2.nc',
                    'ipvh.gdas.201808.grb2.nc',
                    'ipvl.gdas.201809.grb2.nc',
                    'ipvh.gdas.201809.grb2.nc',
                    'ipvl.gdas.201810.grb2.nc',
                    'ipvh.gdas.201810.grb2.nc',
                    'ipvl.gdas.201811.grb2.nc',
                    'ipvh.gdas.201811.grb2.nc',
                    'ipvl.gdas.201812.grb2.nc',
                    'ipvh.gdas.201812.grb2.nc',
                    'ipvl.gdas.201901.grb2.nc',
                    'ipvh.gdas.201901.grb2.nc',
                    'ipvl.gdas.201902.grb2.nc',
                    'ipvh.gdas.201902.grb2.nc',
                    'ipvl.gdas.201903.grb2.nc',
                    'ipvh.gdas.201903.grb2.nc',
                    'ipvl.gdas.201904.grb2.nc',
                    'ipvh.gdas.201904.grb2.nc',
                    'ipvl.gdas.201905.grb2.nc',
                    'ipvh.gdas.201905.grb2.nc',
                    'ipvl.gdas.201906.grb2.nc',
                    'ipvh.gdas.201906.grb2.nc',
                    'ipvl.gdas.201907.grb2.nc',
                    'ipvh.gdas.201907.grb2.nc',
                    'ipvl.gdas.201908.grb2.nc',
                    'ipvh.gdas.201908.grb2.nc',
                    'ipvl.gdas.201909.grb2.nc',
                    'ipvh.gdas.201909.grb2.nc',
                    'ipvl.gdas.201910.grb2.nc',
                    'ipvh.gdas.201910.grb2.nc',
                    'ipvl.gdas.201911.grb2.nc',
                    'ipvh.gdas.201911.grb2.nc',
                    'ipvl.gdas.201912.grb2.nc',
                    'ipvh.gdas.201912.grb2.nc',
                    'ocnf.gdas.201101.grb2.nc',
                    'ocnh.gdas.201101.grb2.nc',
                    'ocnf.gdas.201102.grb2.nc',
                    'ocnh.gdas.201102.grb2.nc',
                    'ocnf.gdas.201103.grb2.nc',
                    'ocnh.gdas.201103.grb2.nc',
                    'ocnf.gdas.201104.grb2.nc',
                    'ocnh.gdas.201104.grb2.nc',
                    'ocnf.gdas.201105.grb2.nc',
                    'ocnh.gdas.201105.grb2.nc',
                    'ocnf.gdas.201106.grb2.nc',
                    'ocnh.gdas.201106.grb2.nc',
                    'ocnf.gdas.201107.grb2.nc',
                    'ocnh.gdas.201107.grb2.nc',
                    'ocnf.gdas.201108.grb2.nc',
                    'ocnh.gdas.201108.grb2.nc',
                    'ocnf.gdas.201109.grb2.nc',
                    'ocnh.gdas.201109.grb2.nc',
                    'ocnf.gdas.201110.grb2.nc',
                    'ocnh.gdas.201110.grb2.nc',
                    'ocnf.gdas.201111.grb2.nc',
                    'ocnh.gdas.201111.grb2.nc',
                    'ocnf.gdas.201112.grb2.nc',
                    'ocnh.gdas.201112.grb2.nc',
                    'ocnf.gdas.201201.grb2.nc',
                    'ocnh.gdas.201201.grb2.nc',
                    'ocnf.gdas.201202.grb2.nc',
                    'ocnh.gdas.201202.grb2.nc',
                    'ocnf.gdas.201203.grb2.nc',
                    'ocnh.gdas.201203.grb2.nc',
                    'ocnf.gdas.201204.grb2.nc',
                    'ocnh.gdas.201204.grb2.nc',
                    'ocnf.gdas.201205.grb2.nc',
                    'ocnh.gdas.201205.grb2.nc',
                    'ocnf.gdas.201206.grb2.nc',
                    'ocnh.gdas.201206.grb2.nc',
                    'ocnf.gdas.201207.grb2.nc',
                    'ocnh.gdas.201207.grb2.nc',
                    'ocnf.gdas.201208.grb2.nc',
                    'ocnh.gdas.201208.grb2.nc',
                    'ocnf.gdas.201209.grb2.nc',
                    'ocnh.gdas.201209.grb2.nc',
                    'ocnf.gdas.201210.grb2.nc',
                    'ocnh.gdas.201210.grb2.nc',
                    'ocnf.gdas.201211.grb2.nc',
                    'ocnh.gdas.201211.grb2.nc',
                    'ocnf.gdas.201212.grb2.nc',
                    'ocnh.gdas.201212.grb2.nc',
                    'ocnf.gdas.201301.grb2.nc',
                    'ocnh.gdas.201301.grb2.nc',
                    'ocnf.gdas.201302.grb2.nc',
                    'ocnh.gdas.201302.grb2.nc',
                    'ocnf.gdas.201303.grb2.nc',
                    'ocnh.gdas.201303.grb2.nc',
                    'ocnf.gdas.201304.grb2.nc',
                    'ocnh.gdas.201304.grb2.nc',
                    'ocnf.gdas.201305.grb2.nc',
                    'ocnh.gdas.201305.grb2.nc',
                    'ocnf.gdas.201306.grb2.nc',
                    'ocnh.gdas.201306.grb2.nc',
                    'ocnf.gdas.201307.grb2.nc',
                    'ocnh.gdas.201307.grb2.nc',
                    'ocnf.gdas.201308.grb2.nc',
                    'ocnh.gdas.201308.grb2.nc',
                    'ocnf.gdas.201309.grb2.nc',
                    'ocnh.gdas.201309.grb2.nc',
                    'ocnf.gdas.201310.grb2.nc',
                    'ocnh.gdas.201310.grb2.nc',
                    'ocnf.gdas.201311.grb2.nc',
                    'ocnh.gdas.201311.grb2.nc',
                    'ocnf.gdas.201312.grb2.nc',
                    'ocnh.gdas.201312.grb2.nc',
                    'ocnf.gdas.201401.grb2.nc',
                    'ocnh.gdas.201401.grb2.nc',
                    'ocnf.gdas.201402.grb2.nc',
                    'ocnh.gdas.201402.grb2.nc',
                    'ocnf.gdas.201403.grb2.nc',
                    'ocnh.gdas.201403.grb2.nc',
                    'ocnf.gdas.201404.grb2.nc',
                    'ocnh.gdas.201404.grb2.nc',
                    'ocnf.gdas.201405.grb2.nc',
                    'ocnh.gdas.201405.grb2.nc',
                    'ocnf.gdas.201406.grb2.nc',
                    'ocnh.gdas.201406.grb2.nc',
                    'ocnf.gdas.201407.grb2.nc',
                    'ocnh.gdas.201407.grb2.nc',
                    'ocnf.gdas.201408.grb2.nc',
                    'ocnh.gdas.201408.grb2.nc',
                    'ocnf.gdas.201409.grb2.nc',
                    'ocnh.gdas.201409.grb2.nc',
                    'ocnf.gdas.201410.grb2.nc',
                    'ocnh.gdas.201410.grb2.nc',
                    'ocnf.gdas.201411.grb2.nc',
                    'ocnh.gdas.201411.grb2.nc',
                    'ocnf.gdas.201412.grb2.nc',
                    'ocnh.gdas.201412.grb2.nc',
                    'ocnf.gdas.201501.grb2.nc',
                    'ocnh.gdas.201501.grb2.nc',
                    'ocnf.gdas.201502.grb2.nc',
                    'ocnh.gdas.201502.grb2.nc',
                    'ocnf.gdas.201503.grb2.nc',
                    'ocnh.gdas.201503.grb2.nc',
                    'ocnf.gdas.201504.grb2.nc',
                    'ocnh.gdas.201504.grb2.nc',
                    'ocnf.gdas.201505.grb2.nc',
                    'ocnh.gdas.201505.grb2.nc',
                    'ocnf.gdas.201506.grb2.nc',
                    'ocnh.gdas.201506.grb2.nc',
                    'ocnf.gdas.201507.grb2.nc',
                    'ocnh.gdas.201507.grb2.nc',
                    'ocnf.gdas.201508.grb2.nc',
                    'ocnh.gdas.201508.grb2.nc',
                    'ocnf.gdas.201509.grb2.nc',
                    'ocnh.gdas.201509.grb2.nc',
                    'ocnf.gdas.201510.grb2.nc',
                    'ocnh.gdas.201510.grb2.nc',
                    'ocnf.gdas.201511.grb2.nc',
                    'ocnh.gdas.201511.grb2.nc',
                    'ocnf.gdas.201512.grb2.nc',
                    'ocnh.gdas.201512.grb2.nc',
                    'ocnf.gdas.201601.grb2.nc',
                    'ocnh.gdas.201601.grb2.nc',
                    'ocnf.gdas.201602.grb2.nc',
                    'ocnh.gdas.201602.grb2.nc',
                    'ocnf.gdas.201603.grb2.nc',
                    'ocnh.gdas.201603.grb2.nc',
                    'ocnf.gdas.201604.grb2.nc',
                    'ocnh.gdas.201604.grb2.nc',
                    'ocnf.gdas.201605.grb2.nc',
                    'ocnh.gdas.201605.grb2.nc',
                    'ocnf.gdas.201606.grb2.nc',
                    'ocnh.gdas.201606.grb2.nc',
                    'ocnf.gdas.201607.grb2.nc',
                    'ocnh.gdas.201607.grb2.nc',
                    'ocnf.gdas.201608.grb2.nc',
                    'ocnh.gdas.201608.grb2.nc',
                    'ocnf.gdas.201609.grb2.nc',
                    'ocnh.gdas.201609.grb2.nc',
                    'ocnf.gdas.201610.grb2.nc',
                    'ocnh.gdas.201610.grb2.nc',
                    'ocnf.gdas.201611.grb2.nc',
                    'ocnh.gdas.201611.grb2.nc',
                    'ocnf.gdas.201612.grb2.nc',
                    'ocnh.gdas.201612.grb2.nc',
                    'ocnf.gdas.201701.grb2.nc',
                    'ocnh.gdas.201701.grb2.nc',
                    'ocnf.gdas.201702.grb2.nc',
                    'ocnh.gdas.201702.grb2.nc',
                    'ocnf.gdas.201703.grb2.nc',
                    'ocnh.gdas.201703.grb2.nc',
                    'ocnf.gdas.201704.grb2.nc',
                    'ocnh.gdas.201704.grb2.nc',
                    'ocnf.gdas.201705.grb2.nc',
                    'ocnh.gdas.201705.grb2.nc',
                    'ocnf.gdas.201706.grb2.nc',
                    'ocnh.gdas.201706.grb2.nc',
                    'ocnf.gdas.201707.grb2.nc',
                    'ocnh.gdas.201707.grb2.nc',
                    'ocnf.gdas.201708.grb2.nc',
                    'ocnh.gdas.201708.grb2.nc',
                    'ocnf.gdas.201709.grb2.nc',
                    'ocnh.gdas.201709.grb2.nc',
                    'ocnf.gdas.201710.grb2.nc',
                    'ocnh.gdas.201710.grb2.nc',
                    'ocnf.gdas.201711.grb2.nc',
                    'ocnh.gdas.201711.grb2.nc',
                    'ocnf.gdas.201712.grb2.nc',
                    'ocnh.gdas.201712.grb2.nc',
                    'ocnf.gdas.201801.grb2.nc',
                    'ocnh.gdas.201801.grb2.nc',
                    'ocnf.gdas.201802.grb2.nc',
                    'ocnh.gdas.201802.grb2.nc',
                    'ocnf.gdas.201803.grb2.nc',
                    'ocnh.gdas.201803.grb2.nc',
                    'ocnf.gdas.201804.grb2.nc',
                    'ocnh.gdas.201804.grb2.nc',
                    'ocnf.gdas.201805.grb2.nc',
                    'ocnh.gdas.201805.grb2.nc',
                    'ocnf.gdas.201806.grb2.nc',
                    'ocnh.gdas.201806.grb2.nc',
                    'ocnf.gdas.201807.grb2.nc',
                    'ocnh.gdas.201807.grb2.nc',
                    'ocnf.gdas.201808.grb2.nc',
                    'ocnh.gdas.201808.grb2.nc',
                    'ocnf.gdas.201809.grb2.nc',
                    'ocnh.gdas.201809.grb2.nc',
                    'ocnf.gdas.201810.grb2.nc',
                    'ocnh.gdas.201810.grb2.nc',
                    'ocnf.gdas.201811.grb2.nc',
                    'ocnh.gdas.201811.grb2.nc',
                    'ocnf.gdas.201812.grb2.nc',
                    'ocnh.gdas.201812.grb2.nc',
                    'ocnf.gdas.201901.grb2.nc',
                    'ocnh.gdas.201901.grb2.nc',
                    'ocnf.gdas.201902.grb2.nc',
                    'ocnh.gdas.201902.grb2.nc',
                    'ocnf.gdas.201903.grb2.nc',
                    'ocnh.gdas.201903.grb2.nc',
                    'ocnf.gdas.201904.grb2.nc',
                    'ocnh.gdas.201904.grb2.nc',
                    'ocnf.gdas.201905.grb2.nc',
                    'ocnh.gdas.201905.grb2.nc',
                    'ocnf.gdas.201906.grb2.nc',
                    'ocnh.gdas.201906.grb2.nc',
                    'ocnf.gdas.201907.grb2.nc',
                    'ocnh.gdas.201907.grb2.nc',
                    'ocnf.gdas.201908.grb2.nc',
                    'ocnh.gdas.201908.grb2.nc',
                    'ocnf.gdas.201909.grb2.nc',
                    'ocnh.gdas.201909.grb2.nc',
                    'ocnf.gdas.201910.grb2.nc',
                    'ocnh.gdas.201910.grb2.nc',
                    'ocnf.gdas.201911.grb2.nc',
                    'ocnh.gdas.201911.grb2.nc',
                    'ocnf.gdas.201912.grb2.nc',
                    'ocnh.gdas.201912.grb2.nc']

        return filelist


    def download_ncep(self, lat, lon, loss_year, path, sensor):
        ncep_parameters = self.get_ncep()
        names = self.names()
        ncep_path = os.path.join(path, 'ncep') #From Irvin et al.
        os.makedirs(ncep_path, exist_ok=True) #From Irvin et al.

        #We look for parameters up to 5 years before loss event: loss_year - 5 to loss_year -1

        for param in ncep_parameters:
            fetch = os.path.join(os.getcwd(), 'input', 'ncep')
            filelist = self.filelist_name()

            list_to_delete = [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]
            for t in range(1, 6):
                if int(loss_year) - t >= 2011 and int(loss_year) - t <= 2019:
                    list_to_delete.remove(int(loss_year) - t)
            for l in filelist:
                for d in list_to_delete:
                    if str(d) in l:
                        filelist.remove(l)
            mean_list = []
            max_list = []
            min_list = []
            for file in filelist:
                #https://spire.com/tutorial/spire-weather-tutorial-intro-to-processing-grib2-data-with-python/
                ds = xr.open_dataset(os.path.join(fetch, file))
                if param in ds:
                    ds_param = ds[param]
                    #warning: pynio not compatible, use netcdf and not grib format
                    df = ds_param.to_dataframe()
                    latitudes = df.index.get_level_values("lat")
                    longitudes = df.index.get_level_values("lon")
                    # -180 to 180 degrees
                    map_function = lambda lon: (lon - 360) if (lon > 180) else lon
                    remapped_longitudes = longitudes.map(map_function)
                    df["longitude"] = remapped_longitudes
                    df["latitude"] = latitudes

                    # Not working because of resolution (2.5 deg)
                    # if sensor == 'landsat':
                    #     pix = 15
                    #
                    # elif sensor == 'planet':
                    #     pix = 4.77
                    #
                    # else:
                    #     print('Sensor not available/known')
                    # 1 deg ~ 111 km around the Equator
                    #min_lat = lat - (pix * 332 * 0.5) / (111 * 1000)
                    #max_lat = lat + (pix * 332 * 0.5) / (111 * 1000)
                    #min_lon = lon - (pix * 332 * 0.5) / (111 * 1000)
                    #max_lon = lon + (pix * 332 * 0.5) / (111 * 1000)

                    df_lat_max = df[df["latitude"] >= lat]
                    max_lat = ((df_lat_max["latitude"]).drop_duplicates()).min()

                    df_lat_min = df[df["latitude"] <= lat]
                    min_lat = ((df_lat_min["latitude"]).drop_duplicates()).max()

                    df_lon_max = df[df["longitude"] >= lon]
                    max_lon = ((df_lon_max["longitude"]).drop_duplicates()).min()

                    df_lon_min = df[df["longitude"] <= lon]
                    min_lon = ((df_lon_min["longitude"]).drop_duplicates()).max()

                    df1 = df[(df["latitude"] == min_lat) & (df["longitude"] == min_lon)]
                    df2 = df[(df["latitude"] == max_lat) & (df["longitude"] == min_lon)]
                    df3 = df[(df["latitude"] == min_lat) & (df["longitude"] == max_lon)]
                    df4 = df[(df["latitude"] == max_lat) & (df["longitude"] == max_lon)]
                    df_conc = pd.concat([df1, df2, df3, df4], axis=0)

                    #suffix = '_' + (list(df)[0]).split('_')[-1] #for GRID

                    if 'SOILW' in param:
                        level = df.index.get_level_values('layer3')
                        df['level'] = level
                        df_select = df_conc.get([param[:-2], 'level'])
                        if param == 'SOILW_Y106_Avg_1':
                            df_param = df_select[df_select['level'] == 0]
                        elif param == 'SOILW_Y106_Avg_2':
                            df_param = df_select[df_select['level'] == 1]
                        elif param == 'SOILW_Y106_Avg_3':
                            df_param = df_select[df_select['level'] == 2]
                        elif param == 'SOILW_Y106_Avg_4':
                            df_param = df_select[df_select['level'] == 3]
                        mean_list.append(df_param.get(param[:-2]).mean())
                        max_list.append(df_param.get(param[:-2]).max())
                        min_list.append(df_param.get(param[:-2]).min())

                    else:
                        df_param = df_conc.get([param])
                        #Take values from bounds
                        mean_list.append(df_param.get(param).mean())
                        max_list.append(df_param.get(param).max())
                        min_list.append(df_param.get(param).min())


                #mean, min, max over 5 years
                val_length = len(mean_list)
                for elem in range(val_length):
                    if mean_list[elem] is None:
                        mean_list.remove(mean_list[elem])
                    if max_list[elem] is None:
                        max_list.remove(max_list[elem])
                    if min_list[elem] is None:
                        min_list.remove(min_list[elem])

                if len(mean_list) == 0:
                    mean_list.append(0)
                if len(max_list) == 0:
                    max_list.append(0)
                if len(min_list) == 0:
                    min_list.append(0)

                mean_param = sum(mean_list)/len(mean_list) #mean of monthly mean over 5 years before forest loss
                max_param = max(max_list) #monthly max over 5 years before forest loss
                min_param = min(min_list) #monthly min over 5 years before forest loss
                #print(mean_param)
                #print(max_param)
                #print(min_param)

                if param == 'SPFH_P8_L103' or param == 'QMAX_P8_L103' or param == 'QMIN_P8_L103' or param == 'APCP_P8_L1' or 'SOILW_P8_2L106' in param:
                    np.save(os.path.join(ncep_path, names[param]+'_avg.npy'), 10000 * np.array(mean_param, dtype='float64'))  # From Irvin et al. (adapted)
                    np.save(os.path.join(ncep_path, names[param]+'_max.npy'), 10000 * np.array(max_param, dtype='float64'))  # From Irvin et al. (adapted)
                    np.save(os.path.join(ncep_path, names[param]+'_min.npy'), 10000 * np.array(min_param, dtype='float64'))  # From Irvin et al. (adapted)
                elif param == 'PRES_P8_L1':
                    np.save(os.path.join(ncep_path, names[param]+'_avg.npy'), 0.1 * np.array(mean_param, dtype='float64'))  # From Irvin et al. (adapted)
                    np.save(os.path.join(ncep_path, names[param]+'_max.npy'), 0.1 * np.array(max_param, dtype='float64'))  # From Irvin et al. (adapted)
                    np.save(os.path.join(ncep_path, names[param]+'_min.npy'), 0.1 * np.array(min_param, dtype='float64'))  # From Irvin et al. (adapted)

                elif param == 'UGRD_P8_L103' or param == 'VGRD_P8_L103' or param == 'WATR_P8_L1' or param == 'ALBDO_P8_L1':
                    np.save(os.path.join(ncep_path, names[param]+'_avg.npy'), 100 * np.array(mean_param, dtype='float64'))  # From Irvin et al. (adapted)
                    np.save(os.path.join(ncep_path, names[param]+'_max.npy'), 100 * np.array(max_param, dtype='float64'))  # From Irvin et al. (adapted)
                    np.save(os.path.join(ncep_path, names[param]+'_min.npy'), 100 * np.array(min_param, dtype='float64'))  # From Irvin et al. (adapted)
                else:
                    np.save(os.path.join(ncep_path, names[param]+'_avg.npy'), np.array(mean_param, dtype='float64')) #From Irvin et al. (adapted)
                    np.save(os.path.join(ncep_path, names[param]+'_max.npy'), np.array(max_param, dtype='float64')) #From Irvin et al. (adapted)
                    np.save(os.path.join(ncep_path, names[param]+'_min.npy'), np.array(min_param, dtype='float64')) #From Irvin et al. (adapted)


def download_ncep_from_images(sensor):
    """
    :param sensor: landsat or planetscope
    :return: ncep parameters for all images created
    """
    years = ['2015', '2016', '2017', '2018', '2019','2020']
    for year in years:
        path_origin = os.path.join(os.path.dirname(os.getcwd()), 'download_images_quick', 'output', str(year))
        list_dir = os.listdir(path_origin)
        for d in list_dir:
            shape = d
            path_sh = os.path.join(path_origin, shape)
            list_dir_sh = os.listdir(path_sh)
            for dn in list_dir_sh:
                index = dn
                if os.path.exists(os.path.join(path_sh, str(index), sensor)) == True:
                    first_image = os.listdir(os.path.join(path_sh, str(index), sensor))[0]
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
                                                
                    if (os.path.exists(os.path.join(path_out_index_sensor, 'ncep')) == False) or (os.path.exists(os.path.join(path_out_index_sensor, 'ncep')) == True and len(os.listdir(os.path.join(path_out_index_sensor, 'ncep'))) <84):
                        print(index)
                        td = NCEPDownloader()
                        td.download_ncep(lat, lon, year, path_out_index_sensor, sensor)
                        print('ok:', path_out_index_sensor)
                    
                elif os.path.exists(os.path.join(path_sh, str(index), sensor)) == False and os.path.exists(os.path.join(path_sh, str(index), sensor + '_fixed')) == True:
                    first_image = os.listdir(os.path.join(path_sh, str(index), sensor + '_fixed'))[0]
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
                                                
                    if (os.path.exists(os.path.join(path_out_index_sensor, 'ncep')) == False) or (os.path.exists(os.path.join(path_out_index_sensor, 'ncep')) == True and len(os.listdir(os.path.join(path_out_index_sensor, 'ncep'))) <84):
                        print(index)
                        td = NCEPDownloader()
                        td.download_ncep(lat, lon, year, path_out_index_sensor, sensor)
                        print('ok:', path_out_index_sensor)

    def download_ncep_from_shape(sensor, shape):
        """
        :param sensor: landsat or planetscope
        :param shape: .shp file
        :return: ncep parameters for shape: all years and all indexes
        """

        years = [2015, 2016, 2017, 2018, 2019, 2020]

        for year in years:
            path_origin = os.path.join(os.path.dirname(os.getcwd()), 'download_images_quick', 'output', str(year), shape)
            list_dir = os.listdir(path_origin)
            if len(list_dir) > 0:
                for i in list_dir:
                    index=int(i)
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

                        td = NCEPDownloader() #From Irvin et al. (adapted)
                        td.download_ncep(lat, lon, year, path_out_index_sensor, sensor) #From Irvin et al. (adapted)
                        print('ok:', path_out_index_sensor)


def download_ncep_from_year(sensor, shape, year):
    """
    :param sensor: landsat or planetscope
    :param shape: .shp file
    :param year: loss year
    :return: ncep parameters for shape and year selected (all indexes)
    """

    path_origin = os.path.join(os.path.dirname(os.getcwd()), 'download_images_quick', 'output', str(year), shape)
    list_dir = os.listdir(path_origin)
    if len(list_dir) > 0:
        for i in list_dir[-6:]:
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

                td = NCEPDownloader()  # From Irvin et al. (adapted)
                td.download_ncep(lat, lon, year, path_out_index_sensor, sensor)  # From Irvin
                print('ok:', path_out_index_sensor)

def download_ncep_from_index(sensor, shape, year):
    """
    :param sensor: landsat or planetscope
    :param shape: .shp file
    :param year: loss year
    :return: ncep parameters for shape selected, year selected and index
    """

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

            td = NCEPDownloader() #From Irvin et al. (adapted)
            td.download_ncep(lat, lon, year, path_out_index_sensor, sensor) #From Irvin et al. (adapted)
            print('ok:', path_out_index_sensor)

if __name__ == "__main__":
    fire.Fire(download_ncep_from_images)
    #fire.Fire(download_ncep_from_shape)
    #fire.Fire(download_ncep_from_year)
    #fire.Fire(download_ncep_from_index)
