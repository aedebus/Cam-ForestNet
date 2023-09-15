import os
import fire

def fix_names(sensor, suffix):
    list_dir = os.listdir(os.path.join(os.getcwd(), 'my_examples_'+str(sensor)+'_'+str(suffix)))

    for d in list_dir:
        path = os.path.join(os.getcwd(), 'my_examples_'+str(sensor)+'_' + str(suffix), d, 'auxiliary', 'ncep')

        if os.path.exists(os.path.join(path,'air_pressure_at_surface_level_min.npy')):
            os.rename(os.path.join(path,'air_pressure_at_surface_level_min.npy'), os.path.join(path,'surface_pressure_min.npy'))
            os.rename(os.path.join(path,'air_pressure_at_surface_level_max.npy'), os.path.join(path,'surface_pressure_max.npy'))
            os.rename(os.path.join(path,'air_pressure_at_surface_level_avg.npy'), os.path.join(path,'surface_pressure_avg.npy'))

        if os.path.exists(os.path.join(path,'surface_level_albedo_min.npy')):
            os.rename(os.path.join(path,'surface_level_albedo_min.npy'), os.path.join(path,'albedo_min.npy'))
            os.rename(os.path.join(path,'surface_level_albedo_max.npy'), os.path.join(path,'albedo_max.npy'))
            os.rename(os.path.join(path,'surface_level_albedo_avg.npy'), os.path.join(path,'albedo_avg.npy'))

        if os.path.exists(os.path.join(path,'clear_sky_longwave_flux_downward_min.npy')):
            os.rename(os.path.join(path,'clear_sky_longwave_flux_downward_min.npy'), os.path.join(path,'clear-sky_downward_longwave_flux_min.npy'))
            os.rename(os.path.join(path,'clear_sky_longwave_flux_downward_max.npy'), os.path.join(path,'clear-sky_downward_longwave_flux_max.npy'))
            os.rename(os.path.join(path,'clear_sky_longwave_flux_downward_avg.npy'), os.path.join(path,'clear-sky_downward_longwave_flux_avg.npy'))

        if os.path.exists(os.path.join(path,'clear_sky_longwave_flux_upward_min.npy')):
            os.rename(os.path.join(path,'clear_sky_longwave_flux_upward_min.npy'), os.path.join(path,'clear-sky_upward_longwave_flux_min.npy'))
            os.rename(os.path.join(path,'clear_sky_longwave_flux_upward_max.npy'), os.path.join(path,'clear-sky_upward_longwave_flux_max.npy'))
            os.rename(os.path.join(path,'clear_sky_longwave_flux_upward_avg.npy'), os.path.join(path,'clear-sky_upward_longwave_flux_avg.npy'))

        if os.path.exists(os.path.join(path,'clear_sky_solar_flux_downward_min.npy')):
            os.rename(os.path.join(path,'clear_sky_solar_flux_downward_min.npy'), os.path.join(path,'clear-sky_downward_solar_flux_min.npy'))
            os.rename(os.path.join(path,'clear_sky_solar_flux_downward_max.npy'), os.path.join(path,'clear-sky_downward_solar_flux_max.npy'))
            os.rename(os.path.join(path,'clear_sky_solar_flux_downward_avg.npy'), os.path.join(path,'clear-sky_downward_solar_flux_avg.npy'))

        if os.path.exists(os.path.join(path,'clear_sky_solar_flux_upward_min.npy')):
            os.rename(os.path.join(path,'clear_sky_solar_flux_upward_min.npy'), os.path.join(path,'clear-sky_upward_solar_flux_min.npy'))
            os.rename(os.path.join(path,'clear_sky_solar_flux_upward_max.npy'), os.path.join(path,'clear-sky_upward_solar_flux_max.npy'))
            os.rename(os.path.join(path,'clear_sky_solar_flux_upward_avg.npy'), os.path.join(path,'clear-sky_upward_solar_flux_avg.npy'))

        if os.path.exists(os.path.join(path,'direct_evaporation_from_bare_soil_min.npy')):
            os.rename(os.path.join(path,'direct_evaporation_from_bare_soil_min.npy'), os.path.join(path,'direct_evaporation_bare_soil_min.npy'))
            os.rename(os.path.join(path,'direct_evaporation_from_bare_soil_max.npy'), os.path.join(path,'direct_evaporation_bare_soil_max.npy'))
            os.rename(os.path.join(path,'direct_evaporation_from_bare_soil_avg.npy'), os.path.join(path,'direct_evaporation_bare_soil_avg.npy'))

        if os.path.exists(os.path.join(path,'longwave_radiation_flux_downward_min.npy')):
            os.rename(os.path.join(path,'longwave_radiation_flux_downward_min.npy'), os.path.join(path,'downward_longwave_radiation_flux_min.npy'))
            os.rename(os.path.join(path,'longwave_radiation_flux_downward_max.npy'), os.path.join(path,'downward_longwave_radiation_flux_max.npy'))
            os.rename(os.path.join(path,'longwave_radiation_flux_downward_avg.npy'), os.path.join(path,'downward_longwave_radiation_flux_avg.npy'))

        if os.path.exists(os.path.join(path,'longwave_radiation_flux_upward_min.npy')):
            os.rename(os.path.join(path,'longwave_radiation_flux_upward_min.npy'), os.path.join(path,'upward_longwave_radiation_flux_min.npy'))
            os.rename(os.path.join(path,'longwave_radiation_flux_upward_max.npy'), os.path.join(path,'upward_longwave_radiation_flux_max.npy'))
            os.rename(os.path.join(path,'longwave_radiation_flux_upward_avg.npy'), os.path.join(path,'upward_longwave_radiation_flux_avg.npy'))

        if os.path.exists(os.path.join(path,'shortwave_radiation_flux_downward_min.npy')):
            os.rename(os.path.join(path,'shortwave_radiation_flux_downward_min.npy'), os.path.join(path,'downward_shortwave_radiation_flux_min.npy'))
            os.rename(os.path.join(path,'shortwave_radiation_flux_downward_max.npy'), os.path.join(path,'downward_shortwave_radiation_flux_max.npy'))
            os.rename(os.path.join(path,'shortwave_radiation_flux_downward_avg.npy'), os.path.join(path,'downward_shortwave_radiation_flux_avg.npy'))

        if os.path.exists(os.path.join(path,'shortwave_radiation_flux_upward_min.npy')):
            os.rename(os.path.join(path,'shortwave_radiation_flux_upward_min.npy'), os.path.join(path,'upward_shortwave_radiation_flux_min.npy'))
            os.rename(os.path.join(path,'shortwave_radiation_flux_upward_max.npy'), os.path.join(path,'upward_shortwave_radiation_flux_max.npy'))
            os.rename(os.path.join(path,'shortwave_radiation_flux_upward_avg.npy'), os.path.join(path,'upward_shortwave_radiation_flux_avg.npy'))

        if os.path.exists(os.path.join(path,'temperature_min.npy')):
            os.rename(os.path.join(path,'temperature_min.npy'), os.path.join(path,'tmin.npy'))
            os.rename(os.path.join(path,'temperature_max.npy'), os.path.join(path,'tmax.npy'))
            os.rename(os.path.join(path,'temperature_avg.npy'),os.path.join(path, 'tavg.npy'))

        if os.path.exists(os.path.join(path,'volumetric_soil_moisture_content1_min.npy')):
            os.rename(os.path.join(path,'volumetric_soil_moisture_content1_min.npy'), os.path.join(path,'soilmoist1_min.npy'))
            os.rename(os.path.join(path,'volumetric_soil_moisture_content1_max.npy'), os.path.join(path,'soilmoist1_max.npy'))
            os.rename(os.path.join(path,'volumetric_soil_moisture_content1_avg.npy'), os.path.join(path,'soilmoist1_avg.npy'))

        if os.path.exists(os.path.join(path,'volumetric_soil_moisture_content2_min.npy')):
            os.rename(os.path.join(path,'volumetric_soil_moisture_content2_min.npy'), os.path.join(path,'soilmoist2_min.npy'))
            os.rename(os.path.join(path,'volumetric_soil_moisture_content2_max.npy'), os.path.join(path,'soilmoist2_max.npy'))
            os.rename(os.path.join(path,'volumetric_soil_moisture_content2_avg.npy'), os.path.join(path,'soilmoist2_avg.npy'))

        if os.path.exists(os.path.join(path,'volumetric_soil_moisture_content3_min.npy')):
            os.rename(os.path.join(path,'volumetric_soil_moisture_content3_min.npy'), os.path.join(path,'soilmoist3_min.npy'))
            os.rename(os.path.join(path,'volumetric_soil_moisture_content3_max.npy'), os.path.join(path,'soilmoist3_max.npy'))
            os.rename(os.path.join(path,'volumetric_soil_moisture_content3_avg.npy'), os.path.join(path,'soilmoist3_avg.npy'))

        if os.path.exists(os.path.join(path,'volumetric_soil_moisture_content4_min.npy')):
            os.rename(os.path.join(path,'volumetric_soil_moisture_content4_min.npy'), os.path.join(path,'soilmoist4_min.npy'))
            os.rename(os.path.join(path,'volumetric_soil_moisture_content4_max.npy'), os.path.join(path,'soilmoist4_max.npy'))
            os.rename(os.path.join(path,'volumetric_soil_moisture_content4_avg.npy'), os.path.join(path,'soilmoist4_avg.npy'))

        if os.path.exists(os.path.join(path,'water_runoff_at_surface_level_min.npy')):
            os.rename(os.path.join(path,'water_runoff_at_surface_level_min.npy'), os.path.join(path,'water_runoff_min.npy'))
            os.rename(os.path.join(path,'water_runoff_at_surface_level_max.npy'), os.path.join(path,'water_runoff_max.npy'))
            os.rename(os.path.join(path,'water_runoff_at_surface_level_avg.npy'), os.path.join(path,'water_runoff_avg.npy'))

        if os.path.exists(os.path.join(path,'wind_component_u_min.npy')):
            os.rename(os.path.join(path,'wind_component_u_min.npy'), os.path.join(path,'u_wind_10m_min.npy'))
            os.rename(os.path.join(path,'wind_component_u_max.npy'), os.path.join(path,'u_wind_10m_max.npy'))
            os.rename(os.path.join(path,'wind_component_u_avg.npy'), os.path.join(path,'u_wind_10m_avg.npy'))

        if os.path.exists(os.path.join(path,'wind_component_v_min.npy')):
            os.rename(os.path.join(path,'wind_component_v_min.npy'), os.path.join(path,'v_wind_10m_min.npy'))
            os.rename(os.path.join(path,'wind_component_v_max.npy'), os.path.join(path,'v_wind_10m_max.npy'))
            os.rename(os.path.join(path,'wind_component_v_avg.npy'), os.path.join(path,'v_wind_10m_avg.npy'))

        if os.path.exists(os.path.join(path,'ground_level_precipitation_min.npy')):
            os.rename(os.path.join(path,'ground_level_precipitation_min.npy'), os.path.join(path,'prec_min.npy'))
            os.rename(os.path.join(path,'ground_level_precipitation_max.npy'), os.path.join(path,'prec_max.npy'))
            os.rename(os.path.join(path,'ground_level_precipitation_avg.npy'), os.path.join(path,'prec_avg.npy'))

if __name__ == "__main__":
    fire.Fire(fix_names)