normalize_mean2_keys = ['2m_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind', 'mean_sea_level_pressure', 'surface_pressure', 
 'geopotential_50', 'geopotential_100', 'geopotential_150', 'geopotential_200', 'geopotential_250', 'geopotential_300', 'geopotential_400', 'geopotential_500', 'geopotential_600', 'geopotential_700', 'geopotential_850', 'geopotential_925', 'geopotential_1000', 
 'u_component_of_wind_50', 'u_component_of_wind_100', 'u_component_of_wind_150', 'u_component_of_wind_200', 'u_component_of_wind_250', 'u_component_of_wind_300', 'u_component_of_wind_400', 'u_component_of_wind_500', 'u_component_of_wind_600', 'u_component_of_wind_700', 'u_component_of_wind_850', 'u_component_of_wind_925', 'u_component_of_wind_1000', 
 'v_component_of_wind_50', 'v_component_of_wind_100', 'v_component_of_wind_150', 'v_component_of_wind_200', 'v_component_of_wind_250', 'v_component_of_wind_300', 'v_component_of_wind_400', 'v_component_of_wind_500', 'v_component_of_wind_600', 'v_component_of_wind_700', 'v_component_of_wind_850', 'v_component_of_wind_925', 'v_component_of_wind_1000', 
 'temperature_50', 'temperature_100', 'temperature_150', 'temperature_200', 'temperature_250', 'temperature_300', 'temperature_400', 'temperature_500', 'temperature_600', 'temperature_700', 'temperature_850', 'temperature_925', 'temperature_1000', 
 'specific_humidity_50', 'specific_humidity_100', 'specific_humidity_150', 'specific_humidity_200', 'specific_humidity_250', 'specific_humidity_300', 'specific_humidity_400', 'specific_humidity_500', 'specific_humidity_600', 'specific_humidity_700', 'specific_humidity_850', 'specific_humidity_925', 'specific_humidity_1000']


var_mapping_for_accMetrics = {
    "HGT_P0_L100_GLC0": "geopotential", # "hgtn",
    "UGRD_P0_L100_GLC0": "u_component_of_wind", # "u",
    "VGRD_P0_L100_GLC0": "v_component_of_wind", # "v",
    "TMP_P0_L100_GLC0": "temperature", # "t",
    "SPFH_P0_L100_GLC0": "specific_humidity", # "q",
    "MSLMA_P0_L101_GLC0": "mean_sea_level_pressure", # "msl",
    "TMP_P0_L103_GLC0": "2m_temperature", # "2t",
    "UGRD_P0_L103_GLC0": "10m_u_component_of_wind", # "10u",
    "VGRD_P0_L103_GLC0": "10m_v_component_of_wind", # "10v",
}

