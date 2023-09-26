# pangu's setting:
# atmos_vars: [
#         "geopotential", 
#         "u_component_of_wind",
#         "v_component_of_wind",
#         "temperature",
#         "specific_humidity",
#       ]
# single_vars: [
#         "2m_temperature",
#         "10m_u_component_of_wind",
#         "10m_v_component_of_wind",
#         "mean_sea_level_pressure",
#       ]

PRESSURE_VARS = [
    "SPFH_P0_L100_GLC0", # specific_humidity, q
    "TMP_P0_L100_GLC0", # temperature, t
    "UGRD_P0_L100_GLC0", # u_component_of_wind, u
    "VGRD_P0_L100_GLC0", # v_component_of_wind, v
    "HGT_P0_L100_GLC0", # geopotential, hgtn
]

SURFACE_VARS = [
    "UGRD_P0_L103_GLC0", # 10m_u_component_of_wind, 10u
    "VGRD_P0_L103_GLC0", # 10m_v_component_of_wind, 10v
    "TMP_P0_L103_GLC0", # 2m_temperature, 2t
    "MSLMA_P0_L101_GLC0", # mean_sea_level_pressure, msl
]

var_mapping = {
    "SPFH_P0_L100_GLC0": "q",
    "TMP_P0_L100_GLC0": "t",
    "UGRD_P0_L100_GLC0": "u",
    "VGRD_P0_L100_GLC0": "v",
    "HGT_P0_L100_GLC0": "hgtn",
    "UGRD_P0_L103_GLC0": "10u",
    "VGRD_P0_L103_GLC0": "10v",
    "TMP_P0_L103_GLC0": "2t",
    "MSLMA_P0_L101_GLC0": "msl",
}

atmos_level = [
    5000, 10000, 15000, 20000, 25000, 30000, 40000,
    50000, 60000, 70000, 85000, 92500, 100000
]

atmos_level_all = [
    5000, 7500, 10000, 12500, 15000, 17500, 20000, 22500, 25000, 27500,
    30000, 32500, 35000, 37500, 40000, 42500, 45000, 47500, 50000, 52500,
    55000, 57500, 60000, 62500, 65000, 67500, 70000, 72500, 75000, 77500,
    80000, 82500, 85000, 87500, 90000, 92500, 95000, 97500, 100000, 101320,
]

index_in_atmos_level_all = [0, 2, 4, 6, 8, 10, 14, 18, 22, 26, 32, 35, 38]
# index_in_atmos_level_all = [atmos_level_all.index(level) for level in atmos_level]

index_level_mapping = {
    0: 5000, 2: 10000, 4: 15000, 6: 20000, 8: 25000, 10: 30000, 14: 40000,
    18: 50000, 22: 60000, 26: 70000, 32: 85000, 35: 92500, 38: 100000,
}

