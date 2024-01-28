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
    "HGT_P0_L100_GLC0", # geopotential, hgtn
    "UGRD_P0_L100_GLC0", # u_component_of_wind, u
    "VGRD_P0_L100_GLC0", # v_component_of_wind, v
    "TMP_P0_L100_GLC0", # temperature, t
    "SPFH_P0_L100_GLC0", # specific_humidity, q
]

SURFACE_VARS = [
    "MSLMA_P0_L101_GLC0", # mean_sea_level_pressure, msl
    "TMP_P0_L103_GLC0", # 2m_temperature, 2t
    "UGRD_P0_L103_GLC0", # 10m_u_component_of_wind, 10u
    "VGRD_P0_L103_GLC0", # 10m_v_component_of_wind, 10v
]

# ⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇
# need to change:
# var to be choosen:
var_mapping = {
    "HGT_P0_L100_GLC0": "hgtn",
    "UGRD_P0_L100_GLC0": "u",
    "VGRD_P0_L100_GLC0": "v",
    "TMP_P0_L100_GLC0": "t",
    "SPFH_P0_L100_GLC0": "q",
    "MSLMA_P0_L101_GLC0": "msl",
    "TMP_P0_L103_GLC0": "2t",
    "UGRD_P0_L103_GLC0": "10u",
    "VGRD_P0_L103_GLC0": "10v",
}
var_mapping_herbie = {
    "HGT": "hgtn",
    "UGRD": "u",
    "VGRD": "v",
    "TMP": "t",
    "SPFH": "q",
    "MSLMA:mean sea level": "msl",
    "TMP:2 m above": "2t",
    "UGRD:10 m above": "10u",
    "VGRD:10 m above": "10v",
}
var_mapping_hrrrlong_herbie = {
    "HGT_P0_L100_GLC0": "HGT",
    "UGRD_P0_L100_GLC0": "UGRD",
    "VGRD_P0_L100_GLC0": "VGRD",
    "TMP_P0_L100_GLC0": "TMP",
    "SPFH_P0_L100_GLC0": "SPFH",
    "MSLMA_P0_L101_GLC0": "MSLMA:mean sea level",
    "TMP_P0_L103_GLC0": "TMP:2 m above",
    "UGRD_P0_L103_GLC0": "UGRD:10 m above",
    "VGRD_P0_L103_GLC0": "VGRD:10 m above",
}
atmos_level_herbie = [
    50, 100, 150, 200, 250, 300, 400,
    500, 600, 700, 850, 925, 1000
]

atmos_level_herbie_new = [
    400, 450, 500, 550, 600, 
    650, 700, 750, 800, 850, 
    900, 950, 1000
]

atmos_level_herbie_all = [
    50, 100, 150, 200, 250, 
    300, 400, 450, 500, 550, 
    600, 650, 700, 750, 800,
    850, 900, 925, 950, 1000
]
# new_var_list = [
#     "msl", "2t", "10u", "10v",
#     "hgtn_400", "hgtn_450", "hgtn_500", "hgtn_550", "hgtn_600",
#     "hgtn_650", "hgtn_700", "hgtn_750", "hgtn_800", "hgtn_850",
#     "hgtn_900", "hgtn_950", "hgtn_1000",
#     "u_400", "u_450", "u_500", "u_550", "u_600",
#     "u_650", "u_700", "u_750", "u_800", "u_850",
#     "u_900", "u_950", "u_1000",
#     "v_400", "v_450", "v_500", "v_550", "v_600",
#     "v_650", "v_700", "v_750", "v_800", "v_850",
#     "v_900", "v_950", "v_1000",
#     "t_400", "t_450", "t_500", "t_550", "t_600",
#     "t_650", "t_700", "t_750", "t_800", "t_850",
#     "t_900", "t_950", "t_1000",
#     "q_400", "q_450", "q_500", "q_550", "q_600",
#     "q_650", "q_700", "q_750", "q_800", "q_850",
#     "q_900", "q_950", "q_1000",
# ]

# level to be choosen:
atmos_level = [
    5000, 10000, 15000, 20000, 25000, 30000, 40000,
    50000, 60000, 70000, 85000, 92500, 100000
]
# ⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆⬆

atmos_level_all = [
    5000, 7500, 10000, 12500, 15000, 17500, 20000, 22500, 25000, 27500,
    30000, 32500, 35000, 37500, 40000, 42500, 45000, 47500, 50000, 52500,
    55000, 57500, 60000, 62500, 65000, 67500, 70000, 72500, 75000, 77500,
    80000, 82500, 85000, 87500, 90000, 92500, 95000, 97500, 100000, 101320,
]
def atmos_level2index_in_atmos_level_all(atmos_level):
    atmos_level_all = [
        5000, 7500, 10000, 12500, 15000, 17500, 20000, 22500, 25000, 27500,
        30000, 32500, 35000, 37500, 40000, 42500, 45000, 47500, 50000, 52500,
        55000, 57500, 60000, 62500, 65000, 67500, 70000, 72500, 75000, 77500,
        80000, 82500, 85000, 87500, 90000, 92500, 95000, 97500, 100000, 101320,
    ]
    # index_in_atmos_level_all = [0, 2, 4, 6, 8, 10, 14, 18, 22, 26, 32, 35, 38]
    index_in_atmos_level_all = [atmos_level_all.index(level) for level in atmos_level]
    index_level_mapping = {}
    for index in index_in_atmos_level_all:
        index_level_mapping[index] = atmos_level_all[index]
    return index_in_atmos_level_all, index_level_mapping

index_in_atmos_level_all, index_level_mapping = atmos_level2index_in_atmos_level_all(atmos_level)


# print(index_level_mapping, index_in_atmos_level_all)
# index_level_mapping = {
#     0: 5000, 2: 10000, 4: 15000, 6: 20000, 8: 25000, 10: 30000, 14: 40000,
#     18: 50000, 22: 60000, 26: 70000, 32: 85000, 35: 92500, 38: 100000,
# }

