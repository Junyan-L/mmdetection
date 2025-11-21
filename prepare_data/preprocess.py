import numpy as np
from sunpy.map import Map
from aiapy.calibrate import correct_degradation
from skimage.exposure import equalize_hist
from sunpy.map.maputils import all_coordinates_from_map


def remove_limb_brightness(solar_map):
    coords = all_coordinates_from_map(solar_map)
    distance = np.sqrt((coords.Tx - solar_map.center.Tx)**2 + (coords.Ty - solar_map.center.Ty)**2)
    distance_rs = (distance/(solar_map.rsun_obs)).value
    solar_disk_mask = (distance_rs <= 1)
    phi_cos = np.sqrt(1-distance_rs[solar_disk_mask]**2)
    ratio = 4.5 - 7.3*phi_cos + 3.9*phi_cos**2
    buffer = solar_map.data.copy()
    buffer[solar_disk_mask] = buffer[solar_disk_mask]/ratio
    return Map(buffer, solar_map.meta)


def preprocess(solar_map, return_map=False, recover_degrad=True, recover_limb=True, equ_hist=True):
    if recover_degrad:
        solar_map = correct_degradation(solar_map, correction_table='aia_V10_20201119_190000_response_table.txt', calibration_version=10)
    if recover_limb:
        solar_map = remove_limb_brightness(solar_map)
    if equ_hist:
        hist_data = equalize_hist(solar_map.data)*255  # data range [0-255]
        if ~return_map:
            return hist_data
        solar_map = Map(hist_data, solar_map.meta)
    return solar_map