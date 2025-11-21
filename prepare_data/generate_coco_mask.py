import re
import os
import cv2
import json
import gzip
import struct
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import astropy.units as u
from astropy.units import Quantity
from astropy.coordinates import SkyCoord
from sunpy.map import Map
from sunpy.time import parse_time
from sunpy.io._fits import read as fits_read
from sunpy.coordinates.ephemeris import get_earth
from sunpy.coordinates import frames, get_body_heliographic_stonyhurst
from sunpy.physics.differential_rotation import solar_rotate_coordinate
from skimage.draw import polygon2mask
import pickle as pkl

CH_pkl = "CH_data_2014.pkl"
with open(CH_pkl, 'rb') as f:
    CH_data = pkl.load(f)

image_id = 0
annotation_count = 0
image_list = []
annotation_list = []

start_time = datetime(2014, 1, 1, 0, 0, 0)
end_time = datetime(2014, 7, 1, 0, 0, 0)
obs_time = start_time

# filament info list
filament_dir = "../../filament/kso/"
kso_list = sorted(os.listdir(filament_dir + 'filament_obs/'))
filament_info = pd.DataFrame(pd.date_range(start_time, end_time, freq='D'), columns=['time'])
filament_info['kso'] = ''
for file_name in kso_list:
    hour = int(file_name[16:18])
    current_index = filament_info.loc[filament_info['time'] == pd.Timestamp(file_name[7:15])].index
    if len(current_index) == 0 or (current_index >= len(filament_info)-1).any():
        continue
    if (filament_info.loc[current_index, 'kso'] == '').all():
        filament_info.loc[current_index, 'kso'] = file_name
    else:
        filament_info.loc[current_index+1, 'kso'] = file_name

while obs_time < end_time:
    current_filament_index = filament_info.loc[filament_info['time'] == obs_time].index[0]
    if (filament_info.loc[current_filament_index, 'kso'] == ''):
        obs_time = obs_time + timedelta(days=1)
        continue  # skip those days without filament data

    aia_193 = Map('/media/ExtHDD/data/AIA-193/aia_lev1_193a_%04d_%02d_%02dt%02d_%02d*_image_lev1.fits'
                  % (obs_time.year, obs_time.month, obs_time.day, obs_time.hour, obs_time.minute))
    aia_193_obs = get_body_heliographic_stonyhurst("earth", time=aia_193.date)
    aia_193_avg = np.average(aia_193.data)
    source_time_str = re.sub("[-:.]", "_", aia_193.date.value[:-1]).replace("T", "t")
    # source_fits = f'aia_lev1_193a_{source_time_str}z_image_lev1.fits'
    source_png = f'aia_193a_{source_time_str}z_image.png'
    coco_image_dict = {"id": image_id, "width": 4096, "height": 4096, "file_name": source_png, "license": 0}
    image_list.append(coco_image_dict)
    # coronal hole
    for j in range(0, len(CH_data), 1):
        CH_event = CH_data[j]
        event_start_time = parse_time(CH_event['event_starttime'])
        event_end_time = parse_time(CH_event['event_endtime'])
        if event_start_time < aia_193.date and event_end_time > aia_193.date and CH_event['area_atdiskcenter'] > 1e9:
            p1 = CH_event["hpc_boundcc"][9:-2]
            p2 = p1.split(',')
            p3 = [v.split(" ") for v in p2]
            ch_boundary = SkyCoord(
                [(float(v[0]), float(v[1])) * u.arcsec for v in p3],
                obstime=event_end_time, observer='earth',
                frame=frames.Helioprojective)
            rotated_ch_boundary = solar_rotate_coordinate(ch_boundary, observer=aia_193_obs)
            pix_aia = aia_193.world_to_pixel(rotated_ch_boundary)
            pix_aia_x = pix_aia.x.value
            pix_aia_y = pix_aia.y.value
            # remove nan
            pix_nan_id = np.isnan(pix_aia_x)
            pix_aia_x = pix_aia_x[~pix_nan_id].tolist()
            pix_aia_y = pix_aia_y[~pix_nan_id].tolist()
            if len(pix_aia_x) < 3:
                continue
            pix_aia_pair = np.vstack([pix_aia_x, pix_aia_y]).transpose()
            aia_pix_mask = polygon2mask((4096, 4096), pix_aia_pair)
            area = float(np.sum(aia_pix_mask))

            coco_annotation_dict = {"id": annotation_count, "image_id": image_id, "category_id": 1, "segmentation": [pix_aia_pair.reshape(-1).tolist()],
                                    "area": area, "iscrowd": 0}
            annotation_list.append(coco_annotation_dict)
            annotation_count = annotation_count + 1
    # filament
    file_name = filament_info.loc[current_filament_index, 'kso']
    with gzip.open(filament_dir+'filament_obs/'+file_name, 'rb') as f:
        xsize, ysize, radius, hh, mm, ss = struct.unpack("6f", f.read(4*6))
        xsize = int(xsize)
        ysize = int(ysize)
        num = xsize*ysize
        filament_data = struct.unpack(f"{num}H", f.read(2*num))
    filament_data = np.array(filament_data).reshape([xsize, ysize])

    source_halpha = filament_dir+'kso_halpha/kanz_halph_fi_%s_%02d%02d%02d.fts.gz' % (file_name[7:15], hh, mm, ss)
    halph_data, halph_header = fits_read(source_halpha)[0]
    e_t = get_earth(halph_header['DATE'])
    halph_header['HGLN_OBS'] = e_t.lon.to('deg').value
    halph_header['HGLT_OBS'] = e_t.lat.to('deg').value
    halph_header['DSUN_OBS'] = e_t.radius.to('m').value
    halph_header['WAVEUNIT'] = 'angstrom'
    halph_map = Map(halph_data, halph_header)

    for k in range(1, 50, 1):
        filament_mask = (filament_data == k).astype(np.uint8)
        if np.sum(filament_mask) == 0:
            break
        contours, hierarchy = cv2.findContours(filament_mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_TC89_KCOS)
        segment = []
        area = 0
        intensity_sum = 0
        for contour in contours:
            pix_x = contour[:, 0, 0]
            pix_y = contour[:, 0, 1]
            pos_world = halph_map.pixel_to_world(Quantity(pix_x, unit='pix'), Quantity(pix_y, unit='pix'))
            rotated_pos = solar_rotate_coordinate(pos_world, observer=aia_193_obs)
            pix_aia = aia_193.world_to_pixel(rotated_pos)
            pix_aia_x = pix_aia.x.value
            pix_aia_y = pix_aia.y.value
            # remove nan
            pix_nan_id = np.isnan(pix_aia_x)
            pix_aia_x = pix_aia_x[~pix_nan_id].tolist()
            pix_aia_y = pix_aia_y[~pix_nan_id].tolist()
            if len(pix_aia_x) < 3:
                continue
            pix_aia_pair = np.vstack([pix_aia_x, pix_aia_y]).transpose()
            aia_pix_mask = polygon2mask((4096, 4096), pix_aia_pair[:, ::-1])  # input polygon coordinates are (row, column)
            area_i = float(np.sum(aia_pix_mask))
            area = area + area_i
            intensity_i = np.sum(aia_193.data[aia_pix_mask])
            intensity_sum = intensity_sum + intensity_i
            segment.append(pix_aia_pair.reshape(-1).tolist())
        if area == 0:
            continue
        intensity_avg = intensity_sum/area
        if intensity_avg > aia_193_avg:
            continue
        coco_annotation_dict = {"id": annotation_count, "image_id": image_id, "category_id": 2, "segmentation": segment,
                                "area": area, "iscrowd": 0}
        annotation_list.append(coco_annotation_dict)
        annotation_count = annotation_count + 1
    print(source_time_str)
    image_id = image_id + 1
    obs_time = obs_time + timedelta(days=1)

info = {"year": 2024, "version": "0.0.2", "description": "solar disk dataset", "date_created": datetime.today().isoformat()}
license_dict = {"url": r"http://creativecommons.org/licenses/by-nc-sa/2.0/", "id": 0, "name": "Attribution-NonCommercial-ShareAlike License"},
category_list = [{"id": 1, "name": "coronal hole", "supercategory": "disk event"},
                 {"id": 2, "name": "filament", "supercategory": "disk event"}]
coco_dataset = {"info": info, "images": image_list, "annotations": annotation_list, "licenses": license_dict, "categories": category_list}

with open("../annotation/solar_event_dataset_coco_v0.json", "w") as f:
    json.dump(coco_dataset, f)