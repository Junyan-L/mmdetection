import re
import numpy as np
from PIL import Image
from sunpy.map import Map
from sunpy.time import TimeRange
import astropy.units as u

from preprocess import preprocess


img_dir_root = "../img_data/"

time_range = TimeRange('2014/01/01', '2014/07/01 00:00:00')
time_range_list = time_range.window(24*u.hour, 24*u.hour)
for delta_time in time_range_list:
    obs_time = delta_time.start.datetime
    aia_193 = Map('/media/ExtHDD/data/AIA-193/aia_lev1_193a_%04d_%02d_%02dt%02d_%02d*_image_lev1.fits'
                  % (obs_time.year, obs_time.month, obs_time.day, obs_time.hour, obs_time.minute))
    preprocessed_matrix = preprocess(aia_193, return_map=False, recover_degrad=True, recover_limb=True, equ_hist=True)
    img = Image.fromarray(preprocessed_matrix.astype(np.uint8))
    source_time_str = re.sub("[-:.]", "_", aia_193.date.value[:-1]).replace("T", "t")
    file_img = f'aia_193a_{source_time_str}z_image.png'
    img.save(img_dir_root+file_img)
    print(source_time_str)