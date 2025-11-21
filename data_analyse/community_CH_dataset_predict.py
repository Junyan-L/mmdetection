from mmdet.apis import DetInferencer
import glob
import numpy as np
from sunpy.map import Map

from prepare_data.preprocess import preprocess

# Choose to use a config
config = 'training_work_dir/convnext.py'
# Setup a checkpoint file to load
checkpoint = 'training_work_dir/best_coco_bbox_mAP_epoch_42_convnext.pth'
# Set the device to be used for evaluation
device = 'cuda:0'
# Initialize the DetInferencer
inferencer = DetInferencer(config, checkpoint, device)

# img = '/home/liujunyan/CH/ch_mmdetection/img_data/aia_193a_2014_01_13t00_00_06_84z_image.png'
# img_list = glob.glob("/home/liujunyan/CH/ch_mmdetection/img_data/aia_193a_2011_01_*_image.png")
img_list = glob.glob("/media/ExtHDD/data/CH_compare_dataset/193/*.fits")

# Use the detector to do inference
for img in img_list:
    solar_map = Map(img)
    out_img_name = img.split('/')[-1][:-5]
    updated_aia_map_data = preprocess(solar_map, return_map=False, recover_degrad=False, recover_limb=True, equ_hist=True)
    flipped_data = np.flipud(updated_aia_map_data)
    model_input_arr = flipped_data.reshape(*(flipped_data.shape), 1).repeat(3, axis=2)
    result = inferencer(model_input_arr, out_dir='data_analyse/compare', out_img_name=out_img_name)