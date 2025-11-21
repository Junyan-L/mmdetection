from mmdet.apis import DetInferencer
import glob

# Choose to use a config
config = 'training_work_dir/convnext.py'
# Setup a checkpoint file to load
# with open("training_work_dir/last_checkpoint") as f:
#     checkpoint = f.readline()
checkpoint = "training_work_dir/best_coco_bbox_mAP_epoch_45.pth"

# Set the device to be used for evaluation
device = 'cuda:0'

# Initialize the DetInferencer
inferencer = DetInferencer(config, checkpoint, device)

# Use the detector to do inference
# img = '/home/liujunyan/CH/ch_mmdetection/img_data/aia_193a_2014_01_13t00_00_06_84z_image.png'
img_list = glob.glob("/home/liujunyan/CH/ch_mmdetection/img_data/aia_193a_2011_01_*_image.png")
result = inferencer(img_list, out_dir='data_analyse')