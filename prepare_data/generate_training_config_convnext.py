from mmengine import Config

cfg = Config.fromfile('projects/ConvNeXt-V2/configs/mask-rcnn_convnext-v2-b_fpn_lsj-3x-fcmae_coco.py')

# Modify dataset classes and color
cfg.metainfo = {
    'classes': ('coronal hole', 'filament'),
    'palette': [
        (0, 255, 0),
        (1, 146, 243)
    ]
}

# Modify dataset type and path
cfg.data_root = './'

cfg.train_dataloader.dataset.ann_file = 'annotation/solar_event_dataset_coco_v0_train.json'
cfg.train_dataloader.dataset.data_root = cfg.data_root
cfg.train_dataloader.dataset.data_prefix.img = 'img_data/'
cfg.train_dataloader.dataset.metainfo = cfg.metainfo

cfg.val_dataloader.dataset.ann_file = 'annotation/solar_event_dataset_coco_v0_val.json'
cfg.val_dataloader.dataset.data_root = cfg.data_root
cfg.val_dataloader.dataset.data_prefix = dict(img='img_data/')
cfg.val_dataloader.dataset.metainfo = cfg.metainfo

cfg.test_dataloader = cfg.val_dataloader

# Modify metric config
cfg.val_evaluator.ann_file = cfg.data_root+'annotation/solar_event_dataset_coco_v0_val.json'
cfg.test_evaluator = cfg.val_evaluator

# Modify num classes of the model in box head and mask head
# cfg.model.roi_head.bbox_head.num_classes = 2
cfg.model.roi_head.bbox_head.num_classes = 2
cfg.model.roi_head.mask_head.num_classes = 2

# We can still the pre-trained Mask RCNN model to obtain a higher performance
cfg.load_from = 'checkpoints/mask-rcnn_convnext-v2-b_fpn_lsj-3x-fcmae_coco_20230113_110947-757ee2dd.pth'

# Set up working dir to save files and logs.
cfg.work_dir = './training_work_dir'

cfg.train_cfg.max_epochs = 50

# We can set the evaluation interval to reduce the evaluation times
cfg.train_cfg.val_interval = 3
# We can set the checkpoint saving interval to reduce the storage cost
cfg.default_hooks.checkpoint.interval = 3
# save the best model
cfg.default_hooks.checkpoint.save_best = "auto"

# The original learning rate (LR) is set for 8-GPU training.
# We divide it by 8 since we only use one GPU.
cfg.optim_wrapper.optimizer.lr = 0.01 / 8
cfg.default_hooks.logger.interval = 10


# Set seed thus the results are more reproducible
cfg.randomness = dict(seed=0, diff_rank_seed=False, deterministic=True)
# set_random_seed(0, deterministic=False)

# We can also use tensorboard to log the training process
cfg.visualizer.vis_backends.append({"type":'TensorboardVisBackend'})

#------------------------------------------------------
config=f'./configs/solar_event/convnext.py'
with open(config, 'w') as f:
    f.write(cfg.pretty_text)
