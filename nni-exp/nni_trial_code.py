import subprocess
import json
import nni
from mmengine import Config
# from mmdet.utils.nnieval import NniHook

cfg = Config.fromfile('configs/solar_event/convnext.py')
hyperparams = nni.get_next_parameter()

cfg.model.roi_head.bbox_head.loss_bbox.loss_weight = hyperparams['loss_bbox_weight']
if hyperparams['loss_cls_algorithm'] == 'CrossEntropy':
    cfg.model.roi_head.bbox_head.loss_cls = dict(type='CrossEntropyLoss', loss_weight=hyperparams['loss_cls_weight'], use_sigmoid=True)
    cfg.model.roi_head.bbox_head.loss_cls = dict(type='CrossEntropyLoss', loss_weight=hyperparams['loss_cls_weight'],
                                                 class_weight=[hyperparams['loss_cls_CH_weight'],
                                                               hyperparams['loss_cls_FI_weight'],
                                                               hyperparams['loss_cls_void_weight']], use_sigmoid=True)
elif hyperparams['loss_cls_algorithm'] == 'Focal':
    cfg.model.roi_head.bbox_head.loss_cls = dict(type='FocalLoss', loss_weight=hyperparams['loss_cls_weight'], use_sigmoid=True)
elif hyperparams['loss_cls_algorithm'] == 'GHMC':
    cfg.model.roi_head.bbox_head.loss_cls = dict(type='GHMC', loss_weight=hyperparams['loss_cls_weight'], use_sigmoid=True)
cfg.model.roi_head.mask_head.loss_mask.loss_weight = hyperparams['loss_mask_weight']
cfg.model.rpn_head.loss_bbox.loss_weight = hyperparams['loss_rpn_bbox_weight']
if hyperparams['loss_rpn_cls_algorithm'] == 'CrossEntropy':
    cfg.model.rpn_head.loss_cls = dict(type='CrossEntropyLoss', loss_weight=hyperparams['loss_rpn_cls_weight'], use_sigmoid=True)
elif hyperparams['loss_rpn_cls_algorithm'] == 'GHMC':
    cfg.model.rpn_head.loss_cls = dict(type='GHMC', loss_weight=hyperparams['loss_rpn_cls_weight'], use_sigmoid=True)

cfg.custom_imports = dict(allow_failed_imports=False, 
                          imports=['mmpretrain.models', 'mmdet.utils.nnieval'])
cfg.work_dir = './nni_work_dir'

trial_id = nni.get_trial_id()
nni_eval_hook = {'custom_hooks': [dict(type='NniHook', trial_id=trial_id)]}
cfg.merge_from_dict(nni_eval_hook)

cfg_file_name = f'nni_trial/trial_cfg_{trial_id}.py'
cfg.dump(cfg_file_name)
training_pipe = subprocess.Popen(f'export CUBLAS_WORKSPACE_CONFIG=:4096:8 && python tools/train.py {cfg_file_name}',
                                 shell=True, stdout=subprocess.PIPE, bufsize=1)
for info in iter(training_pipe.stdout.readline, b''):
    print(info)
with open(f'nni_trial/metric_{trial_id}.json') as f:
    metric = json.load(f)
    nni.report_final_result(metric['best_score'])