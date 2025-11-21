from nni.experiment import Experiment

search_space = {
    'loss_bbox_weight': {'_type': 'loguniform', '_value': [1, 100]},
    'loss_cls_weight': {'_type': 'loguniform', '_value': [0.1, 10]},
    'loss_cls_algorithm': {'_type': 'choice', '_value': ['CrossEntropy', 'Focal', 'GHMC']},
    'loss_cls_CH_weight': {'_type': 'loguniform', '_value': [0.1, 10]},
    'loss_cls_FI_weight': {'_type': 'loguniform', '_value': [0.1, 10]},
    'loss_cls_void_weight': {'_type': 'loguniform', '_value': [0.1, 1]},
    'loss_mask_weight': {'_type': 'loguniform', '_value': [0.1, 10]},
    'loss_rpn_bbox_weight': {'_type': 'loguniform', '_value': [1, 100]},
    'loss_rpn_cls_weight': {'_type': 'loguniform', '_value': [0.1, 10]},
    'loss_rpn_cls_algorithm': {'_type': 'choice', '_value': ['CrossEntropy', 'GHMC']},
}

experiment = Experiment('local')
experiment.config.experiment_name = 'CH detection' 
experiment.config.trial_command = 'python3 nni-exp/nni_trial_code.py'
experiment.config.trial_code_directory = '.'
experiment.config.search_space = search_space
experiment.config.tuner.name = 'SMAC'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
experiment.config.max_trial_number = 50
experiment.config.trial_concurrency = 1

experiment.run(8009)
input('press enter to exit')