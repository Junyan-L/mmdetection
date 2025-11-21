import os
import json
from mmengine.registry import HOOKS
from mmengine.hooks import Hook

@HOOKS.register_module()
class NniHook(Hook):
    def __init__(self, trial_id):
        self.trial_id = trial_id

    def after_run(self, runner):
        print('nni report after run', runner.message_hub.runtime_info['best_score'])
        with open(f'nni_trial/metric_{self.trial_id}.json', 'w') as f:
            json.dump(runner.message_hub.runtime_info, f)
        # move and rename the best checkpoint
        os.system(f"cp {runner.message_hub.runtime_info['best_ckpt']} nni_trial/best_coco_mAP_bbox_{self.trial_id}.pth")