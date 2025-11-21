# The training process of DETACH
## Prepare data

Get into the preparation dir

```bash
cd prepare_data
```

Generate a list of png format preprocessed image data. The generation process includes correct degradation, remove limb brightness, and image histogram equalization.

```bash
python -u generate_preprocessed_data.py
```

## Train data

Download pretrained data

```bash
mim download mmdet --config mask-rcnn_r50-caffe_fpn_ms-poly-3x_coco --dest ./checkpoints
```
Download the `solar_event_dataset_coco_v0.json` from release and place it into the `annotation` dir.

Generate train and val data

```bash
python prepare_data/generate_train_val_coco_json.py
```

Add environment setting

```bash
export CUBLAS_WORKSPACE_CONFIG=:16:8
```

Train:

```bash
python tools/train.py configs/solar_event/convnext_detach.py
```

## Check result

check the prediction of one image

```bash
python data_analyse/prediction_example.py
```

## Annotate the data
The annotated file in DETACH is labeled manually by ourself. Here's the preparation process with the help of HEK database. 

Get into the preparation dir

```bash
cd prepare_data
```
Download coronal hole segmentation from HEK database

```bash
python get_hek_data.py
```
Generate the coco-format json annotation

```bash
python -u generate_coco_mask.py
```

The default annotation boundary is not accurate, so we need to manually correct the annotation.

We annotated the results by using the CVAT tools. The `solar_event_dataset_coco_v0.json` in release is our annotated results.

## Hyperparameter optimization
The hyperparameter in DETACH is selected semi-automatically by the SMAC algorithm with the help of nni toolkit.

Generate training config file:

```bash
python prepare_data/generate_training_config_convnext.py
```

Auto-ML with nni:
```bash
python nni-exp/nni_search.py 
```