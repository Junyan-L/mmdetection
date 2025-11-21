import json
from sklearn.model_selection import train_test_split


origin_coco_file = "annotation/solar_event_dataset_coco_v0.json"
# origin_coco_file = "annotation/cvat_annotation.json"
train_json = "annotation/solar_event_dataset_coco_v0_train.json"
val_json = "annotation/solar_event_dataset_coco_v0_val.json"

with open(origin_coco_file, "r") as f:
    origin_coco = json.load(f)

train_img_list = []
train_annotation_list = []
val_img_list = []
val_annotation_list = []

img_length = len(origin_coco['images'])
train_id_list, val_id_list = train_test_split(range(img_length), test_size=0.3, random_state=0)

for img in origin_coco['images']:
    if img['id'] in train_id_list:
        train_img_list.append(img)
    if img['id'] in val_id_list:
        val_img_list.append(img)

for anno in origin_coco['annotations']:
    seg_length = len(anno['segmentation'])
    x_list = []
    y_list = []
    if anno['iscrowd'] == 0 and 'bbox' not in anno.keys():
        for i in range(seg_length):
            x_list_i = anno['segmentation'][i][::2]
            y_list_i = anno['segmentation'][i][1::2]
            x_list = x_list + x_list_i
            y_list = y_list + y_list_i
        bbox_width = max(x_list) - min(x_list)
        bbox_height = max(y_list) - min(y_list)
        bbox = [min(x_list), min(y_list), bbox_width, bbox_height]
        anno['bbox'] = bbox
    if anno['image_id'] in train_id_list:
        train_annotation_list.append(anno)
    if anno['image_id'] in val_id_list:
        val_annotation_list.append(anno)

train_coco = {"info": origin_coco['info'], "images": train_img_list, "annotations": train_annotation_list,  "licenses": origin_coco['licenses'], "categories": origin_coco['categories']}
with open(train_json, "w") as f:
    json.dump(train_coco, f)

val_coco = {"info": origin_coco['info'], "images": val_img_list, "annotations": val_annotation_list,  "licenses": origin_coco['licenses'], "categories": origin_coco['categories']}
with open(val_json, "w") as f:
    json.dump(val_coco, f)
