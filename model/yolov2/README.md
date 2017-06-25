# How to use

## Download YOLO v2 original model file

```
$ wget http://pjreddie.com/media/files/yolo.weights
```

## Convert YOLO v2 original model to Chainer model

```
$ python darknet_parser.py darkent_weight_file output_model_path
```

Parameters:
* `darkent_weight_file`: (Required)Darknet YOLOv2 weight file path(usually `yolo.weights`)
* `output_model_path`: (Required)Output model file path
* `-c class_num`: (Optional) Number of classes. (default: 80)
* `-b box_num`: (Optional) Number of anchor boxes. (default: 5)

Note that last layer weights are copied only when both `class_num` and `box_num` are same as those of original model.

Example:

```
$ python darknet_parser.py yolo.weights yolov2.model
```

## Predict

```
$ python yolov2_predict.py image_dir output_path model_path [ -c config_file_path ] [ -g gpu_id ]
```

Parameters:
* `image_dir`: (Required) Directory path that has images. Directories can be nested.
* `output_path`: (Required) Output JSON file path.
* `model_path`: (Required) Model file path.
* `-c config_file_path`: (Optional) Configuration file path.
* `-g gpu_id`: (Optional) GPU device id. negative value indicates CPU. (default: -1)

Example:

```
python yolov2_predict.py c:\project_2\dataset\caltech-pedestrian-converter\data\test_images result.json yolov2.model -c config\caltech.json -g 0
```

### Configuration file format

* `categories`: List of categories of detected objects. `null` indicates all categories are detected.
* `input_size`: Input image size. Images are resized to this size. The value must be `[width, height]` or `null` that indicates image is not resized.

Example:

```
{
  "categories": ["person"],
  "input_size": [800, 800], # Input image size. Images are resized to this size. null indicates
}
```
