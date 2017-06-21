#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import cv2
import json
import glob
import os
import time
import numpy as np

import chainer
from chainer import cuda
from chainer import serializers
import chainer.functions as F

from yolov2 import YOLOv2, YOLOv2Predictor
from lib import utils


DEFAULT_CONFIG = {
    'categories': None,
    'input_size': None,
}
IMAGE_PATH_EXTENSIONS = ['.jpg', '.png']


def parse_args():
    parser = argparse.ArgumentParser(description='Object detection')
    parser.add_argument('image_dir', type=str, help='Image directory path')
    parser.add_argument('output_path', type=str, help='Output file path')
    parser.add_argument('model_path', type=str, help='Model file path')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU device ID, negative value indicates CPU')
    parser.add_argument('--config', '-c', type=str, default=None, help='Configuration file path')
    return parser.parse_args()

class CocoPredictor:
    def __init__(self, model_path, config):
        # hyper parameters
        self.n_classes = 80
        self.n_boxes = 5
        self.detection_thresh = 0.01
        self.iou_thresh = 0.5
        self.config = config
        self.labels = ['person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush']
        anchors = [[0.57273, 0.677385], [1.87446, 2.06253], [3.33843, 5.47434], [7.88282, 3.52778], [9.77052, 9.16828]]
        # load model
        print('loading coco model...')
        yolov2 = YOLOv2(n_classes=self.n_classes, n_boxes=self.n_boxes)
        serializers.load_npz(model_path, yolov2)
        model = YOLOv2Predictor(yolov2)
        model.init_anchor(anchors)
        model.predictor.finetune = False
        self.model = model

    def to_gpu(self, device_id=None):
        self.model.to_gpu(device_id)

    def to_cpu(self):
        self.model.to_cpu()

    def __call__(self, orig_img, input_size=None):
        xp = self.model.xp
        orig_input_height, orig_input_width, _ = orig_img.shape
        if input_size is not None:
            img = cv2.resize(orig_img, input_size)
        else:
            img = utils.reshape_to_yolo_size(orig_img)
        input_height, input_width, _ = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.asarray(img, dtype=np.float32) / 255.0
        img = img.transpose(2, 0, 1)

        # forward
        x = xp.asarray(img[np.newaxis, :, :, :])
        x, y, w, h, conf, prob = self.model.predict(x)

        # parse results
        _, _, _, grid_h, grid_w = x.shape

        x = F.reshape(x, (self.n_boxes, grid_h, grid_w)).data
        y = F.reshape(y, (self.n_boxes, grid_h, grid_w)).data
        w = F.reshape(w, (self.n_boxes, grid_h, grid_w)).data
        h = F.reshape(h, (self.n_boxes, grid_h, grid_w)).data
        conf = F.reshape(conf, (self.n_boxes, grid_h, grid_w)).data
        prob = F.transpose(F.reshape(prob, (self.n_boxes, self.n_classes, grid_h, grid_w)), (1, 0, 2, 3)).data
        x = cuda.to_cpu(x)
        y = cuda.to_cpu(y)
        w = cuda.to_cpu(w)
        h = cuda.to_cpu(h)
        conf = cuda.to_cpu(conf)
        prob = cuda.to_cpu(prob)
        detected_indices = (conf * prob).max(axis=0) > self.detection_thresh
        x = x[detected_indices]
        y = y[detected_indices]
        w = w[detected_indices]
        h = h[detected_indices]
        conf = conf[detected_indices]
        prob = prob.transpose(1, 2, 3, 0)[detected_indices]
        categories = self.config.get('categories', None)
        results = []
        for i in range(detected_indices.sum()):
            class_id = prob[i].argmax()
            label = self.labels[class_id]
            results.append({
                'class_id': class_id,
                'label': label,
                'probs': prob[i],
                'conf' : conf[i],
                'objectness': conf[i] * prob[i].max(),
                'box'  : utils.Box(
                            x[i] * orig_input_width,
                            y[i] * orig_input_height,
                            w[i] * orig_input_width,
                            h[i] * orig_input_height).crop_region(orig_input_height, orig_input_width)
            })

        # nms
        nms_results = utils.nms(results, self.iou_thresh)
        return nms_results

def find_files(root_dir, extensions):
    for root, dirs, files in os.walk(root_dir):
        for file_name in files:
            ext = os.path.splitext(file_name)[1]
            if ext in extensions:
                yield os.path.join(root, file_name)

def main():
    args = parse_args()

    chainer.config.train = False
    chainer.config.enable_backprop = False

    image_dir = args.image_dir
    gpu_id = args.gpu
    config = {}
    config.update(DEFAULT_CONFIG)
    config_path = args.config
    if config_path is not None:
        with open(config_path) as f:
            config.update(json.load(f))
    if config['input_size'] is None:
        input_size = None
    else:
        input_size = tuple(config['input_size'])

    predictor = CocoPredictor(args.model_path, config)
    if gpu_id >= 0:
        cuda.get_device(gpu_id).use()
        predictor.to_gpu(gpu_id)

    results = []
    for image_path in find_files(image_dir, IMAGE_PATH_EXTENSIONS):
        # read image
        print('loading image {}...'.format( os.path.relpath(image_path, image_dir)))
        orig_img = cv2.imread(image_path)
        raw_regions = predictor(orig_img, )
        regions = []
        for raw_region in raw_regions:
            box = raw_region['box']
            x = box.x
            y = box.y
            width = box.w
            height = box.h
            regions.append({
                'x': x - width * 0.5,
                'y': y - height * 0.5,
                'width': width,
                'height': height,
                'category': raw_region['label'],
                'confidence': float(raw_region['probs'].max() * raw_region['conf']),
            })
        results.append({
            'file_path': image_path,
            'regions': regions,
        })

    with open(args.output_path, 'w') as f:
        json.dump(results, f)

if __name__ == '__main__':
    main()
