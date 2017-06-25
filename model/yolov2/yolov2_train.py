#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import numpy as np
import os
import six
import time
from PIL import Image

import chainer
from chainer import cuda
from chainer import serializers
from chainer import optimizers

from yolov2 import YOLOv2
from yolov2 import YOLOv2Predictor
from lib import utils


DEFAULT_CONFIG = {
    'categories': [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors','teddy bear','hair drier','toothbrush'
    ],
    'anchors': [[0.57273, 0.677385], [1.87446, 2.06253], [3.33843, 5.47434], [7.88282, 3.52778], [9.77052, 9.16828]],
    'dataset_dir': None,
    'crop_size': 384,
    'train_sizes': [320, 352, 384, 416, 448],
    'initial_model_path': None,
    'model_dir': 'model',
    'prefix': 'yolov2',
    'batch_size': 8,
    'epoch': 30,
    'learning_rate': 5e-4,
    'momentum': 0.9,
    'learning_decay_ratio': 0.1,
    'learning_decay_epoch': 10,
    'burn_in': 1000,
    'weight_decay': 0.0005,
    'save_epoch': 1,
    'min_width': 20,
    'min_height': 20,
    'min_visible_ratio': 0.5,
}
IMAGE_PATH_EXTENSIONS = ['.jpg', '.png']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str,
        help='Configuration file path')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
        help='GPU device ID, negative value indicates CPU')
    return parser.parse_args()

def find_files(root_dir, extensions):
    for root, dirs, files in os.walk(root_dir):
        for file_name in files:
            base_name, ext = os.path.splitext(file_name)
            if not ext in extensions:
                continue
            annotation_path = os.path.join(root, base_name + '.json')
            if not os.path.exists(annotation_path):
                raise FileNotFoundError('File not exists {}'.format(annotation_path))
            file_path = os.path.join(root, file_name)
            yield file_path, annotation_path

def load_annotation(file_path):
    with open(file_path) as f:
        return json.load(f)

def _category_to_index(region, categories):
    category = region['category']
    category_index = categories.index(category)
    one_hot_category = np.zeros(len(categories), dtype=np.float32)
    one_hot_category[category_index] = 1
    result = {}
    result.update(region)
    result.update({
        'category_index': category_index,
        'category_one_hot': one_hot_category,
    })
    return result

def convert_category_to_index(regions, categories):
    regions = filter(lambda x: x['category'] in categories, regions)
    regions = map(lambda x: _category_to_index(x, categories), regions)
    return regions

def convert_regions(regions, crop_rect, input_size, min_width, min_height,
        min_visible_ratio):
    ground_truths = []
    cx, cy, cw, ch = crop_rect
    for region in regions:
        x, y, w, h = region['bbox']
        if region.has_key('visible_bbox'):
            vx, vy, vw, vh = region['visible_bbox']
        else:
            vx, vy, vw, vh = x, y, w, h
        vx1 = min(max(vx - cx, 0), cw)
        vx2 = min(max(vx + vw - cx, 0), cw)
        vy1 = min(max(vy - cy, 0), ch)
        vy2 = min(max(vy + vh - cy, 0), ch)
        if float((vx2 - vx1) * (vy2 - vy1)) / (w * h) < min_visible_ratio:
            continue
        x1 = min(max(x - cx, 0), cw)
        x2 = min(max(x + w - cx, 0), cw)
        y1 = min(max(y - cy, 0), ch)
        y2 = min(max(y + h - cy, 0), ch)
        scale_x = 1.0 / cw
        scale_y = 1.0 / ch
        if x2 - x1 < min_width or y2 - y1 < min_height:
            continue
        ground_truths.append({
            'x': (x1 + x2) * 0.5 * scale_x,
            'y': (y1 + y2) * 0.5 * scale_y,
            'w': (x2 - x1) * scale_x,
            'h': (y2 - y1) * scale_y,
            'label': region['category_index'],
            'one_hot_label': region['category_one_hot']
        })
    return ground_truths

def load_image(file_path):
    return Image.open(file_path).convert('RGB')

def transform_image(image, crop_rect, input_size):
    cx, cy, cw, ch = crop_rect
    image = image.crop((cx, cy, cx + cw, cy + ch)).resize((input_size, input_size))
    image = np.asarray(image, dtype=np.float32) / 255.0
    image = image.transpose(2, 0, 1)
    return image

def make_data(image_path, annotation_path, categories):
    with open(annotation_path) as f:
        annotation = json.load(f)
    return image_path, convert_category_to_index(annotation['regions'], categories)

def randomize_crop_rect(image_width, image_height, crop_size):
    crop_size = min(image_width, image_height, crop_size)
    x = np.random.randint(0, image_width - crop_size + 1)
    y = np.random.randint(0, image_height - crop_size + 1)
    return x, y, crop_size, crop_size

def main():
    args = parse_args()
    config = {}
    config.update(DEFAULT_CONFIG)
    with open(args.config_path) as f:
        config.update(json.load(f))

    categories = config['categories']
    anchors = config['anchors']
    initial_model_path = config['initial_model_path']

    print("loading annotations")
    dataset = find_files(config['dataset_dir'], IMAGE_PATH_EXTENSIONS)
    dataset = [make_data(image_path, annotation_path, categories)
               for image_path, annotation_path in dataset]

    # load model
    yolov2 = YOLOv2(n_classes=len(categories), n_boxes=len(anchors))
    model = YOLOv2Predictor(yolov2)
    model.init_anchor(anchors)
    model.predictor.finetune = False
    if initial_model_path is not None:
        print("loading initial model...")
        serializers.load_npz(initial_model_path, model)

    gpu_id = args.gpu
    xp = np
    if gpu_id >= 0:
        cuda.get_device(gpu_id).use()
        model.to_gpu(gpu_id)
        xp = cuda.cupy

    learning_rate = config['learning_rate']
    learning_decay_epoch = config['learning_decay_epoch']
    learning_decay_ratio = config['learning_decay_ratio']
    burn_in = config['burn_in']
    momentum = config['momentum']
    optimizer = optimizers.MomentumSGD(lr=learning_rate, momentum=momentum)
    optimizer.setup(model)

    batch_size = config['batch_size']
    epoch_num = config['epoch']
    min_width = config['min_width']
    min_height = config['min_height']
    min_visible_ratio = config['min_visible_ratio']
    crop_size = config['crop_size']
    train_sizes = config['train_sizes']
    model_dir = config['model_dir']
    prefix = config['prefix']
    save_epoch = config['save_epoch']
    last_time = time.time()
    iterator = chainer.iterators.SerialIterator(dataset, batch_size)
    trained_num = 0
    iteration = 0
    while iterator.epoch < epoch_num:
        if iteration < burn_in:
            optimizer.lr = learning_rate * ((float)(iteration + 1) / burn_in) ** 4
        if trained_num == 0:
            print('epoch {} start'.format(iterator.epoch))
        epoch = iterator.epoch
        batch = iterator.next()
        train_size = train_sizes[np.random.randint(0, len(train_sizes))]
        image_batch = []
        label_batch = []
        for image_path, regions in batch:
            image = load_image(image_path)
            image_width, image_height = image.size
            crop_rect = randomize_crop_rect(image_width, image_height, crop_size)
            image_batch.append(transform_image(image, crop_rect, train_size))
            label_batch.append(convert_regions(regions, crop_rect, train_size,
                min_width, min_height, min_visible_ratio))

        x = xp.asarray(image_batch)
        x_loss, y_loss, w_loss, h_loss, c_loss, p_loss = model(x, label_batch)
        loss = x_loss + y_loss + w_loss + h_loss + c_loss + p_loss
        model.cleargrads()
        loss.backward()
        optimizer.update()

        trained_num += len(batch)
        print('epoch: {0:d}  {1:d} / {2:d} learning rate: {3} loss: {4:.5f}'.format(
            epoch, trained_num, len(dataset), optimizer.lr, float(loss.data)
        ))
        print('x_loss: {}  y_loss: {}  w_loss: {}  h_loss: {}'.format(
            float(x_loss.data), float(y_loss.data), float(w_loss.data),
            float(h_loss.data)
        ))
        print('c_loss: {}  p_loss: {}'.format(
            float(c_loss.data), float(p_loss.data)
        ))
        iteration += 1
        if iterator.is_new_epoch:
            if isinstance(learning_decay_epoch, list):
                if iterator.epoch in learning_decay_epoch:
                    learning_rate *= learning_decay_ratio
                    optimizer.lr = learning_rate
            elif iterator.epoch % learning_decay_epoch == 0:
                learning_rate *= learning_decay_ratio
                optimizer.lr = learning_rate
            if iterator.epoch % save_epoch == 0:
                save_path = os.path.join(model_dir,
                    '{0}_{1:03d}.model'.format(prefix, iterator.epoch))
                serializers.save_npz(save_path, yolov2)
            trained_num = 0

    save_path = os.path.join(model_dir, '{0}.model'.format(prefix))
    serializers.save_npz(save_path, yolov2)

if __name__ == '__main__':
    main()
