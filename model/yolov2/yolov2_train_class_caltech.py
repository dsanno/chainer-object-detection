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
from chainer import functions as F
from chainer import serializers
from chainer import optimizers

from yolov2_caltech import YOLOv2Classifier


DEFAULT_CONFIG = {
    'categories': None,
    'train_dataset_dir': None,
    'background_dir': None,
    'test_dataset_dir': None,
    'crop_margin': 8,
    'train_sizes': [32, 64, 96, 128, 160, 192, 224],
    'background_iou_threshold': 0.02,
    'test_size': 224,
    'initial_model_path': None,
    'model_dir': 'model',
    'prefix': 'yolov2_class',
    'batch_size': 16,
    'epoch': 30,
    'learning_rate': 0.1,
    'momentum': 0.9,
    'learning_decay_ratio': 0.1,
    'learning_decay_epoch': 10,
    'weight_decay': 0.0005,
    'save_epoch': 1,
}
IMAGE_PATH_EXTENSIONS = ['.jpg', '.png']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str,
        help='Configuration file path')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
        help='GPU device ID, negative value indicates CPU')
    return parser.parse_args()

def find_files(root_dir, extensions, with_annotation=False):
    for root, dirs, files in os.walk(root_dir):
        for file_name in files:
            base_name, ext = os.path.splitext(file_name)
            if not ext in extensions:
                continue
            file_path = os.path.join(root, file_name)
            if with_annotation:
                annotation_path = os.path.join(root, base_name + '.json')
                if not os.path.exists(annotation_path):
                    raise FileNotFoundError('File not exists {}'.format(annotation_path))
                yield file_path, annotation_path
            else:
                yield file_path


def load_image(file_path):
    return Image.open(file_path).convert('RGB')

def iou(bounding_box1, bounding_box2):
    x1, y1, w1, h1 = bounding_box1
    x2, y2, w2, h2 = bounding_box2
    iw = min(x1 + w1, x2 + w2) - max(x1, x2)
    ih = min(y1 + h1, y2 + h2) - max(y1, y2)
    if iw <= 0:
        iw = 0
    if ih <= 0:
        ih = 0
    return float(iw * ih) / (w1 * h1 + w2 * h2 - iw * ih)

def transform_image(image, crop_rect, input_size, mirror=False):
    cx, cy, cw, ch = crop_rect
    image = image.crop((cx, cy, cx + cw, cy + ch)).resize((input_size, input_size))
    image = np.asarray(image, dtype=np.float32)
    image = image.transpose(2, 0, 1) / 255.0
    if mirror and np.random.randint(0, 2) < 1:
        image = image[:,:,::-1]
    return image

def randomize_crop_rect(image_width, image_height, crop_margin):
    width = image_width - crop_margin
    height = image_height - crop_margin
    size = min(width, height)
    x = np.random.randint(0, width - size + 1)
    y = np.random.randint(0, height - size + 1)
    return x, y, size, size

def has_object(bounding_box, regions, iou_threshold):
    for region in regions:
        if iou(bounding_box, region['bbox']) >= iou_threshold:
            return True
    return False

def crop_background(image, avoid_regions, iou_threshold, min_crop_size=64):
    image_width, image_height = image.size
    max_crop_size = min(image_width, image_height)
    scale = np.log(float(max_crop_size) / min_crop_size)
    for i in six.moves.range(100):
        size = int(np.exp(np.random.random() * scale) * min_crop_size)
        x = np.random.randint(0, image_width - size + 1)
        y = np.random.randint(0, image_height - size + 1)
        bounding_box = (x, y, size, size)
        if has_object(bounding_box, avoid_regions, iou_threshold):
            continue
        return image.crop((x, y, x + size, y + size))
    return None

def evaluate(model, dataset, crop_margin, test_size):
    xp = model.xp
    iterator = chainer.iterators.SerialIterator(dataset, 1, repeat=False, shuffle=False)
    acc_sum = 0
    iteration = 0
    for batch in iterator:
        image_batch = []
        label_batch = []
        for image_path, category_id, _ in batch:
            image = load_image(image_path)
            image_width, image_height = image.size
            crop_size = min(image_width, image_height) - crop_margin
            crop_rect = ((image_width - crop_size) // 2, (image_height - crop_size) // 2, crop_size, crop_size)
#            input_size = test_size
            input_size = int(round(crop_size / 32.0) * 32)
            if input_size < 64:
                input_size = 64
            elif input_size > test_size:
                input_size = test_size
            image_batch.append(transform_image(image, crop_rect, input_size))
            label_batch.append(category_id)

        x = xp.asarray(image_batch)
        t = xp.asarray(label_batch)

        with chainer.using_config('enable_backprop', False):
            with chainer.using_config('train', False):
                y = model(x)
        acc = F.accuracy(y, t)
        acc_sum += float(acc.data)
    return acc_sum / len(dataset)

def main():
    args = parse_args()
    config = {}
    config.update(DEFAULT_CONFIG)
    with open(args.config_path) as f:
        config.update(json.load(f))

    train_dataset_dir = config['train_dataset_dir']
    background_dir = config['background_dir']
    test_dataset_dir = config['test_dataset_dir']
    categories = config['categories']
    if categories is None:
        categories = os.listdir(train_dataset_dir)
    avoid_categories = categories + config['avoid_categories']
    initial_model_path = config['initial_model_path']

    print("loading dataset")
    train_dataset = []
    for i, category in enumerate(categories):
        category_dir = os.path.join(train_dataset_dir, category)
        image_paths = find_files(category_dir, IMAGE_PATH_EXTENSIONS)
        for image_path in image_paths:
            train_dataset.append((image_path, i, None))

    if background_dir is not None:
        if 'background' in categories:
            background_id = categories.index('background')
        else:
            background_id = len(categories)
            categories.append('background')
        path_pairs = find_files(background_dir, IMAGE_PATH_EXTENSIONS, with_annotation=True)
        for image_path, annotation_path in path_pairs:
            with open(annotation_path) as f:
                annotation = json.load(f)
            regions = annotation['regions']
            avoid_regions = filter(lambda x: x['category'] in avoid_categories, regions)
            train_dataset.append((image_path, background_id, avoid_regions))

    if test_dataset_dir is not None:
        test_dataset = []
        for i, category in enumerate(categories):
            category_dir = os.path.join(test_dataset_dir, category)
            image_paths = find_files(category_dir, IMAGE_PATH_EXTENSIONS)
            for image_path in image_paths:
                test_dataset.append((image_path, i, None))
    else:
        test_dataset = None

    # load model
    model = YOLOv2Classifier(n_classes=len(categories))
    model.finetune = False
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
    momentum = config['momentum']
    optimizer = optimizers.MomentumSGD(lr=learning_rate, momentum=momentum)
    optimizer.setup(model)

    background_iou_threshold = config['background_iou_threshold']
    crop_margin = config['crop_margin']
    test_size = config['test_size']
    batch_size = config['batch_size']
    epoch_num = config['epoch']
    train_sizes = config['train_sizes']
    model_dir = config['model_dir']
    prefix = config['prefix']
    save_epoch = config['save_epoch']
    last_time = time.time()
    iterator = chainer.iterators.SerialIterator(train_dataset, batch_size)
    trained_num = 0
    loss_sum = 0
    acc_sum = 0
    iteration = 0
    ##
    s = time.time()
    iteration = 0
    ##
    while iterator.epoch < epoch_num:
        if trained_num == 0:
            print('epoch {} start'.format(iterator.epoch))
        epoch = iterator.epoch
        batch = iterator.next()
        train_size = train_sizes[np.random.randint(0, len(train_sizes))]
        image_batch = []
        label_batch = []
        for image_path, category_id, avoid_regions in batch:
            image = load_image(image_path)
            image_width, image_height = image.size
            crop_rect = None
            if avoid_regions is not None:
                max_crop_size = min(image_width, image_height)
                min_crop_size = 64
                scale = np.log(float(max_crop_size) / min_crop_size)
                for i in six.moves.range(100):
                    size = int(np.exp(np.random.random() * scale) * min_crop_size)
                    x = np.random.randint(0, image_width - size + 1)
                    y = np.random.randint(0, image_height - size + 1)
                    bounding_box = (x, y, size, size)
                    if has_object(bounding_box, avoid_regions, background_iou_threshold):
                        continue
                    crop_rect = (x, y, x + size, y + size)
                    break
            else:
                crop_rect = randomize_crop_rect(image_width, image_height, crop_margin)
            if crop_rect is None:
                continue
            image = transform_image(image, crop_rect, train_size, mirror=True)
            image_batch.append(image)
            label_batch.append(category_id)

        x = xp.asarray(image_batch)
        t = xp.asarray(label_batch)

        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        trained_num += len(batch)
        loss_sum += float(loss.data) * batch_size
        acc_sum += float(acc.data) * batch_size
        iteration += 1
        if iterator.is_new_epoch:
            print('epoch {} done'.format(epoch))
            print('  train loss: {0:.5f} train acc: {1:.5f}'.format(
                loss_sum / trained_num, acc_sum / trained_num
            ))
            loss_sum = 0
            acc_sum = 0

            test_acc = evaluate(model, test_dataset, crop_margin, test_size)
            print('  test acc: {0:.5f}'.format(test_acc))

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
                serializers.save_npz(save_path, model)

            current_time = time.time()
            print('{}s elapsed'.format(current_time - last_time))
            last_time = current_time

            trained_num = 0
        ##
##        iteration += 1
##        print float(time.time() - s) / iteration * len(train_dataset) / batch_size
        ##

    save_path = os.path.join(model_dir, '{0}.model'.format(prefix))
    serializers.save_npz(save_path, model)

if __name__ == '__main__':
    main()
