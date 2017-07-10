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

from yolov2 import YOLOv2Classifier


DEFAULT_CONFIG = {
    'categories': None,
    'train_dataset_dir': None,
    'test_dataset_dir': None,
    'crop_margin': 8,
    'train_sizes': [32, 64, 96, 128, 160, 192, 224],
    'max_test_size': 224,
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

def find_files(root_dir, extensions):
    for root, dirs, files in os.walk(root_dir):
        for file_name in files:
            base_name, ext = os.path.splitext(file_name)
            if not ext in extensions:
                continue
            file_path = os.path.join(root, file_name)
            yield file_path

def load_image(file_path):
    return Image.open(file_path).convert('RGB')

def transform_image(image, crop_rect, input_size):
    cx, cy, cw, ch = crop_rect
    image = image.crop((cx, cy, cx + cw, cy + ch)).resize((input_size, input_size))
    image = np.asarray(image, dtype=np.float32) / 255.0
    image = image.transpose(2, 0, 1)
    return image

def randomize_crop_rect(image_width, image_height, crop_margin):
    width = image_width - crop_margin
    height = image_height - crop_margin
    size = min(width, height)
    x = np.random.randint(0, width - size + 1)
    y = np.random.randint(0, height - size + 1)
    return x, y, size, size

def evaluate(model, dataset, crop_margin, max_test_size):
    xp = model.xp
    iterator = chainer.iterators.SerialIterator(dataset, 1, repeat=False, shuffle=False)
    acc_sum = 0
    iteration = 0
    for batch in iterator:
        image_path, category_id = batch[0]
        image = load_image(image_path)
        image_width, image_height = image.size
        crop_size = min(image_width, image_height) - crop_margin
        crop_rect = ((image_width - crop_size) // 2, (image_height - crop_size) // 2, crop_size, crop_size)
        input_size = min(crop_size, max_test_size)
        input_size = int(round(input_size / 32.0)) * 32
        image_batch = [transform_image(image, crop_rect, input_size)]
        label_batch = [category_id]

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
    test_dataset_dir = config['test_dataset_dir']
    categories = config['categories']
    if categories is None:
        categories = os.listdir(train_dataset_dir)
    initial_model_path = config['initial_model_path']

    print("loading dataset")
    train_dataset = []
    for i, category in enumerate(categories):
        category_dir = os.path.join(train_dataset_dir, category)
        image_paths = find_files(category_dir, IMAGE_PATH_EXTENSIONS)
        for image_path in image_paths:
            train_dataset.append((image_path, i))

    if test_dataset_dir is not None:
        test_dataset = []
        for i, category in enumerate(categories):
            category_dir = os.path.join(test_dataset_dir, category)
            image_paths = find_files(category_dir, IMAGE_PATH_EXTENSIONS)
            for image_path in image_paths:
                test_dataset.append((image_path, i))
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

    crop_margin = config['crop_margin']
    max_test_size = config['max_test_size']
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
    while iterator.epoch < epoch_num:
        if trained_num == 0:
            print('epoch {} start'.format(iterator.epoch))
        epoch = iterator.epoch
        batch = iterator.next()
        train_size = train_sizes[np.random.randint(0, len(train_sizes))]
        image_batch = []
        label_batch = []
        for image_path, category_id in batch:
            image = load_image(image_path)
            image_width, image_height = image.size
            crop_rect = randomize_crop_rect(image_width, image_height, crop_margin)
            image_batch.append(transform_image(image, crop_rect, train_size))
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

            test_acc = evaluate(model, test_dataset, crop_margin, max_test_size)
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

    save_path = os.path.join(model_dir, '{0}.model'.format(prefix))
    serializers.save_npz(save_path, model)

if __name__ == '__main__':
    main()
