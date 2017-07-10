#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import json
import numpy as np
import os
from PIL import Image
import six


DEFAULT_CONFIG = {
    'categories': ['person'],
    'background_category': 'background',
    'avoid_categories': [],
    'min_width': 5,
    'min_height': 20,
    'min_visible_ratio': 0.5,
    'avoid_iou_threshold': 0.2,
}

IMAGE_PATH_EXTENSIONS = ['.jpg', '.png']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=str, help='Dataset directory path that has image and label files')
    parser.add_argument('output_dir', type=str, help='Output directory path')
    parser.add_argument('config_path', type=str, help='Configuration file path')
    return parser.parse_args()

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

def crop_image(image, bounding_box):
    image_width, image_height = image.size
    x, y, w, h = bounding_box
    size = min(max(w, h) + 8, image_width, image_height)
    x = x + (w - size) * 0.5
    y = y + (h - size) * 0.5
    if x < 0:
        x = 0
    elif x + size >= image_width:
        x = image_width - size
    if y < 0:
        y = 0
    elif y + size >= image_height:
        y = image_height - size
    x1 = int(x)
    y1 = int(y)
    x2 = x1 + int(size)
    y2 = y1 + int(size)
    return image.crop((x1, y1, x2, y2))

def has_object(bounding_box, regions, iou_threshold):
    for region in regions:
        if iou(bounding_box, region['bbox']) >= iou_threshold:
            return True
    return False

def crop_background(image, avoid_regions, iou_threshold):
    image_width, image_height = image.size
    max_size = min(image_width, image_height)
    for i in six.moves.range(100):
        size = (np.random.randint(max_size // 32) + 1) * 32
        x = np.random.randint(0, image_width - size + 1)
        y = np.random.randint(0, image_height - size + 1)
        bounding_box = (x, y, size, size)
        if has_object(bounding_box, avoid_regions, iou_threshold):
            continue
        return image.crop((x, y, x + size, y + size))
    return None

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

def main():
    args = parse_args()
    dataset_dir = args.dataset_dir
    output_dir = args.output_dir
    config = {}
    config.update(DEFAULT_CONFIG)
    with open(args.config_path) as f:
        config.update(json.load(f))
    categories = config['categories']
    background_category = config['background_category']
    avoid_categories = categories + config['avoid_categories']
    min_width = config['min_width']
    min_height = config['min_height']
    min_visible_ratio = config['min_visible_ratio']
    avoid_iou_threshold = config['avoid_iou_threshold']


    # make output directories
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for category in categories:
        path = os.path.join(output_dir, category)
        os.mkdir(path)
    path = os.path.join(output_dir, background_category)
    os.mkdir(path)


    count = 0
    for image_path, annotation_path in find_files(dataset_dir, IMAGE_PATH_EXTENSIONS):
        image = Image.open(image_path)
        image_width, image_height = image.size
        with open(annotation_path) as f:
            annotation = json.load(f)
        regions = annotation['regions']
        avoid_regions = filter(lambda x: x['category'] in avoid_categories, regions)
        for region in regions:
            category = region['category']
            if not category in categories:
                continue
            x, y, w, h = region['bbox']
            vx, vy, vw, vh = region['visible_bbox']
            if not category in categories:
                continue
            if w < min_width or h < min_height:
                continue
            if float(vw * vh) / (w * h) < min_visible_ratio:
                continue
            object_image = crop_image(image, region['bbox'])
            output_path = os.path.join(output_dir, category, '{0:06d}.jpg'.format(count))
            object_image.save(output_path)
            count += 1
            # add near region to background images
            if np.random.randint(0, 1) < 1:
                bx1 = int(x + w)
                by1 = int(y - 4)
                bx2 = bx1 + int(h) + 8
                by2 = by1 + int(h) + 8
            else:
                bx1 = int(x - h) - 8
                by1 = int(y - 4)
                bx2 = bx1 + int(h) + 8
                by2 = by1 + int(h) + 8
            if by1 < 0:
                by1 = 0
            if by2 > image_height:
                by2 = image_height
            crop_rect = (bx1, by1, bx2, by2)
            bounding_box = (bx1, by1, bx2 - bx1, by2 - by1)
            if bx1 >= 0 and bx2 <= image_width and not has_object(bounding_box, avoid_regions, avoid_iou_threshold):
                background_image = image.crop(crop_rect)
                output_path = os.path.join(output_dir, background_category, '{0:06d}.jpg'.format(count))
                background_image.save(output_path)
                count += 1
        background_image = crop_background(image, avoid_regions, avoid_iou_threshold)
        if background_image is not None:
            output_path = os.path.join(output_dir, background_category, '{0:06d}.jpg'.format(count))
            background_image.save(output_path)
            count += 1


if __name__ == '__main__':
    main()
