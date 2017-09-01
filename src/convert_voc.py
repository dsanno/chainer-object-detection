import argparse
import json
import os
import shutil
from xml.etree import ElementTree


def make_dirs(path):
    if os.path.exists(path):
        return
    os.makedirs(path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str, help='directory path of attnotation e.g. "VOC2007"')
    parser.add_argument('output_dir', type=str, help='output directory path')
    return parser.parse_args()


def parse_object(elem):
    label = elem.find('name').text
    box_elem = elem.find('bndbox')
    left = int(box_elem.find('xmin').text)
    right = int(box_elem.find('xmax').text)
    top = int(box_elem.find('ymin').text)
    bottom = int(box_elem.find('ymax').text)
    region = {
        'category': label,
        'bbox': [left, top, right - left, bottom - top]
    }
    return region


def parse_annotation(file_path):
    record = {}
    tree = ElementTree.parse(file_path)
    root = tree.getroot()
    image_file = root.find('filename').text
    obj = root.findall('object')
    regions = map(parse_object, obj)
    return image_file, { 'regions': regions }


def make_data(list_path, image_dir, annotation_dir, output_dir):
    with open(list_path) as f:
        indices = map(lambda x: x.strip(), f.readlines())
    for index in indices:
        annotation_path = os.path.join(annotation_dir, '{}.xml'.format(index))
        image_file, annotation = parse_annotation(annotation_path)
        image_path = os.path.join(image_dir, image_file)
        out_image_path = os.path.join(output_dir, image_file)
        shutil.copy(image_path, out_image_path)
        base_name = os.path.splitext(image_file)[0]
        out_annotation_path = os.path.join(output_dir, '{}.json'.format(base_name))
        with open(out_annotation_path, 'w') as f:
            json.dump(annotation, f)


def main():
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    annotation_dir = os.path.join(input_dir, 'Annotations')
    image_dir = os.path.join(input_dir, 'JPEGImages')
    train_list = os.path.join(input_dir, 'ImageSets', 'Layout', 'trainval.txt')
    test_list = os.path.join(input_dir, 'ImageSets', 'Layout', 'test.txt')

    out_train_dir = os.path.join(output_dir, 'trainval')
    make_dirs(out_train_dir)
    make_data(train_list, image_dir, annotation_dir, out_train_dir)
    out_test_dir = os.path.join(output_dir, 'test')
    make_dirs(out_test_dir)
    make_data(test_list, image_dir, annotation_dir, out_test_dir)


if __name__ == '__main__':
    main()
