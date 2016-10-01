import argparse
import numpy as np
import os
import six
from xml.etree import ElementTree

pickle = six.moves.cPickle

object_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
                'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
                'tvmonitor']
object_ids = {name: i for i, name in enumerate(object_names)}

def parse_object(elem):
    label = object_ids[elem.find('name').text]
    box_elem = elem.find('bndbox')
    left = int(box_elem.find('xmin').text)
    right = int(box_elem.find('xmax').text)
    top = int(box_elem.find('ymin').text)
    bottom = int(box_elem.find('ymax').text)
    return (label, (left, top, right, bottom))

def parse_file(file_path):
    record = {}
    tree = ElementTree.parse(file_path)
    root = tree.getroot()
    xml_file = os.path.basename(file_path)
    image_file = root.find('filename').text
    elem = root.find('size')
    image_width = int(elem.find('width').text)
    image_height = int(elem.find('height').text)
    elems = root.findall('object')
    objects = map(parse_object, elems)
    return (xml_file, image_file, (image_width, image_height), objects)

def parse_dir(dir_path):
    items = []
    files = os.listdir(dir_path)
    for file_name in files:
        if os.path.splitext(file_name)[1] == '.xml':
            items.append(parse_file(os.path.join(dir_path, file_name)))
    return items

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('annotation_dir', type=str, help='directory path that contains annotation XML files')
    parser.add_argument('out_path', type=str, help='output pickle file path')
    args = parser.parse_args()

    annotations = parse_dir(args.annotation_dir)
    with open(args.out_path, 'wb') as f:
        pickle.dump(annotations, f, pickle.HIGHEST_PROTOCOL)
