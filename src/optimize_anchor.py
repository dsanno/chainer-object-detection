import argparse
import json
import numpy as np
import os
import six


def parse_args():
    parser = argparse.ArgumentParser('Optimize anchor boxes')
    parser.add_argument('dataset_dir', type=str, help='Dataset directory path')
    parser.add_argument('box_num', type=int, help='Number of boxes')
    parser.add_argument('--category', '-c', type=str, default=None, help='Comma separated category names')
    parser.add_argument('--scale', '-s', type=float, default=1, help='Scale factor of box size')
    return parser.parse_args()

def find_files(root_dir, extensions):
    for root, dirs, files in os.walk(root_dir):
        for file_name in files:
            base_name, ext = os.path.splitext(file_name)
            if not ext in extensions:
                continue
            file_path = os.path.join(root, file_name)
            yield file_path

def kmeans(ws, hs, box_num):
    # initiailze anchor boxes
    indices = np.random.randint(0, box_num, len(ws))
    anchor_ws = []
    anchor_hs = []
    for i in six.moves.range(box_num):
        anchor_ws.append(np.mean(ws[indices == i]))
        anchor_hs.append(np.mean(hs[indices == i]))
    anchor_ws = np.asarray(anchor_ws, dtype=np.float32)
    anchor_hs = np.asarray(anchor_hs, dtype=np.float32)

    ws = np.expand_dims(ws, 1)
    hs = np.expand_dims(hs, 1)
    anchor_ws = np.expand_dims(anchor_ws, 0)
    anchor_hs = np.expand_dims(anchor_hs, 0)

    for i in six.moves.range(100):
        unions = np.maximum(ws, anchor_ws) * np.maximum(hs, anchor_hs)
        intersections = np.minimum(ws, anchor_ws) * np.minimum(hs, anchor_hs)
        ious = intersections / unions
        indices = np.argmax(ious, axis=1)
        for j in six.moves.range(box_num):
            anchor_ws[0,j] = np.mean(ws[indices == j])
            anchor_hs[0,j] = np.mean(hs[indices == j])
    sorted_indices = np.argsort(anchor_hs[0])
    return anchor_ws[0][sorted_indices], anchor_hs[0][sorted_indices]

def main():
    args = parse_args()
    if args.category is not None:
        categories = args.category.split(',')
    else:
        categories = None
    scale = args.scale
    widths = []
    heights = []
    for file_path in find_files(args.dataset_dir, '.json'):
        with open(file_path) as f:
            annotation = json.load(f)
        for region in annotation['regions']:
            if categories is not None and not region['category'] in categories:
                continue
            x, y, w, h = region['bbox']
            widths.append(w)
            heights.append(h)
    widths = np.asarray(widths, dtype=np.float32)
    heights = np.asarray(heights, dtype=np.float32)
    anchor_ws, anchor_hs = kmeans(widths, heights, args.box_num)
    for i in six.moves.range(args.box_num):
        print('[{0:.5f}, {1:.5f}],'.format(anchor_ws[i] * scale, anchor_hs[i] * scale))

if __name__ == '__main__':
    main()
