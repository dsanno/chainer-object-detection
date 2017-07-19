#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np

import chainer
from chainer import serializers

from yolov2 import YOLOv2


parser = argparse.ArgumentParser(description='Convert YOLO v2 model file to Chainer model file')
parser.add_argument('file', help='YOLO v2 weights file')
parser.add_argument('out_file', help='Output file path')
parser.add_argument('--class-num', '-c', type=int, default=80, help='Number of classes')
parser.add_argument('--box-num', '-b', type=int, default=5, help='Number of boxes')
args = parser.parse_args()

print('loading', args.file)
file = open(args.file, 'rb')
dat=np.fromfile(file, dtype=np.float32)[4:] # skip header(4xint)

# load model
print('loading initial model...')
n_classes = args.class_num
n_boxes = args.box_num
original_n_classes = 80
original_n_boxes = 5
last_out = (n_classes + 5) * n_boxes

yolov2 = YOLOv2(n_classes=n_classes, n_boxes=n_boxes)
yolov2.train = True
yolov2.finetune = False

layers=[
    [3, 32, 3],
    [32, 64, 3],
    [64, 128, 3],
    [128, 64, 1],
    [64, 128, 3],
    [128, 256, 3],
    [256, 128, 1],
    [128, 256, 3],
    [256, 512, 3],
    [512, 256, 1],
    [256, 512, 3],
    [512, 256, 1],
    [256, 512, 3],
    [512, 1024, 3],
    [1024, 512, 1],
    [512, 1024, 3],
    [1024, 512, 1],
    [512, 1024, 3],
    [1024, 1024, 3],
    [1024, 1024, 3],
    [512, 64, 1],
    [1024 + 64 * 4, 1024, 3],
]

offset=0
for i, l in enumerate(layers):
    in_ch = l[0]
    out_ch = l[1]
    ksize = l[2]

    # load bias(size of Bias.b is same as out_ch)
    txt = 'yolov2.bias%d.b.data[...] = dat[%d:%d]' % (i+1, offset, offset+out_ch)
    offset+=out_ch
    exec(txt)

    # load bn(size of BatchNormalization.gamma is save as out_ch)
    txt = 'yolov2.bn%d.gamma.data[...] = dat[%d:%d]' % (i+1, offset, offset+out_ch)
    offset+=out_ch
    exec(txt)

    # (size of BatchNormalization.avg_mean is same as out_ch)
    txt = 'yolov2.bn%d.avg_mean[...] = dat[%d:%d]' % (i+1, offset, offset+out_ch)
    offset+=out_ch
    exec(txt)

    # (size of BatchNormalization.avg_var is same as out_ch)
    txt = 'yolov2.bn%d.avg_var[...] = dat[%d:%d]' % (i+1, offset, offset+out_ch)
    offset+=out_ch
    exec(txt)

    # load convolution weight(size of Convolution2D.W is outch * in_ch * filter size, and reshape it to (out_ch, in_ch, 3, 3))
    txt = 'yolov2.conv%d.W.data[...] = dat[%d:%d].reshape(%d, %d, %d, %d)' % (i+1, offset, offset+(out_ch*in_ch*ksize*ksize), out_ch, in_ch, ksize, ksize)
    offset+= (out_ch*in_ch*ksize*ksize)
    exec(txt)
    print(i+1, offset)

# load last convolution weight(Load only Biases and Convolution2Ds)
in_ch = 1024
out_ch = last_out
ksize = 1

if n_classes == original_n_classes and n_boxes == original_n_boxes:
    txt = 'yolov2.bias%d.b.data[...] = dat[%d:%d]' % (i+2, offset, offset+out_ch)
    offset+=out_ch
    exec(txt)

    txt = 'yolov2.conv%d.W.data[...] = dat[%d:%d].reshape(%d, %d, %d, %d)' % (i+2, offset, offset+(out_ch*in_ch*ksize*ksize), out_ch, in_ch, ksize, ksize)
    offset+=out_ch*in_ch*ksize*ksize
    exec(txt)
    print(i+2, offset)
    assert len(dat) == offset
else:
    original_last_out = (original_n_classes + 5) * original_n_boxes
    assert len(dat) == offset + original_last_out + original_last_out * in_ch * ksize * ksize

print('save weights file to {}'.format(args.out_file))
serializers.save_npz(args.out_file, yolov2)
