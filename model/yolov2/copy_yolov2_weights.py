import argparse
import six

from chainer import serializers

from yolov2 import YOLOv2
from yolov2 import YOLOv2Classifier

partial_layer = 18

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str, help='Input file path')
    parser.add_argument('output_path', type=str, help='Output file path')
    parser.add_argument('input_class', type=int, help='Number of classes of input model')
    parser.add_argument('output_class', type=int, help='Number of classes of output model')
    parser.add_argument('box', type=int, help='Number of boxes of output model')
    return parser.parse_args()

def copy_conv_layer(src, dst, layer_num):
    for i in six.moves.range(layer_num):
        src_layer = getattr(src, 'conv{}'.format(i + 1))
        dst_layer = getattr(dst, 'conv{}'.format(i + 1))
        dst_layer.W = src_layer.W
        dst_layer.b = src_layer.b

def copy_bias_layer(src, dst, layer_num):
    for i in six.moves.range(layer_num):
        src_layer = getattr(src, 'bias{}'.format(i + 1))
        dst_layer = getattr(dst, 'bias{}'.format(i + 1))
        dst_layer.b = src_layer.b

def copy_bn_layer(src, dst, layer_num):
    for i in six.moves.range(layer_num):
        src_layer = getattr(src, 'bn{}'.format(i + 1))
        dst_layer = getattr(dst, 'bn{}'.format(i + 1))
        dst_layer.N = src_layer.N
        dst_layer.avg_var = src_layer.avg_var
        dst_layer.avg_mean = src_layer.avg_mean
        dst_layer.gamma = src_layer.gamma
        dst_layer.eps = src_layer.eps

def main():
    args = parse_args()

    print("loading classifier model...")
    input_model = YOLOv2Classifier(args.input_class)
    serializers.load_npz(args.input_path, input_model)

    model = YOLOv2(args.output_class, args.box)
    copy_conv_layer(input_model, model, partial_layer)
    copy_bias_layer(input_model, model, partial_layer)
    copy_bn_layer(input_model, model, partial_layer)

    print("saving model to %s" % (args.output_path))
    serializers.save_npz(args.output_path, model)

if __name__ == '__main__':
    main()
