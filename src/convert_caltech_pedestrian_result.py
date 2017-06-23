import argparse
import json
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str, help='Input JSON file path')
    parser.add_argument('output_path', type=str, help='Output directory path')
    return parser.parse_args()

def main():
    args = parse_args()
    output_path = args.output_path
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    results = {}
    with open(args.input_path) as f:
        input_results = json.load(f)
    for result in input_results:
        file_path = result['file_path']
        head, file_name = os.path.split(file_path)
        head, seq_name = os.path.split(head)
        head, set_name = os.path.split(head)
        frame = int(os.path.splitext(file_name)[0]) + 1
        if not results.has_key(set_name):
            results[set_name] = {}
        if not results[set_name].has_key(seq_name):
            results[set_name][seq_name] = []
        boxes = results[set_name][seq_name]
        for region in result['regions']:
            if region['category'] != 'person':
                continue
            x, y, w, h = region['bbox']
            confidence = region['confidence']
            boxes.append((frame, x, y, w, h, confidence))
    for set_name, seqs in results.items():
        set_path = os.path.join(output_path, set_name)
        if not os.path.exists(set_path):
            os.mkdir(set_path)
        for seq_name, boxes in seqs.items():
            seq_path = os.path.join(set_path, '{}.txt'.format(seq_name))
            with open(seq_path, 'w') as f:
                boxes = sorted(boxes, key=lambda x: x[0])
                for box in boxes:
                    f.write(','.join(map(str, box)) + '\n')

if __name__ == '__main__':
    main()
