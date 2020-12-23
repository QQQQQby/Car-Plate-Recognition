# coding: utf-8

import argparse

from detector import PlateDetector


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for Car Plate Recognition.")
    parser.add_argument('--chinese_cnn_path', type=str, default='./models/chinese/79.pth',
                        help='Path to CNN that classifies Chinese characters.')
    parser.add_argument('--others_cnn_path', type=str, default='./models/others/49.pth',
                        help='Path to CNN that classifies letters and digits.')
    parser.add_argument('--image_path', type=str, default='',
                        help='Image path.')
    parser.add_argument('--show_process', action='store_true', default=False,
                        help="Show process when predicting.")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    detector = PlateDetector(chinese_cnn_path=args.chinese_cnn_path, others_cnn_path=args.others_cnn_path,
                             image_path=args.image_path, show_process=args.show_process)
    detector.find_plate_location()
    detector.split_characters()
    detector.classify_characters()
    print(detector.result_list)
