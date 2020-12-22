# coding: utf-8

import argparse
import torch

from classifier import Classifier


def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN for car plate recognition.")
    parser.add_argument('--load_path', type=str, default='',
                        help='Load path to CNN.')
    parser.add_argument('--save_path', type=str, default='',
                        help='Save path to CNN.')
    parser.add_argument('--dataset_path', type=str, default='./data/train',
                        help='Dataset path.')
    parser.add_argument('--is_chinese', action='store_true', default=False,
                        help="Classify Chinese characters, either digits and numbers.")

    parser.add_argument('--train_batch_size', type=int, default=1000,
                        help='Batch size of train set.')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of epochs.')
    parser.add_argument('--method', type=str, default='adam',
                        help='Method of training.')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate.')
    parser.add_argument('--momentum', type=float, default=0,
                        help='Momentum.')

    parser.add_argument('--do_eval', action='store_true', default=False,
                        help="Whether to evaluate the model.")
    parser.add_argument('--train_proportion', type=float, default=0.7,
                        help='Proportion of train set.')
    parser.add_argument('--eval_batch_size', type=int, default=1000,
                        help='Batch size of evaluation set.')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    classifier = Classifier(load_path=args.load_path, dataset_path=args.dataset_path,
                            train_proportion=args.train_proportion, save_path=args.save_path, is_chinese=args.is_chinese)
    classifier.train(num_epochs=args.num_epochs, train_batch_size=args.train_batch_size,
                     method=args.method, lr=args.lr, momentum=args.momentum,
                     do_eval=args.do_eval, eval_batch_size=args.eval_batch_size)
