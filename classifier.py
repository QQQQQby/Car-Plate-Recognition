# coding: utf-8

import os
import random
import time
import cv2
import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm

import modules


class Classifier:
    characters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                  'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
                  'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
                  'W', 'X', 'Y', 'Z',
                  '云', '京', '冀', '吉', '宁', '川', '新', '晋', '桂', '沪',
                  '津', '浙', '渝', '湘', '琼', '甘', '皖', '粤', '苏', '蒙',
                  '藏', '豫', '贵', '赣', '辽', '鄂', '闽', '陕', '青', '鲁',
                  '黑']
    character_dict = dict([(c, i) for i, c in enumerate(characters)])

    def __init__(self, load_path=None, dataset_path=None, train_proportion=0.8, save_path=None):
        self.cnn = modules.CNN1() if not load_path else torch.load(load_path)
        self.train_images, self.train_labels = ([], []) if not dataset_path else self.read_dataset(dataset_path)
        self.eval_images, self.eval_labels = ([], [])
        self.train_proportion = train_proportion
        self.save_path = save_path

    def train(self, num_epochs, train_batch_size=8, method='adam', lr=0.01, momentum=0, do_eval=True,
              eval_batch_size=8):
        assert train_batch_size > 0 and eval_batch_size > 0
        optimizer = self.get_optimizer(method=method, lr=lr, momentum=momentum)
        for epoch in range(num_epochs):
            self.shuffle_dataset()
            # Train
            print('-' * 20 + 'Training epoch %d' % epoch + '-' * 20, flush=True)
            time.sleep(0.1)

            num_correct = 0
            for start in tqdm(range(0, len(self.train_images), train_batch_size), desc='Training batch: '):
                images = self.train_images[start:start + train_batch_size]
                actual_labels = self.train_labels[start:start + train_batch_size]
                # Forward
                images = torch.tensor(np.array(images), dtype=torch.float32)
                outputs = self.cnn(images)
                # Backward
                batch_labels = torch.tensor(actual_labels, dtype=torch.int64)
                self.cnn.zero_grad()
                loss = nn.CrossEntropyLoss()(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                # Calculate metrics
                pred_labels = outputs.softmax(1).argmax(1).tolist()
                num_correct += np.equal(pred_labels, actual_labels).sum()
            print(num_correct / len(self.train_images))
            self.save_cnn(str(epoch) + '.pth')

            # Evaluate
            if not do_eval:
                continue
            num_correct = 0
            print('-' * 20 + 'Evaluating epoch %d' % epoch + '-' * 20, flush=True)
            time.sleep(0.1)
            for start in tqdm(range(0, len(self.eval_images), eval_batch_size), desc='Evaluating batch: '):
                images = self.eval_images[start:start + eval_batch_size]
                actual_labels = self.eval_labels[start:start + eval_batch_size]
                # Forward
                images = torch.tensor(images, dtype=torch.float32)
                outputs = self.cnn(images)
                # Get results
                pred_labels = outputs.softmax(1).argmax(1).tolist()
                num_correct += np.equal(pred_labels, actual_labels).sum()
            print(num_correct / len(self.eval_images))

    def get_optimizer(self, method='adam', lr=0.01, momentum=0):
        if method == 'sgd':
            return optim.SGD(self.cnn.parameters(), lr=lr, momentum=momentum)
        elif method == 'adam':
            return optim.Adam(self.cnn.parameters(), lr=lr)
        else:
            return None

    def shuffle_dataset(self):
        images = self.train_images + self.eval_images
        labels = self.train_labels + self.eval_labels
        seed = time.time()
        random.seed(seed)
        random.shuffle(images)
        random.seed(seed)
        random.shuffle(labels)
        split_index = int(self.train_proportion * len(images))
        self.train_images, self.train_labels = images[:split_index], labels[:split_index]
        self.eval_images, self.eval_labels = images[split_index:], labels[split_index:]

    def save_cnn(self, name):
        if not self.save_path:
            return None
        elif not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        torch.save(self.cnn, os.path.join(self.save_path, name))

    @classmethod
    def read_dataset(cls, path):
        print('-' * 20 + 'Reading data' + '-' * 20, flush=True)
        images, labels = [], []
        for character in tqdm(cls.characters):
            current_dir = os.path.join(path, character)
            for file_name in os.listdir(current_dir):
                file_path = os.path.join(current_dir, file_name)
                image = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                label = cls.character_dict[character]
                images.append(image)
                labels.append(label)
        return images, labels
