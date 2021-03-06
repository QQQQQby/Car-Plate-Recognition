# coding: utf-8

import os
import threading
import cv2
import matplotlib.pyplot as plt
import numpy as np

from classifier import Classifier
from util import find_waves


class PlateDetector:
    def __init__(self, chinese_cnn_path, others_cnn_path, image_path=None, show_process=False):
        if not os.path.exists(chinese_cnn_path) or not os.path.exists(others_cnn_path):
            raise ValueError("CNN path does not exist.")
        self.chinese_classifier = Classifier(chinese_cnn_path, is_chinese=True)
        self.others_classifier = Classifier(others_cnn_path, is_chinese=False)

        self.show_process = show_process

        self.img = None if not image_path else cv2.imread(image_path)
        self.img_after_detected = None
        self.img_plate = None
        self.character_image_list = None
        self.result_list = None

    def find_plate_location(self):
        if self.img is None:
            return None

        # Preprocess
        img_blur = cv2.GaussianBlur(self.img, (5, 5), 0)
        img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
        self.show_image("img_gray", img_gray)

        kernel = np.ones((20, 20), np.uint8)
        img_open = cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, kernel)
        self.show_image("img_open", img_open)

        img_open_weight = cv2.addWeighted(img_gray, 1, img_open, -1, 0)
        self.show_image("img_open_weight", img_open_weight)

        ret, img_binary = cv2.threshold(img_open_weight, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
        self.show_image("img_binary", img_binary)

        img_edge = cv2.Canny(img_binary, 100, 200)
        self.show_image("img_edge", img_edge)

        kernel = np.ones((4, 19), np.uint8)
        img_edge = cv2.morphologyEx(img_edge, cv2.MORPH_CLOSE, kernel)
        img_edge = cv2.morphologyEx(img_edge, cv2.MORPH_OPEN, kernel)
        self.show_image("img_edge_processed", img_edge)

        # Find Contours
        image, contours, hierarchy = cv2.findContours(img_edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 2000]

        # Find a contour that matches our demand
        plate_rect = None
        self.img_after_detected = self.img.copy()
        for index, contour in enumerate(contours):
            rect = cv2.minAreaRect(contour)  # [center(x,y), (w,h), angle of rotation]
            w, h = rect[1]
            if w < h:
                w, h = h, w
            scale = w / h
            if h >= 20 and 2 < scale < 8:
                plate_rect = rect
                box = np.int32(cv2.boxPoints(rect))
                cv2.drawContours(self.img_after_detected, [box], 0, (0, 0, 255), 2)
                break
        if plate_rect is None:
            return None
        print("Plate rect:", plate_rect)
        self.show_image("img_detected", self.img_after_detected)

        # Rotate the rectangle
        rotate_matrix = cv2.getRotationMatrix2D(plate_rect[0], plate_rect[2], 1.0)
        self.img_plate = cv2.warpAffine(self.img, rotate_matrix, (self.img.shape[1], self.img.shape[0]))

        w, h = plate_rect[1]
        x1, x2 = int(plate_rect[0][0] - w / 2), int(plate_rect[0][0] + w / 2)
        y1, y2 = int(plate_rect[0][1] - h / 2), int(plate_rect[0][1] + h / 2)
        self.img_plate = self.img_plate[y1:y2, x1:x2]
        if w < h:
            self.img_plate = np.rot90(self.img_plate)
        self.show_image("plate", self.img_plate)

    def split_characters(self):
        if self.img_plate is None:
            return None

        # Preprocess
        h, w = self.img_plate.shape[:2]
        img_gray = cv2.cvtColor(self.img_plate, cv2.COLOR_BGR2GRAY)
        self.show_image('gray', img_gray)

        _, img_threshold = cv2.threshold(img_gray, 120, 255, cv2.THRESH_BINARY)
        self.show_image('threshold', img_threshold)

        white_array = img_threshold.sum(0) / 255
        white_mean_array = np.convolve(white_array, np.ones(3) / 3, mode='same')

        # Find waves and fix
        white_threshold = (white_mean_array.mean() + white_mean_array.min()) / 2
        wave_list = find_waves(white_mean_array, white_threshold)
        distance = 5
        for idx, wave in enumerate(wave_list):
            i, j = wave
            range_i = range(max(i - distance, 0), min(i + distance + 1, w))
            range_j = range(max(j - distance, 0), min(j + distance + 1, w))
            i, j = (white_mean_array[range_i].argmin() + range_i[0],
                    white_mean_array[range_j].argmin() + range_j[0])
            wave_list[idx] = (i, j)

        # Filter small waves
        wave_list = list(filter(lambda x: x[1] - x[0] > 3, wave_list))

        # Merge some waves
        width_sum = sum([wave[1] - wave[0] for wave in wave_list])
        merge_threshold = w / 24
        for i in range(len(wave_list) - 2, -1, -1):
            # Merge wave[i] and wave[i+1], and remove wave[i+1], if possible
            wave_temp = (wave_list[i][0], wave_list[i + 1][1])
            if abs((wave_temp[1] - wave_temp[0]) - width_sum / 7) < merge_threshold:
                wave_list[i] = wave_temp
                del wave_list[i + 1]
        print("Waves:", wave_list)

        if self.show_process:
            plt.figure(figsize=(10, 10))
            plt.plot(white_array)
            plt.plot(white_mean_array)
            plt.axhline(white_threshold, ls="-", c="green")
            for index_range in wave_list:
                i1, i2 = index_range
                plt.plot((i1, i1, i2, i2), [1, 0, 0, 1], 'r-')
            plt.plot()
            plt.yticks(range(int(white_array.max()) + 1))
            plt.grid()
            plt.show()

        # Generate character images
        self.character_image_list = []
        for idx, wave in enumerate(wave_list):
            self.character_image_list.append(self.img_plate[:, wave[0]:wave[1], :])
            self.show_image('character_' + str(idx), self.character_image_list[idx])

    def classify_characters(self):
        if self.character_image_list is None:
            return None

        # Preprocess
        processed_image_list = []
        for idx, img in enumerate(self.character_image_list):
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = img_gray.shape
            if h > w:
                padding_left = (h - w) // 2
                padding_right = h - w - padding_left
                img_gray = np.pad(img_gray, ((0, 0), (padding_left, padding_right)), 'constant')
            else:
                padding_top = (w - h) // 2
                padding_bottom = w - h - padding_top
                img_gray = np.pad(img_gray, ((padding_top, padding_bottom), (0, 0)), 'constant')
            img_gray = cv2.resize(img_gray, dsize=(20, 20))
            _, img_bin = cv2.threshold(img_gray, 110, 255, cv2.THRESH_BINARY)
            self.show_image('bin_' + str(idx), img_bin)
            processed_image_list.append(img_bin)

        self.result_list = [self.chinese_classifier.predict(processed_image_list[0:1])[0]]
        self.result_list += self.others_classifier.predict(processed_image_list[1:])
        print('Plate numbers:', ''.join(self.result_list))

    def load_img(self, path):
        self.img = cv2.imread(path)
        self.img_after_detected = None
        self.img_plate = None
        self.character_image_list = None
        self.result_list = None

    def clear_img(self):
        self.img = None
        self.img_after_detected = None
        self.img_plate = None
        self.character_image_list = None
        self.result_list = None

    def show_image(self, window_name, img):
        def show(window_name, img):
            cv2.imshow(window_name, img)
            cv2.waitKey(60 * 1000)

        if self.show_process:
            threading.Thread(target=show, args=(window_name, img)).start()
