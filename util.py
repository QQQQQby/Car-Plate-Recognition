# coding: utf-8

from typing import List, Tuple


def find_waves(histogram, threshold, min_width=2):
    up_index = -1
    is_peak = False
    if histogram[0] > threshold:
        up_index = 0
        is_peak = True
    wave_list = []
    for i, x in enumerate(histogram):
        if is_peak and x < threshold:
            if i - up_index >= min_width:
                is_peak = False
                wave_list.append((up_index, i))
        elif not is_peak and x >= threshold:
            is_peak = True
            up_index = i
    if is_peak and up_index != -1 and i - up_index >= min_width:
        wave_list.append((up_index, i))
    return wave_list


def resized_size(size: Tuple[int, int] or List[int, int],
                 target: Tuple[int, int] or List[int, int],
                 mode='scale') -> Tuple[int, int]:
    if mode == 'scale':
        ratio = min(target[0] / size[0], target[1] / size[1])
        return int(size[0] * ratio), int(size[1] * ratio)
    elif mode == 'fill':
        return target[0], target[1]
    else:
        raise ValueError
