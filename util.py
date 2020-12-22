# coding: utf-8

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
