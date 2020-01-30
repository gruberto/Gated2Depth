#  Copyright 2018 Algolux Inc. All Rights Reserved.
import numpy as np
import math

min_val = 1e-7


def threshold(y1, y2, thr=1.25):
    max_ratio = np.maximum(y1 / y2, y2 / y1)
    return np.mean(max_ratio < thr, dtype=np.float64) * 100.


def rmse(y1, y2):
    diff = y1 - y2
    return math.sqrt(np.mean(diff * diff, dtype=np.float64))


def rmse_log(y1, y2):
    return rmse(np.log(y1), np.log(y2))


def ard(y1, y2):
    return np.mean(np.abs(y1 - y2) / y2, dtype=np.float64)


def mae(y1, y2):
    return np.mean(np.abs(y1 - y2), dtype=np.float64)


def calc_metrics(output, groundtruth, min_distance=3., max_distance=150.):
    output = output[groundtruth > 0]
    groundtruth = groundtruth[groundtruth > 0]
    output = np.clip(output, min_distance, max_distance)
    groundtruth = np.clip(groundtruth, min_distance, max_distance)

    return rmse(output, groundtruth), rmse_log(output, groundtruth), \
           ard(output, groundtruth), mae(output, groundtruth), \
           threshold(output, groundtruth, thr=1.25), \
           threshold(output, groundtruth, thr=1.25 ** 2), threshold(output, groundtruth, thr=1.25 ** 3)

metric_str = ['rmse', 'rmse_log', 'ard', 'mae', 'delta1', 'delta2', 'delta3']

if __name__ == '__main__':
    y = np.array(range(10, 130)).reshape((10, 12))
    noise = np.ones_like(y) * 10
    noise[np.random.random(size=noise.shape) > 0.5] *= -2
    y_noisy = y + noise
    print(y)
    print(y_noisy)
    print(calc_metrics(y_noisy, y))
