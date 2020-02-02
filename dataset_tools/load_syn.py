import cv2
import os
import numpy as np


def load_rgb(root_dir, sample):
    path = os.path.join(root_dir, 'rgb_left', sample + '.png')
    img = cv2.imread(path)

    return img


def load_gated(root_dir, sample, slice):
    path = os.path.join(root_dir, 'gated{}_10bit'.format(slice), sample + '.png')
    img = cv2.imread(path, -1)
    img = np.right_shift(img, 2).astype(np.uint8)  # convert from 10bit to 8bit

    return img


def load_depth(root_dir, sample):
    path = os.path.join(root_dir, 'depth_compressed', sample + '.npz')
    depth = np.load(path)['arr_0']

    return depth



if __name__ == '__main__':
    root_dir = '/mnt/fs1/Gated2Depth/data/syn'
    sample = '00000'

    load_rgb(root_dir, sample)
    load_gated(root_dir, sample, slice=0)
    load_gated(root_dir, sample, slice=1)
    load_gated(root_dir, sample, slice=2)
    load_depth(root_dir, sample)