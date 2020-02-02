import cv2
import os
import numpy as np

import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import cv2


def colorize_pointcloud(depth, min_distance=3, max_distance=80, radius=3):
    norm = mpl.colors.Normalize(vmin=min_distance, vmax=max_distance)
    cmap = cm.jet
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    pos = np.argwhere(depth > 0)

    pointcloud_color = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)
    for i in range(pos.shape[0]):
        color = tuple([int(255 * value) for value in m.to_rgba(depth[pos[i, 0], pos[i, 1]])[0:3]])
        cv2.circle(pointcloud_color, (pos[i, 1], pos[i, 0]), radius, (color[0], color[1], color[2]), -1)

    return pointcloud_color


def load_rgb(root_dir, sample, camera='left'):
    path = os.path.join(root_dir, 'rgb_{}_8bit'.format(camera), sample + '.png')
    img = cv2.imread(path)

    return img


def load_gated(root_dir, sample, slice):
    path = os.path.join(root_dir, 'gated{}_10bit'.format(slice), sample + '.png')
    img = cv2.imread(path, -1)
    img = np.right_shift(img, 2).astype(np.uint8) # convert from 10bit to 8bit

    return img


def load_projected_lidar(root_dir, sample, frame='gated'):
    path = os.path.join(root_dir, 'depth_hdl64_{}_compressed'.format(frame), sample + '.npz')
    depth = np.load(path)['arr_0']

    return depth


def load_lidar(root_dir, sample):
    path = os.path.join(root_dir, 'depth_hdl64', sample + '.bin')
    scan = np.fromfile(path, dtype=np.float32)
    pc = np.transpose(scan.reshape((-1, 5))[:, 0:3])

    return pc


def project_lidar(pc, frame='gated'):
    if frame == 'gated':
        shape = (720, 1280)
        mat44 = np.array(
            [[0.02597508, -0.99964206, 0.00640694, -0.03133254],
            [0.00202653, -0.00635643, -0.99997774, -0.44874239],
            [0.99966054, 0.02598749, 0.00186069, -0.7072447],
            [0., 0., 0., 1.]]
        )
        P = np.array(
            [[2322.4, 0.0, 667.777, 0.0],
             [0.0, 2322.4, 261.144, 0.0],
             [0.0, 0.0, 1.0, 0.0]]
        )
    elif frame == 'rgb_left':
        shape = (1024, 1920)
        mat44 = np.array(
            [[0.00816556, -0.99996549, -0.00153346, 0.08749686],
            [-0.00715544, 0.00147505, -0.99997331, -0.41640329],
            [0.99994106, 0.00817631, -0.00714315, -0.6955578],
            [0., 0., 0., 1.]]
        )
        P = np.array(
            [[2355.7228, 0.0, 988.138054, 0.0],
            [0.0, 2355.7228, 508.051838, 0.0],
            [0.0, 0.0, 1.0, 0.0]]
        )

    elif frame == 'rgb_right':
        shape = (1024, 1920)
        mat44 = np.array(
            [[0.00816556, -0.99996549, -0.00153346, 0.08749686],
             [-0.00715544, 0.00147505, -0.99997331, -0.41640329],
             [0.99994106, 0.00817631, -0.00714315, -0.6955578],
             [0., 0., 0., 1.]]
        )
        P = np.array(
            [[2355.7228, 0.0, 988.138054, -478.200589],
             [0.0, 2355.7228, 508.051838, 0.0],
             [0.0, 0.0, 1.0, 0.0]]
        )

    depth_map = np.zeros(shape, dtype=np.float32)

    pc = np.r_[pc, np.ones((1, pc.shape[1]))]
    pc = np.dot(mat44, pc)
    pc_proj = np.dot(P, pc)

    u = np.array(np.round(pc_proj[0, :] / pc_proj[2, :]).astype(int)).flatten()
    v = np.array(np.round(pc_proj[1, :] / pc_proj[2, :]).astype(int)).flatten()
    depth = np.array(pc_proj[2, :]).flatten()

    valid = np.logical_and(depth > 1, np.logical_and(np.logical_and(u >= 0, u < depth_map.shape[1]),
                                   np.logical_and(v >= 0, v < depth_map.shape[0])))

    u = u[valid]
    v = v[valid]
    depth = depth[valid]
    depth_map[v, u] = depth

    return depth_map


if __name__ == '__main__':
    root_dir = '/mnt/fs1/Gated2Depth/data/real'
    sample = '00001'

    rgb_left = load_rgb(root_dir, sample, camera='left')
    rgb_right = load_rgb(root_dir, sample, camera='right')
    gated = load_gated(root_dir, sample, slice=0)
    load_gated(root_dir, sample, slice=1)
    load_gated(root_dir, sample, slice=2)

    load_projected_lidar(root_dir, sample, frame='rgb_left')
    load_projected_lidar(root_dir, sample, frame='gated')

    pc_lidar = load_lidar(root_dir, sample)
    pc_lidar_rgb_left = project_lidar(pc_lidar, frame='rgb_left')
    pc_lidar_rgb_right = project_lidar(pc_lidar, frame='rgb_right')
    pc_lidar_gated = project_lidar(pc_lidar, frame='gated')

    lidar_rgb_left_color = colorize_pointcloud(pc_lidar_rgb_left)
    overlay_rgb_left = cv2.addWeighted(rgb_left, 0.5, lidar_rgb_left_color, 0.5, 0.0)
    cv2.imwrite('overlay_rgb_left.png', overlay_rgb_left)

    lidar_rgb_right_color = colorize_pointcloud(pc_lidar_rgb_right)
    overlay_rgb_right = cv2.addWeighted(rgb_right, 0.5, lidar_rgb_right_color, 0.5, 0.0)
    cv2.imwrite('overlay_rgb_right.png', overlay_rgb_right)

    lidar_gated_color = colorize_pointcloud(pc_lidar_gated)
    gated = cv2.cvtColor(gated, cv2.COLOR_GRAY2BGR)
    overlay_gated = cv2.addWeighted(gated, 0.5, lidar_gated_color, 0.5, 0.0)
    cv2.imwrite('overlay_gated.png', overlay_gated)