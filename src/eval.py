#  Copyright 2018 Algolux Inc. All Rights Reserved.
from __future__ import division
import tensorflow as tf
import numpy as np
import cv2
import os
from metrics import calc_metrics, metric_str
import LSGAN as lsgan
import dataset_util as dsutil
import visualize2D


def run(results_dir, model_dir, base_dir, file_names, data_type, use_multi_scale=False,
        exported_disc_path=None, use_3dconv=False, compute_metrics=False, min_distance=3., max_distance=150., show_result=False):
    sess = tf.Session()
    in_image = tf.placeholder(tf.float32, [None, None, None, 3])

    gt_image = tf.placeholder(tf.float32, [None, None, None, 1])
    gt_mask = None
    if data_type == 'real':
        gt_mask = tf.placeholder(tf.float32, [None, None, None, 1])
        model = lsgan.build_model(in_image, gt_image, data_type=data_type, gt_mask=gt_mask,
                                  smooth_weight=0.1, adv_weight=0.0001, discriminator_ckpt=exported_disc_path,
                                  use_multi_scale=use_multi_scale, use_3dconv=use_3dconv)
        min_eval_distance = min_distance
        max_eval_distance = 80.
    else:
        model = lsgan.build_model(in_image, gt_image, data_type=data_type, gt_mask=gt_mask, smooth_weight=1e-4,
                                  adv_weight=0.0001, use_multi_scale=use_multi_scale, use_3dconv=use_3dconv)
        min_eval_distance = min_distance
        max_eval_distance = max_distance

    out_image = model['out_image']

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, model_dir)

    per_image_metrics = []
    #mae = []

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    results_folder = ['gated2depth', 'gated2depth_img', 'all']
    for result_folder in results_folder:
        if not os.path.exists(os.path.join(results_dir, result_folder)):
            os.makedirs(os.path.join(results_dir, result_folder))

    for ind in range(len(file_names)):
        # get the path from image id
        train_fn = file_names[ind]
        if data_type == 'real':
            img_id = train_fn
            gta_pass = ''

        else:
            img_id = train_fn
            gta_pass = ''

        in_img = dsutil.read_gated_image(base_dir, gta_pass, img_id, data_type)

        input_patch = in_img
        output = sess.run(out_image, feed_dict={in_image: input_patch})
        output = np.clip(output * max_distance, min_distance, max_distance)

        gt_patch, _ = dsutil.read_gt_image(base_dir, gta_pass, img_id, data_type, raw_values_only=True, min_distance=min_distance, max_distance=max_distance)
        
        if compute_metrics:
            #if data_type != 'real':
                #curr_mae = np.mean(np.abs(output - gt_patch), dtype=np.float64)
            curr_metrics = calc_metrics(output[0, :, :, 0], gt_patch, min_distance=min_eval_distance,
                                            max_distance=max_eval_distance)
            per_image_metrics.append(curr_metrics)
                #mae.append(curr_mae)

           # else:
                #depth_lidar1, _ = dsutil.read_gt_image(base_dir, gta_pass, img_id, data_type, raw_values_only=True, min_distance=min_distance, max_distance=max_distance)
                #curr_metrics = calc_metrics(output[0, :, :, 0], gt_patch, min_distance=min_eval_distance,
                #                            max_distance=max_eval_distance)
                #per_image_metrics.append(curr_metrics)
                #mae.append(curr_metrics)

        np.savez_compressed(os.path.join(results_dir, 'gated2depth', '{}'.format(img_id)), output)

        #depth_lidar1, _ = dsutil.read_gt_image(base_dir, gta_pass, img_id, data_type, raw_values_only=True, min_distance=min_distance, max_distance=max_distance)
        
        if data_type != 'real':
            #print(depth_lidar1.shape)
            depth_lidar1_color = visualize2D.colorize_depth(gt_patch, min_distance=min_eval_distance, max_distance=max_eval_distance)
        else:
            #print(depth_lidar1.shape)
            depth_lidar1_color = visualize2D.colorize_pointcloud(gt_patch, min_distance=min_eval_distance,
                                                             max_distance=max_eval_distance, radius=3)

        depth_map_color = visualize2D.colorize_depth(output[0, :, :, 0], min_distance=min_eval_distance,
                                                     max_distance=max_eval_distance)

        in_out_shape = (int(depth_map_color.shape[0] + depth_map_color.shape[0] / 3. +
                            gt_patch.shape[0]), depth_map_color.shape[1], 3)

        input_output = np.zeros(shape=in_out_shape)
        scaled_input = cv2.resize(input_patch[0, :, :, :],
                                  dsize=(int(input_patch.shape[2] / 3), int(input_patch.shape[1] / 3)),
                                  interpolation=cv2.INTER_AREA) * 255

        for i in range(3):
            input_output[:scaled_input.shape[0], :scaled_input.shape[1], i] = scaled_input[:, :, 0]
            input_output[:scaled_input.shape[0], scaled_input.shape[1]: 2 * scaled_input.shape[1], i] = scaled_input[:,
                                                                                                        :, 1]
            input_output[:scaled_input.shape[0], scaled_input.shape[1] * 2:scaled_input.shape[1] * 3, i] = scaled_input[
                                                                                                           :, :, 2]

        input_output[scaled_input.shape[0]: scaled_input.shape[0] + depth_map_color.shape[0], :, :] = depth_map_color
        input_output[scaled_input.shape[0] + depth_map_color.shape[0]:, :, :] = depth_lidar1_color
        cv2.imwrite(os.path.join(results_dir, 'gated2depth_img', '{}.jpg'.format(img_id)), depth_map_color.astype(np.uint8))
        cv2.imwrite(os.path.join(results_dir, 'all', '{}.jpg'.format(img_id)), input_output.astype(np.uint8))

        if show_result:
            import matplotlib.pyplot as plt
            plt.imshow(cv2.cvtColor(input_output.astype(np.uint8), cv2.COLOR_BGR2RGB))
            plt.show()

    if compute_metrics:
        res = np.mean(per_image_metrics, axis=0)
        res_str = ''
        for i in range(res.shape[0]):
            res_str += '{}={:.2f} \n'.format(metric_str[i], res[i])
        print(res_str)
        with open(os.path.join(results_dir, 'results.txt'), 'w') as f:
            f.write(res_str)
        with open(os.path.join(results_dir, 'results.tex'), 'w') as f:
            f.write(' & '.join(metric_str) + '\n')
            f.write(' & '.join(['{:.2f}'.format(r) for r in res])) 
