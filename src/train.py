#  Copyright 2018 Algolux Inc. All Rights Reserved.
from __future__ import division
import tensorflow as tf
import numpy as np
import os
import LSGAN as lsgan
import dataset_util as dsutil


def run(results_dir, model_dir, base_dir, train_file_names, eval_file_names, num_epochs, data_type,
        use_multi_scale=False,
        exported_disc_path=None, use_3dconv=False, smooth_weight=0.5, lrate=1e-4, adv_weight=0.0001, min_distance=3.,
        max_distance=150.):
    train_fns = train_file_names
    val_fns = eval_file_names

    print('num train: %d' % len(train_fns))
    print('num val: %d' % len(val_fns))

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    save_summary_freq = 100
    sess = tf.Session()

    in_image = tf.placeholder(tf.float32, [None, None, None, 3])

    gt_image = tf.placeholder(tf.float32, [None, None, None, 1])
    gt_mask = None
    if data_type == 'real':
        gt_mask = tf.placeholder(tf.float32, [None, None, None, 1])
        model = lsgan.build_model(in_image, gt_image, data_type=data_type, gt_mask=gt_mask,
                                  smooth_weight=smooth_weight, adv_weight=adv_weight,
                                  discriminator_ckpt=exported_disc_path,
                                  use_multi_scale=use_multi_scale, use_3dconv=use_3dconv, lrate=lrate)
    else:
        model = lsgan.build_model(in_image, gt_image, data_type=data_type, gt_mask=gt_mask,
                                  smooth_weight=smooth_weight, adv_weight=adv_weight, use_multi_scale=use_multi_scale,
                                  use_3dconv=use_3dconv, lrate=lrate)

    out_image = model['out_image']
    disc_in = model['disc_in']
    scaled_losses = model['scaled_losses']

    with tf.variable_scope('Generator_loss'):
        l1_loss_sum = tf.summary.scalar('L1', model['l1_loss'])
        grad_loss_sum = tf.summary.scalar('TVariation', model['grad_loss'])
        adv_loss_sum = tf.summary.scalar('Adversarial', model['adv_loss'])
        G_loss_sum = tf.summary.scalar('Total', model['gloss'])
        if len(scaled_losses) > 0:
            l1_half_sum = tf.summary.scalar('L1_half_scale', scaled_losses[0])
            l1_fourth_sum = tf.summary.scalar('L1_fourth_scale', scaled_losses[1])

    with tf.variable_scope('Discriminator_loss'):
        D_real_loss_sum = tf.summary.scalar('Real', model['D_loss_real'])
        D_fake_loss_sum = tf.summary.scalar('Fake', model['D_loss_fake'])
        D_loss_sum = tf.summary.scalar('Total', model['dloss'])

    tf.summary.image('gt_image', gt_image)
    tf.summary.image('output', out_image)
    tf.summary.image('output_clamped', tf.clip_by_value(out_image, 0.0, 1.0))

    with tf.variable_scope('in_image'):
        for i in range(in_image.get_shape()[3]):
            tf.summary.image('chan_%d' % i, in_image[:, :, :, i:(i + 1)])

    train_summary = tf.summary.merge_all()

    if len(scaled_losses) > 0:
        loss_sum = tf.summary.merge([G_loss_sum, D_loss_sum, l1_loss_sum, l1_half_sum, l1_fourth_sum,
                                     grad_loss_sum, adv_loss_sum, D_real_loss_sum, D_fake_loss_sum])
    else:
        loss_sum = tf.summary.merge([G_loss_sum, D_loss_sum, l1_loss_sum, grad_loss_sum,
                                     adv_loss_sum, D_real_loss_sum, D_fake_loss_sum])

    saver = tf.train.Saver(max_to_keep=2)
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(model_dir)

    if ckpt:
        print('loaded ' + ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)

    train_writer = tf.summary.FileWriter(os.path.join(results_dir, 'train'), sess.graph)
    val_writer = tf.summary.FileWriter(os.path.join(results_dir, 'val'))
    global_cnt = 0

    for epoch in range(num_epochs):
        print('epoch: %d ' % epoch)
        cnt = 0
        for ind in np.random.permutation(len(train_fns)):
            global_cnt += 1
            train_fn = train_fns[ind]
            img_id = train_fn
            gta_pass = ''

            in_img = dsutil.read_gated_image(base_dir, gta_pass, img_id, data_type)
            gt_patch, lidar_mask = dsutil.read_gt_image(base_dir, gta_pass, img_id, data_type, min_distance,
                                                        max_distance)
            cnt += 1
            input_patch = in_img
            gt_patch = gt_patch

            if not global_cnt % save_summary_freq:
                out_for_disc = sess.run(out_image, feed_dict={in_image: input_patch})
                if data_type == 'real':
                    summaries, _, G_current = sess.run([train_summary, model['gtrain_op'], model['gloss']],
                                                       feed_dict={in_image: input_patch, gt_image: gt_patch,
                                                                  gt_mask: lidar_mask, disc_in: out_for_disc})
                else:
                    _, G_current = sess.run([model['dtrain_op'], model['dloss']],
                                            feed_dict={gt_image: gt_patch, disc_in: out_for_disc})
                    summaries, _, G_current = sess.run([train_summary, model['gtrain_op'], model['gloss']],
                                                       feed_dict={in_image: input_patch, gt_image: gt_patch,
                                                                  disc_in: out_for_disc})

                print("writing summaries")
                train_writer.add_summary(summaries, global_cnt)
                train_writer.flush()
                idx = np.random.randint(0, len(val_fns))
                val_fn = val_fns[idx]
                img_id = val_fn
                gta_pass = ''

                if data_type == 'real':
                    in_img = dsutil.read_gated_image(base_dir, gta_pass, img_id, data_type)
                    gt_patch, lidar_mask = dsutil.read_gt_image(base_dir, gta_pass, img_id, data_type, min_distance, max_distance)
                    input_patch = in_img
                    out_for_disc = sess.run(out_image, feed_dict={in_image: input_patch})
                    val_loss = sess.run(loss_sum, feed_dict={in_image: input_patch, gt_image: gt_patch,
                                                             gt_mask: lidar_mask, disc_in: out_for_disc})

                else:
                    in_img = dsutil.read_gated_image(base_dir, gta_pass, img_id, data_type)
                    gt_patch, _ = dsutil.read_gt_image(base_dir, gta_pass, img_id, data_type, min_distance, max_distance)
                    input_patch = in_img
                    out_for_disc = sess.run(out_image, feed_dict={in_image: input_patch})
                    val_loss = sess.run(loss_sum,
                                        feed_dict={in_image: input_patch, gt_image: gt_patch, disc_in: out_for_disc})

                val_writer.add_summary(val_loss, global_cnt)
                val_writer.flush()

            else:
                if data_type == 'real':
                    out_for_disc = sess.run(out_image, feed_dict={in_image: input_patch})
                    _, G_current = sess.run([model['gtrain_op'], model['gloss']],
                                            feed_dict={in_image: input_patch, gt_image: gt_patch, gt_mask: lidar_mask,
                                                       disc_in: out_for_disc})
                else:
                    out_for_disc = sess.run(out_image, feed_dict={in_image: input_patch})
                    _, G_current = sess.run([model['dtrain_op'], model['dloss']],
                                            feed_dict={in_image: input_patch, gt_image: gt_patch,
                                                       disc_in: out_for_disc})
                    print("%d %d Loss=%.3f" % (epoch, cnt, G_current))
                    _, G_current = sess.run([model['gtrain_op'], model['gloss']],
                                            feed_dict={in_image: input_patch, gt_image: gt_patch,
                                                       disc_in: out_for_disc})
            print("%d %d Loss=%.3f" % (epoch, cnt, G_current))

        saver.save(sess, os.path.join(model_dir, 'model.ckpt'), global_step=global_cnt)
