#  Copyright 2018 Algolux Inc. All Rights Reserved.
import tensorflow as tf
import tensorflow.contrib.slim as slim
import unet
import numpy as np
from export import export_subgraph, reload_exported_disc
import sys
import os


def generator(image, use_multi_scale, use_3dconv):
    with tf.variable_scope('generator'):
        return unet.network(image, use_multi_scale, use_3dconv)


def discriminator(image, reuse=False, for_G=False):
    with tf.variable_scope('discriminator'):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        df_dim = 64

        net = slim.conv2d(image, df_dim, [4, 4], padding='VALID', stride=2, activation_fn=tf.nn.leaky_relu)
        net = slim.conv2d(net, df_dim * 2, [4, 4], padding='VALID', stride=2, activation_fn=tf.nn.leaky_relu)
        net = slim.conv2d(net, df_dim * 4, [4, 4], padding='VALID', stride=2, activation_fn=tf.nn.leaky_relu)
        net = slim.conv2d(net, df_dim * 8, [4, 4], padding='VALID', stride=2, activation_fn=tf.nn.leaky_relu)
        logits = slim.conv2d(net, 1, [4, 4], stride=2, activation_fn=None)
        n_patches = tf.shape(logits)[1] * tf.shape(logits)[2]
        logits = tf.reshape(logits, [-1, n_patches])
        return logits


def tv_loss(input, g_output):
    dy_out, dx_out = tf.image.image_gradients(g_output)
    dy_out = tf.abs(dy_out)
    dx_out = tf.abs(dx_out)
    dy_input, dx_input = tf.image.image_gradients(tf.reduce_mean(input, axis=3, keepdims=True))
    ep_dy = tf.exp(-tf.abs(dy_input))
    ep_dx = tf.exp(-tf.abs(dx_input))
    grad_loss = tf.reduce_mean(tf.multiply(dy_out, ep_dy) + tf.multiply(dx_out, ep_dx))
    return grad_loss


def loss_l2(D_logits_real, D_logits_fake, input, g_output, targets, data_type, gt_mask, smooth_weight, adv_weight,
            use_multi_scale, multi_scale_weights=[0.8, 0.6]):  # multi_scale_weights=[0.8, 0.6]): #adv_weight=0.0001):

    grad_loss = tv_loss(input, g_output['output'])
    # tf.reduce_mean(tf.multiply(dy_out, ep_dy) + tf.multiply(dx_out, ep_dx))

    if data_type == 'synthetic':
        l1_loss = tf.reduce_mean(tf.abs(g_output['output'] - targets))
    else:
        l1_loss = tf.reduce_sum(tf.multiply(tf.abs(g_output['output'] - targets), gt_mask)) / \
                  (tf.reduce_sum(gt_mask) + np.finfo(float).eps)

    scaled_losses = []

    if use_multi_scale:
        new_targets = []
        new_masks = []
        new_outs = []

        targets2 = slim.avg_pool2d(targets, [2, 2], stride=2, scope='targets2')
        targets3 = slim.avg_pool2d(targets, [4, 4], stride=4, scope='targets3')

        if data_type != 'synthetic':
            mask2 = slim.max_pool2d(gt_mask, [2, 2], stride=2, scope='mask2')
            mask3 = slim.max_pool2d(gt_mask, [4, 4], stride=4, scope='mask3')
            new_masks.append(mask2)
            new_masks.append(mask3)

            avg_mask2 = slim.avg_pool2d(gt_mask, [2, 2], stride=2, scope='avg_mask2')
            avg_mask3 = slim.avg_pool2d(gt_mask, [4, 4], stride=4, scope='avg_mask3')

            targets2 = tf.div_no_nan(targets2, avg_mask2)
            targets3 = tf.div_no_nan(targets3, avg_mask3)

        new_targets.append(targets2)
        new_targets.append(targets3)

        new_outs.append(g_output['half_scale'])
        new_outs.append(g_output['fourth_scale'])

        for i, weight in enumerate(multi_scale_weights):
            if data_type == 'synthetic':
                scaled_losses.append(weight * tf.reduce_mean(tf.abs(new_outs[i] - new_targets[i])))
                l1_loss = l1_loss + scaled_losses[-1]
            else:
                scaled_losses.append(
                    weight * tf.reduce_sum(tf.multiply(tf.abs(new_outs[i] - new_targets[i]), new_masks[i])) /
                    (tf.reduce_sum(new_masks[i]) + np.finfo(float).eps))
                l1_loss = l1_loss + scaled_losses[-1]

    adv_loss = tf.reduce_mean(tf.nn.l2_loss(D_logits_fake - tf.ones_like(D_logits_fake)))
    G_loss = l1_loss + smooth_weight * grad_loss + \
             adv_weight * adv_loss
    D_loss_real = tf.reduce_mean(tf.nn.l2_loss(D_logits_real - tf.ones_like(D_logits_real)))
    D_loss_fake = tf.reduce_mean(tf.nn.l2_loss(D_logits_fake - tf.zeros_like(D_logits_fake)))
    D_loss = 0.5 * D_loss_real + 0.5 * D_loss_fake

    return G_loss, D_loss, l1_loss, grad_loss, adv_loss, D_loss_real, D_loss_fake, scaled_losses


def train_ops(G_loss, D_loss, lrate, beta1=0.5, global_step=tf.Variable(0, trainable=False)):
    t_vars = tf.trainable_variables()
    G_vars = [var for var in t_vars if 'generator' in var.name]
    D_vars = [var for var in t_vars if 'discriminator' in var.name]

    G_optim = tf.train.AdamOptimizer(lrate, beta1=beta1)
    D_optim = tf.train.AdamOptimizer(lrate, beta1=beta1)

    G_grads = G_optim.compute_gradients(G_loss, var_list=G_vars)
    D_grads = D_optim.compute_gradients(D_loss, var_list=D_vars)

    G_train_op = G_optim.apply_gradients(G_grads, global_step=global_step)
    D_train_op = D_optim.apply_gradients(D_grads)

    return G_train_op, D_train_op


def build_model(input, targets, data_type='synthetic', gt_mask=None,
                smooth_weight=1e-4, adv_weight=0.0001,
                discriminator_ckpt=None, use_multi_scale=False,
                use_3dconv=False, lrate=1e-4):
    model = {}
    with tf.name_scope('generator'):
        g_output = generator(input, use_multi_scale, use_3dconv)
    with tf.name_scope('true_discriminator'):
        true_d_output = discriminator(targets)
    if not discriminator_ckpt is None:
        print('loading fake discriminator.')
        disc_in, fake_d_output = reload_exported_disc(discriminator_ckpt)
    else:
        print('Creating fake discriminator')
        with tf.name_scope('fake_discriminator'):
            # fake_d_output = discriminator(targets, reuse=True)
            disc_in = tf.placeholder(tf.float32, [None, None, None, 1])
            fake_d_output = discriminator(disc_in, reuse=True)
    with tf.name_scope('loss'):
        gloss, dloss, l1_loss, grad_loss, adv_loss, D_loss_real, D_loss_fake, scaled_losses = \
            loss_l2(true_d_output, fake_d_output, input, g_output, targets, data_type, gt_mask,
                    smooth_weight, adv_weight, use_multi_scale)
        gtrain_op, dtrain_op = train_ops(gloss, dloss, lrate=lrate)

    model['out_image'] = g_output['output']
    model['l1_loss'] = l1_loss
    model['grad_loss'] = grad_loss
    model['gloss'] = gloss
    model['dloss'] = dloss
    model['adv_loss'] = adv_loss
    model['D_loss_real'] = D_loss_real
    model['D_loss_fake'] = D_loss_fake
    model['gtrain_op'] = gtrain_op
    model['dtrain_op'] = dtrain_op
    model['fake_d_output'] = fake_d_output
    model['disc_in'] = disc_in
    model['scaled_losses'] = scaled_losses
    return model


def export_discriminator(checkpoint_dir, output_dir):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    export_subgraph(checkpoint_dir, 'fake_discriminator/discriminator/Reshape:0',
                    os.path.join(output_dir, 'disc.pb'))


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    export_discriminator(sys.argv[1], sys.argv[2])
