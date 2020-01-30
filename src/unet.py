#  Copyright 2018 Algolux Inc. All Rights Reserved.
from __future__ import division
#import os, time#, scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
#import numpy as np
#import rawpy


def lrelu(x):
    return tf.maximum(x * 0.2, x)


def upsample_and_concat(x1, x2, output_channels, in_channels):
    pool_size = 2
    deconv_filter = tf.Variable(tf.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02))
    deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, pool_size, 1])

    deconv_output = tf.concat([deconv, x2], 3)
    deconv_output.set_shape([None, None, None, output_channels * 2])

    return deconv_output


def build_unet(input_img):
    pool_size = 2

    with tf.variable_scope('unet'):
        conv1 = slim.conv2d(input_img, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_1')
        conv1 = slim.conv2d(conv1, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_2')
        pool1 = slim.max_pool2d(conv1, [pool_size, pool_size], padding='SAME')

        conv2 = slim.conv2d(pool1, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_1')
        conv2 = slim.conv2d(conv2, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_2')
        pool2 = slim.max_pool2d(conv2, [pool_size, pool_size], padding='SAME')

        conv3 = slim.conv2d(pool2, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_1')
        conv3 = slim.conv2d(conv3, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_2')
        pool3 = slim.max_pool2d(conv3, [pool_size, pool_size], padding='SAME')

        conv4 = slim.conv2d(pool3, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_1')
        conv4 = slim.conv2d(conv4, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_2')
        pool4 = slim.max_pool2d(conv4, [pool_size, pool_size], padding='SAME')

        conv5 = slim.conv2d(pool4, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_1')
        conv5 = slim.conv2d(conv5, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_2')

        up6 = upsample_and_concat(conv5, conv4, 256, 512)
        conv6 = slim.conv2d(up6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_1')
        conv6 = slim.conv2d(conv6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_2')

        up7 = upsample_and_concat(conv6, conv3, 128, 256)
        conv7 = slim.conv2d(up7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_1')
        conv7 = slim.conv2d(conv7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_2')

        up8 = upsample_and_concat(conv7, conv2, 64, 128)
        conv8 = slim.conv2d(up8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_1')
        conv8 = slim.conv2d(conv8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_2')

        up9 = upsample_and_concat(conv8, conv1, 32, 64)
        conv9 = slim.conv2d(up9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_1')
        conv9 = slim.conv2d(conv9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_2')

        out = slim.conv2d(conv9, 1, [1, 1], rate=1, activation_fn=None, scope='g_conv10')
    return out, up8, up7


def upsample_and_concat3d(x1, x2, output_channels, in_channels):
    pool_size = 2
    deconv_filter = tf.Variable(
        tf.truncated_normal([1, pool_size, pool_size, output_channels, in_channels], stddev=0.02))
    deconv = tf.nn.conv3d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, 1, pool_size, pool_size, 1])
    deconv_output = tf.concat([deconv, x2], 4)
    deconv_output.set_shape([None, 1, None, None, output_channels * 2])

    return deconv_output


def _variable(name, shape):
    var = tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())
    return var


def conv3d(l_input, w, b, name):
    return tf.nn.bias_add(tf.nn.conv3d(l_input, w, strides=[1, 1, 1, 1, 1], padding='SAME', name=name), b)


def lrelu_conv3d(l_input, w, b, name):
    return lrelu(conv3d(l_input, w, b, name))


def max_pool3d(l_input, k, name):
    return tf.nn.max_pool3d(l_input, ksize=[1, k, 2, 2, 1], strides=[1, k, 2, 2, 1], padding='SAME', name=name)


def build_3d_conv_unet(input_img):
    pool_size = 2

    with tf.variable_scope('unet'):
        input_img = tf.expand_dims(input_img, axis=1)

        conv1 = lrelu_conv3d(input_img, _variable('wconv1_1', [2, 3, 3, 3, 32]), _variable('bconv1_1', [32]),
                             name='g_conv1_1')
        conv2 = lrelu_conv3d(conv1, _variable('wconv1_2', [3, 3, 3, 32, 32]), _variable('bconv1_2', [32]),
                             name='g_conv1_2')
        pool1 = max_pool3d(conv2, 1, name='pool1')

        conv2 = lrelu_conv3d(pool1, _variable('wconv2_1', [3, 3, 3, 32, 64]), _variable('bconv2_1', [64]),
                             name='g_conv2_1')
        conv2 = lrelu_conv3d(conv2, _variable('wconv2_2', [3, 3, 3, 64, 64]), _variable('bconv2_2', [64]),
                             name='g_conv2_2')
        pool2 = max_pool3d(conv2, 1, name='pool2')

        conv3 = lrelu_conv3d(pool2, _variable('wconv3_1', [3, 3, 3, 64, 128]), _variable('bconv3_1', [128]),
                             name='g_conv3_1')
        conv3 = lrelu_conv3d(conv3, _variable('wconv3_2', [3, 3, 3, 128, 128]), _variable('bconv3_2', [128]),
                             name='g_conv3_2')
        pool3 = max_pool3d(conv3, 1, name='pool3')

        conv4 = lrelu_conv3d(pool3, _variable('wconv4_1', [3, 3, 3, 128, 256]), _variable('bconv4_1', [256]),
                             name='g_conv4_1')
        conv4 = lrelu_conv3d(conv4, _variable('wconv4_2', [3, 3, 3, 256, 256]), _variable('bconv4_2', [256]),
                             name='g_conv4_2')
        pool4 = max_pool3d(conv4, 1, name='pool4')

        conv5 = lrelu_conv3d(pool4, _variable('wconv5_1', [3, 3, 3, 256, 512]), _variable('bconv5_1', [512]),
                             name='g_conv5_1')
        conv5 = lrelu_conv3d(conv5, _variable('wconv5_2', [3, 3, 3, 512, 512]), _variable('bconv5_2', [512]),
                             name='g_conv5_2')

        up6 = upsample_and_concat3d(conv5, conv4, 256, 512)
        conv6 = lrelu_conv3d(up6, _variable('wconv6_1', [3, 3, 3, 512, 256]), _variable('bconv6_1', [256]),
                             name='g_conv6_1')
        conv6 = lrelu_conv3d(conv6, _variable('wconv6_2', [3, 3, 3, 256, 256]), _variable('bconv6_2', [256]),
                             name='g_conv6_2')

        up7 = upsample_and_concat3d(conv6, conv3, 128, 256)
        conv7 = lrelu_conv3d(up7, _variable('wconv7_1', [3, 3, 3, 256, 128]), _variable('bconv7_1', [128]),
                             name='g_conv7_1')
        conv7 = lrelu_conv3d(conv7, _variable('wconv_2', [3, 3, 3, 128, 128]), _variable('bconv7_2', [128]),
                             name='g_conv7_2')

        up8 = upsample_and_concat3d(conv7, conv2, 64, 128)
        conv8 = lrelu_conv3d(up8, _variable('wconv8_1', [3, 3, 3, 128, 64]), _variable('bconv8_1', [64]),
                             name='g_conv8_1')
        conv8 = lrelu_conv3d(conv8, _variable('wconv8_2', [3, 3, 3, 64, 64]), _variable('bconv8_2', [64]),
                             name='g_conv8_2')

        up9 = upsample_and_concat3d(conv8, conv1, 32, 64)
        conv9 = lrelu_conv3d(up9, _variable('wconv9_1', [3, 3, 3, 64, 32]), _variable('bconv9_1', [32]),
                             name='g_conv9_1')
        conv9 = lrelu_conv3d(conv9, _variable('wconv9_2', [3, 3, 3, 32, 32]), _variable('bconv9_2', [32]),
                             name='g_conv9_2')

        out = conv3d(conv9, _variable('wconv10_1', [3, 3, 3, 32, 1]), _variable('bconv10_1', [1]), name='g_conv10')

    return tf.squeeze(out, axis=1), tf.squeeze(up8, axis=1), tf.squeeze(up7, axis=1)


def network(input_img, use_multi_scale, use_3dconv=False):
    if use_multi_scale:
        print('Using mult-scale')
        if use_3dconv:
            print('Using 3D convs')
            out, conv8, conv7 = build_3d_conv_unet(input_img)
        else:
            out, conv8, conv7 = build_unet(input_img)
        out2 = tf.contrib.layers.conv2d(conv8, 1, [3, 3], rate=1, padding='SAME',
                                        activation_fn=lrelu, scope='out2', biases_initializer=None)
        out3 = tf.contrib.layers.conv2d(conv7, 1, [3, 3], rate=1, padding='SAME',
                                        activation_fn=lrelu, scope='out3', biases_initializer=None)
        out = {'output': out, 'half_scale': out2, 'fourth_scale': out3}
        return out

    else:
        print('not using multi-scale')
        if use_3dconv:
            print('Using 3D convs')
            out, _, _ = build_3d_conv_unet(input_img)
        else:
            out, _, _ = build_unet(input_img)
        out = {'output': out}
        return out
