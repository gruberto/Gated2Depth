#  Copyright 2018 Algolux Inc. All Rights Reserved.
import glob
import itertools
import os
import time

import tensorflow as tf
import tensorflow.contrib.graph_editor as ge


def export_subgraph(checkpoint, output_tensors, saveto):
    """
    For the current graph, export the subgraph connected to output_tensors to a new graph_def file
    :param checkpoint: path to checkpoint
    :param output_tensors: output tensor names
    :param saveto: path to save graph_def file to
    :return:
    """
    saver = tf.train.import_meta_graph(checkpoint + '.meta', clear_devices=True)
    graph = tf.get_default_graph()
    if isinstance(output_tensors, str):
        output_tensors = [graph.get_tensor_by_name(output_tensors)]
    else:
        assert all([isinstance(out, str) for out in output_tensors])
        output_tensors = [graph.get_tensor_by_name(out) for out in output_tensors]

    def _var_ops(var_op):  # get operations one step ahead of variable ops: read/assign/etc.
        return [var_op.name] + [op.name for t in var_op.outputs for op in t.consumers()]

    keep_op_names = [out.op.name for out in output_tensors]
    var_ops = list({op for out in output_tensors for op in ge.get_backward_walk_ops(out) if op.type == 'VariableV2'})
    keep_op_names += [opname for op in var_ops for opname in _var_ops(op)]
    keep_op_names = [opname for opname in keep_op_names if 'save/' not in opname and 'save_' not in opname]
    graph_def = tf.graph_util.extract_sub_graph(graph.as_graph_def(), keep_op_names)

    with tf.Session() as sess:
        saver.restore(sess, checkpoint)
        new_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, graph_def, [out.op.name for out in output_tensors])
    tf.reset_default_graph()
    tf.train.export_meta_graph(saveto, graph_def=new_graph_def, clear_devices=True)


def reload_exported_generator(checkpoint):
    """
    For a given checkpoint from export_generator, return input and output tensors

    :param checkpoint: path to checkpoint
    :return: input, output
    """
    _ = tf.train.import_meta_graph(checkpoint, clear_devices=True, import_scope='cyclegan')
    graph = tf.get_default_graph()
    tensors = ge.get_tensors(graph)
    input_tensors = [t for t in tensors if 'input' in t.name and 'cyclegan/' in t.name]
    output_tensors = [t for t in tensors if 'output' in t.name and 'cyclegan/' in t.name]
    assert len(input_tensors) == 1
    assert len(output_tensors) == 1
    return input_tensors[0], output_tensors[0]


def reload_exported_disc(checkpoint):
    """
    For a given checkpoint from export_generator, return input and output tensors

    :param checkpoint: path to checkpoint
    :return: input, output
    """
    _ = tf.train.import_meta_graph(checkpoint, clear_devices=True)
    graph = tf.get_default_graph()
    tensors = ge.get_tensors(graph)
    input_tensors = [t for t in tensors if 'fake_discriminator/Placeholder:0' in t.name]
    output_tensors = [t for t in tensors if 'fake_discriminator/discriminator/Reshape:0' in t.name]
    assert len(input_tensors) == 1
    assert len(output_tensors) == 1
    return input_tensors[0], output_tensors[0]
