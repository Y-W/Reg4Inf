# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic evaluation script that evaluates a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import functools
import tensorflow as tf

import imagenet
import resnet_v1
import vgg_preprocessing
import noise

slim = tf.contrib.slim

tf.app.flags.DEFINE_integer(
    'batch_size', 100, 'The number of samples in each batch.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'eval_dir', None, 'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'validation', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 1,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', 224, 'Eval image size')

tf.app.flags.DEFINE_string('noise_type', None, 'Noise type')

tf.app.flags.DEFINE_string('noise_param', None, 'Noise parameter')

tf.app.flags.DEFINE_integer('eval_times', 1, 'number of evaluations to run per sample')

tf.app.flags.DEFINE_string(
    'model_name', 'resnet_v1_50', 'The name of the architecture to evaluate.')

FLAGS = tf.app.flags.FLAGS


def main(_):
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')
  if not FLAGS.checkpoint_path:
    raise ValueError('You must supply the model with --checkpoint_path')

  print('Model name: %s' % FLAGS.model_name)
  print('Checkpoint path: %s' % FLAGS.checkpoint_path)
  print('Eval times: %i' % FLAGS.eval_times)
  print('Noise type: %s' % FLAGS.noise_type)
  print('Noise Param: %s' % FLAGS.noise_param)

  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():
    tf_global_step = slim.get_or_create_global_step()

    ######################
    # Select the dataset #
    ######################
    dataset = imagenet.get_split(FLAGS.dataset_split_name, FLAGS.dataset_dir)

    ####################
    # Select the model #
    ####################
    noise_fn = noise.make_noise_fn(FLAGS.noise_type, FLAGS.noise_param)
    network_fn_maker = eval('resnet_v1.' + FLAGS.model_name)
    network_fn = functools.partial(network_fn_maker,
        num_classes=(dataset.num_classes - FLAGS.labels_offset),
        is_training=False, noise_fn = noise_fn)

    ##############################################################
    # Create a dataset provider that loads data from the dataset #
    ##############################################################
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        shuffle=False,
        common_queue_capacity=2 * FLAGS.batch_size,
        common_queue_min=FLAGS.batch_size)
    [image, label] = provider.get(['image', 'label'])
    label -= FLAGS.labels_offset

    eval_image_size = FLAGS.eval_image_size

    image = vgg_preprocessing.preprocess_image(image, eval_image_size, eval_image_size, is_training=False)

    images, labels = tf.train.batch(
        [image, label],
        batch_size=FLAGS.batch_size,
        num_threads=FLAGS.num_preprocessing_threads,
        capacity=5 * FLAGS.batch_size)
    labels = tf.squeeze(labels)

    images = tf.tile(images, (FLAGS.eval_times, 1, 1, 1), name='duplicated_images')
    


    ####################
    # Define the model #
    ####################
    logits, _ = network_fn(images)
    logits = tf.squeeze(logits)
    prob_output = tf.nn.softmax(logits)
    prob_list = tf.split(prob_output, FLAGS.eval_times, axis=0, name='prob_splits')

    variables_to_restore = slim.get_variables_to_restore()

    metric_maps = {}
    prev_sum = None
    for k in xrange(FLAGS.eval_times):
        if prev_sum is None:
            prev_sum = prob_list[k]
        else:
            prev_sum += prob_list[k]
        metric_maps['rep%i_Accuracy' % (k+1)] = slim.metrics.streaming_accuracy(tf.argmax(prev_sum, axis=1), labels)
        metric_maps['rep%i_Recall5' % (k+1)] = slim.metrics.streaming_recall_at_k(prev_sum, labels, 5)

    # Define the metrics:
    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map(metric_maps)

    # Print the summaries to screen.
    final_op_list = []
    for name, value in names_to_values.iteritems():
      summary_name = 'eval/%s' % name
      op = tf.Print(value, [value], message=summary_name)
      final_op_list.append(op)

    num_batches = math.ceil(dataset.num_samples / float(FLAGS.batch_size))
    assert num_batches * FLAGS.batch_size == dataset.num_samples

    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
      checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
      checkpoint_path = FLAGS.checkpoint_path

    tf.logging.info('Evaluating %s' % checkpoint_path)

    slim.evaluation.evaluate_once('',
        checkpoint_path=checkpoint_path,
        logdir=None,
        summary_op=None,
        num_evals=num_batches,
        eval_op=names_to_updates.values(),
        final_op=final_op_list,
        variables_to_restore=variables_to_restore)


if __name__ == '__main__':
  tf.app.run()
