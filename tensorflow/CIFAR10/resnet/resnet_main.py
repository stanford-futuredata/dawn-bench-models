# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""ResNet Train/Eval module.
"""
import os
import six
import subprocess
import sys
import time

import cifar_input
import numpy as np
import resnet_model
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dataset', 'cifar10', 'cifar10 or cifar100.')
tf.app.flags.DEFINE_string('mode', 'train', 'train or eval.')
tf.app.flags.DEFINE_string('model', '', 'model to train.')
tf.app.flags.DEFINE_string('data_format', 'NHWC',
                           """Data layout to use: NHWC (TF native)
                              or NCHW (cuDNN native).""")
tf.app.flags.DEFINE_string('train_data_path', '',
                           'Filepattern for training data.')
tf.app.flags.DEFINE_string('eval_data_path', '',
                           'Filepattern for eval data')
tf.app.flags.DEFINE_integer('image_size', 32, 'Image side length.')
tf.app.flags.DEFINE_string('train_dir', '',
                           'Directory to keep training outputs.')
tf.app.flags.DEFINE_string('eval_dir', '',
                           'Directory to keep eval outputs.')
tf.app.flags.DEFINE_integer('eval_batch_count', 50,
                            'Number of batches to eval.')
tf.app.flags.DEFINE_bool('eval_once', False,
                         'Whether evaluate the model only once.')
tf.app.flags.DEFINE_string('log_root', '',
                           'Should be a parent directory of FLAGS.train_dir/eval_dir.')
tf.app.flags.DEFINE_string('checkpoint_dir', '',
                           'Directory to store the checkpoints')
tf.app.flags.DEFINE_integer('num_gpus', 0,
                            'Number of gpus used for training. (0 or 1)')
tf.app.flags.DEFINE_bool('use_bottleneck', False,
                         'Use bottleneck module or not.')
tf.app.flags.DEFINE_bool('time_inference', False,
                         'Time inference.')
tf.app.flags.DEFINE_integer('batch_size', -1,
                            'Batch size to use.')


def train(hps):
  """Training loop."""
  images, labels = cifar_input.build_input(
      FLAGS.dataset, FLAGS.train_data_path, hps.batch_size, FLAGS.mode, hps.data_format)
  model = resnet_model.ResNet(hps, images, labels, FLAGS.mode)
  model.build_graph()

  param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
      tf.get_default_graph(),
      tfprof_options=tf.contrib.tfprof.model_analyzer.
          TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
  sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)

  tf.contrib.tfprof.model_analyzer.print_model_analysis(
      tf.get_default_graph(),
      tfprof_options=tf.contrib.tfprof.model_analyzer.FLOAT_OPS_OPTIONS)

  truth = tf.argmax(model.labels, axis=1)
  predictions = tf.argmax(model.predictions, axis=1)
  precision = tf.reduce_mean(tf.to_float(tf.equal(predictions, truth)))

  summary_hook = tf.train.SummarySaverHook(
      save_steps=100,
      output_dir=FLAGS.train_dir,
      summary_op=tf.summary.merge([model.summaries,
                                   tf.summary.scalar('Precision', precision)]))

  num_steps_per_epoch = 391  # TODO: Don't hardcode this.

  logging_hook = tf.train.LoggingTensorHook(
      tensors={'step': model.global_step,
               'loss': model.cost,
               'precision': precision},
      every_n_iter=100)

  class _LearningRateSetterHook(tf.train.SessionRunHook):
    """Sets learning_rate based on global step."""

    def begin(self):
      self._lrn_rate = 0.01

    def before_run(self, run_context):
      return tf.train.SessionRunArgs(
          model.global_step,  # Asks for global step value.
          feed_dict={model.lrn_rate: self._lrn_rate})  # Sets learning rate

    def after_run(self, run_context, run_values):
      train_step = run_values.results
      if train_step < num_steps_per_epoch:
        self._lrn_rate = 0.01
      elif train_step < (91 * num_steps_per_epoch):
        self._lrn_rate = 0.1
      elif train_step < (136 * num_steps_per_epoch):
        self._lrn_rate = 0.01
      elif train_step < (181 * num_steps_per_epoch):
        self._lrn_rate = 0.001
      else:
        self._lrn_rate = 0.0001

  class _SaverHook(tf.train.SessionRunHook):
    """Sets learning_rate based on global step."""

    def begin(self):
      self.saver = tf.train.Saver(max_to_keep=10000)
      subprocess.call("rm -rf %s; mkdir -p %s" % (FLAGS.checkpoint_dir,
                                                  FLAGS.checkpoint_dir), shell=True)
      self.f = open(os.path.join(FLAGS.checkpoint_dir, "times.log"), 'w')

    def after_create_session(self, sess, coord):
      self.sess = sess
      self.start_time = time.time()

    def before_run(self, run_context):
      return tf.train.SessionRunArgs(
          model.global_step  # Asks for global step value.
      )

    def after_run(self, run_context, run_values):
      train_step = run_values.results
      epoch = train_step / num_steps_per_epoch
      if train_step % num_steps_per_epoch == 0:
        end_time = time.time()
        directory = os.path.join(FLAGS.checkpoint_dir, ("%5d" % epoch).replace(' ', '0'))
        subprocess.call("mkdir -p %s" % directory, shell=True)
        ckpt_name = 'model.ckpt'
        self.saver.save(self.sess, os.path.join(directory, ckpt_name),
                        global_step=train_step)
        self.f.write("Step: %d\tTime: %s\n" % (train_step, end_time - self.start_time))
        print("Saved checkpoint after %d epoch(s) to %s..." % (epoch, directory))
        sys.stdout.flush()
        self.start_time = time.time()

    def end(self, sess):
      self.f.close()

  with tf.train.MonitoredTrainingSession(
      checkpoint_dir=FLAGS.log_root,
      hooks=[logging_hook, _LearningRateSetterHook()],
      chief_only_hooks=[summary_hook, _SaverHook()],
      save_checkpoint_secs=None,
      # Since we provide a SummarySaverHook, we need to disable default
      # SummarySaverHook. To do that we set save_summaries_steps to 0.
      save_summaries_steps=None,
      save_summaries_secs=None,
      config=tf.ConfigProto(allow_soft_placement=True)) as mon_sess:
    for i in range(num_steps_per_epoch * 181):
      mon_sess.run(model.train_op)

def evaluate(hps):
  """Eval loop."""
  images, labels = cifar_input.build_input(
      FLAGS.dataset, FLAGS.eval_data_path, hps.batch_size, FLAGS.mode, hps.data_format)
  model = resnet_model.ResNet(hps, images, labels, FLAGS.mode)
  model.build_graph()
  saver = tf.train.Saver()
  summary_writer = tf.summary.FileWriter(FLAGS.eval_dir)

  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
  tf.train.start_queue_runners(sess)

  best_precision = 0.0
  while True:
    try:
      ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)
    except tf.errors.OutOfRangeError as e:
      tf.logging.error('Cannot restore checkpoint: %s', e)
      continue
    if not (ckpt_state and ckpt_state.model_checkpoint_path):
      tf.logging.info('No model to eval yet at %s', FLAGS.log_root)
      break
    tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
    saver.restore(sess, ckpt_state.model_checkpoint_path)

    global_step = ckpt_state.model_checkpoint_path.split('/')[-1].split('-')[-1]
    if not global_step.isdigit():
      global_step = 0
    else:
      global_step = int(global_step)

    total_prediction, correct_prediction, correct_prediction_top5 = 0, 0, 0
    start_time = time.time()
    for _ in six.moves.range(FLAGS.eval_batch_count):
      (summaries, loss, predictions, truth, train_step) = sess.run(
          [model.summaries, model.cost, model.predictions,
           model.labels, model.global_step])

      if not FLAGS.time_inference:
        for (indiv_truth, indiv_prediction) in zip(truth, predictions):
          indiv_truth = np.argmax(indiv_truth)
          top5_prediction = np.argsort(indiv_prediction)[-5:]
          top1_prediction = np.argsort(indiv_prediction)[-1]
          correct_prediction += (indiv_truth == top1_prediction)
          if indiv_truth in top5_prediction:
            correct_prediction_top5 += 1
          total_prediction += 1

    if FLAGS.time_inference:
      print("Time for inference: %.4f" % (time.time() - start_time))
    else:
      precision = 1.0 * correct_prediction / total_prediction
      precision_top5 = 1.0 * correct_prediction_top5 / total_prediction
      best_precision = max(precision, best_precision)

      precision_summ = tf.Summary()
      precision_summ.value.add(
          tag='Precision', simple_value=precision)
      summary_writer.add_summary(precision_summ, train_step)
      best_precision_summ = tf.Summary()
      best_precision_summ.value.add(
          tag='Best Precision', simple_value=best_precision)
      summary_writer.add_summary(best_precision_summ, train_step)
      summary_writer.add_summary(summaries, train_step)
      print('Precision @ 1 = %.4f, Recall @ 5 = %.4f, Global step = %d' %
            (precision, precision_top5, global_step))
      summary_writer.flush()

    if FLAGS.eval_once:
      break

    time.sleep(60)


def main(_):
  if FLAGS.model == '':
    raise Exception('--model must be specified.')

  if FLAGS.num_gpus == 0:
    dev = '/cpu:0'
  elif FLAGS.num_gpus == 1:
    dev = '/gpu:0'
  else:
    raise ValueError('Only support 0 or 1 gpu.')

  if FLAGS.batch_size == -1:
    if FLAGS.mode == 'train':
      batch_size = 128
    elif FLAGS.mode == 'eval':
      batch_size = 100
  else:
    batch_size = FLAGS.batch_size

  if FLAGS.dataset == 'cifar10':
    num_classes = 10
  elif FLAGS.dataset == 'cifar100':
    num_classes = 100

  if FLAGS.model == 'resnet20':
    num_residual_units = 3
  elif FLAGS.model == 'resnet56':
    num_residual_units = 9
  elif FLAGS.model == 'resnet164' and FLAGS.use_bottleneck:
    num_residual_units = 18
  elif FLAGS.model == 'resnet164' and not FLAGS.use_bottleneck:
    num_residual_units = 27
  else:
    raise Exception("Invalid model -- only resnet20, resnet56 and resnet164 supported")

  data_format = FLAGS.data_format

  hps = resnet_model.HParams(batch_size=batch_size,
                             num_classes=num_classes,
                             min_lrn_rate=0.0001,
                             lrn_rate=0.1,
                             num_residual_units=num_residual_units,
                             use_bottleneck=FLAGS.use_bottleneck,
                             weight_decay_rate=0.0005,
                             relu_leakiness=0.1,
                             optimizer='mom',
                             data_format=data_format)

  with tf.device(dev):
    if FLAGS.mode == 'train':
      train(hps)
    elif FLAGS.mode == 'eval':
      evaluate(hps)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
