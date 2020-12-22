#!/usr/bin/python3

from os import mkdir, listdir;
from os.path import join, exists;
import tensorflow as tf;
from models import CANet;
from Data import Data;

nshot = 1;
nquery = 1;

def main(trainset_dir, testset_dir, anno_dir):

  canet = CANet(nshot, pretrain = 'resnet50.h5', drop_rate = 0);
  optimizer = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.ExponentialDecay(0.0025, decay_steps = 110000, decay_rate = 0.9));
  checkpoint = tf.train.Checkpoint(model = canet, optimizer = optimizer);
  train_loss = tf.keras.metrics.Mean(name = 'train loss', dtype = tf.float32);
  train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name = 'train accuracy');
  test_loss = tf.keras.metrics.Mean(name = 'test loss', dtype = tf.float32);
  test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name = 'test accuracy');
  data = Data(trainset_dir, testset_dir, anno_dir);
  # checkpoint
  if False == exists('checkpoints'): mkdir('checkpoints');
  checkpoint.restore(tf.train.latest_checkpoint('checkpoints'));
  # log
  log = tf.summary.create_file_writer('checkpoints');
  # train
  while True:
    supp, supp_lb, qry, qry_lb = data.getTrainBatch(nshot, nquery);
    with tf.GradientTape() as tape:
      preds = canet([qry, supp, supp_lb]);
      if tf.math.reduce_any(tf.math.logical_or(tf.math.is_nan(preds), tf.math.is_inf(preds))) == True:
        print('detected nan in preds, skip current iteration');
        continue;
      loss = tf.keras.losses.SparseCategoricalCrossentropy()(qry_lb, preds);
      if tf.math.reduce_any(tf.math.logical_or(tf.math.is_nan(loss), tf.math.is_inf(loss))) == True:
        print('detected nan in loss, skip current iteration');
        continue;
    grads = tape.gradient(loss, canet.trainable_variables);
    if tf.math.reduce_any([tf.math.reduce_any(tf.math.logical_or(tf.math.is_nan(grad), tf.math.is_inf(grad))) for grad in grads if grad is not None]) == True:
      print('detected nan in grads, skip current iterations');
      continue;
    optimizer.apply_gradients(zip(grads, canet.trainable_variables));
    train_loss.update_state(loss);
    train_accuracy.update_state(qry_lb, preds);
    if tf.equal(optimizer.iterations % 10000, 0):
      # save checkpoint
      checkpoint.save(join('checkpoints', 'ckpt'));
    if tf.equal(optimizer.iterations % 1000, 0):
      # evaluate
      for i in range(10):
        supp, supp_lb, qry, qry_lb = data.getTestBatch(nshot, nquery);
        preds = canet([qry, supp, supp_lb]);
        loss = tf.keras.losses.SparseCategoricalCrossentropy()(qry_lb, preds);
        test_loss.update_state(loss);
        test_accuracy.update_state(qry_lb, preds);
      # write log
      with log.as_default():
        tf.summary.scalar('train loss', train_loss.result(), step = optimizer.iterations);
        tf.summary.scalar('train accuracy', train_accuracy.result(), step = optimizer.iterations);
        tf.summary.scalar('test loss', test_loss.result(), step = optimizer.iterations);
        tf.summary.scalar('test accuracy', test_accuracy.result(), step = optimizer.iterations);
        seg = tf.argmax(preds[0:1,...], axis = -1); # cls.shape = (1, 256, 256)
        palette = tf.constant([[0,0,0],[255,255,255]], dtype = tf.int32); # palette.shape = (2, 3)
        colormap = tf.cast(tf.gather_nd(palette, tf.expand_dims(seg, axis = -1)), dtype = tf.float32); # colormap.shape = (1, 256, 256, 3)
        img = tf.cast(tf.clip_by_value(tf.math.rint(0.5 * colormap + 0.5 * qry[0:1, ...,::-1] * 255.), 0, 255), dtype = tf.uint8);
        tf.summary.image('segmentation', img, step = optimizer.iterations);
      print('Step #%d Train Loss: %.6f Train Accuracy: %.6f Test Loss: %.6f Test Accuracy: %.6f' % \
          (optimizer.iterations, train_loss.result(), train_accuracy.result(), test_loss.result(), test_accuracy.result()));
      # break condition
      if train_loss.result() < 0.01: break;
      # reset
      train_loss.reset_states();
      train_accuracy.reset_states();
      test_loss.reset_states();
      test_accuracy.reset_states();
  canet.save('canet.h5');
  
if __name__ == "__main__":

  assert tf.executing_eagerly();
  from sys import argv;
  if len(argv) != 4:
    print('Usage: %s </path/to/trainset> </path/to/testset> </path/to/annotation>' % argv[0]);
    exit(1);
  main(argv[1], argv[2], argv[3]);
