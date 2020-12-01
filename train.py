#!/usr/bin/python3

import tensorflow as tf;
from create_datasets import parse_function;
from models import CANet;

nshot = 5;

def main():

  canet = CANet(nshot, pretrain = 'resnet50.h5', drop_rate = 0.3);
  optimizer = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.ExponentialDecay(0.0025, decay_steps = 110000, decay_rate = 0.9));
  checkpoint = tf.train.Checkpoint(model = canet, optimizer = optimizer);
  train_loss = tf.keras.metrics.Mean(name = 'train loss', dtype = tf.float32);
  train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name = 'train accuracy');
  test_loss = tf.keras.metrics.Mean(name = 'test loss', dtype = tf.float32);
  test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name = 'test accuracy');
  # TODO
  
if __name__ == "__main__":

  assert tf.executing_eagerly();
  main();
