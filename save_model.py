#!/usr/bin/python3

from os import mkdir;
from os.path import exists, join;
from shutil import rmtree;
import tensorflow as tf;
from models import CANet;

def main():
  
  canet = CANet(nshot, pretrain = 'resnet50.h5', drop_rate = 0.3);
  optimizer = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.ExponentialDecay(0.0025, decay_steps = 110000, decay_rate = 0.9));
  checkpoint = tf.train.Checkpoint(model = canet, optimizer = optimizer);
  checkpoint.restore(tf.train.latest_checkpoint('checkpoints'));
  if False == exists('models'): mkdir('models');
    canet.save(join('models', 'canet.h5'));
    canet.save_weights(join('models', 'canet_weights.h5'));
  
if __name__ == "__main__":

  assert tf.executing_eagerly();
  main();
