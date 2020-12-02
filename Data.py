#!/usr/bin/python3

from os.path import join;
from random import sample;
import numpy as np;
import cv2;
import tensorflow as tf;
from pycocotools.coco import COCO;

class Data(object):

  def __init__(self, trainset_dir = None, testset_dir = None, anno_dir = None):

    self.annotations = {'train': COCO(join(anno_dir, 'instances_train2017.json')),
                        'test': COCO(join(anno_dir, 'instances_val2017.json'))};
    self.dirs = {'train': trainset_dir,
                 'test': testset_dir}

  @tf.function
  def preprocess(self, data, label):
    
    data = tf.cast(data, dtype = tf.float32);
    label = tf.cast(label, dtype = tf.float32);
    # 1) random hsv
    data = tf.expand_dims(data, axis = 0); # data.shape = (1, h, w, 3)
    data = tf.image.random_hue(data, 10 / 180);
    data = tf.image.random_saturation(data, 0, 10);
    data = tf.image.random_brightness(data, 10 / 255);
    # 2) random flip
    comp = tf.concat([data, tf.reshape(label, (1, tf.shape(label)[0], tf.shape(label)[1], 1))], axis = -1); # comp.shape = (1, h, w, 3 + 1)
    comp = tf.cond(tf.math.greater(tf.random.uniform(shape = ()), 0.5), lambda: comp, lambda: tf.image.flip_left_right(comp)); # comp.shape = (1, h, w, 3 + 1)
    data = comp[...,:-1]; # data.shape = (1, h, w, 3)
    label = comp[...,-1:]; # label.shape = (1, h, w, 1)
    # 3) random scale
    scale = tf.random.uniform(minval = 0.5, maxval = 2.0, shape = (), dtype = tf.float32);
    shape = tf.cast([float(tf.shape(data)[1]) * scale, float(tf.shape(data)[2]) * scale], dtype = tf.int32);
    data = tf.image.resize(data, shape, method = tf.image.ResizeMethod.BICUBIC); # data.shape = (1, s*h, s*w, 3)
    label = tf.image.resize(label, shape, method = tf.image.ResizeMethod.NEAREST_NEIGHBOR); # label.shape = (1, s*h, s*w, 1)
    # 4) random crop
    comp = tf.concat([data, label], axis = -1); # comp.shape = (1, s*h, s*w, 3+1)
    crop_h = tf.math.minimum(tf.shape(comp)[1], 512);
    crop_w = tf.math.minimum(tf.shape(comp)[2], 512);
    crop_c = tf.shape(comp)[3];
    comp = tf.image.random_crop(comp, (1, crop_h, crop_w, crop_c)); # data.shape = (1, min(512, s*h), min(512, s*w), 3+1)
    data = comp[...,:-1]; # data.shape = (1, min(512, s*h), min(512, s*w), 3)
    label = comp[...,-1:]; # label.shape = (1, min(512, s*h), min(512, s*w), 1)
    # 5) rescale to 512x512
    data = tf.image.resize(data, (512, 512), method = tf.image.ResizeMethod.BICUBIC); # data.shape = (1, 512, 512, 3)
    label = tf.image.resize(label, (512, 512), method = tf.image.ResizeMethod.NEAREST_NEIGHBOR); # label.shape = (1, 512, 512, 1)
    # 6) squeeze
    data = tf.squeeze(data, axis = 0) / 255.; # data.shape = (512, 512, 3)
    label = tf.reshape(label, (tf.shape(label)[1], tf.shape(label)[2])); # label.shape = (512, 512)
    return data, label;

  def getBatch(self, nshot = 5, nquery = 5, ds = 'train'):
    
    assert ds in ['train', 'test'];
    # pick up a category
    train_cids = self.annotations[ds].getCatIds();
    cid = sample(train_cids, 1);
    # pick up several images containing objects of this category
    img_ids = self.annotations[ds].getImgIds(catIds = cid);
    img_ids = sample(img_ids, nshot + nquery);
    # generate a batch of samples
    imgs = list();
    masks = list();
    for img_id in img_ids:
      img_info = self.annotations[ds].loadImgs([img_id])[0];
      annIds = self.annotations[ds].getAnnIds(imgIds = img_id, catIds = cid);
      anns = self.annotations[ds].loadAnns(annIds);
      img = cv2.imread(join(self.dirs[ds], img_info['file_name']));
      mask = np.zeros((img_info['height'], img_info['width']));
      for ann in anns:
        instance_mask = self.annotations[ds].annToMask(ann);
        mask = np.maximum(mask, instance_mask);
      # preprocess
      img, mask = self.preprocess(img, mask)
      imgs.append(img);
      masks.append(mask);
    supp = tf.stack(imgs[:nshot], axis = 0); # supp.shape = (nshot, 256, 256, 3)
    supp_lb = tf.stack(masks[:nshot], axis = 0); # supp_lb.shape = (nshot, 256, 256)
    qry = tf.stack(imgs[nshot:], axis = 0); # qry.shape = (nquery, 256, 256, 3)
    qry_lb = tf.stack(masks[nshot:], axis = 0); # qry_lb.shape = (nquery, 256, 256)
    return supp, supp_lb, qry, qry_lb;
    
  def getTrainBatch(self, nshot = 5, nquery = 5):
    
    return self.getBatch(nshot, nquery, 'train');
    
  def getTestBatch(self, nshot = 5, nquery = 5):
    
    return self.getBatch(nshot, nquery, 'test');

if __name__ == "__main__":

  assert tf.executing_eagerly();
