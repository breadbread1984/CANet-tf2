#!/usr/bin/python3

import tensorflow as tf;

def Bottleneck(input_shape, filters, stride = 1, dilation = 1):

  # NOTE: either stride or dilation can be over 1
  inputs = tf.keras.Input(input_shape);
  residual = inputs;
  results = tf.keras.layers.Conv2D(filters, (1, 1), padding = 'same', use_bias = False)(inputs);
  results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.ReLU()(results);
  results = tf.keras.layers.Conv2D(filters, (3, 3), padding = 'same', strides = (stride, stride), dilation_rate = (dilation, dilation), use_bias = False)(results);
  results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.ReLU()(results);
  results = tf.keras.layers.Conv2D(filters * 4, (1, 1), padding = 'same', use_bias = False)(results);
  results = tf.keras.layers.BatchNormalization()(results);
  if stride != 1 or inputs.shape[-1] != results.shape[-1]:
    residual = tf.keras.layers.Conv2D(filters * 4, (1, 1), padding = 'same', strides = (stride, stride), use_bias = False)(residual);
    residual = tf.keras.layers.BatchNormalization()(residual);
  results = tf.keras.layers.Add()([results, residual]);
  results = tf.keras.layers.ReLU()(results);
  return tf.keras.Model(inputs = inputs, outputs = results);

def ResNetAtrous(layer_nums = [3, 4, 6, 3], dilations = [1, 2, 1]):

  strides = [2, 1, 1];
  assert layer_nums[-1] == len(dilations);
  assert len(layer_nums) == 1 + len(strides);
  inputs = tf.keras.Input((None, None, 3));
  results = tf.keras.layers.Conv2D(64, (7, 7), strides = (2,2), padding = 'same', use_bias = False)(inputs);
  results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.ReLU()(results);
  results = tf.keras.layers.MaxPool2D(pool_size = (3,3), strides = (2,2), padding = 'same')(results);
  def make_block(inputs, filters, layer_num, stride = 1, dilations = None):
    assert type(dilations) is list or dilations is None;
    results = inputs;
    for i in range(layer_num):
      results = Bottleneck(results.shape[1:], filters, stride = stride if i == 0 else 1, dilation = dilations[i] if dilations is not None else 1)(results);
    return results;
  outputs1 = make_block(results, 64, layer_nums[0]);
  outputs2 = make_block(outputs1, 128, layer_nums[1], stride = strides[0]);
  outputs3 = make_block(outputs2, 256, layer_nums[2], stride = strides[1], dilations = [1] * layer_nums[2]);
  outputs4 = make_block(outputs3, 512, layer_nums[3], stride = strides[2], dilations = dilations);
  return tf.keras.Model(inputs = inputs, outputs = (outputs1, outputs2, outputs3, outputs4));

def ResNet50Atrous():

  # NOTE: (3 + 4 + 6 + 3) * 3 + 2 = 50
  inputs = tf.keras.Input((None, None, 3));
  results = ResNetAtrous([3, 4, 6, 3], [1, 2, 1])(inputs);
  return tf.keras.Model(inputs = inputs, outputs = results, name = 'resnet50');

def ResNet101Atrous():

  # NOTE: (3 + 4 + 23 + 3) * 3 + 2 = 101
  inputs = tf.keras.Input((None, None, 3));
  results = ResNetAtrous([3, 4, 23, 3], [2, 2, 2])(inputs);
  return tf.keras.Model(inputs = inputs, outputs = results, name = 'resnet101');

def Attention(nshot):

  inputs = tf.keras.Input((None, None, nshot * 512)); # inputs.shape = (qn, h / 16, w / 16, nshot * 512)
  # 1) get masks for query images according to this support image
  outputs = tf.keras.layers.Conv2D(nshot * 256, (3, 3), padding = 'same', groups = nshot)(inputs); # outputs.shape = (qn, h / 16, w / 16, nshot * 256)
  outputs = tf.keras.layers.BatchNormalization()(outputs); # outputs.shape = (qn, h / 16, w / 16, nshot * 256)
  outputs = tf.keras.layers.ReLU()(outputs); # outputs.shape = (qn, h / 16, w / 16, nshot * 256)
  outputs = tf.keras.layers.Lambda(lambda x, n: tf.transpose(tf.reshape(x, (tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], n, 256)), (3, 0, 1, 2, 4)), arguments = {'n': nshot})(outputs); # outputs.shape = (nshot, qn, h/16, w/16, 256)
  # 2) get attention weights for query images of this support image
  att = tf.keras.layers.Conv2D(nshot * 256, (3, 3), padding = 'same', groups = nshot)(inputs); # att.shape = (qn, h / 16, w / 16, nshot * 256)
  att = tf.keras.layers.BatchNormalization()(att); # att.shape = (qn, h / 16, w / 16, nshot * 256)
  att = tf.keras.layers.ReLU()(att); # att.shape = (qn, h / 16, w / 16, nshot * 256)
  att = tf.keras.layers.MaxPool2D(pool_size = (3, 3), strides = (1, 1), padding = 'same')(att); # att.shape = (qn, h / 16, w / 16, nshot * 256)
  att = tf.keras.layers.Conv2D(nshot * 256, (3, 3), padding = 'same', groups = nshot)(att); # att.shape = (qn, h / 16, w / 16, nshot * 256)
  att = tf.keras.layers.BatchNormalization()(att); # att.shape = (qn, h / 16, w / 16, nshot * 256)
  att = tf.keras.layers.ReLU()(att); # att.shape = (qn, h / 16, w / 16, nshot * 256)
  att = tf.keras.layers.Lambda(lambda x, n: tf.transpose(tf.reshape(x, (tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], n, 256)), (3, 0, 1, 2, 4)), arguments = {'n': nshot})(att); # att.shape = (nshot, qn, h / 16, w / 16, 256)
  att = tf.keras.layers.Lambda(lambda x: tf.math.reduce_mean(x, axis = [2, 3, 4], keepdims = True))(att); # att.shape = (nshot, qn, 1, 1, 1)
  attended = tf.keras.layers.Lambda(lambda x: x[0] * x[1])([outputs, att]); # attended.shape = (nshot, qn, h / 16, w / 16, 256)
  return tf.keras.Model(inputs = inputs, outputs = attended, name = 'attention');

def DenseComparisonModule(nshot, pretrain = None):

  query = tf.keras.Input((None, None, 3)); # query.shape = (qn, h, w, 3)
  support = tf.keras.Input((None, None, 3), batch_size = nshot); # support.shape = (nshot, h, w, 3)
  labels = tf.keras.Input((None, None, 1), batch_size = nshot); # labels.shape = (nshot, h, w, 1)
  imgs_concat = tf.keras.layers.Concatenate(axis = 0)([support, query]); # imgs_concat.shape = (nshot + qn, h, w, 3)
  resnet50 = ResNet50Atrous();
  # load pretrained model
  if pretrain: resnet50.load_weights(pretrain);
  block1, block2, block3, block4 = resnet50(imgs_concat); # block2.shape = (nshot + qn, h / 8, w / 8, 512), block3.shape = (nshot + qn, h / 8, w / 8, 1024)
  img_fts = tf.keras.layers.Concatenate(axis = -1)([block2, block3]); # img_fts.shape = (nshot + qn, h / 8, w / 8, 512 + 1024)
  img_fts = tf.keras.layers.Conv2D(256, (3, 3), padding = 'same')(img_fts); # img_fts.shape = (nshot + qn, h / 8, w / 8, 256)
  img_fts = tf.keras.layers.BatchNormalization()(img_fts); # img_fts.shape = (nshot + qn, h / 8, w / 8, 256)
  img_fts = tf.keras.layers.ReLU()(img_fts); # img_fts.shape = (nshot + qn, h / 8, w / 8, 256)
  supp_fts, qry_fts = tf.keras.layers.Lambda(lambda x: tf.split(x[0], (tf.shape(x[1])[0], tf.shape(x[2])[0]), axis = 0))([img_fts, support, query]); # supp_fts.shape = (nshot, h / 8, w / 8, 256), qry_fts.shape = (qn, h / 8, w / 8, 256)
  supp_lb = tf.keras.layers.Lambda(lambda x: tf.image.resize(x[0], size = tf.shape(x[1])[1:3], method = tf.image.ResizeMethod.BILINEAR))([labels, img_fts]); # supp_lb.shape = (nshot, h / 8, w / 8, 256)
  proto = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x[0] * x[1], axis = (1,2)) / tf.math.maximum(tf.math.reduce_sum(x[1], axis = (1,2)), 1e-5))([supp_fts, supp_lb]); # proto.shape = (nshot, 256)
  proto = tf.keras.layers.Lambda(lambda x: tf.tile(tf.reshape(x[0], (tf.shape(x[0])[0], 1, 1, tf.shape(x[0])[1])), (1, tf.shape(x[1])[1], tf.shape(x[1])[2], 1)))([proto, qry_fts]); # proto.shape = (nshot, h / 8, w / 8, 256)
  qry_comp_fts = tf.keras.layers.Lambda(lambda x: tf.map_fn(lambda y: tf.concat([x[1], tf.tile(tf.expand_dims(y, axis = 0), (tf.shape(x[1])[0], 1, 1, 1))], axis = -1), x[0]))([proto, qry_fts]); # qry_comp_fts.shape = (nshot, qn, h/8, w/8, 512)
  qry_comp_fts = tf.keras.layers.Lambda(lambda x: tf.reshape(tf.transpose(x, (1,2,3,0,4)), (tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3], -1)))(qry_comp_fts); # qry_comp_fts.shape = (qn, h/8, w/8, nshot * 512)
  qry_comp_fts = Attention(nshot)(qry_comp_fts); # attention.shape = (nshot, qn, h / 8, w / 8, 256)
  weighted = tf.keras.layers.Softmax(axis = 0)(qry_comp_fts); # outputs.shape = (nshot, qn, h / 8, w / 8, 256)
  weighted_sum = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x, axis = 0))(weighted); # weighted_sum.shape = (qn, h / 8, w / 8, 256)
  return tf.keras.Model(inputs = (query, support, labels), outputs = weighted_sum);

def AtrousSpatialPyramidPooling(channel):

  inputs = tf.keras.Input((None, None, channel));
  # global pooling
  # results.shape = (batch, 1, 1, channel)
  results = tf.keras.layers.Lambda(lambda x: tf.math.reduce_mean(x, [1,2], keepdims = True))(inputs);
  # results.shape = (batch, 1, 1, 256)
  results = tf.keras.layers.Conv2D(256, kernel_size = (1,1), padding = 'same', kernel_initializer = tf.keras.initializers.he_normal(), use_bias = False)(results);
  results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.ReLU()(results);
  # pool.shape = (batch, height, width, 256)
  pool = tf.keras.layers.Lambda(lambda x: tf.image.resize(x[0], (tf.shape(x[1])[1], tf.shape(x[1])[2]), method = tf.image.ResizeMethod.BILINEAR))([results, inputs]);
  # results.shape = (batch, height, width, 256)
  results = tf.keras.layers.Conv2D(256, kernel_size = (1,1), dilation_rate = 1, padding = 'same', kernel_initializer = tf.keras.initializers.he_normal(), use_bias = False)(inputs);
  results = tf.keras.layers.BatchNormalization()(results);
  dilated_1 = tf.keras.layers.ReLU()(results);
  # results.shape = (batch, height, width, 256)
  results = tf.keras.layers.Conv2D(256, kernel_size = (3,3), dilation_rate = 6, padding = 'same', kernel_initializer = tf.keras.initializers.he_normal(), use_bias = False)(inputs);
  results = tf.keras.layers.BatchNormalization()(results);
  dilated_6 = tf.keras.layers.ReLU()(results);
  # results.shape = (batch, height, width, 256)
  results = tf.keras.layers.Conv2D(256, kernel_size = (3,3), dilation_rate = 12, padding = 'same', kernel_initializer = tf.keras.initializers.he_normal(), use_bias = False)(inputs);
  results = tf.keras.layers.BatchNormalization()(results);
  dilated_12 = tf.keras.layers.ReLU()(results);
  # results.shape = (batch, height, width, 256)
  results = tf.keras.layers.Conv2D(256, kernel_size = (3,3), dilation_rate = 18, padding = 'same', kernel_initializer = tf.keras.initializers.he_normal(), use_bias = False)(inputs);
  results = tf.keras.layers.BatchNormalization()(results);
  dilated_18 = tf.keras.layers.ReLU()(results);
  # results.shape = (batch, height, width, 256 * 5)
  results = tf.keras.layers.Concatenate(axis = -1)([pool, dilated_1, dilated_6, dilated_12, dilated_18]);
  results = tf.keras.layers.Conv2D(256, kernel_size = (1,1), dilation_rate = 1, padding = 'same', kernel_initializer = tf.keras.initializers.he_normal(), use_bias = False)(results);
  results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.ReLU()(results);
  return tf.keras.Model(inputs = inputs, outputs = results);

def IterativeOptimizationModule():

  inputs = tf.keras.Input((None, None, 256)); # inputs.shape = (qn, h / 8, w / 8, 256)
  mask = tf.keras.Input((None, None, 2)); # mask.shape = (qn, h / 8, w / 8, 2)
  def make_vanilla_residual_block(inputs, mask = None):
    residual = inputs;
    if mask is not None: inputs = tf.keras.layers.Concatenate(axis = -1)([inputs, mask]);
    results = tf.keras.layers.Conv2D(256, (3, 3), padding = 'same')(inputs);
    results = tf.keras.layers.BatchNormalization()(results);
    results = tf.keras.layers.ReLU()(results);
    results = tf.keras.layers.Conv2D(256, (3, 3), padding = 'same')(results);
    results = tf.keras.layers.BatchNormalization()(results);
    results = tf.keras.layers.Add()([residual, results]);
    results = tf.keras.layers.ReLU()(results);
    return results;
  results = make_vanilla_residual_block(inputs, mask); # results.shape = (qn, h / 8, w / 8, 256)
  results = make_vanilla_residual_block(results); # results.shape = (qn, h / 8, w / 8, 256)
  results = make_vanilla_residual_block(results); # results.shape = (qn, h / 8, w / 8, 256)
  results = AtrousSpatialPyramidPooling(256)(results); # results.shape = (qn, h / 8, w / 8, 256)
  results = tf.keras.layers.Conv2D(2, (1, 1), padding = 'same', activation = tf.keras.activations.softmax)(results); # results.shape = (qn, h / 8, w / 8, 2)
  return tf.keras.Model(inputs = (inputs, mask), outputs = results);

def CANet(nshot, iter_num = 3, pretrain = None):

  assert type(iter_num) is int and iter_num > 1;
  query = tf.keras.Input((None, None, 3)); # query.shape = (qn, h, w, 3)
  support = tf.keras.Input((None, None, 3), batch_size = nshot); # support.shape = (nshot, h, w, 3)
  labels = tf.keras.Input((None, None, 1), batch_size = nshot); # labels.shape = (nshot, h, w, 1)
  feature = DenseComparisonModule(nshot, pretrain)([query, support, labels]); # feature.shape = (qn, h / 8, w / 8, 256)
  mask = tf.keras.layers.Conv2D(2, (1, 1), padding = 'same', activation = tf.keras.activations.softmax)(feature); # mask.shape = (qn, h / 8, w / 8, 2)
  iom = IterativeOptimizationModule();
  for i in range(iter_num):
    mask = iom([feature, mask]); # mask.shape = (qn, h / 8, w / 8, 2)
  # scale the prediction to image size
  mask = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, (8 * tf.shape(x)[1], 8 * tf.shape(x)[2]), tf.image.ResizeMethod.BILINEAR))(mask); # mask.shape = (qn, h, w, 2)
  # get foreground from the prediction
  mask = tf.keras.layers.Lambda(lambda x: tf.math.argmax(x, axis = -1))(mask); # mask.shape = (qn, h, w)
  return tf.keras.Model(inputs = (query, support, labels), outputs = mask);

if __name__ == "__main__":

  assert tf.executing_eagerly();
  import numpy as np;
  query = np.random.normal(size = (2, 224, 224, 3))
  support = np.random.normal(size = (4, 224, 224, 3))
  labels = np.random.normal(size = (4, 224, 224, 1))
  canet = CANet(4, 2);
  canet.save('canet.h5');
  results = canet([query, support, labels]);
  print(results.shape);
