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

def Attention():

  inputs = tf.keras.Input((None, None, 512)); # inputs.shape = (qn, h / 16, w / 16, 256 + 256)
  # 1) get masks for query images according to this support image
  outputs = tf.keras.layers.Conv2D(256, (3, 3), padding = 'same')(inputs); # outputs.shape = (qn, h / 16, w / 16, 256)
  # 2) get attention weights for query images of this support image
  att = tf.keras.layers.Conv2D(256, (3, 3), padding = 'same')(inputs); # att.shape = (qn, h / 16, w / 16, 256)
  att = tf.keras.layers.MaxPool2D(pool_size = (3, 3), strides = (1, 1), padding = 'same')(att); # att.shape = (qn, h / 16, w / 16, 256)
  att = tf.keras.layers.Conv2D(256, (3, 3), padding = 'same')(att); # att.shape = (qn, h / 16, w / 16, 256)
  att = tf.keras.layers.Lambda(lambda x: tf.math.reduce_mean(x, axis = [1, 2, 3], keepdims = True))(att); # att.shape = (qn, 1, 1, 1)
  attended = tf.keras.layers.Lambda(lambda x: x[0] * x[1])([outputs, att]); # attended.shape = (qn, h / 16, w / 16, 256)
  return tf.keras.Model(inputs = inputs, outputs = attended);

def DenseComparisonModule(pretrain = None):

  query = tf.keras.Input((None, None, 3)); # query.shape = (qn, h, w, 3)
  support = tf.keras.Input((None, None, 3)); # support.shape = (nshot, h, w, 3)
  labels = tf.keras.Input((None, None, 1)); # labels.shape = (nshot, h, w, 1)
  imgs_concat = tf.keras.layers.Concatenate(axis = 0)([support, query]); # imgs_concat.shape = (nshot + qn, h, w, 3)
  resnet50 = ResNet50Atrous();
  # load pretrained model
  if pretrain: resnet50.load_weights(pretrain);
  block1, block2, block3, block4 = resnet50(imgs_concat); # block2.shape = (nshot + qn, h / 8, w / 8, 512), block3.shape = (nshot + qn, h / 8, w / 8, 1024)
  img_fts = tf.keras.layers.Concatenate(axis = -1)([block2, block3]); # img_fts.shape = (nshot + qn, h / 8, w / 8, 512 + 1024)
  img_fts = tf.keras.layers.Conv2D(256, (3, 3), padding = 'same')(img_fts); # img_fts.shape = (nshot + qn, h / 8, w / 8, 256)
  supp_fts, qry_fts = tf.keras.layers.Lambda(lambda x: tf.split(x[0], (tf.shape(x[1])[0], tf.shape(x[2])[0]), axis = 0))([img_fts, support, query]); # supp_fts.shape = (nshot, h / 16, w / 16, 256), qry_fts.shape = (qn, h / 16, w / 16, 256)
  supp_lb = tf.keras.layers.Lambda(lambda x: tf.image.resize(x[0], size = tf.shape(x[1])[1:3], method = tf.image.ResizeMethod.BILINEAR))([labels, img_fts]); # supp_lb.shape = (nshot, h / 16, w / 16, 256)
  proto = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x[0] * x[1], axis = (1,2)) / tf.math.maximum(tf.math.reduce_sum(x[1], axis = (1,2)), 1e-5))([supp_fts, supp_lb]); # proto.shape = (nshot, 256)
  proto = tf.keras.layers.Lambda(lambda x: tf.tile(tf.reshape(x[0], (tf.shape(x[0])[0], 1, 1, 1, tf.shape(x[0])[1])), (1, tf.shape(x[1])[0], tf.shape(x[1])[1], tf.shape(x[1])[2], 1)))([proto, qry_fts]); # proto.shape = (nshot, qn, h / 16, w / 16, 256)
  qry_fts = tf.keras.layers.Lambda(lambda x: tf.tile(tf.expand_dims(x[0], axis = 0), (tf.shape(x[1])[0],1,1,1,1)))([qry_fts, proto]); # qry_fts.shape = (nshot, qn, h / 16, w / 16, 256)
  qry_comp_fts = tf.keras.layers.Concatenate(axis = -1)([qry_fts, proto]); # qry_comp_fts.shape = (nshot, qn, h / 16, w / 16, 256 + 256)
  weighted = tf.keras.backend.map_fn(fn = Attention(), elems = qry_comp_fts); # qry_comp_fts.shape = (nshot, qn, h / 16, w / 16, 256)
  weighted = tf.keras.layers.Softmax(axis = 0)(qry_comp_fts); # outputs.shape = (nshot, qn, h / 16, w / 16, 256)
  weighted_sum = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x, axis = 0))(weighted); # weighted_sum.shape = (qn, h / 16, w / 16, 256)
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

  inputs = tf.keras.Input((None, None, 256));
  

if __name__ == "__main__":

  assert tf.executing_eagerly();
  dcm = DenseComparisonModule();
  dcm.save('dcm.h5');
