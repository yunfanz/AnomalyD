

"""functions used to construct different architectures  
"""

import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_float('weight_decay', 0.0005,
                          """ """)
xinitializer = tf.contrib.layers.xavier_initializer(uniform=True)
def _activation_summary(x):
  """Helper to create summaries for activations.
  Creates a summary that provides a histogram of activations.
  Creates a summary that measure the sparsity of activations.
  Args:
    x: Tensor
  Returns:
    nothing
  """
  tensor_name = x.op.name
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.
  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.
  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
  Returns:
    Variable Tensor
  """
  var = tf.get_variable(name, shape,
                       initializer=xinitializer)
  # var = tf.get_variable(name, shape,
  #                        initializer=tf.truncated_normal_initializer(stddev=stddev))
  if wd:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    weight_decay.set_shape([])
    tf.add_to_collection('losses', weight_decay)
  return var

def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)

def _padding_size(dim, stride, kernel_size):
    """
    compute padding needed to maintain size in conv,
    with given dimension size, stride, kernel size  
    """
    return [(dim * (stride - 1) + kernel_size - 1) // 2,
            (dim * (stride - 1) + kernel_size - 2) // 2 + 1]

def conv2d(inputs, kernel_size, stride, num_features, name, linear=False):
  with tf.variable_scope(name) as scope:
    input_channels = inputs.get_shape()[3]

    weights = _variable_with_weight_decay('weights', shape=[kernel_size[0],kernel_size[1],input_channels,num_features],stddev=0.01, wd=FLAGS.weight_decay)
    biases = tf.get_variable('biases',[num_features], initializer=tf.constant_initializer(0.01))

    # custom reflective padding to reduce edge artifacts
    inputs = tf.pad(inputs, [[0, 0],
                             _padding_size(inputs.get_shape()[1], stride[0], kernel_size[1]),
                             _padding_size(inputs.get_shape()[2], stride[0], kernel_size[1]),
                             [0, 0]], mode='REFLECT')
    conv = tf.nn.conv2d(inputs, weights, strides=[1, stride[0], stride[1], 1], padding='VALID')
    conv_biased = tf.nn.bias_add(conv, biases)
    if linear:
      return conv_biased
    conv_rect = tf.nn.elu(conv_biased, name=name)
    return conv_rect

def transpose_conv_layer(inputs, kernel_size, stride, num_features, idx, linear = False):
  with tf.variable_scope('{0}_trans_conv'.format(idx)) as scope:
    input_channels = inputs.get_shape()[3]
    
    weights = _variable_with_weight_decay('weights', shape=[kernel_size[0],kernel_size[1],num_features,input_channels], stddev=0.01, wd=FLAGS.weight_decay)
    biases = tf.get_variable('biases',[num_features],initializer=tf.constant_initializer(0.01))
    batch_size = tf.shape(inputs)[0]
    output_shape = tf.stack([tf.shape(inputs)[0], tf.shape(inputs)[1]*stride[0], tf.shape(inputs)[2]*stride[1], num_features]) 
    conv = tf.nn.conv2d_transpose(inputs, weights, output_shape, strides=[1,stride[0],stride[1],1], padding='SAME')
    conv_biased = tf.nn.bias_add(conv, biases)
    if linear:
      return conv_biased
    conv_rect = tf.nn.elu(conv_biased,name='{0}_transpose_conv'.format(idx))
    return conv_rect
     
def conv_layer_1D(inputs, kernel_size, stride, num_features, idx, linear = False):
  with tf.variable_scope('{0}_conv'.format(idx)) as scope:
    input_channels = inputs.get_shape()[-1]

    weights = _variable_with_weight_decay('weights', shape=[kernel_size,input_channels,num_features],stddev=0.01, wd=FLAGS.weight_decay)
    biases = tf.get_variable('biases',[num_features],initializer=tf.constant_initializer(0.01))

    conv = tf.nn.conv1d(inputs, weights, stride=stride, padding='SAME')
    conv_biased = tf.nn.bias_add(conv, biases)
    if linear:
      return conv_biased
    conv_rect = tf.nn.elu(conv_biased,name='{0}_conv'.format(idx))
    return conv_rect

def transpose_conv_layer_1D(inputs, kernel_size, stride, num_features, idx, linear=False):
  with tf.variable_scope('{0}_trans_conv'.format(idx)) as scope:
    input_shape = inputs.get_shape()
    input_channels = input_shape[-1]
    weights = _variable_with_weight_decay('weights', shape=[1,kernel_size,num_features,input_channels], stddev=0.01, wd=FLAGS.weight_decay)
    biases = tf.get_variable('biases',[num_features],initializer=tf.constant_initializer(0.01))
    
    batch_size = input_shape[0]
    inputs = tf.expand_dims(inputs, 1)
    output_shape = tf.stack([input_shape[0], 1, input_shape[1]*stride, num_features]) 
    deconv = tf.nn.conv2d_transpose(inputs, filter=weights, output_shape=output_shape, strides=[1, 1, stride, 1], padding='SAME')
    deconv = tf.squeeze(deconv, [1]) #removes the first dummy dimension
    conv_biased = tf.nn.bias_add(deconv, biases)
    if linear:
      return conv_biased
    conv_rect = tf.nn.elu(conv_biased,name='{0}_transpose_conv'.format(idx))
    return conv_rect

def fc_layer(inputs, hiddens, name, flat=False, linear=False, input_shape=None):
  with tf.variable_scope(name) as scope:
    if input_shape is None:
      input_shape = inputs.get_shape().as_list()
    if flat:
      dim = input_shape[1]*input_shape[2]*input_shape[3]
      inputs_processed = tf.reshape(inputs, [-1,dim])
    else:
      dim = input_shape[1]
      inputs_processed = inputs
    
    weights = _variable_with_weight_decay('weights', shape=[dim,hiddens],stddev=0.01, wd=FLAGS.weight_decay)
    biases = tf.get_variable('biases', [hiddens], initializer=tf.constant_initializer(0.01))
    if linear:
      return tf.add(tf.matmul(inputs_processed,weights),biases,name=name)
  
    ip = tf.add(tf.matmul(inputs_processed,weights),biases)
    return tf.nn.elu(ip,name=name)
    
