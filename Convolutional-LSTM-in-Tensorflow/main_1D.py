
from __future__ import print_function
import os.path
import time
import math
import numpy as np
import tensorflow as tf
#import cv2
from fits_reader import *
import bouncing_balls as b
import layer_def as ld
from ConvLSTM1D import BasicConvLSTMCell
from BasicConvLSTMCell2d import BasicConvLSTMCell2d
import matplotlib.pyplot as plt
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', './checkpoints/train_store_conv_lstm',
                            """dir to store trained net""")
tf.app.flags.DEFINE_string('data_dir', './Data',
                            """dir to load data""")
tf.app.flags.DEFINE_integer('seq_length', 16,
                            """size of hidden layer""")
tf.app.flags.DEFINE_integer('seq_start', 5,
                            """ start of seq generation""")
tf.app.flags.DEFINE_integer('max_step', 200000,
                            """max num of steps""")
tf.app.flags.DEFINE_float('keep_prob', 1.,
                            """for dropout""")
tf.app.flags.DEFINE_float('lr', .001,
                            """for dropout""")
tf.app.flags.DEFINE_integer('batch_size', 32,
                            """batch size for training""")
tf.app.flags.DEFINE_boolean('resume', False,
                            """whether to load saved wieghts""")
#fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v') 

def generate_bouncing_ball_sample(batch_size, seq_length, shape, num_balls):
  dat = np.zeros((batch_size, seq_length, shape, shape, 3))
  for i in xrange(batch_size):
    dat[i, :, :, :, :] = b.bounce_vec(32, num_balls, seq_length)
  return dat 

def generate_x_batch(batch_size, t, f, n_sig=2, ret=False, noise=True):
    f_0 = tf.random_uniform(shape=[batch_size, n_sig,1,1],minval=0., maxval=512.) #in ms 
    amp = tf.random_uniform(shape=[batch_size,n_sig,1,1],minval=0.25,maxval= 2.5)
    width = tf.random_uniform(shape=[batch_size,n_sig,1,1],minval=1., maxval=20.)
    slope = tf.random_uniform(shape=[batch_size,n_sig,1,1],minval=-15, maxval=15)
    f0_all = f_0 + slope * t
    pulse = tf.reduce_sum(amp*tf.exp(-0.5 * (f - f0_all) ** 2 / width ** 2.), axis=1)
    if noise:
        noise_level = 0.2
        pulse += noise_level *tf.random_uniform(pulse.get_shape(), minval=0, maxval=1)
    return pulse[..., tf.newaxis]

def encode_stack(inputs, i):
  conv1 = ld.conv_layer_1D(inputs, 3, 2, 8, "encode_{}".format(i+1))
  # conv2
  conv2 = ld.conv_layer_1D(conv1, 3, 1, 8, "encode_{}".format(i+2))
  return conv2

def decode_stack(inputs, i):
  conv1 = ld.transpose_conv_layer_1D(inputs, 3, 2, 8, "decode_{}".format(i+1))
  # conv2
  conv2 = ld.transpose_conv_layer_1D(conv1, 3, 1, 8, "decode_{}".format(i+2))
  return conv2

def network(inputs, hidden, lstm_depth=4):
  #inputs is 3D tensor (batch, )
  conv1 = ld.conv_layer_1D(inputs, 5, 2, 8, "encode_1")
  # conv2
  conv = ld.conv_layer_1D(conv1, 3, 1, 8, "encode_2")
  for i in xrange(2,10,2):
    conv = encode_stack(conv, i)

  conv = ld.conv_layer_1D(conv, 1, 1, 4, "encode_{}".format(i+3))
  y_0 = conv
  

  with tf.variable_scope('conv_lstm_0', initializer=tf.contrib.layers.xavier_initializer(uniform=True)):
    #with tf.variable_scope('conv_lstm', initializer = tf.random_uniform_initializer(-.01, 0.1)):
      cell = BasicConvLSTMCell([16,], [5,], 4)
      if hidden is None:
        hidden = cell.zero_state(FLAGS.batch_size, tf.float32) 
      y_0, _ = cell(y_0, hidden)


  for l in range(1, lstm_depth):
    # conv lstm cell 
    with tf.variable_scope('conv_lstm_{}'.format(l), initializer=tf.contrib.layers.xavier_initializer(uniform=True)):
    #with tf.variable_scope('conv_lstm', initializer = tf.random_uniform_initializer(-.01, 0.1)):
      cell = BasicConvLSTMCell([16,], [5,], 4)
      y_0, _ = cell(y_0, hidden)

  y_1 = y_0
  #import IPython; IPython.embed()
  dconv = ld.transpose_conv_layer_1D(y_1, 1, 1, 8, "decode_1")
  #import IPython; IPython.embed()
  for i in xrange(1,9,2):
    #print(dconv.get_shape())
    dconv = decode_stack(dconv, i)
  #import IPython; IPython.embed()
  # x_1 
  x_1 = ld.transpose_conv_layer_1D(dconv, 5, 2, 1, "decode_{}".format(i+3), True) # set activation to linear

  return x_1, hidden

def network_simple(inputs, encoder_state, decoder_state):
  #inputs is 3D tensor (batch, )
  conv = ld.conv_layer_1D(inputs, 8, 4, 8, "embedding")
  
  # encoder convlstm 
  with tf.variable_scope('encoder_1', initializer=tf.contrib.layers.xavier_initializer(uniform=True)):
    cell1 = BasicConvLSTMCell([128,], [8,], 4)
    if encoder_state is None:
      encoder_state = cell1.zero_state(FLAGS.batch_size, tf.float32) 
    conv1, encoder_state = cell1(conv, encoder_state)
  with tf.variable_scope('encoder_2', initializer=tf.contrib.layers.xavier_initializer(uniform=True)):
    cell2 = BasicConvLSTMCell([128,], [8,], 4)
    conv2, encoder_state = cell2(conv1, encoder_state)
  with tf.variable_scope('encoder_3', initializer=tf.contrib.layers.xavier_initializer(uniform=True)):
    cell3 = BasicConvLSTMCell([128,], [8,], 4)
    conv3, encoder_state = cell3(conv2, encoder_state)

  # decoder convlstm 
  with tf.variable_scope('decoder_1', initializer=tf.contrib.layers.xavier_initializer(uniform=True)):
    fcell = BasicConvLSTMCell([128,], [8,], 4)
    if decoder_state is None:
      decoder_state = fcell.zero_state(FLAGS.batch_size, tf.float32) 
    fconv, decoder_state = fcell(conv3, decoder_state)

  # future output
  output = ld.transpose_conv_layer_1D(fconv, 8, 4, 1, "output", True)

  return output, encoder_state, decoder_state

# make a template for reuse
network_template = tf.make_template('network', network_simple)

def _plot_samples(samples, fname):
    batch_size = samples.shape[0]
    plt.figure(1, figsize=(16,10))
    n_columns = 4
    n_rows = min(math.ceil(batch_size / n_columns) + 1, 6)
    for i in range(min(batch_size, n_columns*n_rows)):
        plt.subplot(n_rows, n_columns, i+1)
        plt.imshow(samples[i], interpolation="nearest", cmap="hot", aspect='auto')
    print('saving', fname)
    plt.savefig(fname)

def train(with_gan=True, load_x=True):
  """Train ring_net for a number of steps."""
  with tf.Graph().as_default():
    x = tf.placeholder(tf.float32, [None, FLAGS.seq_length, 512, 1])

    # possible dropout inside
    keep_prob = tf.placeholder("float")
    x_dropout = tf.nn.dropout(x, keep_prob)

    # create network
    futures = []
    
    # conv network
    encoder_state = None
    decoder_state = None
    future = x_dropout[:,0,:,:]

    for i in range(FLAGS.seq_length-1):
      #conditional generation
      if i < FLAGS.seq_start:
        concat = tf.concat([future, x_dropout[:,i,:,:]], 2)
      else:
        concat = tf.concat([future, future], 2)
      future, encoder_state, decoder_state = network_template(concat, encoder_state, decoder_state)
      futures.append(future)

    # pack them all together 
    futures = tf.stack(futures)
    futures = tf.transpose(futures, [1,0,2,3])

    # loss
    future_gt = x_dropout[:,1:,:,:]
    loss = tf.nn.l2_loss(future_gt - futures)
    tf.summary.scalar('loss', loss)

    # training
    optimizer = tf.train.AdamOptimizer(FLAGS.lr)
    gvs = optimizer.compute_gradients(loss)
    # gradient clipping
    capped_gvs = [(tf.clip_by_value(grad, -3., 3.), var) for grad, var in gvs]
    train_op = optimizer.apply_gradients(capped_gvs)

    # List of all Variables
    variables = tf.global_variables()

    # Build a saver
    saver = tf.train.Saver(tf.global_variables())   

    # Summary op
    summary_op = tf.summary.merge_all()
 
    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()

    # Start running operations on the Graph.
    sess = tf.Session()

    # init if this is the very time training
    
    sess.run(init)
    if FLAGS.resume:
      latest = tf.train.latest_checkpoint(FLAGS.train_dir)
      if not latest:
          print("No checkpoint to continue from in", FLAGS.train_dir)
          sys.exit(1)
      print("resume", latest)
      saver.restore(sess, latest)
    else:
      print("init network from scratch")
    
    # Summary op
    graph_def = sess.graph.as_graph_def(add_shapes=True)
    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, graph_def=graph_def)
    files = find_files(FLAGS.data_dir)
    sample_dir = FLAGS.train_dir + '/samples/'
    
    if not os.path.exists(sample_dir):
      os.makedirs(sample_dir)
    for step in range(FLAGS.max_step):
      dat = load_batch(FLAGS.batch_size, files, step)
      t = time.time()
      _, loss_r = sess.run([train_op, loss],feed_dict={x:dat, keep_prob:FLAGS.keep_prob})
      elapsed = time.time() - t
      
      if step%100 == 0 and step != 0:
        summary_str = sess.run(summary_op, feed_dict={x:dat, keep_prob:FLAGS.keep_prob})
        summary_writer.add_summary(summary_str, step) 
        print("time per batch is " + str(elapsed))
        print(step)
        print(loss_r)
      
      assert not np.isnan(loss_r), 'Model diverged with loss = NaN'

      if step%1000 == 0:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)  
        print("saved to " + FLAGS.train_dir)

        print("now saving sample!")
        future_ims = sess.run(futures,feed_dict={x:dat, keep_prob:FLAGS.keep_prob})
        _plot_samples(dat[:,1:,:,:].squeeze(), sample_dir+'step_{}_future_gt.png'.format(step))
        _plot_samples(future_ims.squeeze(), sample_dir+'step_{}_future.png'.format(step))

def main(argv=None):  # pylint: disable=unused-argument
  if not FLAGS.resume:
    if tf.gfile.Exists(FLAGS.train_dir):
      tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
  train()

if __name__ == '__main__':
  tf.app.run()
