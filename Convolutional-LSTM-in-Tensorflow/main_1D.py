
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

tf.app.flags.DEFINE_string('train_dir', './checkpoints/gan-loss',
                            """dir to store trained net""")
tf.app.flags.DEFINE_string('data_dir', './Data',
                            """dir to load data""")
tf.app.flags.DEFINE_string('mode', 'train',
                            """train or test""")
tf.app.flags.DEFINE_integer('seq_length', 32,
                            """size of hidden layer""")
tf.app.flags.DEFINE_integer('seq_start', 16,
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
  for i in range(batch_size):
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
  for i in range(2,10,2):
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
  for i in range(1,9,2):
    #print(dconv.get_shape())
    dconv = decode_stack(dconv, i)
  #import IPython; IPython.embed()
  # x_1 
  x_1 = ld.transpose_conv_layer_1D(dconv, 5, 2, 1, "decode_{}".format(i+3), True) # set activation to linear

  return x_1, hidden

def network_2d(inputs, encoder_state, past_state, future_state):
  #inputs is 3D tensor (batch, )
  conv = ld.conv2d(inputs, (4,4), (2,2), 16, "encode")
  #conv = inputs
  # encoder convlstm 
  with tf.variable_scope('conv_lstm_encoder_1', initializer=tf.contrib.layers.xavier_initializer(uniform=True)):
    cell1 = BasicConvLSTMCell2d([8, 256], [4, 4], 16)
    if encoder_state is None:
      encoder_state = cell1.zero_state(FLAGS.batch_size, tf.float32) 
    conv1, encoder_state = cell1(conv, encoder_state)
  with tf.variable_scope('conv_lstm_encoder_2', initializer=tf.contrib.layers.xavier_initializer(uniform=True)):
    cell2 = BasicConvLSTMCell2d([8, 256], [4, 4], 16)
    conv2, encoder_state = cell2(conv1, encoder_state)
  with tf.variable_scope('conv_lstm_encoder_3', initializer=tf.contrib.layers.xavier_initializer(uniform=True)):
    cell3 = BasicConvLSTMCell2d([8, 256], [4, 4], 16)
    conv3, encoder_state = cell3(conv2, encoder_state)
  
  # past decoder convlstm 
  with tf.variable_scope('past_decoder_1', initializer=tf.contrib.layers.xavier_initializer(uniform=True)):
    pcell1 = BasicConvLSTMCell2d([8, 256], [4, 4], 16)
    if past_state is None:
      past_state = pcell1.zero_state(FLAGS.batch_size, tf.float32) 
    pconv1, past_state = pcell1(conv1, past_state)
  with tf.variable_scope('past_decoder_2', initializer=tf.contrib.layers.xavier_initializer(uniform=True)):
    pcell2 = BasicConvLSTMCell2d([8, 256], [4, 4], 16)
    pconv2, past_state = pcell2(conv2, past_state)
  with tf.variable_scope('past_decoder_3', initializer=tf.contrib.layers.xavier_initializer(uniform=True)):
    pcell3 = BasicConvLSTMCell2d([8, 256], [4, 4], 16)
    pconv3, past_state = pcell3(conv3, past_state)

  # future decoder convlstm 
  with tf.variable_scope('future_decoder_1', initializer=tf.contrib.layers.xavier_initializer(uniform=True)):
    fcell1 = BasicConvLSTMCell2d([8, 256], [4, 4], 16)
    if future_state is None:
      future_state = fcell1.zero_state(FLAGS.batch_size, tf.float32) 
    fconv1, future_state = fcell1(conv1, future_state)
  with tf.variable_scope('future_decoder_2', initializer=tf.contrib.layers.xavier_initializer(uniform=True)):
    fcell2 = BasicConvLSTMCell2d([8, 256], [4, 4], 16)
    fconv2, future_state = fcell2(conv2, future_state)
  with tf.variable_scope('future_decoder_3', initializer=tf.contrib.layers.xavier_initializer(uniform=True)):
    fcell3 = BasicConvLSTMCell2d([8, 256], [4, 4], 16)
    fconv3, future_state = fcell3(conv3, future_state)
  # present output
  x_1 = ld.transpose_conv_layer(pconv3, 8, 2, 1, "present_output", True)
  # # future output
  y_1 = ld.transpose_conv_layer(fconv3, 8, 2, 1, "future_output", True)
  #x_1 = pconv3; y_1 = fconv3
  #import IPython; IPython.embed()
  return x_1, y_1, encoder_state, past_state, future_state

# make a template for reuse
network_template = tf.make_template('network', network_2d)
def discriminator(image, df_dim=32, reuse=False, fc_shape=None):
  with tf.variable_scope("discriminator") as scope:
    if reuse:
      scope.reuse_variables()
    image = ld.conv2d(image, (3, 5),(1,2),df_dim, name='d_input_conv')
    image = tf.nn.avg_pool(image, 
                                    ksize=[1, 1, 3, 1],
                                    strides=[1, 1, 2, 1],
                                    padding='SAME')
    h0 = ld.conv2d(image, (3, 3),(1,1),df_dim, name='d_h0_conv')
    h0 = ld.conv2d(h0, (3, 3),(1,2),df_dim, name='d_h00_conv')
    h1 = ld.conv2d(h0, (3, 3),(1,1),df_dim*2, name='d_h1_conv')
    h1 = ld.conv2d(h1, (3, 3),(1,2),df_dim*2, name='d_h11_conv')
    h2 = ld.conv2d(h1, (3, 3),(1,1),df_dim*4, name='d_h2_conv')
    h2 = ld.conv2d(h2, (3, 3),(1,2),df_dim*4, name='d_h22_conv')
    h3 = ld.conv2d(h2, (3, 3),(1,1),df_dim*8, name='d_h3_conv')
    h3 = ld.conv2d(h3, (3, 3),(1,2),df_dim*8, name='d_h33_conv')
    h4 = ld.conv2d(h3, (3, 3),(1,1),df_dim*16, name='d_h4_conv')
    h5 = ld.conv2d(h4, (3, 3),(2,2),df_dim*16, name='d_h5_conv')

    h5 = tf.nn.dropout(h5, 0.5)

    #import IPython; IPython.embed()
    h6 = ld.fc_layer(h5, 1, name='d_h6_lin', linear=True, flat=True, input_shape=fc_shape)

    return tf.nn.sigmoid(h6), h6, h5


def _plot_samples(samples, fname):
    batch_size = samples.shape[0]
    plt.figure(1, figsize=(16,10))
    n_columns = 4
    n_rows = min(math.ceil(batch_size / n_columns) + 1, 4)
    for i in range(min(batch_size, n_columns*n_rows)):
        plt.subplot(n_rows, n_columns, i+1)
        plt.imshow(samples[i], interpolation="nearest", cmap="hot", aspect='auto')
    print('saving', fname)
    plt.savefig(fname)

def train(with_gan=True, load_x=True, with_y=True):
  """Train ring_net for a number of steps."""
  with tf.Graph().as_default():
    x_all = tf.placeholder(tf.float32, [None, FLAGS.seq_length, 512, 1])

    # possible dropout inside
    keep_prob = tf.placeholder("float")
    #x_dropout = tf.nn.dropout(x, keep_prob)

    y = x_all[:,FLAGS.seq_start:,:,:]
    x = x_all[:,:FLAGS.seq_start,:,:]
    # conv network
    encoder_state = None
    past_state = None
    future_state = None
    x_1, y_1, encoder_state, past_state, future_state = network_template(x, encoder_state, past_state, future_state)

    past_loss_l2 = tf.nn.l2_loss(x - x_1)
    future_loss_l2 = tf.nn.l2_loss(y - y_1)

    
    if with_gan:

      #import IPython; IPython.embed()
      D, D_logits, D3 = discriminator(y, reuse=False)
      #import IPython; IPython.embed()
      D_, D_logits_, D3_ = discriminator(y_1, reuse=True, fc_shape=D3.get_shape().as_list())
      d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=D_logits, labels=tf.ones_like(D)))
      d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=D_logits_, labels=tf.zeros_like(D_)))
      d_loss = d_loss_real + d_loss_fake
      g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=D_logits_, labels=tf.ones_like(D_)))
      D3_loss = tf.nn.l2_loss(D3-D3_)
      t_vars = tf.trainable_variables()
      d_vars = [var for var in t_vars if 'd_' in var.name]
      g_vars = [var for var in t_vars if 'd_' not in var.name]
      tf.summary.scalar('loss_g', g_loss)
      tf.summary.scalar('loss_d', d_loss)
      tf.summary.scalar('loss_feature', D3_loss)
      loss = 0.05*(past_loss_l2 + future_loss_l2) + g_loss + 0.001*D3_loss
      tf.summary.scalar('past_loss_l2', past_loss_l2)
      tf.summary.scalar('future_loss_l2', future_loss_l2)
      d_optim = tf.train.AdamOptimizer(FLAGS.lr).minimize(d_loss, var_list=d_vars)
      g_optim = tf.train.AdamOptimizer(FLAGS.lr).minimize(loss, var_list=g_vars)
      #import IPython; IPython.embed()
      train_op = tf.group(d_optim, d_optim, g_optim)

    else:
      loss = past_loss_l2 + future_loss_l2
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
    if not with_y:
      files = find_files(FLAGS.data_dir)
    else:
      files = find_pairs(FLAGS.data_dir)
    sample_dir = FLAGS.train_dir + '/samples/'
    
    if not os.path.exists(sample_dir):
      os.makedirs(sample_dir)
    for step in range(FLAGS.max_step):
      dat = load_batch(FLAGS.batch_size, files, step, with_y=with_y)

      t = time.time()
      errG, errD = sess.run([g_loss, d_loss], feed_dict={x_all:dat, keep_prob:FLAGS.keep_prob})
      i = 0
      while errG > 0.9:
                            
          _ = sess.run(g_optim, feed_dict={x_all:dat, keep_prob:FLAGS.keep_prob})
          i+=1
          if i > 2: break
          else:
              errG = sess.run(g_loss, feed_dict={x_all:dat, keep_prob:FLAGS.keep_prob})

      i = 0
      while errD > 0.45:
          # only update discriminator if loss are within given bounds
          _ = sess.run(d_optim, feed_dict={x_all:dat, keep_prob:FLAGS.keep_prob})
          i+=1
          if i > 2: break
          else:
              errD = sess.run(d_loss, feed_dict={x_all:dat, keep_prob:FLAGS.keep_prob})
      _, loss_r = sess.run([train_op, loss],feed_dict={x_all:dat, keep_prob:FLAGS.keep_prob})
      elapsed = time.time() - t
      
      if step%100 == 0 and step != 0:
        summary_str = sess.run(summary_op, feed_dict={x_all:dat, keep_prob:FLAGS.keep_prob})
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
        im_x, im_y = sess.run([x_1, y_1],feed_dict={x_all:dat, keep_prob:FLAGS.keep_prob})
        _plot_samples(dat[:,:FLAGS.seq_start,:,:].squeeze(), sample_dir+'step_{}_past_t.png'.format(step))
        _plot_samples(im_x.squeeze(), sample_dir+'step_{}_past.png'.format(step))
        _plot_samples(dat[:,FLAGS.seq_start:,:,:].squeeze(), sample_dir+'step_{}_future_t.png'.format(step))
        _plot_samples(im_y.squeeze(), sample_dir+'step_{}_future.png'.format(step))

def test(test_mode='anomaly'):
  with tf.Graph().as_default():
    x = tf.placeholder(tf.float32, [None, FLAGS.seq_length, 512, 1])


    # possible dropout inside
    keep_prob = tf.placeholder("float")
    #x_dropout = tf.nn.dropout(x, keep_prob)
    x_dropout = x

    # conv network
    encoder_state = None
    past_state = None
    future_state = None
    x_1, y_1, encoder_state, past_state, future_state = network_template(x_dropout[:,:FLAGS.seq_start,:,:], encoder_state, past_state, future_state)

    #past_loss_l2 = tf.nn.l2_loss(x[:,:FLAGS.seq_start,:,:] - x_1)
    future_loss_l2 = tf.nn.l2_loss(x[:,FLAGS.seq_start:,:,:] - y_1)
    anomaly_loss_l2 = [tf.nn.l2_loss(x[i,FLAGS.seq_start:,:,:] - y_1[i]) for i in range(FLAGS.batch_size)]
    fake_loss_l2 = [tf.nn.l2_loss(x[FLAGS.batch_size-1-i,FLAGS.seq_start:,:,:] - y_1[i]) for i in range(FLAGS.batch_size)]

    img = x[:,FLAGS.seq_start:,:,:]
    img_ = y_1
    #import IPython; IPython.embed()
    D, D_logits, D3 = discriminator(img, reuse=False)
    #import IPython; IPython.embed()
    D_, D_logits_, D3_ = discriminator(img_, reuse=True, fc_shape=D3.get_shape().as_list())
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                  logits=D_logits, labels=tf.ones_like(D)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                  logits=D_logits_, labels=tf.zeros_like(D_)))
    d_loss = d_loss_real + d_loss_fake
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                  logits=D_logits_, labels=tf.ones_like(D_)))
    D3_loss = tf.nn.l2_loss(D3-D3_)
    anomaly_loss_D3 = [tf.nn.l2_loss(D3[i] - D3_[i]) for i in range(FLAGS.batch_size)]
    fake_loss_D3 = [tf.nn.l2_loss(D3[FLAGS.batch_size-1-i] - D3_[i]) for i in range(FLAGS.batch_size)]

    variables = tf.global_variables()
    saver = tf.train.Saver(tf.global_variables())

    # Summary op
    summary_op = tf.summary.merge_all()
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    latest = tf.train.latest_checkpoint(FLAGS.train_dir)
    if not latest:
        print("No checkpoint to continue from in", FLAGS.train_dir)
        sys.exit(1)
    print("resume", latest)
    saver.restore(sess, latest)
    files = find_files(FLAGS.data_dir)
    sample_dir = FLAGS.train_dir + '/samples/'
    
    if not os.path.exists(sample_dir):
      os.makedirs(sample_dir)
    for step in range(FLAGS.max_step):
      dat = load_batch(FLAGS.batch_size, files, step)
      im_x, im_y, dloss, e_loss, f_loss, eD3_loss, fD3_loss = sess.run([x_1, y_1, d_loss,
                                                            anomaly_loss_l2, fake_loss_l2,
                                                            anomaly_loss_D3, fake_loss_D3,],
                                                            feed_dict={x:dat, keep_prob:1.})
      #_plot_samples(dat[:,:FLAGS.seq_start,:,:].squeeze(), sample_dir+'test_{}_past_t.png'.format(step))
      #_plot_samples(im_x.squeeze(), sample_dir+'test_{}_past.png'.format(step))
      #_plot_samples(dat[:,FLAGS.seq_start:,:,:].squeeze(), sample_dir+'test_{}_future_t.png'.format(step))
      #_plot_samples(im_y.squeeze(), sample_dir+'test_{}_future.png'.format(step))
      print("loss", d_loss)
      import IPython; IPython.embed()

def main(argv=None):  # pylint: disable=unused-argument
  if FLAGS.mode == "train":
    if not FLAGS.resume:
      if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
      tf.gfile.MakeDirs(FLAGS.train_dir)
    train(with_y=True)
  elif FLAGS.mode == "test":
    test()

if __name__ == '__main__':
  tf.app.run()
