from __future__ import print_function
import os, sys
import time
import math
import numpy as np
import tensorflow as tf
from tensorflow.contrib.distributions import percentile
import bouncing_balls as b
import layer_def as ld
from six_stack_reader import *
from ConvLSTM1D import BasicConvLSTMCell
from BasicConvLSTMCell2d import BasicConvLSTMCell2d
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 16}

matplotlib.rc('font', **font)
plt.switch_backend('agg')

#os.environ["CUDA_VISIBLE_DEVICES"] = '0'

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', './checkpoints/gan-loss',
                            """dir to store trained net""")
# note: /datax/scratch/ayu/6-stacked is NOT same as /datax/scratch/yzhang/6-stacked
# has been reformatted
tf.app.flags.DEFINE_string('data_dir', '/datax/scratch/ayu/6-stacked',
                            """dir to load data""")
tf.app.flags.DEFINE_string('norm_input', 'shift_std',
                            """normalization for data""")
tf.app.flags.DEFINE_string('train_data_index', './train_data',
                            """index to load train data""")
tf.app.flags.DEFINE_string('test_data_index', './test_data',
                            """index to load test data""")
tf.app.flags.DEFINE_float('split', .9,
                            """train data proportion""")
tf.app.flags.DEFINE_string('mode', 'train',
                            """train or test""")
tf.app.flags.DEFINE_string('test_mode', 'ROC',
                            """ROC or hallucinate""")
tf.app.flags.DEFINE_string('train_mode', 'with_gan',
                            """train or test""")
tf.app.flags.DEFINE_integer('seq_length', 6,
                            """size of hidden layer""")
tf.app.flags.DEFINE_integer('seq_start', 1,
                            """ start of seq generation""")
tf.app.flags.DEFINE_integer('max_step', 200000,
                            """max num of steps""")
tf.app.flags.DEFINE_float('keep_prob', 1.,
                            """for dropout""")
tf.app.flags.DEFINE_float('lr', .001,
                            """for dropout""")
tf.app.flags.DEFINE_integer('batch_size', 32,
                            """batch size for training""")
tf.app.flags.DEFINE_integer('print_every', 10,
                            """batch size for training""")
tf.app.flags.DEFINE_integer('save_every', 100,
                            """batch size for training""")
tf.app.flags.DEFINE_boolean('resume', False,
                            """whether to load saved weights""")
tf.app.flags.DEFINE_boolean('match_mask', False,
                            """whether to load saved weights""")
tf.app.flags.DEFINE_float('past_loss_weight', 5e-5,
                           """weight for past loss""")
tf.app.flags.DEFINE_float('future_loss_weight', 5e-5,
                           """weight for future loss""")
tf.app.flags.DEFINE_float('d3_loss_weight', 1e-5,
                           """weight for D3 loss""")
#fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v') 

sample_dir = FLAGS.train_dir + '/samples/'
if not os.path.exists(sample_dir):
  os.makedirs(sample_dir)

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
  conv1 = ld.conv_layer_1D(inputs, 3, 2, 8, "encode_{}".format(i+1)) # conv2
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
  conv = ld.conv2d(inputs, [1,1], [1,1], 8, "encode")
  #conv = inputs
  # encoder convlstm 
  with tf.variable_scope('conv_lstm_encoder_1', initializer=tf.contrib.layers.xavier_initializer(uniform=True)):
    cell1 = BasicConvLSTMCell2d([16, 128], [8, 8], 8)
    if encoder_state is None:
      encoder_state = cell1.zero_state(FLAGS.batch_size, tf.float32) 
    conv1, encoder_state = cell1(conv, encoder_state)
  with tf.variable_scope('conv_lstm_encoder_2', initializer=tf.contrib.layers.xavier_initializer(uniform=True)):
    cell2 = BasicConvLSTMCell2d([16, 128], [8, 8], 8)
    conv2, encoder_state = cell2(conv1, encoder_state)
  with tf.variable_scope('conv_lstm_encoder_3', initializer=tf.contrib.layers.xavier_initializer(uniform=True)):
    cell3 = BasicConvLSTMCell2d([16, 128], [8, 8], 8)
    conv3, encoder_state = cell3(conv2, encoder_state)
  
  # past decoder convlstm 
  with tf.variable_scope('past_decoder_1', initializer=tf.contrib.layers.xavier_initializer(uniform=True)):
    pcell1 = BasicConvLSTMCell2d([16, 128], [8, 8], 8)
    if past_state is None:
      past_state = pcell1.zero_state(FLAGS.batch_size, tf.float32) 
    pconv1, past_state = pcell1(conv1, past_state)
  with tf.variable_scope('past_decoder_2', initializer=tf.contrib.layers.xavier_initializer(uniform=True)):
    pcell2 = BasicConvLSTMCell2d([16, 128], [8, 8], 8)
    pconv2, past_state = pcell2(conv2, past_state)
  with tf.variable_scope('past_decoder_3', initializer=tf.contrib.layers.xavier_initializer(uniform=True)):
    pcell3 = BasicConvLSTMCell2d([16, 128], [8, 8], 8)
    pconv3, past_state = pcell3(conv3, past_state)

  # future decoder convlstm 
  with tf.variable_scope('future_decoder_1', initializer=tf.contrib.layers.xavier_initializer(uniform=True)):
    fcell1 = BasicConvLSTMCell2d([16, 128], [8, 8], 8)
    if future_state is None:
      future_state = fcell1.zero_state(FLAGS.batch_size, tf.float32) 
    fconv1, future_state = fcell1(conv1, future_state)
  with tf.variable_scope('future_decoder_2', initializer=tf.contrib.layers.xavier_initializer(uniform=True)):
    fcell2 = BasicConvLSTMCell2d([16, 128], [8, 8], 8)
    fconv2, future_state = fcell2(conv2, future_state)
  with tf.variable_scope('future_decoder_3', initializer=tf.contrib.layers.xavier_initializer(uniform=True)):
    fcell3 = BasicConvLSTMCell2d([16, 128], [8, 8], 8)
    fconv3, future_state = fcell3(conv3, future_state)
  # present output
  x_1 = ld.conv2d(pconv3, [1,1], [1,1], 1, "present_output", True)
  # # future output
  y_1 = ld.conv2d(fconv3, [1,1], [1,1], 1, "future_output", True)

  #x_1_s = tf.nn.sigmoid(x_1)
  #y_1_s = tf.nn.sigmoid(y_1)
  #x_1 = pconv3; y_1 = fconv3
  #import IPython; IPython.embed()
  return x_1, y_1, encoder_state, past_state, future_state

# make a template for reuse
network_template = tf.make_template('network', network_2d)
def discriminator(image, df_dim=16, reuse=False, fc_shape=None):
  # the discriminator network
  with tf.variable_scope("discriminator") as scope:
    if reuse:
      scope.reuse_variables() # 80 128
    h0 = ld.conv2d(image, (3, 3),(1,1),df_dim, name='d_h0_conv') 
    h00 = ld.conv2d(h0, (3, 3),(1,2),df_dim, name='d_h00_conv') # 80 64
    h1 = ld.conv2d(h00, (3, 3),(1,1),df_dim*2, name='d_h1_conv') 
    h11 = ld.conv2d(h1, (3, 3),(2,2),df_dim*2, name='d_h11_conv') # 40 32
    h2 = ld.conv2d(h11, (3, 3),(1,1),df_dim*4, name='d_h2_conv')
    h22 = ld.conv2d(h2, (3, 3),(2,2),df_dim*4, name='d_h22_conv') # 20 16
    h3 = ld.conv2d(h22, (3, 3),(1,1),df_dim*8, name='d_h3_conv') 
    h33 = ld.conv2d(h3, (3, 3),(2,2),df_dim*8, name='d_h33_conv') # 10 8 
    h4 = ld.conv2d(h33, (3, 3),(1,1),df_dim*16, name='d_h4_conv') 
    h5 = ld.conv2d(h4, (3, 3),(2,2),df_dim*16, name='d_h5_conv') # 5 4

    #import IPython; IPython.embed()
    h6 = ld.fc_layer(h5, 1, name='d_h6_lin', linear=True, flat=True, input_shape=fc_shape) # 1 1

    return tf.nn.sigmoid(h6), h6, h5

def discriminator_buff(image, df_dim=32, reuse=False, fc_shape=None):
  # wtf is this?
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


def _plot_samples(dat, im_x, im_y, fname, pad='m', t_range=[0,10]):

    #blank_im = np.zeros(dat[0].shape[1:-1])
    past = im_x[0, :,:,:,:]
    past = past.reshape(past.shape[:-1])
    past_im = np.vstack(past)

    future = im_y[0, :, :,:,:]
    future = future.reshape(future.shape[:-1])
    future_im = np.vstack(future)

    inputs_list = dat[0].reshape(dat[0].shape[:-1])
    inputs = np.vstack(inputs_list)

    if not os.path.exists(sample_dir):
      os.makedirs(sample_dir)
    
    # npy
    ims = np.hstack([inputs, past_im, future_im])
    # normalize
    min_val, max_val = np.amin(ims), np.amax(ims) 
    ims /= max_val
    np.save(os.path.join(sample_dir, fname + ".npy"), ims)

    # png
    plt.figure(1, figsize=(16,10))
    gs1 = gridspec.GridSpec(FLAGS.seq_length, 3)
    gs1.update(wspace=0.01, hspace=0.01)
    for i in range(FLAGS.seq_length):
        ax = plt.subplot(gs1[3*i])
        if i < FLAGS.seq_length - 1:
            ax.set_xticklabels([])
        ax.set_yticklabels([])
        plt.imshow(inputs_list[i], interpolation="nearest", cmap="viridis", aspect='auto', vmin=min_val, vmax=max_val)
    for i in range(FLAGS.seq_length):
        ax = plt.subplot(gs1[3*i+1])
        if i < FLAGS.seq_length - 1:
            ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(past[i], interpolation="nearest", cmap="viridis", aspect='auto', vmin=min_val, vmax=max_val)
    for i in range(FLAGS.seq_length):
        ax = plt.subplot(gs1[3*i+2])
        if i < FLAGS.seq_length - 1:
            ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(future[i], interpolation="nearest", cmap="viridis", aspect='auto', vmin=min_val, vmax=max_val)

    plt.savefig(os.path.join(sample_dir, fname + ".png"))

def _binary_cross_entropy(x, x_hat):
    return -tf.reduce_mean((1 - x) * tf.log(1 - x_hat) + x * tf.log(x_hat))

def train(with_gan=True, load_x=True, match_mask=False):
  """Train ring_net for a number of steps."""
  with tf.Graph().as_default():
    x_all = tf.placeholder(tf.float32, [None, FLAGS.seq_length, 16, 128, 1])
    if match_mask: with_gan = False
    # possible dropout inside
    keep_prob = tf.placeholder("float")
    #x_dropout = tf.nn.dropout(x, keep_prob)

    x_in = x_all[:,:FLAGS.seq_start,:,:,:]

    # create network
    x_unwrap, y_unwrap = [], []

    # conv network
    encoder_state = None
    past_state = None
    future_state = None

    for i in range(FLAGS.seq_length):
      #if i < FLAGS.seq_start:
      x_1, y_1, encoder_state, past_state, future_state = network_template(x_all[:,i,:,:,:], encoder_state, past_state, future_state)
      #else:
      # x_1, y_1, encoder_state, past_state, future_state = network_template(y_1, encoder_state, past_state, future_state)
      x_unwrap.append(x_1)
      y_unwrap.append(y_1)

    # pack them all together 
    x_unwrap = tf.stack(x_unwrap)
    x_unwrap = tf.transpose(x_unwrap, [1,0,2,3,4])
    y_unwrap = tf.stack(y_unwrap)
    y_unwrap = tf.transpose(y_unwrap, [1,0,2,3,4])

    if not match_mask:
      past_loss_l2 = tf.nn.l2_loss(x_all[:, FLAGS.seq_start-1:-1, :,:,:] - x_unwrap[:, FLAGS.seq_start:, :,:,:]) * FLAGS.past_loss_weight
      future_loss_l2 = tf.nn.l2_loss(x_all[:, FLAGS.seq_start:,:,:,:] - y_unwrap[:, FLAGS.seq_start-1:-1, :,:,:]) * FLAGS.future_loss_weight
      #past_loss_l2 = _binary_cross_entropy(x_all[:, :-2, :,:,:], x_unwrap[:, 1:, :,:,:])
      #future_loss_l2 = _binary_cross_entropy(x_all[:, FLAGS.seq_start:,:,:,:], y_unwrap[:, FLAGS.seq_start-1:, :,:,:])
    else:
      x_mask = x_all > percentile(x_all, q=95.)
      x_mask = tf.one_hot(tf.cast(x_mask, tf.int32), depth=2, axis=-1)
      x_logit = tf.stack([x_1, 1./x_1], axis=-1)
      y_logit = tf.stack([y_1, 1./y_1], axis=-1)
      x_1 = tf.nn.softmax(logits=x_logit)
      y_1 = tf.nn.softmax(logits=y_logit)
      y = x_mask[:,FLAGS.seq_start:,:,:]
      x = x_mask[:,:FLAGS.seq_start,:,:]
      past_loss_l2 = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=x_logit, labels=x))
      future_loss_l2 = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=y_logit, labels=y))
      #import IPython; IPython.embed()
    if with_gan:
      img = x_all[:,FLAGS.seq_start:,:,:,:]
      img_ = y_unwrap[:, :-1, :,:,:]
      # cut off channel dimension and use the 6 stack hidden dim as channel
      shape_no_chnl = [-1, (FLAGS.seq_length - FLAGS.seq_start), 16, 128]
      img = tf.reshape(img, shape_no_chnl)
      img_ = tf.reshape(img_, shape_no_chnl)
      img = tf.transpose(img, [0, 2, 3, 1])
      img_ = tf.transpose(img_, [0, 2, 3, 1])
      #import IPython; IPython.embed(i)
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
      D3_loss = tf.nn.l2_loss(D3-D3_) * FLAGS.d3_loss_weight
      t_vars = tf.trainable_variables()
      d_vars = [var for var in t_vars if 'd_' in var.name]
      g_vars = [var for var in t_vars if 'd_' not in var.name]
      tf.summary.scalar('loss_g', g_loss)
      tf.summary.scalar('loss_d', d_loss)
      tf.summary.scalar('loss_feature', D3_loss)

      # loss def
      loss = (past_loss_l2 + future_loss_l2) + g_loss + D3_loss

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
    files = find_files(FLAGS.train_data_index)
    for step in range(FLAGS.max_step):
      dat = load_batch(FLAGS.batch_size, files, step, normalize=FLAGS.norm_input)
      dat = random_flip(dat)
      t = time.time()
      errG, errD = sess.run([g_loss, d_loss], feed_dict={x_all:dat, keep_prob:FLAGS.keep_prob})
      if errG > 0.6 and errD>0.6:
        _, loss_r = sess.run([train_op, loss],feed_dict={x_all:dat, keep_prob:FLAGS.keep_prob})
      else:
        i = 0
        print('   partial G', end=' ')
        while errG > 0.6:
            _ = sess.run(g_optim, feed_dict={x_all:dat, keep_prob:FLAGS.keep_prob})
            i+=1
            if i > 3: break
            else:
                errG = sess.run(g_loss, feed_dict={x_all:dat, keep_prob:FLAGS.keep_prob})
                if i < 3:
                    print(i, errG, end=', ')
        print(i-1, errG, end='; ')

        i = 0
        print('D', end=' ')
        while errD > 0.6:
            # only update discriminator if loss are within given bounds
            _ = sess.run(d_optim, feed_dict={x_all:dat, keep_prob:FLAGS.keep_prob})
            i+=1
            if i > 3: break
            else:
                errD = sess.run(d_loss, feed_dict={x_all:dat, keep_prob:FLAGS.keep_prob})
                if i < 3:
                    print(i, errD, end=', ')
        print(i-1, errD)
        loss_r = sess.run(loss, feed_dict={x_all:dat, keep_prob:FLAGS.keep_prob})
      #_, loss_r = sess.run([train_op, loss],feed_dict={x:dat, keep_prob:FLAGS.keep_prob})
      elapsed = time.time() - t
      
      if step % FLAGS.print_every == 0: #and step != 0:
        summary_str, g_loss_r, past_loss_l2_r, future_loss_l2_r, D3_loss_r, d_loss_r = \
                sess.run([summary_op, g_loss, past_loss_l2, future_loss_l2, D3_loss, d_loss], feed_dict={x_all:dat, keep_prob:FLAGS.keep_prob})
        summary_writer.add_summary(summary_str, step) 
        print("{}: loss: {}, time: {}".format(step, loss_r, elapsed))
        print("   G errG: {}, past: {}, fut: {}, D3: {}".format(g_loss_r, past_loss_l2_r, future_loss_l2_r, D3_loss_r))
        print("   D errD: {}".format(d_loss_r))
      
      assert not np.isnan(loss_r), 'Model diverged with loss = NaN'

      if step % FLAGS.save_every == 0:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)  
        print("saved to " + FLAGS.train_dir)

        print("now saving sample!", end='')
        sys.stdout.flush()
        im_x, im_y = sess.run([x_unwrap, y_unwrap],feed_dict={x_all:dat, keep_prob:FLAGS.keep_prob})
        if match_mask:
            im_x = im_x[...,1]
            im_y = im_y[...,1]
        _plot_samples(dat, im_x, im_y, '%05d' % step)
        print(" - saved")
 
def _plot_roc(data, percent, save):
  plt.clf()
  plt.figure(1, figsize=(8,4))
  for i in range(len(data)):
    fpr, tpr = data[i][:,0], data[i][:,1]
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='AUC = %0.2f, top-%d%%' % (roc_auc, percent[i]))
  plt.title('ROC by pixel coverage')
  plt.legend(loc='lower right')
  plt.plot([0, 1], [0, 1],'r--')
  plt.xlim([0, 1])
  plt.ylim([0, 1])
  plt.ylabel('True Positive Rate')
  plt.xlabel('False Positive Rate')
  plt.tight_layout()
  plt.savefig(save + "_roc.pdf")

def test():
  # test implementation not fixed
  with tf.Graph().as_default():
    x_all = tf.placeholder(tf.float32, [None, FLAGS.seq_length, 16, 128, 1])

    # create network
    x_unwrap, y_unwrap = [], []

    # conv network
    encoder_state = None
    past_state = None
    future_state = None

    for i in range(10):
      if i < FLAGS.seq_start:
        x_1, y_1, encoder_state, past_state, future_state = network_template(x_all[:,i,:,:,:], encoder_state, past_state, future_state)
      else:
        x_1, y_1, encoder_state, past_state, future_state = network_template(x_1, encoder_state, past_state, future_state)
      x_unwrap.append(x_1)
      y_unwrap.append(y_1)

    # pack them all together 
    x_unwrap = tf.stack(x_unwrap)
    x_unwrap = tf.transpose(x_unwrap, [1,0,2,3,4])
    y_unwrap = tf.stack(y_unwrap)
    y_unwrap = tf.transpose(y_unwrap, [1,0,2,3,4])

    # possible dropout inside
    keep_prob = tf.placeholder("float")
    #x_dropout = tf.nn.dropout(x, keep_prob)
    x_dropout = x

    # conv network

    #past_loss_l2 = tf.nn.l2_loss(x[:,:FLAGS.seq_start,:,:] - x_1)
    future_loss_l2 = tf.nn.l2_loss(x_all[:-1] - y_unwrap)
    #anomaly_loss_l2 = [tf.nn.l2_loss(y[i] - y_1[i]) for i in range(FLAGS.batch_size)]
    #fake_loss_l2 = [tf.nn.l2_loss(y[FLAGS.batch_size-1-i] - y_1[i]) for i in range(FLAGS.batch_size)]

    img = x
    img_ = y_1
    #import IPython; IPython.embed()
    """
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
    """
    #fake_loss_D3 = [tf.nn.l2_loss(D3[FLAGS.batch_size-1-i] - D3_[i]) for i in range(FLAGS.batch_size)]

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
    files = find_files(FLAGS.test_data_index)

    if FLAGS.test_mode == 'ROC':
      nsteps = min(500, FLAGS.max_step)
      thresh_list = np.arange(0., 1., 0.02)**4
      fill_percent_list = [1,2,5,10]
      ROC = np.zeros((nsteps, len(fill_percent_list), thresh_list.size, 2))
      for step in range(nsteps):
        dat = load_batch(FLAGS.batch_size, files, step, normalize=FLAGS.norm_input)
        x_t, y_t, im_x, im_y, dloss, e_loss, eD3_loss = sess.run([x, y, x_1, y_1, d_loss,
                                                              anomaly_loss_l2,
                                                              anomaly_loss_D3,],
                                                              feed_dict={x_all:dat, keep_prob:1.})
        for fi, fill_p in enumerate(fill_percent_list):
          m1 = y_t>np.percentile(y_t, axis=(1,2,3), q=100-fill_p, keepdims=True) #True for top 5% pixels
          m2 = im_y>np.percentile(im_y, axis=(1,2,3), q=100-fill_p, keepdims=True) #True for top 5% pixels

          val_normal = np.sum(m1&m2, axis=(1,2,3))/np.sum(m1|m2, axis=(1,2,3))
          val_anomaly = np.sum(m1[::-1]&m2, axis=(1,2,3))/np.sum(m1[::-1]|m2, axis=(1,2,3))
          #print(val_normal)
          #print(val_anomaly)
          
          for ti, thresh in enumerate(thresh_list):
            n_correct = np.sum(val_normal>thresh) + np.sum(val_anomaly<thresh)
            acc = float(n_correct)/FLAGS.batch_size/2
            false_alarm_rate = np.sum(val_normal<thresh).astype(float)/FLAGS.batch_size
            true_positive_rate = np.sum(val_anomaly<thresh).astype(float)/FLAGS.batch_size
            #print(thresh, acc, false_alarm, missed_detection)
            ROC[step, fi, ti, 0] = false_alarm_rate
            ROC[step, fi, ti, 1] = true_positive_rate
        #import IPython; IPython.embed()
        if step < 2:
          pass
          #_plot_samples(dat.squeeze(), sample_dir+'GT{}.pdf'.format(step))
          #_plot_samples(np.concatenate([x_t.squeeze(), y_t.squeeze()], axis=1), sample_dir+'G2_{}.png'.format(step))
          #_plot_samples(np.concatenate([im_x.squeeze(), im_y.squeeze()], axis=1), sample_dir+'PRED_{}.pdf'.format(step))
          #print("loss", dloss)
          #import IPython; IPython.embed()
      ROC = np.mean(ROC, axis=0)
      print("save roc")
      np.save(FLAGS.train_dir+"roc.npy", ROC)
      _plot_roc(ROC, fill_percent_list, FLAGS.train_dir)

    elif FLAGS.test_mode == 'hallucinate':
      nsteps = 5
      frames = []
      for step in range(nsteps):
        dat = load_batch(FLAGS.batch_size, files, step, normalize=FLAGS.norm_input)
        im_x, im_y = sess.run([x_1, y_1], feed_dict={x_all:dat, keep_prob:1.})
        dat = np.concatenate([im_x, im_y], axis=1)
        frames.append(im_y)
      frames = np.concatenate(frames, axis=1).squeeze()
      for i in range(min(10, FLAGS.batch_size//2)):
        pass
        #_plot_samples(frames[i:i+2], sample_dir+'hallucinate{}.pdf'.format(i), pad=None, t_range=[0,5*nsteps], n_columns=2, max_row=1)


def main(argv=None):  # pylint: disable=unused-argument
  # create train/test split
  if not os.path.isfile(FLAGS.train_data_index) or \
      not os.path.isfile(FLAGS.test_data_index):
    train_test_split(FLAGS.data_dir, FLAGS.train_data_index, FLAGS.test_data_index, FLAGS.split)
  if FLAGS.mode == "train":
    if not FLAGS.resume:
      if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
      tf.gfile.MakeDirs(FLAGS.train_dir)
    if FLAGS.train_mode == 'with_gan':
      train(with_gan=True)
    elif FLAGS.train_mode == 'match_mask':
      train(match_mask=True)
    else:
      train()
  elif FLAGS.mode == "test":
    test()

if __name__ == '__main__':
  tf.app.run()
