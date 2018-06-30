from __future__ import print_function
import os
import time
import math
import numpy as np
import tensorflow as tf
from tensorflow.contrib.distributions import percentile
from sklearn.metrics import auc
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from fits_reader import *
import layer_def as ld
from BasicConvLSTMCell import BasicConvLSTMCell2d

#os.environ["CUDA_VISIBLE_DEVICES"] = '0'

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', './checkpoints/iter-loss',
                            """dir to store trained net""")
tf.app.flags.DEFINE_string('data_dir', './Data',
                            """dir to load data""")
tf.app.flags.DEFINE_string('norm_input', 'max',
                            """dir to load data""")
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
tf.app.flags.DEFINE_integer('seq_length', 32,
                            """size of hidden layer""")
tf.app.flags.DEFINE_integer('seq_start', 8,
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


def network(inputs, encoder_state, decoder_state):
  # conv
  conv1 = ld.conv2d(inputs, (4,8), (1,2), 4, "conv_1") # (?, 8, 256, 4)
  conv2 = ld.conv2d(conv1, (8,16), (1,2), 2, "conv_2") # (?, 8, 128, 2)
  conv3 = ld.conv2d(conv2, (16,16), (1,2), 1, "conv_3") # (?, 8, 64, 1)

  # convlstm 
  with tf.variable_scope('encoder', initializer=tf.contrib.layers.xavier_initializer(uniform=True)):
    cell1 = BasicConvLSTMCell2d([8, 64], [3, 3], 4)
    if encoder_state is None:
      encoder_state = cell1.zero_state(FLAGS.batch_size, tf.float32)
    out1, encoder_state = cell1(conv3, encoder_state) # (?, 8, 64, 4)

  with tf.variable_scope('decoder', initializer=tf.contrib.layers.xavier_initializer(uniform=True)):
    cell2 = BasicConvLSTMCell2d([8, 64], [3, 3], 4)
    if decoder_state is None:
      decoder_state = cell2.zero_state(FLAGS.batch_size, tf.float32)
    out2, decoder_state = cell2(out1, decoder_state) # (?, 8, 64, 4)

  # deconv
  y1 = ld.transpose_conv_layer(out2, (4,4), (1,2), 4, "dconv_1", True)
  y2 = ld.transpose_conv_layer(y1, (4,2), (1,2), 2, "deconv_2", True)
  y3 = ld.transpose_conv_layer(y2, (2,1), (1,2), 1, "deconv_3", True)

  return y3, encoder_state, decoder_state


# make a template for reuse
network_template = tf.make_template('network', network)


def discriminator(image, df_dim=16, reuse=False, fc_shape=None):
  with tf.variable_scope("discriminator") as scope:
    if reuse: scope.reuse_variables()
    image = tf.nn.avg_pool(image, 
                           ksize=[1, 1, 4, 1],
                           strides=[1, 1, 4, 1],
                           padding='SAME')
    h0 = ld.conv2d(image, (3, 7),(1,4),df_dim, name='d_h0_conv')
    h1 = ld.conv2d(h0, (3, 7),(1,2),df_dim*2, name='d_h1_conv')
    h2 = ld.conv2d(h1, (3, 5),(1,2),df_dim*4, name='d_h2_conv')
    h3 = ld.conv2d(h2, (3, 5),(1,2),df_dim*8, name='d_h3_conv')
    h4 = ld.conv2d(h3, (3, 3),(1,2),df_dim*16, name='d_h4_conv')
    h5 = ld.conv2d(h4, (3, 3),(1,1),df_dim*16, name='d_h5_conv')
    h6 = ld.fc_layer(h5, 1, name='d_h6_lin', linear=True, flat=True, input_shape=fc_shape)

    return tf.nn.sigmoid(h6), h6, h5


def _plot_samples(samples, fname, pad='m', t_range=[0,10], n_columns=3, max_row=3):
  batch_size = samples.shape[0]
  if pad == 'mid':
    print(np.zeros_like(samples[:,0,:])[:,np.newaxis,:].shape, samples.shape)
    samples = np.concatenate([samples[:,:FLAGS.seq_start,:], np.zeros_like(samples[:,0,:])[:,np.newaxis,:],samples[:,FLAGS.seq_start:,:]], axis=1)
  plt.figure(1, figsize=(16,10))
  n_rows = min(math.ceil(batch_size / n_columns) + 1, max_row)
  for i in range(min(batch_size, n_columns*n_rows)):
    plt.subplot(n_rows, n_columns, i+1)
    plt.imshow(samples[i], interpolation="nearest", cmap="hot", aspect='auto', extent=[-3*.256,3*0.256, t_range[1],t_range[0]])
  print('saving', fname)
  plt.savefig(fname)


def train(with_gan=True, load_x=True, with_y=True, match_mask=False):
  """Train for a number of steps."""
  with tf.Graph().as_default():
    x_all = tf.placeholder(tf.float32, [None, FLAGS.seq_length, 512, 1])
    if match_mask: with_gan = False

    # possible dropout inside
    keep_prob = tf.placeholder("float")

    # conv network
    encoder_state = None
    decoder_state = None
    x_in = x_all[:,:FLAGS.seq_start,:,:]
    y_1 = []

    for _ in range(FLAGS.seq_length - FLAGS.seq_start):
      pred, encoder_state, decoder_state = network_template(x_in, encoder_state, decoder_state)
      y_1.append(pred[:, -1, :, :])
      pred = tf.reshape(pred[:, -1, :, :], [-1, 1, 512, 1])
      x_in = tf.concat([x_in[:, 1:, :, :], pred], 1)

    y_1 = tf.stack(y_1, 1)

    if not match_mask:
      y = x_all[:,FLAGS.seq_start:,:,:]
      x = x_all[:,:FLAGS.seq_start,:,:]
      future_loss_l2 = tf.nn.l2_loss(y - y_1)
    else:
      x_mask = x_all > percentile(x_all, q=95.)
      x_mask = tf.one_hot(tf.cast(x_mask, tf.int32), depth=2, axis=-1)
      y_logit = tf.stack([y_1, 1./y_1], axis=-1)
      y_1 = tf.nn.softmax(logits=y_logit)
      y = x_mask[:,FLAGS.seq_start:,:,:]
      x = x_mask[:,:FLAGS.seq_start,:,:]
      future_loss_l2 = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=y_logit, labels=y))
    if with_gan:
      img = x_all[:,FLAGS.seq_start:,:,:]
      img_ = y_1
      D, D_logits, D3 = discriminator(img, reuse=False)
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
      loss = 0.05*future_loss_l2 + g_loss + D3_loss*1.e-4
      tf.summary.scalar('future_loss_l2', future_loss_l2)
      d_optim = tf.train.AdamOptimizer(FLAGS.lr).minimize(d_loss, var_list=d_vars)
      g_optim = tf.train.AdamOptimizer(FLAGS.lr).minimize(loss, var_list=g_vars)
      train_op = tf.group(d_optim, d_optim, g_optim)
    else:
      loss = future_loss_l2
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
      files = find_files(FLAGS.train_data_index)
    else:
      files = find_pairs(FLAGS.train_data_index)
    sample_dir = FLAGS.train_dir + '/samples/'
    if not os.path.exists(sample_dir):
      os.makedirs(sample_dir)
    for step in range(FLAGS.max_step):
      dat = load_batch(FLAGS.batch_size, files, step, with_y=with_y, normalize=FLAGS.norm_input)
      dat = random_flip(dat)
      t = time.time()
      errG, errD = sess.run([g_loss, d_loss], feed_dict={x_all:dat, keep_prob:FLAGS.keep_prob})
      if errG > 0.6 and errD>0.6:
        _, loss_r = sess.run([train_op, loss],feed_dict={x_all:dat, keep_prob:FLAGS.keep_prob})
      else:
        i = 0
        while errG > 0.6:   
            _ = sess.run(g_optim, feed_dict={x_all:dat, keep_prob:FLAGS.keep_prob})
            i+=1
            if i > 2: break
            else:
                errG = sess.run(g_loss, feed_dict={x_all:dat, keep_prob:FLAGS.keep_prob})
        print('G', i, errG)

        i = 0
        while errD > 0.6:
            # only update discriminator if loss are within given bounds
            _ = sess.run(d_optim, feed_dict={x_all:dat, keep_prob:FLAGS.keep_prob})
            i+=1
            if i > 2: break
            else:
                errD = sess.run(d_loss, feed_dict={x_all:dat, keep_prob:FLAGS.keep_prob})
        print('D', i, errD)
        loss_r = sess.run(loss, feed_dict={x_all:dat, keep_prob:FLAGS.keep_prob})

      elapsed = time.time() - t
      
      if step%1000 == 0 and step != 0:
        summary_str = sess.run(summary_op, feed_dict={x_all:dat, keep_prob:FLAGS.keep_prob})
        summary_writer.add_summary(summary_str, step) 
        print("time per batch is " + str(elapsed))
        print(step)
        print(loss_r)
      
      assert not np.isnan(loss_r), 'Model diverged with loss = NaN'

      if step%4000 == 0:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)  
        print("saved to " + FLAGS.train_dir)

        print("now saving sample!")
        im_y = sess.run(y_1,feed_dict={x_all:dat, keep_prob:FLAGS.keep_prob})
        if match_mask:
          im_y = im_y[...,1]
        _plot_samples(dat[:,:FLAGS.seq_start,:,:].squeeze(), sample_dir+'step_{}_past_t.png'.format(step))
        _plot_samples(dat[:,FLAGS.seq_start:,:,:].squeeze(), sample_dir+'step_{}_future_t.png'.format(step))
        _plot_samples(im_y.squeeze(), sample_dir+'step_{}_future.png'.format(step))


def _plot_roc(data, percent, save):
  plt.clf()
  plt.figure(1, figsize=(16,10))
  for i in range(len(data)):
    fpr, tpr = data[i][:,0], 1 - data[i][:,1]
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
  plt.savefig(save + "_roc.png")


def test(with_y=True):
  with tf.Graph().as_default():
    x_all = tf.placeholder(tf.float32, [None, FLAGS.seq_length, 512, 1])

    y = x_all[:,FLAGS.seq_start:,:,:]
    x = x_all[:,:FLAGS.seq_start,:,:]
    # possible dropout inside
    keep_prob = tf.placeholder("float")
    x_dropout = x

    # conv network
    encoder_state = None
    decoder_state = None
    x_in = x_all[:,:FLAGS.seq_start,:,:]
    y_1 = []

    for _ in range(FLAGS.seq_length - FLAGS.seq_start):
      pred, encoder_state, decoder_state = network_template(x_in, encoder_state, decoder_state)
      pred = tf.reshape(pred[:, -1, :, :], [-1, 1, 512, 1])
      y_1.append(pred[:, -1, :, :])
      x_in = tf.concat([x_in[:, 1:, :, :], pred], 1)

    y_1 = tf.stack(y_1, 1)

    future_loss_l2 = tf.nn.l2_loss(y - y_1)
    anomaly_loss_l2 = [tf.nn.l2_loss(y[i] - y_1[i]) for i in range(FLAGS.batch_size)]
    fake_loss_l2 = [tf.nn.l2_loss(y[FLAGS.batch_size-1-i] - y_1[i]) for i in range(FLAGS.batch_size)]

    img = x
    img_ = y_1
    D, D_logits, D3 = discriminator(img, reuse=False)
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
    if not with_y:
      files = find_files(FLAGS.test_data_index)
    else:
      files = find_pairs(FLAGS.test_data_index)
    sample_dir = FLAGS.train_dir + '/samples/'
    
    if not os.path.exists(sample_dir):
      os.makedirs(sample_dir)

    if FLAGS.test_mode == 'ROC':
      nsteps = min(100, FLAGS.max_step)
      thresh_list = np.arange(0., 1., 0.02)
      fill_percent_list = [1,2,5,10]
      ROC = np.zeros((nsteps, len(fill_percent_list), thresh_list.size, 2))
      for step in range(nsteps):
        dat = load_batch(FLAGS.batch_size, files, step, with_y=with_y, normalize=FLAGS.norm_input)
        x_t, y_t, im_y, dloss, e_loss, f_loss, eD3_loss, fD3_loss = sess.run([x, y, y_1, d_loss,
                                                              anomaly_loss_l2, fake_loss_l2,
                                                              anomaly_loss_D3, fake_loss_D3,],
                                                              feed_dict={x_all:dat, keep_prob:1.})
        for fi, fill_p in enumerate(fill_percent_list):
          m1 = y_t>np.percentile(y_t, axis=(1,2,3), q=100-fill_p, keepdims=True) #True for top 5% pixels
          m2 = im_y>np.percentile(im_y, axis=(1,2,3), q=100-fill_p, keepdims=True) #True for top 5% pixels

          val_normal = np.sum(m1&m2, axis=(1,2,3))/np.sum(m1|m2, axis=(1,2,3))
          val_anomaly = np.sum(m1[::-1]&m2, axis=(1,2,3))/np.sum(m1[::-1]|m2, axis=(1,2,3))
          
          for ti, thresh in enumerate(thresh_list):
            n_correct = np.sum(val_normal>thresh) + np.sum(val_anomaly<thresh)
            acc = float(n_correct)/FLAGS.batch_size/2
            false_alarm_rate = np.sum(val_normal<thresh).astype(float)/FLAGS.batch_size
            true_positive_rate = np.sum(val_anomaly>thresh).astype(float)/FLAGS.batch_size
            ROC[step, fi, ti, 0] = false_alarm_rate
            ROC[step, fi, ti, 1] = true_positive_rate
        if step < 2:
          _plot_samples(dat.squeeze(), sample_dir+'GT{}.pdf'.format(step))
          _plot_samples(np.concatenate([x_t.squeeze(), im_y.squeeze()], axis=1), sample_dir+'PRED_{}.pdf'.format(step))

      ROC = np.mean(ROC, axis=0)
      print("save roc")
      np.save(FLAGS.train_dir+"roc.npy", ROC)
      _plot_roc(ROC, fill_percent_list, FLAGS.train_dir)

    elif FLAGS.test_mode == 'hallucinate':
      nsteps = 5
      frames = []
      for step in range(nsteps):
        dat = load_batch(FLAGS.batch_size, files, step, with_y=with_y, normalize=FLAGS.norm_input)
        im_y = sess.run(y_1, feed_dict={x_all:dat, keep_prob:1.})
        dat = np.concatenate([x_t, im_y], axis=1)
        frames.append(im_y)
      frames = np.concatenate(frames, axis=1).squeeze()
      for i in range(min(10, FLAGS.batch_size//2)):
        _plot_samples(frames[i:i+2], sample_dir+'hallucinate{}.pdf'.format(i), pad=None, t_range=[0,5*nsteps], n_columns=2, max_row=1)


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
