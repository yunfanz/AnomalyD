import os, sys
import glob
import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

# import bouncing_balls as b
import layer_def as ld
import BasicConvLSTMCell, BasicConvLSTMCell2d

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', 'train',
                            """dir to store trained net""")
tf.app.flags.DEFINE_string('data_dir', '/datax/scratch/yzhang/6-stacked/',
                            """dir to find dataset""")
tf.app.flags.DEFINE_string('data_file_format', '/*/*.npy',
                            """data file path format, for glob""")
tf.app.flags.DEFINE_string('preview_png_path', 'preview_png/',
                            """path to save preview images, in png format""")
tf.app.flags.DEFINE_string('preview_npy_path', 'preview_npy/',
                            """path to save preview images, in npy format""")
tf.app.flags.DEFINE_integer('seq_length', 6,
                            """size of hidden layer""")
tf.app.flags.DEFINE_integer('seq_start', 1,
                            """ start of seq generation""")
tf.app.flags.DEFINE_integer('max_step', 200000,
                            """max num of steps""")
tf.app.flags.DEFINE_float('keep_prob', .8,
                            """for dropout""")
tf.app.flags.DEFINE_float('lr', .001,
                            """for dropout""")
tf.app.flags.DEFINE_integer('batch_size', 32,
                            """batch size for training""")
tf.app.flags.DEFINE_float('weight_init', .1,
                            """weight init for fully connected layers""")
                            
if not os.path.exists(FLAGS.preview_png_path):
    os.mkdir(FLAGS.preview_png_path)
if not os.path.exists(FLAGS.preview_npy_path):
    os.mkdir(FLAGS.preview_npy_path)
                            
# def generate_bouncing_ball_sample(batch_size, seq_length, shape, num_balls):
  # dat = np.zeros((batch_size, seq_length, shape, shape, 3))
  # for i in range(batch_size):
    # dat[i, :, :, :, :] = b.bounce_vec(32, num_balls, seq_length)
  # return dat 
  
def npy_data_loader(max_len):
  files = glob.glob(FLAGS.data_dir + FLAGS.data_file_format)
  if not files:
    print("No data files found!")
    sys.exit(1)
  all_data = []
  while True:
      for file in files:
        data = np.load(file)
        all_data.extend(data)
        while len(all_data) >= max_len:
            yield np.array(all_data[:max_len])
            all_data = all_data[max_len:]
  if all_data:
    yield np.array(all_data)

def network(inputs, hidden):
  conv1 = ld.conv2d(inputs, (3,3), (1,2), 8, "encode_1")
  # conv2
  # conv2 = ld.conv2d(conv1, (3,3), (1,2), 8, "encode_2")
  # conv3
  # conv3 = ld.conv2d(conv2, (3,3), (1,2), 8, "encode_3")
  # conv4
  #conv4 = ld.conv2d(conv3, (1,1), (1,1), 4, "encode_4")
  #y_0 = conv4
  y_0 = conv1
  # conv lstm cell 
  with tf.variable_scope('conv_lstm', initializer = tf.random_uniform_initializer(-.01, 0.1)):
    cell = BasicConvLSTMCell.BasicConvLSTMCell([16,64], [8,32], 8)
    if hidden is None:
      hidden = cell.zero_state(FLAGS.batch_size, tf.float32) 
    y_1, hidden = cell(y_0, hidden)
  
  with tf.variable_scope('conv_lstm_2', initializer = tf.random_uniform_initializer(-.01, 0.1)):  
    cell2 = BasicConvLSTMCell.BasicConvLSTMCell([16,64], [8,32], 8)
    if hidden is None:
      hidden = cell2.zero_state(FLAGS.batch_size, tf.float32) 
    y_2, hidden = cell2(y_1, hidden)
    
  with tf.variable_scope('conv_lstm_3', initializer = tf.random_uniform_initializer(-.01, 0.1)):  
    cell3 = BasicConvLSTMCell.BasicConvLSTMCell([16,64], [8,32], 8)
    if hidden is None:
      hidden = cell3.zero_state(FLAGS.batch_size, tf.float32) 
    y_3, hidden = cell3(y_2, hidden)
 
  # conv5
  #conv5 = ld.transpose_conv_layer(y_3, (1,1), (1,1), 8, "decode_5")
  # conv6
  # conv6 = ld.transpose_conv_layer(conv5, (3,3), (1,2), 8, "decode_6")
  # conv7
  # conv7 = ld.transpose_conv_layer(conv6, (3,3), (1,2), 8, "decode_7")
  # x_1 
  conv7 = y_3
  x_1 = ld.transpose_conv_layer(conv7, (3,3), (1,2), 1, "decode_8") # set activation to linear

  return x_1, hidden

# make a template for reuse
network_template = tf.make_template('network', network)

def train(loader):
  """Train ring_net for a number of steps."""
  with tf.Graph().as_default():
    # make inputs
    x = tf.placeholder(tf.float32, (None, FLAGS.seq_length, 16, 128, 1))

    # possible dropout inside
    keep_prob = tf.placeholder("float")
    x_dropout = tf.nn.dropout(x, keep_prob)

    # create network
    x_unwrap = []

    # conv network
    hidden = None
    for i in range(FLAGS.seq_length-1):
      if i < FLAGS.seq_start:
        x_1, hidden = network_template(x[:,i,:,:,:], hidden)
      else:
        x_1, hidden = network_template(x_1, hidden)
      x_unwrap.append(x_1)

    # pack them all together 
    x_unwrap = tf.stack(x_unwrap)
    x_unwrap = tf.transpose(x_unwrap, [1,0,2,3,4])

    # this part will be used for generating video
    x_unwrap_g = []
    hidden_g = None
    for i in range(50):
      if i < FLAGS.seq_start:
        x_1_g, hidden_g = network_template(x[:,i,:,:,:], hidden_g)
      else:
        x_1_g, hidden_g = network_template(x_1_g, hidden_g)
      x_unwrap_g.append(x_1_g)

    # pack them generated ones
    x_unwrap_g = tf.stack(x_unwrap_g)
    x_unwrap_g = tf.transpose(x_unwrap_g, [1,0,2,3,4])

    # calc total loss (compare x_t to x_t+1)
    loss = tf.nn.l2_loss(x[:,FLAGS.seq_start+1:,:,:,:] - x_unwrap[:,FLAGS.seq_start:,:,:,:])
    tf.summary.scalar('loss', loss)

    # training
    train_op = tf.train.AdamOptimizer(FLAGS.lr).minimize(loss)
    
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
    print("init network from scratch")
    sess.run(init)

    # Summary op
    graph_def = sess.graph.as_graph_def(add_shapes=True)
    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, graph_def=graph_def)

    for step, dat in enumerate(loader):
      #dat = generate_bouncing_ball_sample(FLAGS.batch_size, FLAGS.seq_length, 32, FLAGS.num_balls)
      dat = dat.reshape(*dat.shape, 1)
      t = time.time()
      _, loss_r = sess.run([train_op, loss],feed_dict={x:dat, keep_prob:FLAGS.keep_prob})
      elapsed = time.time() - t

      if step%1 == 0 and step != 0:
        summary_str = sess.run(summary_op, feed_dict={x:dat, keep_prob:FLAGS.keep_prob})
        summary_writer.add_summary(summary_str, step)
        print("time per batch is " + str(elapsed))
        print(step)
        print(loss_r)
      
      assert not np.isnan(loss_r), 'Model diverged with loss = NaN'

      if step%10 == 0:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
        print("saved to " + FLAGS.train_dir)

        # make video
        print("now generating video!")
        
        #video = cv2.VideoWriter()
        #success = video.open("generated_conv_lstm_video.mov", fourcc, 4, (180, 180), True)
        dat_gif = dat
        out = sess.run([x_unwrap_g],feed_dict={x:dat_gif, keep_prob:FLAGS.keep_prob})
        out = out[0][0][:7]
        out = out.reshape(out.shape[:-1])
        out = np.vstack(out)
        out /= np.max(out)
        
        inputs = np.vstack(dat_gif[0].reshape(dat_gif[0].shape[:-1]))
        inputs = np.vstack([inputs, np.zeros(dat_gif[0].shape[1:-1])])
        inputs /= np.max(inputs)
        
        ims = np.hstack([inputs, out])
        for i in range(ims.shape[0]):
            ims[i, dat_gif[0].shape[2]] = 0.5
            
        for i in range(dat_gif[0].shape[1], ims.shape[0], dat_gif[0].shape[1]):
            for j in range(ims.shape[1]):
                ims[i, j] = 0.5
        
        pil_im = Image.fromarray((ims * 255).astype('uint8'))
        pil_im.save(os.path.join(FLAGS.preview_png_path, 'step_{}.png'.format(step)))
        np.save(os.path.join(FLAGS.preview_npy_path, 'step_{}.npy'.format(step)), ims)
        
        #plt.figure()
        #plt.imshow(ims)
        #plt.savefig('ball_samples/step_{}.png'.format(step))
        #print(ims.shape)
        #for i in range(50 - FLAGS.seq_start):
        #  x_1_r = np.uint8(np.maximum(ims[i,:,:,:], 0) * 255)
        #  new_im = cv2.resize(x_1_r, (180,180))
        #  video.write(new_im)
        #video.release()
      if step > FLAGS.max_step:
        break


def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  loader = npy_data_loader(FLAGS.batch_size)
  train(loader)

if __name__ == '__main__':
  tf.app.run()


