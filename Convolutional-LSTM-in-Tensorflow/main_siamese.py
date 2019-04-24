from __future__ import print_function
import numpy as np
import tensorflow as tf
import layer_def as ld
from six_stack_reader import *
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', './checkpoints/siamese',
        """dir to store trained net""")
tf.app.flags.DEFINE_string('data_dir', '/datax/scratch/ayu/6-stacked-max200',
        """dir to load data""")
tf.app.flags.DEFINE_string('norm_input', 'batch',
        """normalization for data""")
tf.app.flags.DEFINE_string('train_data_index', './train_data',
        """index to load train data""")
tf.app.flags.DEFINE_string('test_data_index', './test_data',
        """index to load test data""")
tf.app.flags.DEFINE_float('split', .9,
        """train data proportion""")
tf.app.flags.DEFINE_integer('seq_length', 6,
        """size of hidden layer""")
tf.app.flags.DEFINE_integer('batch_size', 32,
        """batch size for training""")
tf.app.flags.DEFINE_boolean('resume', False,
        """whether to load saved weights""")
tf.app.flags.DEFINE_boolean('export', False,
        """whether to export tensorflow model to models/. Only if --resume also set""")
tf.app.flags.DEFINE_float('lr', .0001,
        """for dropout""")
tf.app.flags.DEFINE_integer('max_step', 5000000,
        """max num of steps""")
tf.app.flags.DEFINE_integer('print_every', 20,
                            """print loss every ... steps""")
tf.app.flags.DEFINE_integer('save_every', 500,
                            """save model and sample ever ... steps""")

sample_dir = FLAGS.train_dir + '/samples/'

def _plot_samples(dat_fake, dat_real, conf_fake, conf_real, latent_f, latent_r, fname):
    dat_fake = np.squeeze(dat_fake)
    dat_real = np.squeeze(dat_real)

    plt.figure(1, figsize=(16,10))
    gs1 = gridspec.GridSpec(FLAGS.seq_length + 1, 2)
    gs1.update(wspace=0.01, hspace=0.01)

    min_val = min(np.amin(dat_fake), np.amin(dat_real)) 
    max_val = max(np.amax(dat_fake), np.amax(dat_real)) 

    for i in range(FLAGS.seq_length):
        ax = plt.subplot(gs1[2*i])
        if i < FLAGS.seq_length - 1:
            ax.set_xticklabels([])
        ax.set_yticklabels([])
        if i == 0:
            ax.set_title(conf_fake)
        plt.imshow(dat_fake[i], interpolation="nearest", cmap="viridis", aspect='auto', vmin=min_val, vmax=max_val)

    for i in range(FLAGS.seq_length):
        ax = plt.subplot(gs1[2*i+1])
        if i < FLAGS.seq_length - 1:
            ax.set_xticklabels([])
        ax.set_yticklabels([])
        if i == 0:
            ax.set_title(conf_real)
        plt.imshow(dat_real[i], interpolation="nearest", cmap="viridis", aspect='auto', vmin=min_val, vmax=max_val)

    ax = plt.subplot(gs1[2*FLAGS.seq_length])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.imshow(latent_f, interpolation="nearest", cmap="viridis", aspect='auto')

    ax = plt.subplot(gs1[2*FLAGS.seq_length + 1])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.imshow(latent_r, interpolation="nearest", cmap="viridis", aspect='auto')

    plt.savefig(os.path.join(sample_dir, fname + ".png"))

def network(six_stack, df_dim=2, reuse=False, training=False):
    with tf.variable_scope("network", initializer=tf.contrib.layers.variance_scaling_initializer(uniform=True)) as outer_scope:
        confs = []
        for i in range(6):
            #, initializer=tf.contrib.layers.xavier_initializer(uniform=True)
            with tf.variable_scope("img_{}".format(i), initializer=tf.contrib.layers.variance_scaling_initializer(uniform=True)) as scope:
                if reuse:
                    scope.reuse_variables() # 16 128 1
                image = six_stack[:,i,:,:,:]
                h0 = ld.conv2d(image, (3, 12),(1, 2),32, name='h0_conv' + str(i)) # 16 64 32
                with tf.variable_scope('h1_pool'+ str(i)) as scope:
                    h1 = tf.nn.max_pool(h0,
                                           ksize=[1, 1, 2, 1],
                                           strides=[1, 1, 2, 1],
                                           padding='VALID') # 16 32 32
                h2 = ld.conv2d(h1, (3, 6),(1, 2),128, name='h2_conv' + str(i)) # 16 16 128
                with tf.variable_scope('h3_pool'+ str(i)) as scope:
                    h3 = tf.nn.max_pool(h2,
                                           ksize=[1, 2, 2, 1],
                                           strides=[1, 2, 2, 1],
                                           padding='VALID') # 8 8 128
                h4 = ld.spatial_pyramid_pool(h3, [1, 2, 4, 8], 64, name='h4_spp'+ str(i)) # 8 8 256
                h5 = ld.conv2d(h4, (3, 3), (2, 2), 256, name='h5_conv'+ str(i)) # 4 4 256
                with tf.variable_scope('h6_pool'+ str(i)) as scope:
                    h6 = tf.nn.max_pool(h5,
                                           ksize=[1, 2, 2, 1],
                                           strides=[1, 2, 2, 1],
                                           padding='VALID') # 2 2 256
                h7 = ld.conv2d(h6, (3, 3),(2, 2), 256, name='h7_conv'+ str(i)) # 1 1 256
                h8 = ld.fc_layer(h7, 256, name='h8_fc'+ str(i), linear=False, flat=True) # 256
                h9 = ld.fc_layer(h8, 16, name='h9_fc'+ str(i), linear=False, flat=False) # 16
                confs.append(h9)
        stacked_confs = tf.stack(confs, axis=1) 
        stacked_confs = tf.reshape(stacked_confs, [-1, 6, 16, 1])
        confs_lin = ld.conv2d(stacked_confs, (3, 6), (2, 4), 8, name='h_conv') # 3 4 16
        confs_lin = tf.reshape(confs_lin, [-1, 96]) # 6 64 2
        combined_conf = ld.fc_layer(confs_lin, 96, name='h_fc', linear=False, flat=False, input_shape=[-1, 96])
        combined_conf = ld.fc_layer(combined_conf, 1, name='h_fc_lin', linear=True, flat=False, input_shape=[-1, 96])
        combined_conf = tf.squeeze(combined_conf)
        return tf.sigmoid(combined_conf), combined_conf, stacked_confs

#network_template = tf.make_template('network', network, create_scope_now_=True)

def train():
    with tf.Graph().as_default():
        x1 = tf.placeholder(tf.float32, [None, FLAGS.seq_length, 16, 128, 1])
        y1 = tf.placeholder(tf.float32, [None])
        training = tf.placeholder(tf.bool, name='training')
        conf, logits, latent = network(x1, reuse=False, training=training)
        tf.summary.scalar('logits', tf.reduce_mean(logits))
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y1))
        tf.summary.scalar('loss', loss)
        t_vars = tf.trainable_variables()
        #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        # optimizer
        optim = tf.train.AdamOptimizer(FLAGS.lr).minimize(loss, var_list=t_vars)
        #train_op = tf.group([optim, update_ops])

        # Build a saver
        saver = tf.train.Saver(tf.global_variables())   
        # summary op
        summary_op = tf.summary.merge_all()

        # initialize session
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        if FLAGS.resume:
            # restore latest savepoint
            latest = tf.train.latest_checkpoint(FLAGS.train_dir)
            if not latest:
                print("No checkpoint to continue from in", FLAGS.train_dir)
                sys.exit(1)
            print("resume", latest)
            saver.restore(sess, latest)
            if FLAGS.export:
                # freeze model to pb
                export_dir = os.path.join('models', time.strftime("%Y%m%d-%H%M%S"))
                builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
                builder.add_meta_graph_and_variables(sess,
                      [tf.saved_model.tag_constants.TRAINING],
                      strip_default_attrs=True)
                builder.add_meta_graph([tf.saved_model.tag_constants.SERVING], strip_default_attrs=True)
                builder.save()
                sys.exit(0)
        else:
            print("init network from scratch")

        # create sample dir
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)
        graph_def = sess.graph.as_graph_def(add_shapes=True)
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, graph_def=graph_def)

        # load lists of train, test files
        files = find_files(FLAGS.train_data_index)
        files_by_channel = bin_files_by_channel(files)
        test_files = find_files(FLAGS.test_data_index)
        test_files_by_channel = bin_files_by_channel(test_files)

        for step in range(FLAGS.max_step):
            # load batch and train
            dat, is_real = load_batch_with_fakes(FLAGS.batch_size, files, files_by_channel, step, normalize=FLAGS.norm_input)
            _ = sess.run([optim], feed_dict={x1:dat, y1:is_real, training:True})
            if step % FLAGS.print_every == 0:
                summary_str, loss_r = sess.run([summary_op, loss], feed_dict={x1:dat, y1:is_real, training:False})
                print('step {}: loss={}'.format(step, loss_r))
                summary_writer.add_summary(summary_str, step) 
            if step % FLAGS.save_every == 0:
                # save checkpoint and plot
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)  
                print("saved to " + FLAGS.train_dir)
                print("now saving sample!", end='')
                test_dat_fake, test_dat_real = load_batch_fr_paired(1, test_files, test_files_by_channel, -1, normalize=FLAGS.norm_input)
                latent_f, test_conf_fake = sess.run([latent, conf], feed_dict={x1:test_dat_fake, training:False})
                latent_r, test_conf_real = sess.run([latent, conf], feed_dict={x1:test_dat_real, training:False})

                latent_img_f = np.squeeze(latent_f[0, :,:,:])
                latent_img_r = np.squeeze(latent_r[0, :,:,:])
                _plot_samples(test_dat_fake, test_dat_real, test_conf_fake, test_conf_real, latent_img_f, latent_img_r, '%07d' % step)
                print(" - saved")

def main(argv=None):  # pylint: disable=unused-argument
    # create train/test split
    if not os.path.isfile(FLAGS.train_data_index) or \
          not os.path.isfile(FLAGS.test_data_index):
              train_test_split(FLAGS.data_dir, FLAGS.train_data_index, FLAGS.test_data_index, FLAGS.split)
    if not FLAGS.resume:
        if tf.gfile.Exists(FLAGS.train_dir):
            tf.gfile.DeleteRecursively(FLAGS.train_dir)
        tf.gfile.MakeDirs(FLAGS.train_dir)
    train()

if __name__ == '__main__':
    tf.app.run()
