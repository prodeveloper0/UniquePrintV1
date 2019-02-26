import tensorflow as tf
import numpy as np
import os

import Visualize


def numfmt(maxvalue, value):
    return ('{:0%d}' % (len(str(maxvalue)) + 1)).format(value)


def sample_z(number_of_classes, samples_per_class, z_dim):
    return np.random.uniform(-1., 1., size=(number_of_classes * samples_per_class, z_dim))  # bs x z_dim


def sample_y(number_of_classes, samples_per_class):
    y = np.zeros([number_of_classes * samples_per_class, samples_per_class])

    for i in range(number_of_classes * samples_per_class):
        y[i, int(i % samples_per_class)] = 1

    return y


def concat(z, y):
    return tf.concat([z, y], 1)


def conv_concat(x, y, size, y_dim):
    bz = tf.shape(x)[0]
    y = tf.reshape(y, [bz, 1, 1, y_dim])
    return tf.concat([x, y * tf.ones([bz, size, size, y_dim])], 3)  # bs x size x size x (channel + y_dim)


class CGAN(object):
    def __init__(self, generator, discriminator, data, max_to_keep=999, d_learning_rate=0.0005, g_learning_rate=0.0005):
        self.generator = generator
        self.discriminator = discriminator
        self.data = data

        # Structure declaration
        self.z_dim = self.data.z_dim
        self.y_dim = self.data.y_dim
        self.size = self.data.size
        self.channel = self.data.channel

        # Input placeholder
        self.X = tf.placeholder(tf.float32, shape=[None, self.size, self.size, self.channel])
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim])
        self.y = tf.placeholder(tf.float32, shape=[None, self.y_dim])

        # Nets
        self.G_sample = self.generator(concat(self.z, self.y))
        self.D_real, _ = self.discriminator(conv_concat(self.X, self.y, self.size, self.y_dim))
        self.D_fake, _ = self.discriminator(conv_concat(self.G_sample, self.y, self.size, self.y_dim), reuse=True)

        # Objective functions for cGAN
        d_term_loss_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_real,
                                                                               labels=tf.ones_like(self.D_real)))

        g_term_loss_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake,
                                                                               labels=tf.zeros_like(self.D_fake)))

        # Loss functions for cGAN
        self.D_loss = d_term_loss_d + g_term_loss_g
        self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake,
                                                                             labels=tf.ones_like(self.D_fake)))

        # Optimizer for cGAN
        self.D_solver = tf.train.AdamOptimizer(d_learning_rate).minimize(self.D_loss, var_list=self.discriminator.vars)
        self.G_solver = tf.train.AdamOptimizer(g_learning_rate).minimize(self.G_loss, var_list=self.generator.vars)

        # TensorFlow session and saver
        self.saver = tf.train.Saver(discriminator.vars + generator.vars, max_to_keep=max_to_keep)
        self.sess = tf.Session()

    def train(self, name, epoch=10, batch_size=100, checkpoint_period=100, count_per_type=5):
        self.sess.run(tf.global_variables_initializer())

        loss_log = list()
        sample_dir = os.path.join(name, 'Samples')
        weight_dir = os.path.join(name, 'Weights')

        for ep in range(epoch):
            x_batches, y_batches = self.data(batch_size)

            # Train all batches
            for x_batch, y_batch in zip(x_batches, y_batches):
                # Update D
                self.sess.run(self.D_solver, feed_dict={self.X: x_batch,
                                                        self.y: y_batch,
                                                        self.z: sample_z(len(x_batch), 1, self.z_dim)})

                # Update G
                self.sess.run(self.G_solver, feed_dict={self.y: y_batch,
                                                        self.z: sample_z(len(y_batch), 1, self.z_dim)})

            # Print loss
            if (ep % checkpoint_period) == 0 or ep < checkpoint_period:
                # Print losses
                d_loss_value = self.sess.run(self.D_loss, feed_dict={self.X: x_batches[0],
                                                                     self.y: y_batches[0],
                                                                     self.z: sample_z(len(x_batches[0]), 1,
                                                                                      self.z_dim)})

                g_loss_value = self.sess.run(self.G_loss, feed_dict={self.y: y_batches[0],
                                                                     self.z: sample_z(len(y_batches[0]), 1,
                                                                                      self.z_dim)})

                log = 'Iter: {}; D loss: {:.4}; G_loss: {:.4}'.format(numfmt(epoch, ep), d_loss_value, g_loss_value)
                loss_log.append(log)
                print(log)

            # Save checkpoint and image periodically
            if (ep % checkpoint_period) == 0:
                # Save images
                y_s = sample_y(count_per_type, self.y_dim)
                samples = self.sess.run(self.G_sample, feed_dict={self.y: y_s,
                                                                  self.z: sample_z(self.y_dim, count_per_type,
                                                                                   self.z_dim)}) * 255

                if not os.path.exists(sample_dir):
                    os.makedirs(sample_dir)

                fig = Visualize.sample2fig(samples, count_per_type, self.y_dim)
                Visualize.savefig(fig, os.path.join(sample_dir, '{}.png'.format(numfmt(epoch, ep))))

                # Save checkpoint
                self.saver.save(self.sess, os.path.join(weight_dir, 'weights.ckpt'),
                                global_step=ep, write_meta_graph=False)

        # Save checkpoint finally
        self.saver.save(self.sess, os.path.join(weight_dir, 'weights.ckpt'), write_meta_graph=False)

        # Save logs
        with open(os.path.join(name, 'losses.txt'), 'w') as f:
            for log in loss_log:
                f.write('%s\n' % log)

    def generate(self, demo=True, name='UniquePrintV1_FingerNet', count_per_type=5, checkpoint_epoch=None):
        ckpt = tf.train.get_checkpoint_state(os.path.join(name, 'Weights'))
        self.sess.run(tf.global_variables_initializer())

        if checkpoint_epoch is not None:
            for checkpoint in ckpt.all_model_checkpoint_paths:
                if str(checkpoint_epoch) == checkpoint.split(os.path.sep)[-1].split('-')[-1]:
                    self.saver.restore(self.sess, checkpoint)
                    break
        else:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)

        y_s = sample_y(count_per_type, self.y_dim)
        samples = self.sess.run(self.G_sample, feed_dict={self.y: y_s,
                                                          self.z: sample_z(self.y_dim, count_per_type, self.z_dim)})
        samples *= 255.

        if demo:
            fig = Visualize.sample2fig(samples, count_per_type, self.y_dim)
            Visualize.showfig(fig)

        return samples
