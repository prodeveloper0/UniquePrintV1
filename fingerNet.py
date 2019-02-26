import tensorflow as tf
import tensorflow.contrib.layers as tcl


def leaky_relu(x, alpha=0.2, name='leaky_relu'):
    with tf.variable_scope(name):
        return tf.nn.leaky_relu(x, alpha)


class Generator(object):
    def __init__(self):
        self.name = 'Generator_fingerNet'

    def __call__(self, z):
        with tf.variable_scope(self.name):
            g = tcl.fully_connected(z, 16 * 16 * 128, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm,
                                    weights_initializer=tf.random_normal_initializer(0, 0.02))

            # bsx32768 -> bsx16x16x128
            g = tf.reshape(g, (-1, 16, 16, 128))

            # bsx16x16x128 -> bsx32x32x64
            g = tcl.conv2d_transpose(g, 64, 4, stride=2, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm,
                                     padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))

            # bsx32x32x64 -> bsx64x64x32
            g = tcl.conv2d_transpose(g, 32, 4, stride=2, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm,
                                     padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))

            # bsx64x64x32 -> bsx128x128x1
            g = tcl.conv2d_transpose(g, 1, 4, stride=2, activation_fn=tf.nn.sigmoid, normalizer_fn=tcl.batch_norm,
                                     padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))

            return g

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


class Discriminator(object):
    def __init__(self):
        self.name = 'Discriminator_fingerNet'

    def __call__(self, x, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            # bsx128x128x1 -> bsx64x64x32
            shared = tcl.conv2d(x, num_outputs=32, kernel_size=4, stride=2, activation_fn=leaky_relu,
                                normalizer_fn=tcl.batch_norm)

            # bsx64x64x32 -> bsx32x32x64
            shared = tcl.conv2d(shared, num_outputs=64, kernel_size=4, stride=2, activation_fn=leaky_relu,
                                normalizer_fn=tcl.batch_norm)

            # bsx32x32x64 -> bsx16x16x128
            shared = tcl.conv2d(shared, num_outputs=128, kernel_size=4, stride=2, activation_fn=leaky_relu,
                                normalizer_fn=tcl.batch_norm)

            # bsx16x16x128 -> bsx32768
            shared = tcl.flatten(shared)

            q = tcl.fully_connected(shared, 128, activation_fn=leaky_relu, normalizer_fn=tcl.batch_norm)
            q = tcl.fully_connected(q, 5, activation_fn=None)    # Five types fingerprints

            d = tcl.fully_connected(shared, 1, activation_fn=None,
                                    weights_initializer=tf.random_normal_initializer(0, 0.02))
            return d, q

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
