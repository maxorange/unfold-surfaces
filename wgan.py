import argparse
import datasets
import numpy as np
from ops import *
import plotutil
from pydoc import locate
import tensorflow as tf

class WGAN(object):

    def __init__(self, args):
        # Placeholders
        self.x_real = tf.placeholder(tf.float32, [args.batch_size, args.nx], 'x_real')
        z_noise = tf.placeholder(tf.float32, [args.batch_size, args.nz], 'z_noise')
        train_G = tf.placeholder(tf.bool, name='train_G')
        train_D = tf.placeholder(tf.bool, name='train_D')

        # Generator network
        self.x_fake = self.generator(args, z_noise, train_G)

        # Discriminator network
        self.y_real = self.discriminator(args, self.x_real, train_D)
        self.y_fake = self.discriminator(args, self.x_fake, train_D, True)

        # Get variables of generator and discriminator
        t_vars = tf.trainable_variables()
        G_vars = [var for var in t_vars if var.name.startswith('G')]
        D_vars = [var for var in t_vars if var.name.startswith('D')]

        # Loss functions
        self.G_loss = -tf.reduce_mean(self.y_fake)
        self.D_loss = -tf.reduce_mean(self.y_real) + tf.reduce_mean(self.y_fake)

        # WGAN gradient penalty
        alpha = tf.random_uniform((args.batch_size, 1), minval=0., maxval=1.)
        interps = alpha*self.x_real + (1 - alpha)*self.x_fake
        disc_interps = self.discriminator(args, interps, train_D, True)
        grads = tf.gradients(disc_interps, [interps])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(grads), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes - 1)**2)
        self.D_loss += args.lam*gradient_penalty

        # Optimization
        opt = tf.train.AdamOptimizer(args.learning_rate, 0.5)
        self.G_opt = opt.minimize(self.G_loss, var_list=G_vars)
        self.D_opt = opt.minimize(self.D_loss, var_list=D_vars)

    def generator(self, args, z, train, reuse=False):
        with tf.variable_scope('G', reuse=reuse):
            h = tf.nn.relu(batch_norm(linear(z, [args.nz, 128], 'h1'), train, 'bn1'))
            h = tf.nn.relu(batch_norm(linear(h, [128, 128], 'h2'), train, 'bn2'))
            h = tf.nn.relu(batch_norm(linear(h, [128, 128], 'h3'), train, 'bn3'))
            h = tf.nn.relu(batch_norm(linear(h, [128, 128], 'h4'), train, 'bn4'))
            h = tf.nn.relu(batch_norm(linear(h, [128, 128], 'h5'), train, 'bn5'))
            h = tf.nn.relu(batch_norm(linear(h, [128, 128], 'h6'), train, 'bn6'))
            h = tf.nn.relu(batch_norm(linear(h, [128, 128], 'h7'), train, 'bn7'))
            x = linear(h, [128, args.nx], 'h8', bias=True)
            return x

    def discriminator(self, args, x, train, reuse=False):
        with tf.variable_scope('D', reuse=reuse):
            h = leaky_relu(linear(x, [args.nx, 64], 'h1', bias=True))
            h = leaky_relu(batch_norm(linear(h, [64, 64], 'h2'), train, 'bn2'))
            h = leaky_relu(batch_norm(linear(h, [64, 64], 'h3'), train, 'bn3'))
            h = leaky_relu(batch_norm(linear(h, [64, 64], 'h4'), train, 'bn4'))
            h = leaky_relu(batch_norm(linear(h, [64, 64], 'h5'), train, 'bn5'))
            y = linear(h, [64, 1], 'h6', bias=True)
            return y

    def train(self, args):
        noise = locate('noises.{0}'.format(args.noise))(args)
        dataset = datasets.Dataset(args)

        x_real = dataset.training_data[:args.n_test_data].copy()
        z_test = noise.test_data

        config_proto = tf.ConfigProto()
        config_proto.gpu_options.allow_growth = True

        with tf.Session(config=config_proto) as sess:
            sess.run(tf.global_variables_initializer())
            for step in range(1, args.n_iters+1):
                # Update discriminator
                for _ in range(args.n_critic_iters):
                    x = dataset.get_training_data(args.batch_size)
                    z = noise.sample(args.batch_size)
                    D_loss, _ = sess.run([self.D_loss, self.D_opt], {
                        'x_real:0':x,
                        'z_noise:0':z,
                        'train_D:0':True,
                        'train_G:0':True,
                    })

                # Update generator
                z = noise.sample(args.batch_size)
                G_loss, _ = sess.run([self.G_loss, self.G_opt], {
                    'z_noise:0':z,
                    'train_D:0':True,
                    'train_G:0':True,
                })

                if step % args.log_interval == 0:
                    print 'step:{0:>6}, Ld:{1:>9.6f}, Lg:{2:>9.6f}'.format(step, D_loss, G_loss)

                if step % args.plot_interval == 0:
                    x_fake = self.sample_data(args, sess, z_test)
                    plotutil.save_plot(args, step, x_real, x_fake, z_test)

    def sample_data(self, args, sess, z_test):
        x_fake = np.zeros([args.n_test_data, args.nx])
        total_batch = args.n_test_data / args.batch_size
        stride = args.batch_size

        for i in range(total_batch):
            # Generated samples
            x_fake[stride*i:stride*(i+1)] = sess.run(self.x_fake, {
                'z_noise:0':z_test[stride*i:stride*(i+1)],
                'train_D:0':False,
                'train_G:0':False,
            })
        return x_fake

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nz', type=int, default=2)
    parser.add_argument('--nx', type=int, default=3)
    parser.add_argument('--n_test_data', type=int, default=10000)
    parser.add_argument('--n_iters', type=int, default=500000)
    parser.add_argument('--n_critic_iters', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--lam', type=float, default=0.1)
    parser.add_argument('--log_interval', type=int, default=1000)
    parser.add_argument('--plot_interval', type=int, default=100)
    parser.add_argument('--noise', type=str, default='Uniform')
    parser.add_argument('--filename', type=str, default='data/bunny.xyz')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    gan = WGAN(args)
    gan.train(args)
