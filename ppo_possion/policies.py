import numpy as np
import tensorflow as tf
from baselines.a2c.utils import conv, ortho_init, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm
from distributions import make_pdtype
import tensorflow_probability as tfp
from ordinal_utils import action_mask, construct_mask


class MlpDiscretePolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, tau, reuse=False):  # pylint: disable=W0613
        bins = 1 + ac_space.nvec.max()
        print('S is {}, A is {}, making policy bins size {}'.format(ob_space.shape, ac_space.shape, bins))
        assert bins is not None
        ob_shape = (nbatch,) + ob_space.shape
        actdim = ac_space.shape[0]
        self.pdtype = pdtype = make_pdtype(ac_space)
        X = tf.compat.v1.placeholder(tf.float32, ob_shape, name='Ob')  # obs
        with tf.compat.v1.variable_scope("model", reuse=reuse):
            h1 = fc(X, 'pi_fc1', nh = 128, init_scale = np.sqrt(2), act = tf.nn.tanh)
            h2 = fc(h1, 'pi_fc2', nh = 128, init_scale = np.sqrt(2), act = tf.nn.softplus)
            m = fc(h2, 'pi', actdim, act = tf.nn.softplus, init_scale = 0.01)  # of size [batchsize, num-actions*bins]

            #Poisson process
            norm_softm_tiled = tf.tile(tf.expand_dims(m, axis=-1), [1, 1, bins])
            log_possion = tf.math.log(norm_softm_tiled) #log f(x)
            nature = tf.expand_dims(tf.linspace(start = 1.0, stop = bins, num = bins), 0)
            nature = tf.tile(tf.expand_dims(nature, 0), [norm_softm_tiled.shape[0], actdim, 1] )# batch * dimension * bins, 1,2,3,4,....,
            factorial_tensor = tf.expand_dims([tf.constant(np.math.factorial(i), dtype=tf.float32) for i in range(1, bins+1, 1)], axis = 0 )
            factorial_tensor = tf.math.log(tf.tile(tf.expand_dims(factorial_tensor, axis = 0), [norm_softm_tiled.shape[0], actdim, 1])) # batch * dimension * bins, log(n!)
            norm_softm_tiled_new = tf.multiply(nature, log_possion) - norm_softm_tiled - factorial_tensor


            pdparam = gumbel_softmax(norm_softm_tiled_new,tau)
            # pdparam = tf.reshape(pdparam, [nbatch, -1])
            norm_softm_tiled = tf.tile(tf.expand_dims(pdparam, axis=-1), [1, 1, 1, 1])

            # construct the mask
            am_numpy = construct_mask(bins)
            am_tf = tf.constant(am_numpy, dtype=tf.float32)
            #
            # # construct pdparam
            pdparam = tf.reduce_sum(
                 tf.math.log(norm_softm_tiled + 1e-8) * am_tf + tf.math.log(1 - norm_softm_tiled + 1e-8) * (1 - am_tf),
                 axis=-1)
            pdparam = tf.reshape(pdparam, [nbatch, -1])
            # value function
            h1 = fc(X, 'vf_fc1', nh=128, init_scale=np.sqrt(2), act=tf.tanh)
            h2 = fc(h1, 'vf_fc2', nh=128, init_scale=np.sqrt(2), act=tf.tanh)
            vf = fc(h2, 'vf', 1, act=lambda x: x)[:, 0]

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pdparam)
        a0 = self.pd.sample()  # [0,1]
        # neglogp0 = tf.reduce_sum(-self.pd.log_prob(a0), axis=-1)
        neglogp0 = self.pd.neglogp(a0)

        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X: ob})

            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X: ob})

        self.X = X

        self.vf = vf
        self.step = step
        self.value = value


def fc(x, scope, nh, act=tf.nn.relu, init_scale=1.0):
    with tf.variable_scope(scope):
        nin = x.get_shape()[1].value
        w = tf.compat.v1.get_variable("w", [nin, nh], initializer=ortho_init(init_scale))
        b = tf.compat.v1.get_variable("b", [nh], initializer=tf.constant_initializer(0.0))
        z = tf.matmul(x, w) + b
        h = act(z)
        return h

def possion_mask(x, n_dim, bins):
    possion = tf.Variable(tf.zeros_like(x))
    batch = (x.shape)[0]
    for h in range(batch):
        for i in range(n_dim):
            for j in range(bins):
                factorial_tensor =  tf.constant(np.math.factorial(j+1), dtype = tf.float32)
                tf.assign(possion[h,i,j], (((x[h,i,j]) ** (j+1) )* tf.math.exp(-x[h,i,j])) / (factorial_tensor))

    return possion

def possion_log(x, n_dim, bins):
    possion = tf.Variable(tf.ones_like(x))
    batch = (x.shape)[0]
    for h in range(batch):
        for i in range(n_dim):
            for j in range(bins):
                factorial_tensor =  tf.constant(np.math.factorial(j+1), dtype = tf.float32)
                tf.assign(possion[h,i,j], -((j+1) * tf.math.log(x[h,i,j]) - x[h,i,j] - tf.math.log(factorial_tensor)))


    return possion

def gumbel_softmax(x, tau=0.4 ):
    return tf.nn.softmax(x/tau)
