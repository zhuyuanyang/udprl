import numpy as np
import tensorflow as tf, tensorflow_probability as tfp
from baselines.a2c.utils import ortho_init
from baselines.common.distributions import make_pdtype


class MlpPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False): #pylint: disable=W0613
        ob_shape = (nbatch,) + ob_space.shape
        actdim = ac_space.shape[0]
        X = tf.placeholder(tf.float32, ob_shape, name='Ob') #obs
        with tf.variable_scope("model", reuse=reuse):
            h1 = fc(X, 'pi_fc1', nh=64, init_scale=np.sqrt(2), act=tf.tanh)
            h2 = fc(h1, 'pi_fc2', nh=64, init_scale=np.sqrt(2), act=tf.tanh)
            alpha = tf.nn.softplus(fc(h2, 'alpha', actdim, act=lambda x:x, init_scale=0.001)) + 1.0
            beta = tf.nn.softplus(fc(h2, 'beta', actdim, act=lambda x:x, init_scale=0.001)) + 1.0
            h1 = fc(X, 'vf_fc1', nh=64, init_scale=np.sqrt(2), act=tf.tanh)
            h2 = fc(h1, 'vf_fc2', nh=64, init_scale=np.sqrt(2), act=tf.tanh)
            vf = fc(h2, 'vf', 1, act=lambda x:x)[:,0]

        self.pd = tfp.distributions.Beta(alpha, beta)

        a0 = self.pd.sample() # [0,1]
        neglogp0 = tf.reduce_sum(-self.pd.log_prob(a0), axis=-1)

        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})

        self.X = X
        self.alpha = alpha
        self.beta = beta
        self.vf = vf
        self.step = step
        self.value = value

def fc(x, scope, nh, act=tf.nn.relu, init_scale=1.0):
    with tf.variable_scope(scope):
        nin = x.get_shape()[1].value
        w = tf.get_variable("w", [nin, nh], initializer=ortho_init(init_scale))
        b = tf.get_variable("b", [nh], initializer=tf.constant_initializer(0.0))
        z = tf.matmul(x, w)+b
        h = act(z)
        return h