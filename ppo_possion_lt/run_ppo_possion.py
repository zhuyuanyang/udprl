#!/usr/bin/env python
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from baselines import bench, logger
import sys
sys.path.append("..")
from envs.mujoco import *
import pybulletgym
import time
def train(env_id, num_timesteps, seed, lr, entcoef, continue_train, nsteps, bins,cpus):
    from baselines.common import set_global_seeds
    from baselines.common.vec_env.vec_normalize import VecNormalize
    import ppo2
    from policies import MlpDiscretePolicy
    import gym
    import tensorflow as tf
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
    ncpu = cpus
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    tf.compat.v1.Session(config=config).__enter__()
    import wrapper
    def make_env():
        env = gym.make(env_id)
        env = wrapper.discretizing_wrapper(env, bins)
        env = bench.Monitor(env, logger.get_dir())
        return env
    env = DummyVecEnv([make_env])
    env = VecNormalize(env)

    set_global_seeds(seed)
    policy = MlpDiscretePolicy

    ppo2.learn(policy=policy, env=env, nsteps=nsteps, nminibatches=32,
        lam=0.95, gamma=0.99, noptepochs=10, log_interval=1,
        ent_coef=entcoef,
        lr=lr,
        cliprange=0.2,
        total_timesteps=num_timesteps)

time_record = time.strftime("%d%h", time.localtime(time.time()))
def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='Humanoid-v3')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(5e6))
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--entcoef', type=float, default=0)
    parser.add_argument('--continue-train', type=int, default=1) # 1 for continued training
    parser.add_argument('--nsteps', type=int, default=2048)
    parser.add_argument('--bins', type=int, default=11)
    parser.add_argument('--cpus', type=int, default=10)
    args = parser.parse_args()
    logger.configure(dir = '../../exp/'+ str(args.env) + '/ppo_poisson_lt' + '-'+ str(args.bins)+ '-'+ str(args.seed)+ '-' +str(time_record))
    # logger.configure(dir = 'ppo_poisson'+'-' + str(args.env) + '-'+str(args.bins)+ '-'+ time_record)
    train(args.env, nsteps=args.nsteps, entcoef=args.entcoef, num_timesteps=args.num_timesteps, seed=args.seed, lr=args.lr, continue_train=args.continue_train, bins=args.bins, cpus = args.cpus)

if __name__ == '__main__':
    main()

