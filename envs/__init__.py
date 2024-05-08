#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 15:52:14 2018

@author: qiutian
"""
from gym.envs.registration import register

# 2D Navigation
# ----------------------------------------

register(
    'Navigation2D-v1',
    entry_point='envs.navigation:Navigation2DEnvV1',
    max_episode_steps=1000
)

register(
        'Navigation2D-v2',
        entry_point='envs.navigation:Navigation2DEnvV2',
        max_episode_steps=1000
        )

# register(
#         'Navigation2DDis-v2',
#         entry_point='envs.navigation:Navigation2DDisEnvV2',
#         max_episode_steps=100
#         )

register(
        'Navigation2D-v3',
        entry_point='envs.navigation:Navigation2DEnvV3',
        max_episode_steps=1000
        )
# register(
#         'Navigation2DDis-v3',
#         entry_point='envs.navigation:Navigation2DDisEnvV3',
#         max_episode_steps=100
#         )

register(
        'HalfCheetahVel-v1',
        entry_point = 'envs.mujoco.half_cheetah:HalfCheetahVelEnv',
        #entry_point = 'envs.utils:mujoco_wrapper',
        #kwargs = {'entry_point': 'envs.mujoco.half_cheetah:HalfCheetahVelEnv'},
        max_episode_steps=1000
        )

register(
        'ReacherDyna-v1',
        entry_point = 'envs.mujoco.reacher:ReacherDynaEnvV1',
        #entry_point = 'envs.utils:mujoco_wrapper',
        #kwargs = {'entry_point': 'envs.mujoco.reacher:ReacherDynaEnvV1'},
        max_episode_steps=1000
        )
# register(
#         'ReacherDynaDis-v1',
#         entry_point = 'envs.mujoco.reacher:ReacherDynaEnvDisV1',
#         #entry_point = 'envs.utils:mujoco_wrapper',
#         #kwargs = {'entry_point': 'envs.mujoco.reacher:ReacherDynaEnvV1'},
#         max_episode_steps = 100
#         )

register(
        'ReacherDyna-v2',
        entry_point = 'envs.mujoco.reacher:ReacherDynaEnvV2',
        #entry_point = 'envs.utils:mujoco_wrapper',
        #kwargs = {'entry_point': 'envs.mujoco.reacher:ReacherDynaEnvV2'},
        max_episode_steps=1000
        )

register(
        'ReacherDyna-v3',
        entry_point = 'envs.mujoco.reacher:ReacherDynaEnvV3',
        #entry_point = 'envs.utils:mujoco_wrapper',
        #kwargs = {'entry_point': 'envs.mujoco.reacher:ReacherDynaEnvV3'},
        max_episode_steps=1000
        )

register(
        'HopperVel-v1',
        entry_point = 'envs.mujoco.hopper:HopperVelEnv',
        #entry_point = 'envs.utils:mujoco_wrapper',
        #kwargs = {'entry_point': 'envs.mujoco.hopper:HopperVelEnv'},
        max_episode_steps=1000
        )

register(
        'HopperEnv3-v3',
        entry_point = 'envs.mujoco.hopperv3:HopperEnv3',
        #entry_point = 'envs.utils:mujoco_wrapper',
        #kwargs = {'entry_point': 'envs.mujoco.hopper:HopperVelEnv'},
        max_episode_steps=1000
        )
# register(
#         'HopperDisEnv3-v3',
#         entry_point = 'envs.mujoco.hopperv3:HopperDisEnv3',
#         #entry_point = 'envs.utils:mujoco_wrapper',
#         #kwargs = {'entry_point': 'envs.mujoco.hopper:HopperVelEnv'},
#         max_episode_steps = 100
#         )
register(
        'SwimmerVel-v1',
        entry_point = 'envs.mujoco.swimmer:SwimmerVelEnv',
        #entry_point = 'envs.utils:mujoco_wrapper',
        #kwargs = {'entry_point': 'envs.mujoco.swimmer:SwimmerVelEnv'},
        max_episode_steps = 1000
        )

register(
        'SwimmerEnv-v1',
        entry_point = 'envs.mujoco.swimmer_v3:SwimmerEnv',
        #entry_point = 'envs.utils:mujoco_wrapper',
        #kwargs = {'entry_point': 'envs.mujoco.swimmer:SwimmerVelEnv'},
        max_episode_steps=1000
        )
register(
    'InvertedPendulumEnv-v1',
    entry_point='envs.mujoco.inverted_pendulum:InvertedPendulumEnv',
    max_episode_steps=1000
    # reward_threshold=90.0,
)
register(
    'InvertedDoublePendulumEnv-v1',
    entry_point='envs.mujoco.inverted_double_pendulum:InvertedDoublePendulumEnv',
    max_episode_steps=1000
    # reward_threshold=90.0,
)
register(
    'HumanoidStandupEnv-v1',
    entry_point='envs.mujoco.humanoidstandup:HumanoidStandupEnv',
    max_episode_steps=1000
    # reward_threshold=90.0,
)
register(
    'BipedalWalker-v5',
    entry_point='envs.box2d.bipedal_walker:BipedalWalker',
    max_episode_steps=1000
    # reward_threshold=90.0,

)
register(
    'BipedalWalkerHardcore-v1',
    entry_point='envs.box2d.bipedal_walker:BipedalWalkerHardcore',
    max_episode_steps=1000
    # reward_threshold=90.0,

)
register(
    'LunarLanderContinuous-v1',
    entry_point='envs.box2d.lunar_lander:LunarLanderContinuous',
    max_episode_steps=1000
    # reward_threshold=90.0,
)
register(
    'HalfCheetahEnv-v1',
    entry_point='envs.mujoco.half_cheetah:HalfCheetahEnv',
    max_episode_steps=1000
    # reward_threshold=90.0,
)
register(
    'AntVelEnv-v1',
    entry_point='envs.mujoco.ant:AntVelEnv',
    max_episode_steps=1000
    # reward_threshold=90.0,
)

register(
    'AntEnv-v3',
    entry_point='envs.mujoco.ant_v3:AntEnv',
    max_episode_steps=1000
    # reward_threshold=90.0,
)
register(
    'HumanoidEnv-v3',
    entry_point='envs.mujoco.humanoid_v3:HumanoidEnv',
    max_episode_steps=1000
    # reward_threshold=90.0,
)
register(
    'HumanoidEnv-v1',
    entry_point='envs.mujoco.humanoid:HumanoidEnv',
    max_episode_steps=1000
    # reward_threshold=90.0,
)
register(
    'Walker2dEnv-v3',
    entry_point='envs.mujoco.walker2d_v3:Walker2dEnv',
    max_episode_steps=1000
    # reward_threshold=90.0,
)

register(
        'Walker2dEnv-v1',
        entry_point = 'envs.mujoco.walker2d:Walker2dEnv',
        #entry_point = 'envs.utils:mujoco_wrapper',
        #kwargs = {'entry_point': 'envs.mujoco.hopper:HopperVelEnv'},
        max_episode_steps=1000
        )