#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 13:22:59 2018

@author: qiutian
"""
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding

class Navigation2DEnvV1(gym.Env):
    """2D navigation problems, as described in [1]. The code is adapted from 
    https://github.com/cbfinn/maml_rl/blob/9c8e2ebd741cb0c7b8bf2d040c4caeeb8e06cc95/maml_examples/point_env_randgoal.py
    At each time step, the 2D agent takes an action (its velocity, clipped in
    [-0.1, 0.1]), and receives a penalty equal to its L2 distance to the goal 
    position (ie. the reward is `-distance`). The 2D navigation tasks are 
    generated by sampling goal positions from the uniform distribution 
    on [-0.5, 0.5]^2.
    [1] Chelsea Finn, Pieter Abbeel, Sergey Levine, "Model-Agnostic 
        Meta-Learning for Fast Adaptation of Deep Networks", 2017 
        (https://arxiv.org/abs/1703.03400)
    """
    def __init__(self):
        super(Navigation2DEnvV1, self).__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
            shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(low=-0.1, high=0.1,
            shape=(2,), dtype=np.float32)

        ### the default goal position
        self._goal =  np.array([0.25,0.25], dtype=np.float32)
        self._state = np.zeros(2, dtype=np.float32)
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset_task(self, task):
        ### task: a 2-dimensional array
        self._goal = task[0:2]

    def reset(self, env=True):
        self._state = np.zeros(2, dtype=np.float32)
        return self._state

    def step(self, action):
        action = np.clip(action, -0.1, 0.1)
        self._state = self._state + action

        x = self._state[0] - self._goal[0]
        y = self._state[1] - self._goal[1]
        reward_dist = - np.sqrt(x ** 2 + y ** 2)
        reward_ctrl = - 0.01 * np.square(action).sum()
        reward = reward_dist + reward_ctrl

        done = ((np.abs(x) < 0.03) and (np.abs(y) < 0.03))

        return self._state, reward, done, {}

class Navigation2DDisEnvV1(gym.Env):
    """2D navigation problems, as described in [1]. The code is adapted from
    https://github.com/cbfinn/maml_rl/blob/9c8e2ebd741cb0c7b8bf2d040c4caeeb8e06cc95/maml_examples/point_env_randgoal.py
    At each time step, the 2D agent takes an action (its velocity, clipped in
    [-0.1, 0.1]), and receives a penalty equal to its L2 distance to the goal
    position (ie. the reward is `-distance`). The 2D navigation tasks are
    generated by sampling goal positions from the uniform distribution
    on [-0.5, 0.5]^2.
    [1] Chelsea Finn, Pieter Abbeel, Sergey Levine, "Model-Agnostic
        Meta-Learning for Fast Adaptation of Deep Networks", 2017
        (https://arxiv.org/abs/1703.03400)
    """
    def __init__(self):
        super(Navigation2DDisEnvV1, self).__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
            shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(9)
        # self.action_space = spaces.Box(low=-0.1, high=0.1,
        #     shape=(2,), dtype=np.float32)
        self.action_distance = 0.1
        ### the default goal position
        self._goal =  np.array([0.25,0.25], dtype=np.float32)
        self._state = np.zeros(2, dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset_task(self, task):
        ### task: a 2-dimensional array
        self._goal = task[0:2]
        self.action_distance = task[-1]


    def reset(self, env=True):
        self._state = np.zeros(2, dtype=np.float32)
        return self._state

    def step(self, action):
        if action ==0 :
            action = [-self.action_distance,self.action_distance]
        elif action == 1:
            action = [-self.action_distance, 0]
        elif action == 2:
            action = [-self.action_distance, -self.action_distance]
        elif action == 3:
            action = [0, self.action_distance]
        elif action == 4:
            action = [0, 0]
        elif action == 5:
            action = [0, -self.action_distance]
        elif action == 6:
            action = [self.action_distance, self.action_distance]
        elif action == 7:
            action = [self.action_distance, 0]
        elif action == 8:
            action = [self.action_distance, -self.action_distance]

        action = np.clip(action, -0.1, 0.1)
        self._state = self._state + action

        x = self._state[0] - self._goal[0]
        y = self._state[1] - self._goal[1]
        reward_dist = - np.sqrt(x ** 2 + y ** 2)
        reward_ctrl = - 0.01 * np.square(action).sum()
        reward = reward_dist + reward_ctrl

        done = ((np.abs(x) < 0.03) and (np.abs(y) < 0.03))

        return self._state, reward, done, {}

class Navigation2DEnvV2(gym.Env):
    def __init__(self):
        super(Navigation2DEnvV2, self).__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
            shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(low=-0.1, high=0.1,
            shape=(2,), dtype=np.float32)

        ### the default goal position 
        self._goal = np.array([0.5, 0.5], dtype=np.float32)
        self._state = np.array([-0.1, -0.1], dtype=np.float32)

        ### three puddles with different sizes
        self._r_small = 0.1; self._r_medium = 0.2; self._r_large = 0.3
        self._small = np.array([-0.2, 0.2], dtype=np.float32)
        self._medium = np.array([0.2, 0.3], dtype=np.float32)
        self._large = np.array([0.2, -0.3], dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset_task(self, task):
        ### task: a 6-dimensional array
        task = np.array(task, dtype=np.float32).reshape(-1)
        self._small = task[0:2]
        self._medium = task[2:4]
        self._large = task[4:6]

    def reset(self, env=True):
        self._state = np.array([-0.1,-0.1], dtype=np.float32)
        return self._state

    def step(self, action):
        action = np.clip(action, -0.1, 0.1)
        # assert self.action_space.contains(action)
        temp_state = self._state + action
        navigable = self.check_puddle(temp_state)
        if navigable:
            self._state = temp_state 
            reward_puddle = 0.0
        else:
            reward_puddle = -0.1

        x = self._state[0] - self._goal[0]
        y = self._state[1] - self._goal[1]
        reward_dist = -np.sqrt(x ** 2 + y ** 2)
        reward_ctrl = - 0.01 * np.square(action).sum()
        reward = reward_dist + reward_puddle + reward_ctrl 
        done = ((np.abs(x) < 0.03) and (np.abs(y) < 0.03))

        return self._state, reward, done, {}

    def check_puddle(self, pos):
        navigable = True 
        x = pos[0]; y = pos[1]
        dist_small = np.sqrt((x-self._small[0])**2 + (y-self._small[1])**2)
        if dist_small <= self._r_small:
            navigable = False 
        dist_medium = np.sqrt((x-self._medium[0])**2 + (y-self._medium[1])**2)
        if dist_medium <= self._r_medium:
            navigable = False 
        dist_large = np.sqrt((x-self._large[0])**2 + (y-self._large[1])**2)
        if dist_large <= self._r_large:
            navigable = False 
        return navigable 

class Navigation2DDisEnvV2(gym.Env):
    def __init__(self):
        super(Navigation2DDisEnvV2, self).__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
            shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(49)
        self.action_distance = 0.1
        ### the default goal position
        self._goal = np.array([0.5, 0.5], dtype=np.float32)
        self._state = np.array([-0.1, -0.1], dtype=np.float32)

        ### three puddles with different sizes
        self._r_small = 0.1; self._r_medium = 0.2; self._r_large = 0.3
        self._small = np.array([-0.2, 0.2], dtype=np.float32)
        self._medium = np.array([0.2, 0.3], dtype=np.float32)
        self._large = np.array([0.2, -0.3], dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset_task(self, task):
        ### task: a 6-dimensional array
        task = np.array(task, dtype=np.float32).reshape(-1)
        self._small = task[0:2]
        self._medium = task[2:4]
        self._large = task[4:6]
        self.action_distance = task[-1]


    def reset(self, env=True):
        self._state = np.array([-0.1,-0.1], dtype=np.float32)
        return self._state

    def step(self, action):
        dim_1 = action//7
        dim_2 = action%7
        action = [(-1 + (1/3) * dim_1) * self.action_distance, (-1 + (1/3) * dim_2)* self.action_distance]
        action = np.clip(action, -0.1, 0.1)
        # assert self.action_space.contains(action)
        temp_state = self._state + action
        navigable = self.check_puddle(temp_state)
        if navigable:
            self._state = temp_state
            reward_puddle = 0.0
        else:
            reward_puddle = -0.1

        x = self._state[0] - self._goal[0]
        y = self._state[1] - self._goal[1]
        reward_dist = -np.sqrt(x ** 2 + y ** 2)
        reward_ctrl = - 0.01 * np.square(action).sum()
        reward = reward_dist + reward_puddle + reward_ctrl
        done = ((np.abs(x) < 0.03) and (np.abs(y) < 0.03))

        return self._state, reward, done, {}

    def check_puddle(self, pos):
        navigable = True
        x = pos[0]; y = pos[1]
        dist_small = np.sqrt((x-self._small[0])**2 + (y-self._small[1])**2)
        if dist_small <= self._r_small:
            navigable = False
        dist_medium = np.sqrt((x-self._medium[0])**2 + (y-self._medium[1])**2)
        if dist_medium <= self._r_medium:
            navigable = False
        dist_large = np.sqrt((x-self._large[0])**2 + (y-self._large[1])**2)
        if dist_large <= self._r_large:
            navigable = False
        return navigable

class Navigation2DEnvV3(gym.Env):
    def __init__(self):
        super(Navigation2DEnvV3, self).__init__()

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
            shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(low=-0.1, high=0.1,
            shape=(2,), dtype=np.float32)

        self._goal = np.array([0.5,0.5], dtype=np.float32)
        self._state = np.zeros(2, dtype=np.float32)
        self.seed()
        self._r_small = 0.1; self._r_medium = 0.15; self._r_large = 0.2
        self._small = np.array([-0.25, 0.25], dtype=np.float32)
        self._medium = np.array([0, -0.25], dtype=np.float32)
        self._large = np.array([0.25, 0.25], dtype=np.float32)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset_task(self, task):
        ### task: a 8-dimensional array
        task = np.array(task, dtype=np.float32).reshape(-1)
        self._small = task[:2]
        self._medium = task[2:4]
        self._large = task[4:6]
        self._goal = task[6:8]

    def reset(self, env=True):
        self._state = np.zeros(2, dtype=np.float32)
        return self._state

    def step(self, action):
        action = np.clip(action, -0.1, 0.1)
        # assert self.action_space.contains(action)
        temp_state = self._state + action
        navigable = self.check_puddle(temp_state)
        if navigable:
            self._state = temp_state 
            reward_puddle = 0.0
        else:
            reward_puddle = -0.1

        x = self._state[0] - self._goal[0]
        y = self._state[1] - self._goal[1]
        reward_dist = -np.sqrt(x ** 2 + y ** 2)
        reward = reward_dist + reward_puddle 
        done = ((np.abs(x) < 0.01) and (np.abs(y) < 0.01))

        return self._state, reward, done, {}

    def check_puddle(self, pos):
        navigable = True 
        x = pos[0]; y = pos[1]
        dist_small = np.sqrt((x-self._small[0])**2 + (y-self._small[1])**2)
        if dist_small <= self._r_small:
            navigable = False 
        dist_medium = np.sqrt((x-self._medium[0])**2 + (y-self._medium[1])**2)
        if dist_medium <= self._r_medium:
            navigable = False 
        dist_large = np.sqrt((x-self._large[0])**2 + (y-self._large[1])**2)
        if dist_large <= self._r_large:
            navigable = False 
        return navigable 

class Navigation2DDisEnvV3(gym.Env):
    def __init__(self):
        super(Navigation2DDisEnvV3, self).__init__()

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
            shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(9)


        self._goal = np.array([0.5,0.5], dtype=np.float32)
        self._state = np.zeros(2, dtype=np.float32)
        self.seed()
        self._r_small = 0.1; self._r_medium = 0.15; self._r_large = 0.2
        self._small = np.array([-0.25, 0.25], dtype=np.float32)
        self._medium = np.array([0, -0.25], dtype=np.float32)
        self._large = np.array([0.25, 0.25], dtype=np.float32)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset_task(self, task):
        ### task: a 8-dimensional array
        task = np.array(task, dtype=np.float32).reshape(-1)
        self._small = task[:2]
        self._medium = task[2:4]
        self._large = task[4:6]
        self._goal = task[6:8]
        self.action_distance = task[8:9][0]

    def reset(self, env=True):
        self._state = np.zeros(2, dtype=np.float32)
        return self._state

    def step(self, action):
        if action == 0:
            action = [-self.action_distance, self.action_distance]
        elif action == 1:
            action = [-self.action_distance, 0]
        elif action == 2:
            action = [-self.action_distance, -self.action_distance]
        elif action == 3:
            action = [0, self.action_distance]
        elif action == 4:
            action = [0, 0]
        elif action == 5:
            action = [0, -self.action_distance]
        elif action == 6:
            action = [self.action_distance, self.action_distance]
        elif action == 7:
            action = [self.action_distance, 0]
        elif action == 8:
            action = [self.action_distance, -self.action_distance]
        action = np.clip(action, -0.1, 0.1)
        # assert self.action_space.contains(action)
        temp_state = self._state + action
        navigable = self.check_puddle(temp_state)
        if navigable:
            self._state = temp_state
            reward_puddle = 0.0
        else:
            reward_puddle = -0.1

        x = self._state[0] - self._goal[0]
        y = self._state[1] - self._goal[1]
        reward_dist = -np.sqrt(x ** 2 + y ** 2)
        reward = reward_dist + reward_puddle
        done = ((np.abs(x) < 0.01) and (np.abs(y) < 0.01))

        return self._state, reward, done, {}

    def check_puddle(self, pos):
        navigable = True
        x = pos[0]; y = pos[1]
        dist_small = np.sqrt((x-self._small[0])**2 + (y-self._small[1])**2)
        if dist_small <= self._r_small:
            navigable = False
        dist_medium = np.sqrt((x-self._medium[0])**2 + (y-self._medium[1])**2)
        if dist_medium <= self._r_medium:
            navigable = False
        dist_large = np.sqrt((x-self._large[0])**2 + (y-self._large[1])**2)
        if dist_large <= self._r_large:
            navigable = False
        return navigable