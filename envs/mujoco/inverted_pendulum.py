import numpy as np
from envs.mujoco.inverted_pendulum_env import InvertedPendulumEnv

class InvertedPendulumEnv(InvertedPendulumEnv):
    def __init__(self):
        self.stage = 'continuous'
        self._goal = np.array([0, 0], dtype=np.float32)
        super(InvertedPendulumEnv, self).__init__()

    def step(self, a):
        reward = 1.0
        action = np.clip(a, -1.0, 1.0)
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        notdone = np.isfinite(ob).all() and (np.abs(ob[1]) <= .2)
        done = not notdone
        return ob, reward, done, {}

    def reset_model(self):
        self.init_qpos = self._goal
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent

    def reset_task(self, task):
        ### task: a scalar
        self._goal = task

class InvertedPendulumDisEnv(InvertedPendulumEnv):
    def __init__(self):
        self.stage = 'dis9'
        self._goal = np.array([0, 0], dtype=np.float32)
        self.action_distance = 1
        super(InvertedPendulumEnv, self).__init__()

    def step(self, a):
        if a == 0:
            action = [-self.action_distance*0.25]
        elif a == 1:
            action = [-self.action_distance*0.5]
        elif a == 2:
            action = [-self.action_distance*0.75]
        elif a == 3:
            action = [-self.action_distance]
        elif a == 4:
            action = [0]
        elif a == 5:
            action = [self.action_distance*0.25]
        elif a == 6:
            action = [self.action_distance*0.5]
        elif a == 7:
            action = [self.action_distance*0.75]
        elif a == 8:
            action = [self.action_distance]
        reward = 1.0
        action = np.clip(action, -1.0, 1.0)
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        notdone = np.isfinite(ob).all() and (np.abs(ob[1]) <= .2)
        done = not notdone
        return ob, reward, done, {}

    def reset_model(self):
        self.init_qpos = self._goal
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent

    def reset_task(self, task):
        ### task: a scalar
        self._goal = task[0:2]
        self.action_distance = task[2]