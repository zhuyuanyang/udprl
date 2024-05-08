import numpy as np
from envs.mujoco.inverted_double_pendulum_env import InvertedDoublePendulumEnv

class InvertedDoublePendulumEnv(InvertedDoublePendulumEnv):

    def __init__(self):
        self.stage = 'continuous'
        self._goal = np.array([0, 0, 0], dtype=np.float32)
        super(InvertedDoublePendulumEnv, self).__init__()

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        x, _, y = self.sim.data.site_xpos[0]
        dist_penalty = 0.01 * x ** 2 + (y - 2) ** 2
        v1, v2 = self.sim.data.qvel[1:3]
        vel_penalty = 1e-3 * v1**2 + 5e-3 * v2**2
        alive_bonus = 10
        r = alive_bonus - dist_penalty - vel_penalty
        done = bool(y <= 1)
        return ob, r, done, {}

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos[:1],  # cart x pos
            np.sin(self.sim.data.qpos[1:]),  # link angles
            np.cos(self.sim.data.qpos[1:]),
            np.clip(self.sim.data.qvel, -10, 10),
            np.clip(self.sim.data.qfrc_constraint, -10, 10)
        ]).ravel()

    def reset_model(self):
        self.init_qpos = self._goal
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
            self.init_qvel + self.np_random.randn(self.model.nv) * .1
        )
        return self._get_obs()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent * 0.5
        v.cam.lookat[2] = 0.12250000000000005  # v.model.stat.center[2]

    def reset_task(self, task):
        ### task: a scalar
        self._goal = task

class InvertedDoublePendulumDisEnv(InvertedDoublePendulumEnv):
    def __init__(self):
        self.stage = 'dis3'
        self._goal = np.array([0, 0, 0], dtype=np.float32)
        self.action_distance = 1
        super(InvertedDoublePendulumEnv, self).__init__()

    def step(self, action):
        if action == 0:
            action = [-self.action_distance]
        elif action == 1:
            action = [-self.action_distance*0.75]
        elif action == 2:
            action = [-self.action_distance*0.5]
        elif action == 3:
            action = [-self.action_distance*0.25]
        elif action == 4:
            action = [0]
        elif action == 5:
            action = [self.action_distance*0.25]
        elif action == 6:
            action = [self.action_distance*0.5]
        elif action == 7:
            action = [self.action_distance*0.75]
        elif action == 8:
            action = [self.action_distance]
        a = np.clip(action, -1.0, 1.0)
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        x, _, y = self.sim.data.site_xpos[0]
        dist_penalty = 0.01 * x ** 2 + (y - 2) ** 2
        v1, v2 = self.sim.data.qvel[1:3]
        vel_penalty = 1e-3 * v1**2 + 5e-3 * v2**2
        alive_bonus = 10
        r = alive_bonus - dist_penalty - vel_penalty
        done = bool(y <= 1)
        return ob, r, done, {}

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos[:1],  # cart x pos
            np.sin(self.sim.data.qpos[1:]),  # link angles
            np.cos(self.sim.data.qpos[1:]),
            np.clip(self.sim.data.qvel, -10, 10),
            np.clip(self.sim.data.qfrc_constraint, -10, 10)
        ]).ravel()

    def reset_model(self):
        self.init_qpos = self._goal
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
            self.init_qvel + self.np_random.randn(self.model.nv) * .1
        )
        return self._get_obs()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent * 0.5
        v.cam.lookat[2] = 0.12250000000000005  # v.model.stat.center[2]

    def reset_task(self, task):
        ### task: a scalar
        self._goal = task[0:3]
        self.action_distance = task[3]
