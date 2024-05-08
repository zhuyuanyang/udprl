import numpy as np
from envs.mujoco.hopper_env import HopperEnv


class HopperVelEnv(HopperEnv):
    def __init__(self):
        self.stage = 'continuous'
        self._goal_vel = 1.0

        super(HopperVelEnv, self).__init__()

    def step(self, a):

        a = np.clip(a, -1.0, 1.0)
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        done = not (
            np.isfinite(s).all()
            and (np.abs(s[2:]) < 100).all()
            and (height > 0.7)
            and (abs(ang) < 0.2)
        )
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            np.clip(self.sim.data.qvel.flat, -10, 10)
        ]).astype(np.float32).flatten()

    def reset_task(self, task):
        ### task: a scalar
        self._goal_vel = task[0]


class HopperVelEnvDis(HopperEnv):
    def __init__(self):
        self.stage = 'dis3'
        self._goal_vel = 1.0
        self.action_distance = 1
        super(HopperVelEnvDis, self).__init__()

    def step(self, a):
        K = 11
        action_table = np.reshape([np.linspace(-1, 1, K) for i in range(3)], [3, K])
        action_cont = action_table[np.arange(3), a]
        a = np.clip(action_cont, -1.0, 1.0)
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]

        alive_bonus = 1.0
        forward_vel = (posafter - posbefore) / self.dt
        forward_reward = 1.0 - 1.0 * abs(forward_vel - self._goal_vel)
        ctrl_cost = 0.15 - 0.05 * np.square(a).sum()
        reward = forward_reward + ctrl_cost + alive_bonus
        reward_norm = np.clip(reward, 0, np.inf)

        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .7) and (abs(ang) < .2))
        ob = self._get_obs()

        return ob, reward_norm, done, {}

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            np.clip(self.sim.data.qvel.flat, -10, 10)
        ]).astype(np.float32).flatten()

    def reset_task(self, task):
        ### task: a scalar
        self._goal_vel = task[0]
        self.action_distance = task[-1]
