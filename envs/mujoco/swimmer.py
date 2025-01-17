import numpy as np
from envs.mujoco.swimmer_env import SwimmerEnv

class SwimmerVelEnv(SwimmerEnv):
    def __init__(self):
        ### default goal velocity
        self._goal_vel = 0.1
        self.stage = 'continuous'
        super(SwimmerVelEnv, self).__init__()

    def step(self, a):
        a = np.clip(a, -1.0, 1.0)
        ctrl_cost_coeff = 0.05
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        vel_fwd = (xposafter - xposbefore) / self.dt
        reward_fwd = -1.0 * abs(vel_fwd - self._goal_vel)
        # reward_ctrl = - 0.05 * np.square(a).sum()
        reward_ctrl = - ctrl_cost_coeff * np.square(a).sum()
        reward = reward_fwd + reward_ctrl
        ob = self._get_obs()
        return ob, reward, False, dict(reward_fwd=reward_fwd, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos.flat[2:], 
            qvel.flat]).astype(np.float32).flatten()

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.1, high=.1, size=self.model.nv)
        )
        return self._get_obs()

    def reset_task(self, task):
        ### task: a scalar
        self._goal_vel = task

class SwimmerVelEnvDis(SwimmerEnv):
    def __init__(self):
        ### default goal velocity
        self._goal_vel = 1.0
        self.stage = 'dis'
        self.action_distance = 1
        super(SwimmerVelEnvDis, self).__init__()

    def step(self, a):
        dim_1 = a // 7
        dim_2 = a % 7
        action = [(-1 + (1 / 3) * dim_1) * self.action_distance, (-1 + (1 / 3) * dim_2) * self.action_distance]
        a = np.clip(action, -1.0, 1.0)
        ctrl_cost_coeff = 0.05
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        vel_fwd = (xposafter - xposbefore) / self.dt
        reward_fwd = -1.0 * abs(vel_fwd - self._goal_vel)
        # reward_ctrl = - 0.05 * np.square(a).sum()
        reward_ctrl = - ctrl_cost_coeff * np.square(a).sum()
        reward = reward_fwd + reward_ctrl
        ob = self._get_obs()
        return ob, reward, False, dict(reward_fwd=reward_fwd, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos.flat[2:],
            qvel.flat]).astype(np.float32).flatten()

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.1, high=.1, size=self.model.nv)
        )
        return self._get_obs()

    def reset_task(self, task):
        ### task: a scalar
        self._goal_vel = task[0]
        self.action_distance = task[-1]