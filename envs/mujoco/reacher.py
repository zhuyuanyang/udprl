import numpy as np
from envs.mujoco.reacher_env import ReacherEnv
from envs.mujoco import mujoco_env
from gym import utils


class ReacherDynaEnvV1(ReacherEnv):
    def __init__(self):
        self.stage = 'continuous'
        self.goal = np.array([0.1,0.1], dtype=np.float32)
        super(ReacherDynaEnvV1, self).__init__()

    def reset_task(self, task):
        ### task: a 2-dimensional array
        task = np.array(task, dtype=np.float32).reshape(-1)
        self.goal = task

    def reset_model(self):
        qpos = self.np_random.uniform(low=-.005, high=.005, size=self.model.nq) + self.init_qpos
        qpos[-2:] = self.goal 
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def step(self, a):
        a = np.clip(a, -1.0, 1.0)
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - 0.01 * np.square(a).sum()
        reward = reward_dist + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat[:2],
            self.get_body_com("fingertip") - self.get_body_com("target")
            ]).astype(np.float32).flatten()

class ReacherDynaEnvDisV1(ReacherEnv):
    def __init__(self):
        self.goal = np.array([0.1,0.1], dtype=np.float32)
        self.stage = 'dis'
        self.action_distance = 1
        super(ReacherDynaEnvDisV1, self).__init__()



    def reset_task(self, task):
        ### task: a 2-dimensional array
        task = np.array(task, dtype=np.float32).reshape(-1)
        self.goal = task[0:2]
        self.action_distance = task[-1]

    def reset_model(self):
        qpos = self.np_random.uniform(low=-.005, high=.005, size=self.model.nq) + self.init_qpos
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def step(self, a):
        dim_1 = a // 7
        dim_2 = a % 7
        action = [(-1 + (1 / 3) * dim_1) * self.action_distance, (-1 + (1 / 3) * dim_2) * self.action_distance]
        # if a == 0:
        #     action = [-self.action_distance,self.action_distance]
        # elif a == 1:
        #     action = [-self.action_distance, 0]
        # elif a == 2:
        #     action = [-self.action_distance, -self.action_distance]
        # elif a == 3:
        #     action = [0, self.action_distance]
        # elif a == 4:
        #     action = [0, 0]
        # elif a == 5:
        #     action = [0, -self.action_distance]
        # elif a == 6:
        #     action = [self.action_distance, self.action_distance]
        # elif a == 7:
        #     action = [self.action_distance, 0]
        # elif a == 8:
        #     action = [self.action_distance, -self.action_distance]
        a = np.clip(action, -1.0, 1.0)
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - 0.01 * np.square(a).sum()
        reward = reward_dist + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat[:2],
            self.get_body_com("fingertip") - self.get_body_com("target")
            ]).astype(np.float32).flatten()


class ReacherDynaEnvV2(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.stage = 'continuous'
        self.goal = np.array([0.1,0.1], dtype=np.float32)

    def step(self, a):
        a = np.clip(a, -1.0, 1.0)
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat[:2],
            self.get_body_com("fingertip") - self.get_body_com("target")
        ]).astype(np.float32).flatten()

    def reset_task(self, task):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'reacher_%d.xml' % task, 2)


class ReacherDynaEnvV3(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.stage = 'continuous'
        self.goal = np.array([0.3,0.3], dtype=np.float32)

    def step(self, a):
        a = np.clip(a, -1.0, 1.0)
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat[:2],
            self.get_body_com("fingertip") - self.get_body_com("target")
        ]).astype(np.float32).flatten()

    def reset_task(self, task):
        self.goal = np.array(task[:2], dtype=np.float32).reshape(-1)
        utils.EzPickle.__init__(self)
        phy_index = int(task[2])
        mujoco_env.MujocoEnv.__init__(self, 'reacher_%d.xml' % phy_index, 2)
        



