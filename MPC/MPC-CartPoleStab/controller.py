import numpy as np
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import torch.utils.data as data
import pickle
from Hive import Hive
from Hive import Utilities
import time


class MPC(object):
    def __init__(self, env, config):
        self.env = env
        mpc_config = config["mpc_config"]
        self.horizon = mpc_config["horizon"]
        self.numb_bees = mpc_config["numb_bees"]
        self.max_itrs = mpc_config["max_itrs"]
        self.gamma = mpc_config["gamma"]
        self.action_low = mpc_config["action_low"]
        self.action_high = mpc_config["action_high"]
        self.evaluator = Evaluator(self.gamma)

    def act(self, state, dynamic_model):
        self.evaluator.update(state, dynamic_model)
        optimizer = Hive.BeeHive( lower = [float(self.action_low)] * self.horizon,
                                  upper = [float(self.action_high)] * self.horizon,
                                  fun = self.evaluator.evaluate,
                                  numb_bees = self.numb_bees,
                                  max_itrs = self.max_itrs,
                                  verbose=False)
        cost = optimizer.run()
        #print("Solution: ",optimizer.solution[0])
        #print("Fitness Value ABC: {0}".format(optimizer.best))
        #Utilities.ConvergencePlot(cost)
        return optimizer.solution[0]

class Evaluator(object):
    def __init__(self, gamma=0.8):
        self.gamma = gamma

    def update(self, state, dynamic_model):
        self.state = state
        self.dynamic_model = dynamic_model

    def evaluate(self, actions):
        actions = np.array(actions)
        horizon = actions.shape[0]
        rewards = 0
        state_tmp = self.state.copy()
        for j in range(horizon):
            input_data = np.concatenate( (state_tmp,[actions[j]]) )
            state_dt = self.dynamic_model.predict(input_data)
            state_tmp = state_tmp + state_dt[0]
            rewards -= (self.gamma ** j) * self.get_reward(state_tmp, actions[j])
        return rewards

    # need to change this function according to different environment
    def get_reward(self,obs, action_n):
        x, sin_th, cos_th, x_dot, theta_dot = obs
        cos_th = min(max(cos_th, -1), 1)
        reward = -cos_th + 1
        return reward

