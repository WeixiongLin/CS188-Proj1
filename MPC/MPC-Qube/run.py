# coding: utf-8
import gym
import torch.utils.data as data
from dynamics import *
from controller import *
from utils import *
from quanser_robots.common import GentlyTerminating
import time
import random
import numpy as np


def predict(model, x):
    '''
    Given the current state and action, predict the next state

    :param x: (numpy array) current state and action in one array
    :return: (numpy array) next state numpy array
    '''
    x = np.array(x)
    # x = self.pre_process(x)
    # x_tensor = Variable(torch.FloatTensor(x).unsqueeze(0), volatile=True) # not sure here
    x_tensor = torch.Tensor(x)
    print("x_tensor", x_tensor)
    out_tensor = model(x_tensor)
    out = out_tensor.cpu().detach().numpy()
    # out = self.after_process(out)
    return out


def get_reward(obs, action_n):
    cos_th, sin_th, cos_al, sin_al, th_d, al_d = obs
    cos_th = min(max(cos_th, -1), 1)
    cos_al = min(max(cos_al, -1), 1)
    al=np.arccos(cos_al)
    th=np.arccos(cos_th)
    al_mod = al % (2 * np.pi) - np.pi
    action = action_n * 5
    cost = al_mod**2 + 5e-3*al_d**2 + 1e-1*th**2 + 2e-2*th_d**2 + 3e-3*action**2
    reward = np.exp(-cost)*0.02
    return reward


# 重写 evaluate 函数, 因为 load 进来的 model 是 MPC 而不是 Dynamic_model
# 所以需要进行精简和改写
def evaluate(model, gamma, state, action):
    """
    model = MPC
    gamma: decay rate
    state = [] 5 个元素
    action = [] 5 个元素, 表示之后 5 个 step 的动作
    """
    acc = 0  # 统计5 step 的 acc
    actions = np.array(action)
    horizon = actions.shape[0]
    rewards = 0
    state_tmp = state.copy()
    for j in range(horizon):
        # input_data = np.concatenate( (state_tmp, [actions[j]]) )
        input_data = state_tmp
        state_dt = predict(model, input_data)
        print("state_dt", state_dt[0][0])
        print("state_tmp", state_tmp)
        state_tmp = state_tmp + state_dt[0][0]
        rewards -= (gamma ** j) * get_reward(state_tmp, actions[j])
    return rewards


# datasets:  numpy array, size:[sample number, input dimension]
# labels:  numpy array, size:[sample number, output dimension]

env_id ="Qube-100-v0" # "CartPole-v0"
env = GentlyTerminating(gym.make(env_id))
config_path = "config.yml"
config = load_config(config_path)
print_config(config_path)


env = GentlyTerminating(gym.make(env_id))
n_max_steps = 500
render = False
num_actions = 10
action_tmp = []  # 用来存放 num_actions=10 次探索结果, 然后选取 step 数最小的为最优值

action_low = config["mpc_config"]["action_low"]
action_high = config["mpc_config"]["action_high"]
gamma = config["mpc_config"]["gamma"]

# 定义对 NewState 的评价函数
state = env.reset()
# state = torch.Tensor([0,0,0,0,0,0])

model = torch.load('exp_1.ckpt')
model = model.cpu()
# 预测 NewState
# out = model(state+action)

# evaluator = Evaluator(gamma=gamma)
# evaluator.update(state=state, dynamic_model=model)


# 共进行 num_actions=10 次对最佳 policy 的探索
for i in range(num_actions):
    action = [random.random() for x in range(5)]  # 随机生成初始的 action
    data_tmp = []
    label_tmp = []
    state_old = env.reset()  # 最开始的 old_state 是环境的初状态
    reward_episode = 0  # 一个 episode 之后 actions 带来的总的 reward

    left_action = np.array([action_low] * 5)
    right_action = np.array([action_high] * 5)
    cur_action = []
    for i in range(5):
        cur_action.append( left_action + random.random() * (right_action - left_action) )
    cur_action = np.array(cur_action)
    
    epsilon = 0.1  # 只要 action 的变化量大于 epsilon 就说明没有收敛
    action_delta = 1  # action 更新的变化量
    
    # 每一次探索的上限是 500 steps
    for j in range(n_max_steps):
        print("step: {j}")
        if render:
            env.render()
        # 更新 action, 向前走5步
        # 要 minimize reward
        while(action_delta > epsilon):
            action_delta = 0
            for step in range(5):  # 依次更新每一个 action
                left_reward = evaluate(model, gamma, state, left_action)
                right_reward = evaluate(model, gamma, state, right_action)
                if left_reward > right_reward:
                    action_delta += abs(cur_action[step] - right_action[step])
                    left_action[step] = cur_action[step]
                    cur_action[step] = (right_action[step] + cur_action[step]) / 2
                else:
                    action_delta += abs(cur_action[step] - left_action[step])
                    right_action[step] = cur_action[step]
                    cur_action[step] = (left_action[step] + cur_action[step]) / 2

        action = cur_action
        data_tmp.append(np.concatenate((state_old, action)))
        state_new, reward, done, info = env.step(action)
        reward_episode += reward
        label_tmp.append(state_new - state_old)
        if done:
            break
        state_old = state_new
