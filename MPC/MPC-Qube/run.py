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

# datasets:  numpy array, size:[sample number, input dimension]
# labels:  numpy array, size:[sample number, output dimension]

env_id ="Qube-100-v0" # "CartPole-v0"
env = GentlyTerminating(gym.make(env_id))
config_path = "config.yml"
config = load_config(config_path)
print_config(config_path)

# model = DynamicModel(config)

# data_fac = DatasetFactory(env,config)
# data_fac.collect_random_dataset()

# loss = model.train(data_fac.random_trainset,data_fac.random_testset)

# mpc = MPC(env,config)

# rewards_list = []
# for itr in range(config["dataset_config"]["n_mpc_itrs"]):
#     t = time.time()
#     print("**********************************************")
#     print("The reinforce process [%s], collecting data ..." % itr)
    # rewards = data_fac.collect_mpc_dataset(mpc, model)
    # trainset, testset = data_fac.make_dataset()
    # rewards_list += rewards

    # 正式开始编写 random shooting 的算法
    # 1. 建立 action[5]
    # gamma = 0.999
    # evaluator = Evaluator(gamma)

    # 2. 根据 reward 进行 action 的收敛工作
    # 3. reward_list append

    # plt.close("all")
    # plt.figure(figsize=(12, 5))
    # plt.title('Reward Trend with %s iteration' % itr)
    # plt.plot(rewards_list)
    # plt.savefig("storage/reward-" + str(model.exp_number) + ".png")
    # print("Consume %s s in this iteration" % (time.time() - t))
    # loss = model.train(trainset, testset)

env = GentlyTerminating(gym.make(env_id))
n_max_steps = 500
render = False
num_actions = 10
action_tmp = []  # 用来存放 num_actions=10 次探索结果, 然后选取 step 数最小的为最优值

action_low = config["mpc_config"]["action_low"]
action_high = config["mpc_config"]["action_high"]


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
                left_reward = evaluate(left_action)
                right_reward = evaluate(right_action)
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