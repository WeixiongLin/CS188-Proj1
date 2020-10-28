import gym
from quanser_robots.common import GentlyTerminating


for i in range(10):
    for j in range(10):
        print(i, j)
        if j==5:
            break