import pickle
from dynamics import *


# F = open('data_exp_7.pkl', 'rb')
# F = open('exp_7.ckpt', 'rb')
F = open('exp_1.ckpt', 'rb')

content = pickle.load(F)
print(type(content))

print(content)
# print(len(content['data']))
# print(len(content['label']))
# print(content['data'][0])


model = DynamicModel(config)
out = model([1, 1])
print(out)
