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

config_path = "config.yml"
config = load_config(config_path)
print_config(config_path)

config["model_config"]["load_model"] = True
config["dataset_config"]["load_flag"] = True


model = DynamicModel(config)
input_data = [0.998783, -0.014414, 1.002026, 0.003621, 0.216799, -0.226104, 1]

out = model.predict(input_data)
print(out)
