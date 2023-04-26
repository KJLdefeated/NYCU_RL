import gym
from ddpg_lunarlander import train
from skopt.space import Real, Integer
from skopt import gp_minimize

#gamma_, tau_, lr_a_, lr_c_, lr_a_decay_, lr_c_decay_, noise_scale_, batch_size_
search_space = [
    Real(0.9, 1, name='gamma'),
    Real(0, 0.01, name='tau'),
    Real(0, 0.005, name='lr_a'),
    Real(0, 0.005, name='lr_c'),
    Real(0.7, 1, name='lr_a_decay'),
    Real(0.7, 1, name='lr_c_decay'),
    Real(0, 0.5, name='noise_scale'),
    Integer(64, 256, name='batch_size')
]

def objective(params):
    print(params)
    episodes_num = train(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7])
    return episodes_num

result = gp_minimize(objective, search_space, n_calls=10, random_state=0)

print("Best hyperparameters: ", result.x)
print("Best objective value: ", result.fun)
print("Hyperparameters tried: ", result.x_iters)
print("Objective values at each step: ", result.func_vals)