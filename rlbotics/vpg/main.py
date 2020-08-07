from rlbotics.vpg.vpg import *
import gym

env = gym.make('CartPole-v1')

model = Reinforce(env)

render = False

max_reward_per_epoch = []

for epoch in range(100):
    if epoch % 10 == 0:
        print("epoch : ", epoch)
        render = False

    else:
        render = False

    model.train(100, render=False)
