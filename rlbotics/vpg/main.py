from rlbotics.vpg.vpg import *
import gym
import matplotlib.pyplot as plt

env = gym.make('CartPole-v1')

model = Reinforce(env)

render = False

max_reward_per_epoch = []

for epoch in range(1000):
    if epoch % 10 == 0:
        print("epoch : ", epoch)
        render = False

    else:
        render = False

    obs, acts, rews = model.collect_experience(1000, render=render)
    max_reward_per_epoch.append(max(rews))
    model.policy_update(obs, acts, rews)

#obs, acts, rews = model.collect_experience(1000, render=True)
plt.xlabel('epochs')
plt.ylabel('score')

plt.plot(max_reward_per_epoch, label = "max score in epoch")

plt.legend()
plt.show()
