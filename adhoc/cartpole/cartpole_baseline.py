import gym
env = gym.make('CartPole-v0')
ret = 0
n_episode = 100
for i_episode in range(n_episode):
    observation = env.reset()
    for t in range(100):
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        ret = ret + reward
        if done:
            break
env.close()
print(ret/n_episode)