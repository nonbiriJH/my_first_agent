import gym
env = gym.make('MountainCar-v0')

print(env.action_space)
print(env.observation_space)
observation = env.reset()
for t in range(5):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print(action)
        print(observation, reward, done, info)
        if done:
            break
env.close()

# n_episode = 1000
# ret = 0
# for i_episode in range(n_episode):
#     observation = env.reset()
#     for t in range(200):
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         ret = ret + reward
#         if done:
#             break
# env.close()
# print(ret/n_episode)