import gym
import numpy as np
class policy_interaction:
    
    def __init__(self,w,step,naction):
        self.step = step
        self.naction = naction
        self.theta = w
    def get_feature(self,obs,a):
            #mask action as -1 and 1.
            feat_vec = []
            if (a==0):
                act = -1
            else:
                act = 1
            for i in obs:
                #linear on si*a
                feat_vec.append(i*act)
            return [float(i) for i in feat_vec]

    def prob(self,obs):
        pref = []
        for a in range(self.naction):
            feat_vec = self.get_feature(obs,a)
            #calculate preference within +-500
            pref = np.append(pref,np.exp(np.clip(np.dot(feat_vec,self.theta), -500.,500.)))
        if sum(pref) == 0:
            print(pref)
            return [0.5,0.5]
        else:
            return [i/sum(pref) for i in pref]

    def update(self,state,action,reward):
        ret = sum(reward)
        for i in range(len(state)):
            act_inx = int(action[i])
            feat_vec = self.get_feature(state[i],act_inx)
            act_dist = self.prob(state[i])
            #calculate gradient of ln_pi
            if act_inx == 0:
                grad_act_dependency = act_dist[1] 
            elif act_inx == 1:
                grad_act_dependency = act_dist[0]
            #mc update
            self.theta = [i[0] + self.step * ret * i[1] * grad_act_dependency for i in zip(self.theta, feat_vec)]
            ret = ret - reward[i]


#training
p = policy_interaction(np.ones(4), 0.0002, 2)
env = gym.make('CartPole-v0')
nepisode = 1000
ncomplete = 0
ret_list = []
for i_episode in range(nepisode):
    obs_list = []
    act_list = []
    rew_list = []
    observation = env.reset()
    for t in range(1000):
        #take action
        prob_distr = p.prob(observation)
        action = np.random.choice(2,1,p=prob_distr)[0]
        #store episode trajectory
        obs_list.append(observation)
        act_list = np.append(act_list,int(action))
        observation, reward, done, info = env.step(action)
        rew_list = np.append(rew_list,reward)
        if done:
            p.update(obs_list,act_list,rew_list)
            ret_list.append(sum(rew_list))
            ncomplete = ncomplete + 1
            break
env.close()


print(sum(ret_list)/ncomplete)
print(ncomplete)
print(p.theta)


#rendering
observation = env.reset()
for t in range(5000):
    env.render()
    prob_distr = p.prob(observation)
    print(prob_distr)
    action = np.random.choice(2,1,p=prob_distr)[0]
    observation, reward, done, info = env.step(action)
    if done:
        print("Episode finished after {} timesteps".format(t+1))
        break
env.close()
    



