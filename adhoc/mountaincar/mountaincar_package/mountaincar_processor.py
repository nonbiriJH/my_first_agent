from rl.core import Processor
import math

class MountaincarProcessor(Processor):
    def process_step(self, observation, reward, done, info):
        """Processes an entire step by applying the processor to the observation, reward, and info arguments.

        # Arguments
            observation (object): An observation as obtained by the environment.
            reward (float): A reward as obtained by the environment.
            done (boolean): `True` if the environment is in a terminal state, `False` otherwise.
            info (dict): The debug info dictionary as obtained by the environment.

        # Returns
            The tupel (observation, reward, done, reward) with with all elements after being processed.
        """
        observation = self.process_observation(observation)
        if observation[0] < 0.5:  #focus on speed before close to goal,
            reward = self.process_reward(reward) + 1000.0 * math.pow(observation[1],2)
        else: #after close to goal focus on position
            reward = self.process_reward(reward) + observation[0]
        # if observation[0] < 0.5:
        #     reward = self.process_reward(reward) + math.pow(observation[1],2)/math.pow(0.07,2) #focus on speed before close to goal, normalise to [0,1]
        # else:
        #     #after close to goal add position, scale to [0,1].
        #     reward = self.process_reward(reward) + 0.1 * math.pow(observation[1],2)/math.pow(0.07,2) + (observation[0] - 0.5) * 10
        info = self.process_info(info)
        return observation, reward, done, info
    