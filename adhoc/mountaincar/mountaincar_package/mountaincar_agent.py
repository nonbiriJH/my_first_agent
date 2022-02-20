
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from .mountaincar_processor import MountaincarProcessor


class agent:
    def __init__(self, model, action, nb_steps_warmup, target_model_update, memory_recent_length, memory_sample_limit):
        self.model = model
        self.action = action
        self.nb_steps_warmup = nb_steps_warmup
        self.tau = target_model_update
        self.memory_recent_length = memory_recent_length
        self.memory_sample_limit = memory_sample_limit

    def build_agent(self):
        return DQNAgent(  # update action value function as per TD target r_t + gamma * max_a Q(s_t+1, a)
            # choose action stochastically on softmax of q value
            processor=MountaincarProcessor()
            , model=self.model
            , policy=BoltzmannQPolicy()        # required for mini-batch
            , memory=SequentialMemory(limit=self.memory_sample_limit #for backward
                , window_length=self.memory_recent_length #for get_recent in forward 
                )
            , nb_actions=self.action, nb_steps_warmup=self.nb_steps_warmup
            , target_model_update=self.tau # tau * source_weight + (1. - tau) * target_weight\,
            , enable_dueling_network=True #expected sarsa
            )
