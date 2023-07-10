import torch
import torch.nn as nn
from torch.distributions import Categorical, kl_divergence

class Distribution():
    def sample_action(self,policy,action_mask):
        distribution = Categorical(logits=policy.masked_fill(action_mask==0,float("-1e20")))
        action = distribution.sample()
        return action.item(), distribution.log_prob(action)
        
    def log_prob(self,policy,action,action_mask):
        distribution = Categorical(logits=policy.masked_fill(action_mask==0,float("-1e20")))
        return distribution.log_prob(action), distribution.entropy()

    def kl_divergence(self,policy,policy_new):
        return kl_divergence(Categorical(logits=policy),Categorical(logits=policy_new))
    
