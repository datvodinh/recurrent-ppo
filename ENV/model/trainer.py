from torch.distributions import Categorical,kl_divergence
import torch
import torch.nn as nn
import numpy as np
import time
import os

from setup import make
from model.model import LSTMPPOModel
from model.agent import Agent
from model.writer import Writer

class Trainer:
    """Train the model"""
    def __init__(self,config,game_name,writer_path=None) -> None:
        self.config        = config
        self.env           = make(game_name)

        self.model         = LSTMPPOModel(config,self.env.getStateSize(),self.env.getActionSize())
        self.optimizer     = torch.optim.AdamW(self.model.parameters(),lr=config['lr'])
        self.memory_length = config["memory_length"]
        if writer_path is not None:
            self.writer    = Writer(writer_path)
        self.agent         = Agent(self.env,self.model,config)
    
    def _cal_loss(self,value,value_new,entropy,log_prob,log_prob_new,advantage):
        """
        Overview:
            Calculate Total Loss

        Arguments:
            - value: (`torch.Tensor`):  
            - value_new: (`torch.Tensor`): 
            - entropy: (`torch.Tensor`): 
            - log_prob: (`torch.Tensor`): 
            - log_prob_new: (`torch.Tensor`): 
            - advantage: (`torch.Tensor`):  

        Return:
            - actor_loss: (`torch.Tensor`): 
            - critic_loss: (`torch.Tensor`): 
            - total_loss: (`torch.Tensor`): 

        
        """
        #Calculate returns and advantage
        returns               = value + advantage
        ratios                = torch.exp(torch.clamp(log_prob_new-log_prob.detach(),min=-20.,max=5.))
        weighted_prob         = ratios * advantage
        weighted_clipped_prob = torch.clamp(ratios,1-0.2,1+0.2) * advantage
        actor_loss            = -torch.min(weighted_prob,weighted_clipped_prob).mean()

        value_clipped         = value + torch.clamp(value_new - value, -self.config["value_clip"], self.config["value_clip"])
        critic_loss           = 0.5 * torch.max((returns-value_new)**2,(returns-value_clipped)**2).mean()

        entropy               = entropy.mean()
        
        total_loss            = actor_loss + self.config["critic_coef"] * critic_loss - self.config["entropy_coef"] * entropy

        return actor_loss, critic_loss, total_loss, entropy.mean()
    

    


    def train(self,write_data=True):
        training = True

        step = 0
        while training:

            win_rate = self.agent.run(num_games=self.config["num_game_per_batch"])
            self.agent.rollout.cal_advantages(self.config["gamma"],self.config["gae_lambda"])
            
            self.model.train()
            
            for _ in range(self.config["num_epochs"]):
                mini_batch_loader   = self.agent.rollout.mini_batch_loader(self.config)
                for mini_batch in mini_batch_loader:
                    pol_new,val_new = self.model(mini_batch["states"])
                    val_new         = val_new.squeeze(1)
                    # print(pol_new, mini_batch["action_mask"])
                    B,M,A = mini_batch["action_mask"].shape
                    # print(pol_new.shape,(B,M,A))
                    categorical_new = Categorical(logits=pol_new.masked_fill(mini_batch["action_mask"].view(B*M,A)==0,float('-1e20')))
                    log_prob_new    = categorical_new.log_prob(mini_batch["actions"].view(1,-1)).squeeze(0)
                    entropy         = categorical_new.entropy()

                    padding = mini_batch["padding"].reshape(-1)
                    actor_loss, critic_loss, total_loss,entropy = self._cal_loss(
                        value        = self._remove_padding(mini_batch["values"].reshape(-1),padding).detach(),
                        value_new    = self._remove_padding(val_new,padding),
                        entropy      = self._remove_padding(entropy,padding),
                        log_prob     = self._remove_padding(mini_batch["probs"].reshape(-1),padding).detach(),
                        log_prob_new = self._remove_padding(log_prob_new,padding),
                        advantage    = self._remove_padding(mini_batch["advantages"].reshape(-1),padding).detach(),
                    )
                    with torch.autograd.set_detect_anomaly(self.config["set_detect_anomaly"]):
                        if not torch.isnan(total_loss).any():
                            self.optimizer.zero_grad()
                            total_loss.backward()
                            nn.utils.clip_grad_norm_(self.model.parameters(),max_norm=self.config["max_grad_norm"])
                            self.optimizer.step()
                    if write_data:
                        
                        with torch.no_grad():
                            self.writer.add(
                                step        = step,
                                win_rate    = win_rate,
                                reward      = self.agent.rollout.batch["rewards"].mean(),
                                entropy     = entropy,
                                actor_loss  = actor_loss,
                                critic_loss = critic_loss,
                                total_loss  = total_loss
                            )
                            step+=1
            
            self.agent.rollout.reset_data()

    def _save_model(model:LSTMPPOModel,path):
        torch.save(model.state_dict(),path)

