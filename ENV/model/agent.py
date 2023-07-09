import torch
import torch.nn.functional as F
import numpy as np
from numba import njit
from model.rolloutBuffer import RolloutBuffer
from torch.distributions import Categorical
class Agent():
    """Agent"""
    def __init__(self,env,model,config):
        super().__init__()
        self.env                = env
        self.model              = model
        self.reward             = config["rewards"]
        self.max_eps_length     = config["max_eps_length"]
        self.rollout            = RolloutBuffer(config,env.getStateSize(),env.getActionSize())
        self.h_state            = torch.zeros(1,1,config["hidden_size"])
        self.c_state            = torch.zeros(1,1,config["hidden_size"])
        self.hidden_size        = config["hidden_size"]

    def reset_hidden(self):
        """
        Overview:
            Reset Hidden State and Candidate State
        """
        self.h_state            = torch.zeros(1,1,self.hidden_size)
        self.c_state            = torch.zeros(1,1,self.hidden_size)

    @torch.no_grad()
    def play(self,state,per):
        """
        Overview:
            Agent's play function

        Arguments:
            - state: (`np.array`): state
            - per: (`List`): per file

        Returns:
            - action: (`int`): Agent's action
            - per: (`List`): per file
        """
        self.model.eval()
        tensor_state = torch.tensor(state.reshape(1,1,-1),dtype=torch.float32)
        policy,value,h,c = self.model(tensor_state,self.h_state,self.c_state)
        policy       = policy.squeeze()
        list_action  = self.env.getValidActions(state)
        actions      = torch.tensor(list_action,dtype=torch.float32)
        categorical  = Categorical(logits=policy.masked_fill(actions==0,float('-1e20')))
        action       = categorical.sample().item()
        if actions[action] != 1:
            action   = np.random.choice(np.where(list_action==1)[0])

        log_prob     = categorical.log_prob(torch.tensor([action]).view(1,-1)).squeeze()
        
        if self.env.getReward(state)==-1:
            if self.rollout.step_count < self.max_eps_length:
                self.rollout.add_data(state      = torch.from_numpy(state),
                                    h_state      = self.h_state.squeeze(),
                                    c_state      = self.c_state.squeeze(),
                                    action       = action,
                                    value        = value.item(),
                                    reward       = self.reward[int(self.env.getReward(state))] * 1.0,
                                    done         = 0,
                                    valid_action = torch.from_numpy(list_action),
                                    prob         = log_prob
                                    )
            self.rollout.step_count+=1
        else:
            if self.rollout.step_count < self.max_eps_length:
                self.rollout.add_data(state      = torch.from_numpy(state),
                                    h_state      = self.h_state.squeeze(),
                                    c_state      = self.c_state.squeeze(),
                                    action       = action, 
                                    value        = value.item(),
                                    reward       = self.reward[int(self.env.getReward(state))] * 1.0,
                                    done         = 1,
                                    valid_action = torch.from_numpy(list_action),
                                    prob         = log_prob
                                    )
                
            self.rollout.batch["dones_indices"][self.rollout.game_count] = self.rollout.step_count
            self.rollout.game_count+=1
            self.rollout.step_count=0

        self.h_state,self.c_state = h,c
        if self.env.getReward(state)!=-1:
            self.reset_hidden()
        
        return action,per 
    
    def run(self,num_games:int)->float:
        """
        Overview:
            Run custom environment and return win rate.

        Arguments:
            - num_games: (`int`): number of games.
            
        """
        win_rate =  self.env.run(self.play,num_games,np.array([0.]),1)[0] / num_games
        return win_rate
    
    @njit()
    def bot_max_eps_length(self,state, perData):
        validActions = self.env.getValidActions(state)
        arr_action = np.where(validActions == 1)[0]
        idx = np.random.randint(0, arr_action.shape[0])
        perData[0]+=1
        if self.env.getReward(state)!=-1:
            if perData[0] > perData[1]:
                perData[1] = perData[0]
            perData[0] = 0
        return arr_action[idx], perData
    