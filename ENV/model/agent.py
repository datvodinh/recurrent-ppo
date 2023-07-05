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
        self.rollout            = RolloutBuffer(config["seq_length"],config["max_eps_length"],config["num_game_per_batch"],env.getStateSize(),env.getActionSize())
        self.h_state            = None
        self.c_state            = None
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
        with torch.no_grad():
            tensor_state = torch.tensor(state.reshape(1,1,-1),dtype=torch.float32)
            policy,value,self.h_state,self.c_state = self.model(tensor_state,self.h_state,self.c_state)
            policy       = policy.squeeze()
            list_action  = self.env.getValidActions(state)
            actions  = torch.tensor(list_action,dtype=torch.float32)
            categorical  = Categorical(logits=policy.masked_fill(actions==0,float('-inf')))
            action       = categorical.sample().item()
            if actions[action] != 1:
                action   = np.random.choice(np.where(list_action==1)[0])

            log_prob     = categorical.log_prob(torch.tensor([action]).view(1,-1)).squeeze()
            

            # print(action)
            
            if self.env.getReward(state)==-1:
                self.rollout.add_data(state      = torch.from_numpy(state),
                                    h_state      = self.h_state,
                                    c_state      = self.c_state,
                                    action       = action,
                                    value        = value.item(),
                                    reward       = 0.0,
                                    done         = 0,
                                    valid_action = torch.from_numpy(list_action),
                                    prob         = log_prob
                                    )
                self.rollout.step_count+=1
            else:
                self.rollout.add_data(state      = torch.from_numpy(state),
                                    h_state      = self.h_state,
                                    c_state      = self.c_state,
                                    action       = action, 
                                    value        = value.item(),
                                    reward       = self.reward[int(self.env.getReward(state))] * 1.0,
                                    done         = 1,
                                    valid_action = torch.from_numpy(list_action),
                                    prob         = log_prob
                                    )
                self.rollout.game_count+=1
                self.rollout.step_count=0
                self.h_state = None
                self.c_state = None
        
        return action,per 
    
    @staticmethod
    def stable_softmax(x):
        """
        Overview:
            Return the stable softmax

        Arguments:
            - x: (`Optional[torch.Tensor]`): input Logits.

        Returns:
            - softmax: (`Optional[torch.Tensor]`): stable softmax.
        """
        max_val           = np.max(x)
        scaled_values     = x - max_val
        exp_scaled_values = np.exp(scaled_values)
        softmax           = exp_scaled_values / np.sum(exp_scaled_values)
        return softmax
    
    def run(self,num_games:int)->float:
        """
        Overview:
            Run custom environment and return win rate.

        Arguments:
            - num_games: (`int`): number of games.
            
        """
        win_rate =  self.env.run(self.play,num_games,np.array([0.]),1)[0] / num_games
        # print(num_games,win_rate)
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
    