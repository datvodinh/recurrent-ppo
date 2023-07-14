import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LSTMPPOModel(nn.Module):
    """LSTM PPO Model class for the policy and value networks"""
    def __init__(self, config, state_size, action_size):
        """
        Overview:
            Initializes the LSTMPPOModel instance.

        Arguments:
            - config: (`dict`): Configuration settings for the model.
            - state_size: (`int`): The size of the input state.
            - action_size: (`int`): The size of the action space.
        """
        super().__init__()
        self.encoder = nn.Sequential(
            self._layer_init(nn.Linear(state_size,config["embed_size"])),
            nn.Tanh()
            )
        self.lstm= nn.LSTM(config["embed_size"],config["hidden_size"],batch_first=True)

        self.fc_pol   = nn.Sequential(
            nn.Tanh(),
            self._layer_init(nn.Linear(config["hidden_size"],config["hidden_size"])),
            nn.Tanh(),
            self._layer_init(nn.Linear(config["hidden_size"],action_size),std=0.01),
        )

        self.fc_val   = nn.Sequential(
            nn.Tanh(),
            self._layer_init(nn.Linear(config["hidden_size"],config["hidden_size"])),
            nn.Tanh(),
            self._layer_init(nn.Linear(config["hidden_size"],1),std=1),
        )

    @staticmethod
    def _layer_init(layer, std=np.sqrt(2), bias_const=0.0):
        """
        Overview:
            Init Weight and Bias with Constraint

        Arguments:
            - layer: Layer.
            - std: (`float`): Standard deviation.
            - bias_const: (`float`): Bias

        Return:
        
        """
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer    
    
    def forward(self,state,hidden_state,candidate_state):
        """
        Overview:
            Forward method.

        Arguments:
            - state: (`torch.Tensor`): shape (batch_size, len_seq, state_len)
            - hidden_state: (`torch.Tensor`): shape (batch_size, len_seq, hidden_len)
            - candidate_state: (`torch.Tensor`):shape (batch_size, len_seq, hidden_len)

        Return:
            - policy: (torch.Tensor): policy with shape (batch_size,num_action)
            - value: (torch.Tensor): value with shape (batch_size,1)
            - h: 
        """
        out        = self.encoder(state)
        out, (h,c) = self.lstm(out,(hidden_state,candidate_state))
        out        = out.reshape(out.size(0) * out.size(1),out.size(2))
        policy     = self.fc_pol(out)
        value      = self.fc_val(out)

        return policy,value,h,c

