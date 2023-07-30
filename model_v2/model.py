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

        self.config = config
        
        self.fc_pol   = nn.Sequential(
            self._layer_init(nn.Linear(state_size,config["LSTM"]["hidden_size"])),
            nn.ReLU(),
            self._layer_init(nn.Linear(config["LSTM"]["hidden_size"],config["LSTM"]["hidden_size"])),
            nn.ReLU(),
        )

        self.fc_val   = nn.Sequential(
            self._layer_init(nn.Linear(state_size,config["LSTM"]["hidden_size"])),
            nn.ReLU(),
            self._layer_init(nn.Linear(config["LSTM"]["hidden_size"],config["LSTM"]["hidden_size"])),
            nn.ReLU(),
        )

        self.lstm_pol = nn.GRU(config["LSTM"]["hidden_size"],config["LSTM"]["hidden_size"],batch_first=True)
        self.lstm_val = nn.GRU(config["LSTM"]["hidden_size"],config["LSTM"]["hidden_size"],batch_first=True)

        self.pol      = nn.Sequential(
            nn.ReLU(),
            self._layer_init(nn.Linear(config["LSTM"]["hidden_size"],action_size),std=0.01)
        )

        self.val      = nn.Sequential(
            nn.ReLU(),
            self._layer_init(nn.Linear(config["LSTM"]["hidden_size"],1),std=1)
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
    
    def forward(self,state,p_state,v_state,seq_length = 1):
        out_p               = self.fc_pol(state)
        out_v               = self.fc_val(state)
        out_p               = out_p.reshape(out_p.shape[0] // seq_length, seq_length, out_p.shape[1])
        out_v               = out_v.reshape(out_v.shape[0] // seq_length, seq_length, out_v.shape[1])
        out_p, p_state_new  = self.lstm_pol(out_p,p_state)
        out_v, v_state_new  = self.lstm_val(out_v,v_state)
        out_p               = out_p.reshape(out_p.size(0) * out_p.size(1),out_p.size(2))
        out_v               = out_v.reshape(out_v.size(0) * out_v.size(1),out_v.size(2))
        policy              = self.pol(out_p)
        value               = self.val(out_v)

        return policy,value,p_state_new,v_state_new
    
    def get_policy(self,state,p_state,seq_length=1):
        out_p               = self.fc_pol(state)
        out_p               = out_p.reshape(out_p.shape[0] // seq_length, seq_length, out_p.shape[1])
        out_p, p_state_new  = self.lstm_pol(out_p,p_state)
        policy              = self.pol(out_p)
        return policy,p_state_new
    


