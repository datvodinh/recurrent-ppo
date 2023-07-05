import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LSTMPPOModel(nn.Module):
    def __init__(self,config,state_size,action_size):
        super().__init__()
        self.lstm_pol = nn.LSTM(state_size,config["hidden_size"])
        self.lstm_val = nn.LSTM(state_size,config["hidden_size"])

        self.fc_pol   = nn.Sequential(
            nn.Tanh(),
            self._layer_init(nn.Linear(config["hidden_size"],config["hidden_size"])),
            nn.Tanh(),
            self._layer_init(nn.Linear(config["hidden_size"],config["hidden_size"]),std=0.01),
        )

        self.fc_val   = nn.Sequential(
            nn.Tanh(),
            self._layer_init(nn.Linear(config["hidden_size"],config["hidden_size"])),
            nn.Tanh(),
            self._layer_init(nn.Linear(config["hidden_size"],config["hidden_size"]),std=1),
        )

        for submodule in self.modules():
            submodule.register_forward_hook(self.nan_hook)

    @staticmethod
    def nan_hook(self, inp, output):
        if not isinstance(output, tuple):
            outputs = [output]
        else:
            outputs = output

        for i, out in enumerate(outputs):
            nan_mask = torch.isnan(out)
            if nan_mask.any():
                print("Hook: Nan occured In", self.__class__.__name__)

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
            - state: (torch.Tensor): shape (batch_size, len_seq, state_len)
            - hidden_state: (torch.Tensor): shape (batch_size, len_seq, state_len)
            - candidate_state: (torch.Tensor):shape (batch_size, len_seq, state_len)

        Return:
            - policy: (torch.Tensor): policy with shape (batch_size,num_action)
            - value: (torch.Tensor): value with shape (batch_size,1)
        """
        out_pol, (h,c) = self.fc_pol(state,(hidden_state,candidate_state))
        out_val, (h,c) = self.fc_val(state,(hidden_state,candidate_state))
        policy     = self.fc_pol(out_pol)
        value      = self.fc_val(out_val)

        return policy,value,h,c

