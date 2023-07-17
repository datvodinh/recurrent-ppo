import torch
import torch.nn.functional as F
import numpy as np
from numba import njit
from model.rolloutBuffer import RolloutBuffer
from model.distribution import Distribution

class Agent():
    """Agent class representing an AI agent"""
    def __init__(self, env, model, config):
        """
        Overview:
            Initializes the Agent instance.

        Arguments:
            - env: (`object`): The environment object.
            - model: (`object`): The model object.
            - config: (`dict`): Configuration settings for the agent.
        """
        super().__init__()
        self.env            = env
        self.model          = model
        self.reward         = config["rewards"]
        self.max_eps_length = config["LSTM"]["max_eps_length"]
        self.h_state        = torch.zeros(1, 1, config["LSTM"]["hidden_size"])
        self.c_state        = torch.zeros(1, 1, config["LSTM"]["hidden_size"])
        self.hidden_size    = config["LSTM"]["hidden_size"]
        self.rollout        = RolloutBuffer(config, env.getStateSize(), env.getActionSize())
        self.distribution   = Distribution()

    def reset_hidden(self):
        """
        Overview:
            Resets the hidden state and candidate state.
        """
        self.h_state = torch.zeros(1, 1, self.hidden_size)
        self.c_state = torch.zeros(1, 1, self.hidden_size)

    @torch.no_grad()
    def play(self, state, per):
        """
        Overview:
            Agent's play function.

        Arguments:
            - state: (`np.array`): The current state.
            - per: (`List`): The per file.

        Returns:
            - action: (`int`): The agent's chosen action.
            - per: (`List`): The per file.
        """
        self.model.eval()
        tensor_state        = torch.tensor(state.reshape(1, 1, -1), dtype=torch.float32)
        policy, value, h, c = self.model(tensor_state, self.h_state, self.c_state)
        policy              = policy.squeeze()
        list_action         = self.env.getValidActions(state)
        action_mask         = torch.tensor(list_action, dtype=torch.float32)
        action, log_prob    = self.distribution.sample_action(policy, action_mask)
        
        if action_mask[action] != 1:
            action = np.random.choice(np.where(list_action == 1)[0])
        
        if self.env.getReward(state) == -1:
            if self.rollout.step_count < self.max_eps_length:
                self.rollout.add_data(
                    state        = torch.from_numpy(state),
                    h_state      = self.h_state.squeeze(),
                    c_state      = self.c_state.squeeze(),
                    action       = action,
                    value        = value.item(),
                    reward       = self.reward[int(self.env.getReward(state))] * 1.0,
                    done         = 0,
                    valid_action = torch.from_numpy(list_action),
                    prob         = log_prob,
                    policy       = policy
                )
            self.rollout.step_count += 1
        else:
            if self.rollout.step_count < self.max_eps_length:
                self.rollout.add_data(
                    state        = torch.from_numpy(state),
                    h_state      = self.h_state.squeeze(),
                    c_state      = self.c_state.squeeze(),
                    action       = action,
                    value        = value.item(),
                    reward       = self.reward[int(self.env.getReward(state))] * 1.0,
                    done         = 1,
                    valid_action = torch.from_numpy(list_action),
                    prob         = log_prob,
                    policy       = policy
                )
                
            self.rollout.batch["dones_indices"][self.rollout.game_count] = self.rollout.step_count
            self.rollout.game_count += 1
            self.rollout.step_count = 0

        self.h_state, self.c_state = h, c
        if self.env.getReward(state) != -1:
            self.reset_hidden()
        
        return action, per 
    
    def run(self, num_games: int) -> float:
        """
        Overview:
            Runs the custom environment and returns the win rate.

        Arguments:
            - num_games: (`int`): The number of games to run.

        Returns:
            - win_rate: (`float`): The win rate of the agent.
        """
        win_rate = self.env.run(self.play, num_games, np.array([0.]), 1)[0] / num_games
        return win_rate
