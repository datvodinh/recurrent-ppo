from torch.distributions import Categorical, kl_divergence
import torch
import torch.nn as nn
import numpy as np
import json
import os

torch.manual_seed(9999)
np.random.seed(9999)

from model.model import LSTMPPOModel
from model.agent import Agent
from model.writer import Writer
from model.distribution import Distribution

class Trainer:
    """Trainer class for training the model"""
    def __init__(self, config, env, writer_path=None,save_path=None):
        """
        Overview:
            Initializes the Trainer instance.

        Arguments:
            - config: (`dict`): Configuration settings for the trainer.
            - env: (`object`): The environment object.
            - writer_path: (`str`): The path to save Tensorboard logs (optional).
        """
        self.config = config
        self.env = env
        self.model = LSTMPPOModel(config, self.env.getStateSize(), self.env.getActionSize())
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config['lr'])
        
        
        if writer_path is not None:
            self.writer = Writer(writer_path)
        if save_path is not None:
            self.save_path = save_path

        self.agent = Agent(self.env, self.model, config)
        self.distribution = Distribution()

        if not os.path.exists(save_path):
            os.makedirs(save_path)
            print("Directory created:", save_path)
        else:
            print("Directory already exists:", save_path)

        try:
            self.model.load_state_dict(torch.load(f'{save_path}model.pt'))
            with open(f"{save_path}stat.json","r") as f:
                self.data = json.load(f)
            print('PROGRESS RESTORED!')
        except:
            print("TRAIN FROM BEGINING!")
            self.data = {
                "step":0,
                "entropy_coef":config["entropy_coef"]["start"]
            }

        self.entropy_coef = self.data["entropy_coef"]
        self.entropy_coef_step = (config['entropy_coef']["start"] - config['entropy_coef']['end']) / config['entropy_coef']['step']
        

    def _entropy_coef_schedule(self):
        self.entropy_coef -= self.entropy_coef_step
        if self.entropy_coef <= self.config['entropy_coef']['end']:
            self.entropy_coef = self.config['entropy_coef']['end']

    def _truly_loss(self, value, value_new, entropy, log_prob, log_prob_new, Kl, advantage):
        """
        Overview:
            Calculates the total loss using Truly PPO method.

        Arguments:
            - value: (`torch.Tensor`): The predicted values.
            - value_new: (`torch.Tensor`): The updated predicted values.
            - entropy: (`torch.Tensor`): The entropy.
            - log_prob: (`torch.Tensor`): The log probabilities of the actions.
            - log_prob_new: (`torch.Tensor`): The updated log probabilities of the actions.
            - Kl: (`torch.Tensor`): The KL divergence.
            - advantage: (`torch.Tensor`): The advantages.

        Returns:
            - actor_loss: (`torch.Tensor`): The actor loss.
            - critic_loss: (`torch.Tensor`): The critic loss.
            - total_loss: (`torch.Tensor`): The total loss.
            - entropy: (`torch.Tensor`): The entropy.
        """
        returns = value + advantage

        if self.config["normalize_advantage"]:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        ratios     = torch.exp(torch.clamp(log_prob_new - log_prob.detach(), min=-20., max=5.))

        R_dot_A    = ratios * advantage
        actor_loss = -torch.where(
            (Kl >= self.config["PPO"]["policy_kl_range"]) & (R_dot_A > advantage),
            R_dot_A - self.config["PPO"]["policy_params"] * Kl,
            R_dot_A
        )

        value_clipped = value + torch.clamp(value_new - value, -self.config["PPO"]["value_clip"], self.config["PPO"]["value_clip"])
        critic_loss   = 0.5 * torch.max((returns - value_new) ** 2, (returns - value_clipped) ** 2)

        total_loss    = actor_loss + self.config["PPO"]["critic_coef"] * critic_loss - self.entropy_coef * entropy

        return actor_loss.mean(), critic_loss.mean(), total_loss.mean(), entropy.mean()

    def train(self, write_data=True):
        """
        Overview:
            Trains the model.

        Arguments:
            - write_data: (`bool`): Whether to write data to Tensorboard or not.
        """
        training = True

        while training:
            win_rate = self.agent.run(num_games=self.config["num_game_per_batch"])
            self.agent.rollout.cal_advantages(self.config["PPO"]["gamma"], self.config["PPO"]["gae_lambda"])
            self.agent.rollout.prepare_batch()

            self.model.train()

            for _ in range(self.config["num_epochs"]):
                mini_batch_generator = self.agent.rollout.mini_batch_generator()

                for mini_batch in mini_batch_generator:
                    B, S                   = mini_batch["states"].shape
                    mini_batch["states"]   = mini_batch["states"].view(B // self.agent.rollout.actual_sequence_length, self.agent.rollout.actual_sequence_length, S)
                    pol_new, val_new, _, _ = self.model(mini_batch["states"], mini_batch["h_states"].unsqueeze(0), mini_batch["c_states"].unsqueeze(0))
                    val_new                = val_new.squeeze(1)

                    log_prob_new, entropy  = self.distribution.log_prob(pol_new, mini_batch["actions"].view(1, -1), mini_batch["action_mask"])
                    log_prob_new           = log_prob_new.squeeze(0)[mini_batch["loss_mask"]]
                    val_new                = val_new[mini_batch["loss_mask"]]
                    entropy                = entropy[mini_batch["loss_mask"]]

                    Kl                     = self.distribution.kl_divergence(mini_batch["policy"], pol_new)
                    Kl                     = Kl[mini_batch["loss_mask"]]

                    actor_loss, critic_loss, total_loss, entropy = self._truly_loss(
                        value        = mini_batch["values"].reshape(-1).detach(),
                        value_new    = val_new,
                        entropy      = entropy,
                        log_prob     = mini_batch["probs"].reshape(-1).detach(),
                        log_prob_new = log_prob_new,
                        Kl           = Kl,
                        advantage    = mini_batch["advantages"].reshape(-1).detach(),
                    )

                    with torch.autograd.set_detect_anomaly(self.config["set_detect_anomaly"]):
                        if not torch.isnan(total_loss).any():
                            self.optimizer.zero_grad()
                            total_loss.backward()
                            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config["max_grad_norm"])
                            self.optimizer.step()
                            self._entropy_coef_schedule()

                    if write_data:
                        try:
                            with torch.no_grad():
                                self.writer.add(
                                    step        = self.data["step"],
                                    win_rate    = win_rate,
                                    reward      = self.agent.rollout.batch["rewards"].mean(),
                                    entropy     = entropy,
                                    actor_loss  = actor_loss,
                                    critic_loss = critic_loss,
                                    total_loss  = total_loss,
                                    kl_mean     = Kl.mean().item(),
                                    kl_max      = Kl.max().item(),
                                    kl_min      = Kl.min().item()
                                )
                                
                                self._save_log()
                        except:
                            pass
                        
            if (self.data["step"]%200)==0:
                self._save_model()
            self.agent.rollout.reset_data()

    def _save_model(self):
        """
        Overview:
            Saves the model and other data.
        """
        torch.save(self.model.state_dict(), f'{self.save_path}model.pt')
        with open(f"{self.save_path}stat.json","w") as f:
                json.dump(self.data,f)

    def _save_log(self):
        self.data["step"]+=1
        self.data["entropy_coef"] = self.entropy_coef
