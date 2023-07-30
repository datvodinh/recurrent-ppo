import torch
import torch.nn as nn
from torch.distributions import Categorical, kl_divergence

class Distribution():
    """Distribution class for handling probability distributions"""
    def sample_action(self, policy, action_mask):
        """
        Overview:
            Samples an action from the given policy distribution.

        Arguments:
            - policy: (`torch.Tensor`): The policy distribution logits.
            - action_mask: (`torch.Tensor`): The action mask.

        Returns:
            - action: (`int`): The sampled action.
            - log_prob: (`torch.Tensor`): The log probability of the sampled action.
        """
        distribution = Categorical(logits=policy.masked_fill(action_mask == 0, float("-1e20")))
        action = distribution.sample()
        return action.item(), distribution.log_prob(action)

    def log_prob(self, policy, action, action_mask):
        """
        Overview:
            Computes the log probability and entropy of an action in the given policy distribution.

        Arguments:
            - policy: (`torch.Tensor`): The policy distribution logits.
            - action: (`torch.Tensor`): The action to compute log probability and entropy for.
            - action_mask: (`torch.Tensor`): The action mask.

        Returns:
            - log_prob: (`torch.Tensor`): The log probability of the action.
            - entropy: (`torch.Tensor`): The entropy of the distribution.
        """
        distribution = Categorical(logits=policy.masked_fill(action_mask == 0, float("-1e20")))
        return distribution.log_prob(action), distribution.entropy()

    def kl_divergence(self, policy, policy_new):
        """
        Overview:
            Computes the KL-divergence between two policy distributions.

        Arguments:
            - policy: (`torch.Tensor`): The original policy distribution logits.
            - policy_new: (`torch.Tensor`): The updated policy distribution logits.

        Returns:
            - kl_div: (`torch.Tensor`): The KL-divergence between the two distributions.
        """
        return kl_divergence(Categorical(logits=policy), Categorical(logits=policy_new))
