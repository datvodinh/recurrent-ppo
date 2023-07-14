from torch.utils.tensorboard import SummaryWriter
import os

class Writer:
    """Writer class for saving stats and plotting using Tensorboard"""
    def __init__(self, path):
        """
        Overview:
            Initializes the Writer instance.

        Arguments:
            - path: (`str`): The path to save the Tensorboard logs.
        """
        self.writer = SummaryWriter(log_dir=path)

    def add(self, step, win_rate, reward, entropy, actor_loss, critic_loss, total_loss, kl_mean, kl_max, kl_min):
        """
        Overview:
            Adds scalar values to Tensorboard.

        Arguments:
            - step: (`int`): The current training step.
            - win_rate: (`float`): The win rate.
            - reward: (`float`): The reward.
            - entropy: (`float`): The entropy.
            - actor_loss: (`float`): The actor loss.
            - critic_loss: (`float`): The critic loss.
            - total_loss: (`float`): The total loss.
            - kl_mean: (`float`): The mean KL divergence.
            - kl_max: (`float`): The maximum KL divergence.
            - kl_min: (`float`): The minimum KL divergence.
        """
        self.writer.add_scalar("A.Train/Win Rate", win_rate, step)
        self.writer.add_scalar("A.Train/Reward", reward, step)
        self.writer.add_scalar("B.Loss/Entropy", entropy, step)
        self.writer.add_scalar("B.Loss/ActorLoss", actor_loss, step)
        self.writer.add_scalar("B.Loss/CriticLoss", critic_loss, step)
        self.writer.add_scalar("B.Loss/TotalLoss", total_loss, step)
        self.writer.add_scalar("C.Kl_divergence/mean", kl_mean, step)
        self.writer.add_scalar("C.Kl_divergence/max", kl_max, step)
        self.writer.add_scalar("C.Kl_divergence/min", kl_min, step)

    def close(self):
        """
        Overview:
            Closes the writer and releases resources.
        """
        self.writer.close()
