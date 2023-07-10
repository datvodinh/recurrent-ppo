from torch.utils.tensorboard import SummaryWriter
import os
class Writer:
    """Save stat and Plot using Tensorboard"""
    def __init__(self,path) -> None:

        self.writer = SummaryWriter(log_dir=path)
    def add(self,step,win_rate,reward,entropy,actor_loss,critic_loss,total_loss,kl_mean,kl_max,kl_min):
        self.writer.add_scalar("A.Train/Win Rate",win_rate,step)
        self.writer.add_scalar("A.Train/Reward",reward,step)
        self.writer.add_scalar("B.Loss/Entropy",entropy,step)
        self.writer.add_scalar("B.Loss/ActorLoss",actor_loss,step)
        self.writer.add_scalar("B.Loss/CriticLoss",critic_loss,step)
        self.writer.add_scalar("B.Loss/TotalLoss",total_loss,step)
        self.writer.add_scalar("C.Kl_divergence/mean",kl_mean,step)
        self.writer.add_scalar("C.Kl_divergence/max",kl_max,step)
        self.writer.add_scalar("C.Kl_divergence/min",kl_min,step)
        
    def close(self):
        self.writer.close()