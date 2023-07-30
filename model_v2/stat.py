import torch

class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = torch.zeros(shape)
        self.var = torch.ones(shape)
        self.count = epsilon
        self.active = False
        self.x = []

    def update(self):
        x = torch.tensor(self.x)
        batch_mean = torch.mean(x, dim=0)  # Compute mean along the first dimension (batch)
        batch_var = torch.var(x, dim=0)    # Compute variance along the first dimension (batch)
        batch_count = x.size(0)           # Get the number of data points in the batch
        self.update_from_moments(batch_mean, batch_var, batch_count)
        self.active = True
        self.x = []

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + torch.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

    def normalize(self,x):
        self.x.append(x.tolist())
        if self.active:
            return (x - self.mean) / (torch.sqrt(self.var) + 1e-5)
        else:
            return x
    