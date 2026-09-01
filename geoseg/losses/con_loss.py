import torch.nn as nn

__all__ = ["con_loss"]

class con_loss(nn.Module):
    def __init__(self):
        super(con_loss, self).__init__()


    def forward(self, logit, target):
        loss = nn.BCEWithLogitsLoss()(logit, target)
        return loss