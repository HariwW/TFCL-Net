import torch

from .functional import soft_dice_score


__all__ = ["dice_bce_loss"]


class dice_bce_loss(torch.nn.Module):

    
    __constants__ = ["batch", "log_loss", "smooth", "eps"]


    def __init__(self, batch=True, log_loss=False, smooth=0.0, eps=1e-7):
        super(dice_bce_loss, self).__init__()
        self.batch = batch
        self.log_loss = log_loss
        self.smooth = smooth
        self.eps = eps

    def soft_dice_coeff(self, y_true, y_pred):
        if self.batch:
            dims = None
        else:
            dims = (1, 2, 3)
        score = soft_dice_score(y_pred, y_true, smooth=self.smooth, eps=self.eps, dims=dims)
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        score = self.soft_dice_coeff(y_true, y_pred)
        if self.log_loss:
            return -torch.log(score.clamp_min(self.eps))
        return 1.0 - score

    def __call__(self, y_pred, y_true):
        # bce loss  
        a = torch.nn.functional.binary_cross_entropy(y_pred, y_true)
        # dice loss
        b = self.soft_dice_loss(y_true, y_pred)
        return a + b
