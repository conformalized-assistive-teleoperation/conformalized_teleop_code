import pdb

import torch
import torch.nn as nn


#pinball loss class
# Credit for PinballLoss implementation: Anastasios Angelopoulos, Repo: im2im-uq
# https://github.com/aangelopoulos/im2im-uq/blob/92124f5f5ac6954eb66f03e20735d8f12f47b797/core/models/losses/pinball.py#L4
# Code associated with @article{angelopoulos2022image,
#   title={Image-to-Image Regression with Distribution-Free Uncertainty Quantification and Applications in Imaging},
#   author={Angelopoulos, Anastasios N and Kohli, Amit P and Bates, Stephen and Jordan, Michael I and Malik,
#   Jitendra and Alshaabi, Thayer and Upadhyayula, Srigokul and Romano, Yaniv},
#   journal={arXiv preprint arXiv:2202.05265},
#   year={2022}
# }
class PinballLoss():

  def __init__(self, quantile=0.10, reduction='mean'):
      self.quantile = quantile
      assert 0 < self.quantile
      assert self.quantile < 1
      self.reduction = reduction

  def __call__(self, output, target):
      # pdb.set_trace()
      assert output.shape == target.shape
      loss = torch.zeros_like(target, dtype=torch.float)
      error = output - target
      smaller_index = error < 0
      bigger_index = 0 < error
      loss[smaller_index] = self.quantile * (abs(error)[smaller_index])
      loss[bigger_index] = (1-self.quantile) * (abs(error)[bigger_index])

      if self.reduction == 'sum':
        loss = loss.sum()
      if self.reduction == 'mean':
        loss = loss.mean()

      return loss



def qr_criterion(inputs_med, inputs_low, inputs_high, target, mse_loss_function):
    qlow_loss = PinballLoss(quantile=0.05)
    qhigh_loss = PinballLoss(quantile=0.95)

    loss = qlow_loss(inputs_low, target) + qhigh_loss(inputs_high, target) + mse_loss_function(inputs_med, target)
    loss_components = (qlow_loss(inputs_low, target), qhigh_loss(inputs_high, target), mse_loss_function(inputs_med, target))

    return loss, loss_components


def ensemble_criterion(inputs_mean, inputs_var, target, mse_loss_function):
    # compute negative log likelihood
    loss = torch.log(inputs_var) * 0.5 + 0.5 * (inputs_mean - target) ** 2 / inputs_var
    loss = loss.mean()

    return loss, None