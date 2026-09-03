import torch
import torch.nn as nn
import torch.nn.functional as F

class TransitionFocusedLoss(nn.Module):
    def __init__(self, transition_weight=3.0, certain_weight=0.2):
        super().__init__()
        self.transition_weight = transition_weight
        self.certain_weight = certain_weight

    def forward(self, pred_alpha, gt_alpha, base_alpha=None, sem_label=None):
        l1_loss = F.l1_loss(pred_alpha, gt_alpha, reduction='none')

        if sem_label is not None:
            weight = torch.ones_like(pred_alpha)

            transition_mask = (sem_label == 1) | (sem_label == 2)
            weight[transition_mask.unsqueeze(1)] = self.transition_weight

            certain_mask = (sem_label == 0) | (sem_label == 3)
            weight[certain_mask.unsqueeze(1)] = self.certain_weight

        else:
            transition_mask = (gt_alpha > 0.05) & (gt_alpha < 0.95)
            weight = torch.where(transition_mask,
                                 torch.tensor(self.transition_weight).to(pred_alpha.device),
                                 torch.tensor(self.certain_weight).to(pred_alpha.device))

        weighted_loss = (l1_loss * weight).mean()

        if transition_mask.any() if sem_label is None else transition_mask.any():
            dx = torch.abs(pred_alpha[:, :, :, 1:] - pred_alpha[:, :, :, :-1])
            dy = torch.abs(pred_alpha[:, :, 1:, :] - pred_alpha[:, :, :-1, :])
            smooth_loss = (dx.mean() + dy.mean()) * 0.1
            weighted_loss = weighted_loss + smooth_loss

        return weighted_loss


class HybridLoss(nn.Module):
    def __init__(self, transition_weight=3.0, certain_weight=0.2, grad_weight=0.1):
        super().__init__()
        self.transition_weight = transition_weight
        self.certain_weight = certain_weight
        self.grad_weight = grad_weight

    def forward(self, pred_alpha, gt_alpha, base_alpha=None, sem_label=None):
        l1_loss = F.l1_loss(pred_alpha, gt_alpha, reduction='none')

        if sem_label is not None:
            weight = torch.ones_like(pred_alpha)
            transition_mask = (sem_label == 1) | (sem_label == 2)
            weight[transition_mask.unsqueeze(1)] = self.transition_weight
            certain_mask = (sem_label == 0) | (sem_label == 3)
            weight[certain_mask.unsqueeze(1)] = self.certain_weight
        else:
            transition_mask = (gt_alpha > 0.05) & (gt_alpha < 0.95)
            weight = torch.where(transition_mask,
                                 torch.tensor(self.transition_weight).to(pred_alpha.device),
                                 torch.tensor(self.certain_weight).to(pred_alpha.device))

        weighted_l1 = (l1_loss * weight).mean()

        if (sem_label is not None and transition_mask.any()) or (sem_label is None and transition_mask.any()):
            pred_grad_x = torch.abs(pred_alpha[:, :, :, 1:] - pred_alpha[:, :, :, :-1])
            pred_grad_y = torch.abs(pred_alpha[:, :, 1:, :] - pred_alpha[:, :, :-1, :])
            gt_grad_x = torch.abs(gt_alpha[:, :, :, 1:] - gt_alpha[:, :, :, :-1])
            gt_grad_y = torch.abs(gt_alpha[:, :, 1:, :] - gt_alpha[:, :, :-1, :])

            grad_loss = (F.l1_loss(pred_grad_x, gt_grad_x, reduction='mean') +
                        F.l1_loss(pred_grad_y, gt_grad_y, reduction='mean'))
        else:
            grad_loss = torch.tensor(0.0).to(pred_alpha.device)

        total_loss = weighted_l1 + self.grad_weight * grad_loss

        return total_loss