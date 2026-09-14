import torch
import torch.nn as nn

from refine_net import RefineNet

class RVMWithRefine(nn.Module):
    def __init__(self, rvm_model, refine_net=None, low=0.05, high=0.95):
        super().__init__()
        self.rvm = rvm_model
        self.low = low
        self.high = high

        if refine_net is None:
            refine_net = RefineNet(in_channels=5)
        
        self.refine_net = refine_net

    def make_boundary_mask(self, alpha):
        mask = (alpha > self.low) & (alpha < self.high)
        return mask.float()
    
    def forward(self, src, depth, *rec, downsample_ratio=0.25):
        fgr, pha, *rec = self.rvm(src, *rec, downsample_ratio=downsample_ratio)

        ref_input = torch.cat([src, pha, depth], dim=1) # [B, 5, H, W]
        delta = self.refine_net(ref_input) # [B, 1, H, W]

        boundary_mask = self.make_boundary_mask(pha) # [B, 1, H, W]
        pha_refined = torch.clamp(pha + boundary_mask * delta, 0.0, 1.0)

        return {
            "fgr": fgr,
            "pha": pha,
            "delta": delta,
            "boundary_mask": boundary_mask,
            "pha_refined": pha_refined,
            "rec": rec,
        }