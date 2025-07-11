import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.loss import _Loss
from typing import Optional, Union
from torch.nn import functional as F
from torchvision.transforms import functional as TF




def get_patches(x, x_wind=143):
    kh, dh = (x_wind*2)+1, 1
    patches = x.unfold(2, kh, dh)
    patches = torch.squeeze(patches,dim=1).permute(0,1,3,2)
    return patches

def get_strain(disp, x_wind=143):
    d = x_wind*2+1
    Uxx_list = []
    disp = get_patches(disp,x_wind=x_wind)
    depthX = torch.linspace(1,d,d)
    depthX = torch.stack([depthX,torch.ones_like(depthX)]).float().permute(1,0).cuda()
    depthX = depthX.unsqueeze(0).repeat(disp.shape[1],1,1)
    XtX = depthX.permute(0,2,1).bmm(depthX)
    for i in range(len(disp)):
        # Cholesky decomposition
        XtY = depthX.permute(0,2,1).bmm(disp[i,...])
        betas_cholesky = torch.linalg.solve(XtX, XtY)
        Uxx = torch.squeeze(betas_cholesky[:,0,:])
        # pad to original size
        Uxx_list += [F.pad(Uxx, (0,0,x_wind, x_wind))]
    return torch.stack(Uxx_list).unsqueeze(1)

def warp(img, flow):
    # img: [B, C, H, W], flow: [B, 2, H, W] (displacement in pixel units)
    B, C, H, W = img.shape
    grid_y, grid_x = torch.meshgrid(torch.arange(0, H, device=img.device),
                                    torch.arange(0, W, device=img.device), indexing='ij')
    grid = torch.stack((grid_x, grid_y), dim=2).float()  # [H, W, 2]
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, H, W, 2]
    # Convert grid to normalized coordinates in [-1, 1]
    grid_norm = torch.zeros_like(grid)
    grid_norm[:, :, :, 0] = 2.0 * grid[:, :, :, 0] / (W - 1) - 1.0
    grid_norm[:, :, :, 1] = 2.0 * grid[:, :, :, 1] / (H - 1) - 1.0
    # Normalize flow to the same scale and add
    flow_norm = torch.zeros_like(flow)
    flow_norm[:, 0, :, :] = flow[:, 0, :, :] * (2.0 / (W - 1))
    flow_norm[:, 1, :, :] = flow[:, 1, :, :] * (2.0 / (H - 1))
    warped_grid = grid_norm + flow_norm.permute(0, 2, 3, 1)  # [B, H, W, 2]
    warped_img = F.grid_sample(img, warped_grid, align_corners=True, padding_mode='border')