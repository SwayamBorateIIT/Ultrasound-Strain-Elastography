import torch
import torch.nn.functional as F

from torch import Tensor
from typing import Optional, Union
from torch.nn.modules.loss import _Loss
from torch.nn import functional as F

class ZNSSDLoss(_Loss):
    def __init__(self, patch_size: int = 32, stride: int = 16, epsilon: float = 1e-8):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.epsilon = epsilon

    def forward(self, I: Tensor, I_warp: Tensor) -> Tensor:
        return patch_znssd_loss(I, I_warp, self.patch_size, self.stride, self.epsilon)
    
    def znssd_loss(I: Tensor, I_warp: Tensor, patch_size: int = 32, stride: int = 16, epsilon: float = 1e-8) -> Tensor:
        """
        Computes the zero-normalized sum of squared differences (ZNSSD) loss
        between image I and its warped version I_warp.
        I and I_warp are assumed to be of shape [B, 1, H, W].
        """
        # Extract patches; output shape: [B, patch_size*patch_size, L]
        patches_I = F.unfold(I, kernel_size=patch_size, stride=stride)
        patches_I_warp = F.unfold(I_warp, kernel_size=patch_size, stride=stride)
        # Compute per-patch mean and standard deviation.
        mean_I = patches_I.mean(dim=1, keepdim=True)
        std_I = patches_I.std(dim=1, keepdim=True) + epsilon
        mean_I_warp = patches_I_warp.mean(dim=1, keepdim=True)
        std_I_warp = patches_I_warp.std(dim=1, keepdim=True) + epsilon
        # Normalize patches
        norm_I = (patches_I - mean_I) / std_I
        norm_I_warp = (patches_I_warp - mean_I_warp) / std_I_warp
        loss = torch.mean((norm_I - norm_I_warp) ** 2)
        return loss
    
class PatchZNSSDLoss(_Loss):
    def __init__(self, patch_size: int = 32, stride: int = 16, epsilon: float = 1e-8):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.epsilon = epsilon

    def forward(self, I: Tensor, I_warp: Tensor) -> Tensor:
        return patch_znssd_loss(I, I_warp, self.patch_size, self.stride, self.epsilon)
    
class SmoothnessLoss(_Loss):
    def __init__(self):
        super().__init__()

    def forward(self, flow: Tensor, img: Tensor) -> Tensor:
        return smoothness_loss(flow, img)

class CensusLoss(_Loss):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, I: Tensor, I_warp: Tensor) -> Tensor:
        return census_loss(I, I_warp, self.kernel_size)



def patch_znssd_loss(I, I_warp, patch_size=32, stride=16, epsilon=1e-8):
    """
    Computes the patch-based zero-normalized sum of squared differences (ZNSSD)
    between image I and its warped version I_warp.
    I and I_warp are assumed to be of shape [B, 1, H, W].
    """
    # Extract patches; output shape: [B, patch_size*patch_size, L]
    patches_I = F.unfold(I, kernel_size=patch_size, stride=stride)
    patches_I_warp = F.unfold(I_warp, kernel_size=patch_size, stride=stride)
    # Compute per-patch mean and standard deviation.
    mean_I = patches_I.mean(dim=1, keepdim=True)
    std_I = patches_I.std(dim=1, keepdim=True) + epsilon
    mean_I_warp = patches_I_warp.mean(dim=1, keepdim=True)
    std_I_warp = patches_I_warp.std(dim=1, keepdim=True) + epsilon
    # Normalize patches
    norm_I = (patches_I - mean_I) / std_I
    norm_I_warp = (patches_I_warp - mean_I_warp) / std_I_warp
    loss = torch.mean((norm_I - norm_I_warp) ** 2)
    return loss


def smoothness_loss(flow, img):
    """
    Computes an edge-aware smoothness loss on the flow.
    flow: [B, 2, H, W]
    img: [B, 1, H, W] used to weight the gradients.
    """
    # Calculate gradients of flow along x and y directions
    grad_flow_x = torch.abs(flow[:, :, :, 1:] - flow[:, :, :, :-1])
    grad_flow_y = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :])
    # Calculate image gradients, averaged over channel
    grad_img_x = torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]), dim=1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]), dim=1, keepdim=True)
    loss_x = grad_flow_x * torch.exp(-grad_img_x)
    loss_y = grad_flow_y * torch.exp(-grad_img_y)
    return torch.mean(loss_x) + torch.mean(loss_y)

def census_loss(I, I_warp, kernel_size=7):
    """
    Computes a simplified census loss between I and I_warp.
    This loss compares the local structure by forming binary descriptors.
    """
    pad = kernel_size // 2
    # Extract local patches
    patches_I = F.unfold(I, kernel_size=kernel_size, padding=pad)
    patches_I_warp = F.unfold(I_warp, kernel_size=kernel_size, padding=pad)
    center_idx = (kernel_size * kernel_size) // 2
    # Get the center pixel intensity for each patch
    center_I = patches_I[:, center_idx:center_idx+1, :]
    center_I_warp = patches_I_warp[:, center_idx:center_idx+1, :]
    # Form binary descriptors by comparing each pixel with the center pixel
    desc_I = torch.sign(patches_I - center_I)
    desc_I_warp = torch.sign(patches_I_warp - center_I_warp)
    # Compute the (normalized) Hamming distance as the census loss
    diff = torch.abs(desc_I - desc_I_warp) / 2.0
    return diff.mean()



