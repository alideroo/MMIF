import itertools
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm
from typing import Tuple
from model.net import OverlapPatchEmbed, TransformerBlock, BaseFeatureExtraction, DetailFeatureExtraction
class LeakySiLU(nn.Module):

    def __init__(self, negative_slope=0.01) -> None:
        super().__init__()

        self.negative_slope = float(negative_slope)

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        return (1 - self.negative_slope) * F.silu(input) + self.negative_slope * input
    

class Sampler(nn.Module):

    def __init__(self, z_dim: int):
        super().__init__()

        self.z_dim = int(z_dim)
        assert self.z_dim > 0

    def forward(self, batch_size: int): # no input

        raise NotImplementedError("return (batch_size, z_dim)-shaped tensor")
    
class NeuralSampler(Sampler):

    def __init__(self, z_dim: int, inp_channels=1,
                 out_channels=1,
                 dim=64,
                 num_blocks=[4, 4],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',):
        super().__init__(z_dim)

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                            bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.baseFeature = BaseFeatureExtraction(dim=dim, num_heads = heads[2])
        self.detailFeature = DetailFeatureExtraction()
        
    def forward(self, batch_size: int):
        
        eps = torch.randn(size=[batch_size, 1, self.z_dim, self.z_dim]).cuda()
        inp_enc_level1 = self.patch_embed(eps)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        base_feature = self.baseFeature(out_enc_level1)
        detail_feature = self.detailFeature(out_enc_level1)
        return base_feature, detail_feature
    
def logit_gradient(x: torch.Tensor, z: torch.Tensor, logit: torch.Tensor):

    batch_size = x.shape[0]

    grad = torch.autograd.grad(
        outputs=logit,
        inputs=[x, z],
        grad_outputs=torch.ones_like(logit),
        retain_graph=True,
        create_graph=True,
    )
    grad_x = grad[0].view(batch_size, -1)
    grad_z = grad[1].view(batch_size, -1)
    grad_cat = torch.cat([grad_x, grad_z], dim=1)

    return grad_cat

class Log_distance_coef(nn.Module):

    def __init__(self):
        super().__init__()
        self.log_distance_coef = nn.Parameter(torch.tensor(1.0).log(), requires_grad=True)

    def forward(self):
        return self.log_distance_coef
    

def gradient_penalty_one_centered(
    x: torch.Tensor, z: torch.Tensor,
    x2s: torch.Tensor, zs: torch.Tensor,
    disc_block: nn.Module
) -> torch.Tensor:

    batch_size = x.shape[0]
    
    tp = torch.rand(size=[batch_size, ], device="cuda")
    xp = x * tp[:,None,None,None] + x2s * (1 - tp[:,None,None,None])
    zp = z * tp[:,None] + zs * (1 - tp[:,None])
    
    xp.requires_grad_(True)
    zp.requires_grad_(True)

    logit_interpolation = disc_block(xp, zp)
    grad_cat = logit_gradient(xp, zp, logit_interpolation)
    grad_norm = grad_cat.norm(p=2, dim=1)
    loss_gp = torch.mean((grad_norm - 1) ** 2)

    return loss_gp


class Discriminator(nn.Module):

    def __init__(self, x_shape: Tuple[int, int, int], z_dim: int) -> None:
        super().__init__()

        self.x_net = nn.Sequential(
            spectral_norm(nn.Conv2d(1, 32, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            
            spectral_norm(nn.Conv2d(32, 64, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            
            spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            
            nn.Flatten(),
            spectral_norm(nn.Linear(128 * 16 * 16, 64)),
        )


        self.z_net = nn.Sequential(
            spectral_norm(nn.Conv2d(64, 64, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            
            spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)), 
            nn.LeakyReLU(0.2),
            
            spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)), 
            nn.LeakyReLU(0.2),
            
            nn.Flatten(),
            spectral_norm(nn.Linear(256 * 16 * 16, 64)), 
        )


        self.xz_net = nn.Sequential(
            spectral_norm(nn.Linear(128, 256)),
            nn.LeakyReLU(0.2),

            spectral_norm(nn.Linear(256, 1)),
        )

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:

        h = torch.cat([self.x_net(x), self.z_net(z)], dim=1) / 2
        return self.xz_net(h)
    
def logit_gradient(x: torch.Tensor, z: torch.Tensor, logit: torch.Tensor):

    batch_size = x.shape[0]

    grad = torch.autograd.grad(
        outputs=logit,
        inputs=[x, z],
        grad_outputs=torch.ones_like(logit),
        retain_graph=True,
        create_graph=True,
    )
    grad_x = grad[0].view(batch_size, -1)
    grad_z = grad[1].view(batch_size, -1)
    grad_cat = torch.cat([grad_x, grad_z], dim=1)

    return grad_cat

def gradient_penalty_one_centered(
    x: torch.Tensor, z: torch.Tensor,
    x2s: torch.Tensor, zs: torch.Tensor,
    disc_block: nn.Module
) -> torch.Tensor:

    batch_size = x.shape[0]

    tp = torch.rand(size=[batch_size, ]).cuda()

    xp = x * tp[:, None, None, None] + x2s * (1 - tp[:, None, None, None])
    zp = z * tp[:, None, None, None] + zs * (1 - tp[:, None, None, None])
    
    xp.requires_grad_(True)
    zp.requires_grad_(True)

    logit_interpolation = disc_block(xp, zp)
    grad_cat = logit_gradient(xp, zp, logit_interpolation)
    grad_norm = grad_cat.norm(p=2, dim=1)
    loss_gp = torch.mean((grad_norm - 1) ** 2)

    return loss_gp