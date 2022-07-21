""" Image to Patch Embedding using Conv2d

A convolution based approach to patchifying a 2D image w/ embedding projection.

Based on the impl in https://github.com/google-research/vision_transformer

Hacked together by / Copyright 2020 Ross Wightman
"""

import torch
from torch import nn as nn

from .helpers import to_2tuple


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # print(f'preproject {x.shape}')
        x = self.proj(x)
        # print(f'after project {x.shape}')
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
            # print(f'after flatten {x.shape}')
        x = self.norm(x)
        # print(x.shape)
        return x


class PatchEmbedSeq(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        # self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.num_patches = 9
        self.flatten = flatten

        self.feat = nn.Conv2d(in_channels=in_chans, out_channels=in_chans, kernel_size=3, padding=1)
        self.flat = nn.Flatten()
        self.reduct = nn.Linear(150528, 768)

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # B, C, H, W = x.shape
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        z = []
        for x_i in x:
            z.append(torch.unsqueeze(self.reduct(self.flat(self.feat(x_i))), dim=1))

        z_out = torch.cat((z[0], z[1], z[2], z[3], z[4], z[5], z[6], z[7], z[8]), dim=1)

        # x = self.proj(x)
        # if self.flatten:
        #     x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        z_out = self.norm(z_out)
        return z_out


class PatchEmbedCNNViT(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, backbone=None, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        # img_size = to_2tuple(img_size)
        # patch_size = to_2tuple(patch_size)
        # self.img_size = img_size
        # self.patch_size = patch_size
        # self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        # self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.num_patches = 9
        self.flatten = flatten

        # self.feat = nn.Conv2d(in_channels=in_chans, out_channels=in_chans, kernel_size=3, padding=1)
        # self.flat = nn.Flatten()
        # self.reduct = nn.Linear(150528, 768)

        self.proj = backbone

        # self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # cnn feature extraction
        z = self.proj(x)
        # reshape for transformer input
        bs_patch, d_embed = z.shape
        bs = int(bs_patch / self.num_patches)
        z = z.reshape(self.num_patches, bs, d_embed).transpose(1, 0)   # (n_patch x bs, d_embed) -> (n_patch, bs, d_embed) -> (bs, n_patch, d_embed)
        # norm
        z = self.norm(z)

        return z


class PatchEmbedSeqResNet(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, backbone=None, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        # img_size = to_2tuple(img_size)
        # patch_size = to_2tuple(patch_size)
        # self.img_size = img_size
        # self.patch_size = patch_size
        # self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        # self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.num_patches = 9
        self.flatten = flatten

        # self.feat = nn.Conv2d(in_channels=in_chans, out_channels=in_chans, kernel_size=3, padding=1)
        # self.flat = nn.Flatten()
        # self.reduct = nn.Linear(150528, 768)

        self.proj = backbone

        # self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # cnn feature extraction
        z = self.proj(x)
        # reshape for transformer input
        bs_patch, d_embed = z.shape
        bs = int(bs_patch / self.num_patches)
        z = z.reshape(self.num_patches, bs, d_embed).transpose(1, 0)   # (n_patch x bs, d_embed) -> (n_patch, bs, d_embed) -> (bs, n_patch, d_embed)
        # norm
        z = self.norm(z)

        return z