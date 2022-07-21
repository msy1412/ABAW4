import torch
import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout, MaxPool2d, \
    AdaptiveAvgPool2d, Sequential, Module
from collections import namedtuple

from .helpers import named_apply
from .layers import PatchEmbed, DropPath, trunc_normal_, lecun_normal_
from functools import partial
import math
from einops import rearrange
from models import layers


class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output

class SEModule(Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0, bias=False)

        nn.init.xavier_uniform_(self.fc1.weight.data)

        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0, bias=False)

        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return module_input * x

class bottleneck_IR(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False), BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False), PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False), BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut

class bottleneck_IR_SE(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth),
            SEModule(depth, 16)
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut

class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''

def get_block(in_channel, depth, num_units, stride=2):

    return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]

def get_blocks(num_layers):
    if num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            # get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]

    return blocks

class Backbone(Module):
    def __init__(self, input_size, num_layers, mode='ir'):
        super(Backbone, self).__init__()
        assert input_size[0] in [112, 224], "input_size should be [112, 112] or [224, 224]"
        assert num_layers in [50, 100, 152], "num_layers should be 50, 100 or 152"
        assert mode in ['ir', 'ir_se'], "mode should be ir or ir_se"
        blocks = get_blocks(num_layers)

        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        # if input_size[0] == 112:
        #     self.output_layer = Sequential(BatchNorm2d(512),
        #                                    Dropout(),
        #                                    Flatten(),
        #                                    Linear(512 * 7 * 7, 512),
        #                                    BatchNorm1d(512))
        # else:
        #     self.output_layer = Sequential(BatchNorm2d(512),
        #                                    Dropout(),
        #                                    Flatten(),
        #                                    Linear(512 * 14 * 14, 512),
        #                                    BatchNorm1d(512))

        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))
        self.body = Sequential(*modules)

        self._initialize_weights()

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        # x = self.output_layer(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.in_features = in_features
        if in_features == 512:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.act = act_layer()
            self.fc2 = nn.Linear(hidden_features, out_features)
            self.drop = nn.Dropout(drop)
        else:
            self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
            self.bn1 = nn.BatchNorm2d(hidden_features)
            self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, padding=1, groups=hidden_features)
            self.bn2 = nn.BatchNorm2d(hidden_features)
            self.act = act_layer()
            self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
            self.bn3 = nn.BatchNorm2d(out_features)
            self.drop = nn.Dropout(drop)

    def forward(self, x):
        if self.in_features == 512:
            x = self.fc1(x)
            x = self.act(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = self.drop(x)
        else:
            B,N,C = x.shape
            x = x.reshape(B, int(N**0.5), int(N**0.5), C).permute(0,3,1,2)
            x = self.bn1(self.fc1(x))
            x = self.act(x)
            x = self.drop(x)
            x = self.act(self.bn2(self.dwconv(x)))
            x = self.bn3(self.fc2(x))
            x = self.drop(x)
            x = x.permute(0,2,3,1).reshape(B, -1, C)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Transformer(nn.Module):
    def __init__(self, base_dim, depth, heads, mlp_ratio,
                 drop_rate=.0, attn_drop_rate=.0, drop_path_prob=None, use_mask=False, masked_block=None):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])
        self.depth = depth
        embed_dim = base_dim * heads

        if drop_path_prob is None:
            drop_path_prob = [0.0 for _ in range(depth)]

        if use_mask==True:
            assert masked_block is not None
            self.blocks = nn.ModuleList()
            for i in range(depth):
                if i < masked_block:
                    self.blocks.append(Block(
                        dim=embed_dim,
                        num_heads=heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=True,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=drop_path_prob[i],
                        norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        use_mask=use_mask
                    ))
                else:
                    self.blocks.append(Block(
                        dim=embed_dim,
                        num_heads=heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=True,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=drop_path_prob[i],
                        norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        use_mask=False
                    ))
        else:
            self.blocks = nn.ModuleList([
                Block(
                    dim=embed_dim,
                    num_heads=heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path_prob[i],
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                )
                for i in range(depth)])


    def forward(self, x):
        B,C,H,W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        # x = x.permute(0,2,3,1).reshape(B, H * W, C)
        for i in range(self.depth):
            x = self.blocks[i](x)
        # x = x.reshape(B, H, W, C).permute(0,3,1,2)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        return x

class conv_head_pooling(nn.Module):
    def __init__(self, in_feature, out_feature, stride,
                 padding_mode='zeros'):
        super(conv_head_pooling, self).__init__()

        self.conv = nn.Conv2d(in_feature, out_feature, kernel_size=stride + 1,
                              padding=stride // 2, stride=stride,
                              padding_mode=padding_mode, groups=in_feature)

    def forward(self, x):

        x = self.conv(x)

        return x

class PosConv(nn.Module):
    # PEG  from https://arxiv.org/abs/2102.10882
    def __init__(self, in_chans, embed_dim=512, stride=1):
        super(PosConv, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, 3, stride, 1, bias=True, groups=embed_dim), )
        self.stride = stride

    def forward(self, x):#, H, W
        # B, N, C = x.shape
        # cnn_feat_token = x.transpose(1, 2).view(B, C, H, W)
        cnn_feat_token = x
        x = self.proj(cnn_feat_token)
        if self.stride == 1:
            x += cnn_feat_token
        # x = x.flatten(2).transpose(1, 2)
        return x

    def no_weight_decay(self):
        return ['proj.%d.weight' % i for i in range(4)]

class PoolingTransformer(nn.Module):
    def __init__(self, base_dims, depth, heads,
                 mlp_ratio, num_classes=7,
                 attn_drop_rate=.0, drop_rate=.0, drop_path_rate=.0, use_mask=False, masked_block=None,num_AU_patch=4):
        super(PoolingTransformer, self).__init__()

        total_block = sum(depth)
        block_idx = 0

        self.base_dims = base_dims
        self.heads = heads
        self.num_classes = num_classes

        self.pos_embed = nn.Parameter(torch.zeros(1, 14*14 , base_dims[0] * heads[0]))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.transformers = nn.ModuleList([])
        self.pools = nn.ModuleList([])
        self.pos_block = nn.ModuleList([])
        self.num_AU_patch=num_AU_patch
        self.patch_embed=nn.Conv2d(256, base_dims[0] * heads[0], 3, padding=1)

        for stage in range(len(depth)):
            drop_path_prob = [drop_path_rate * i / total_block
                              for i in range(block_idx, block_idx + depth[stage])]
            block_idx += depth[stage]
            self.pos_block.append(
                PosConv(base_dims[stage] * heads[stage], base_dims[stage] * heads[stage])
            )
            self.transformers.append(
                Transformer(base_dims[stage], depth[stage], heads[stage],
                            mlp_ratio,
                            drop_rate, attn_drop_rate, drop_path_prob)
            )
            if stage < len(heads) - 1:
                self.pools.append(
                    conv_head_pooling(base_dims[stage] * heads[stage],
                                      base_dims[stage + 1] * heads[stage + 1],
                                      stride=2
                                      )
                )

        self.norm = nn.LayerNorm(base_dims[-1] * heads[-1], eps=1e-6)
        self.embed_dim = base_dims[-1] * heads[-1]
        if num_AU_patch==7:
            self.gap_AU11 = nn.AdaptiveAvgPool2d(1)
            self.gap_AU12 = nn.AdaptiveAvgPool2d(1)
            self.gap_AU13 = nn.AdaptiveAvgPool2d(1)
            self.gap_AU21 = nn.AdaptiveAvgPool2d(1)
            self.gap_AU22 = nn.AdaptiveAvgPool2d(1)
            self.gap_AU23 = nn.AdaptiveAvgPool2d(1)
            self.gap_AU3  = nn.AdaptiveAvgPool2d(1)
            self.norm_AU11 = nn.LayerNorm(self.embed_dim, eps=1e-6)
            self.norm_AU12 = nn.LayerNorm(self.embed_dim, eps=1e-6)
            self.norm_AU13 = nn.LayerNorm(self.embed_dim, eps=1e-6)
            self.norm_AU21 = nn.LayerNorm(self.embed_dim, eps=1e-6)
            self.norm_AU22 = nn.LayerNorm(self.embed_dim, eps=1e-6)
            self.norm_AU23 = nn.LayerNorm(self.embed_dim, eps=1e-6)
            self.norm_AU3  = nn.LayerNorm(self.embed_dim, eps=1e-6)
            self.head_AU11= nn.Linear(self.embed_dim, 4)
            self.head_AU12= nn.Linear(self.embed_dim, 1)
            self.head_AU13= nn.Linear(self.embed_dim, 4)
            self.head_AU21= nn.Linear(self.embed_dim, 1)
            self.head_AU22= nn.Linear(self.embed_dim, 1)
            self.head_AU23= nn.Linear(self.embed_dim, 1)
            self.head_AU3= nn.Linear(self.embed_dim, 14)
        elif num_AU_patch==5:
            self.gap_AU1 = nn.AdaptiveAvgPool2d(1)
            self.gap_AU21 = nn.AdaptiveAvgPool2d(1)
            self.gap_AU22 = nn.AdaptiveAvgPool2d(1)
            self.gap_AU23 = nn.AdaptiveAvgPool2d(1)
            self.gap_AU3  = nn.AdaptiveAvgPool2d(1)
            self.norm_AU1 = nn.LayerNorm(self.embed_dim, eps=1e-6)
            self.norm_AU21 = nn.LayerNorm(self.embed_dim, eps=1e-6)
            self.norm_AU22 = nn.LayerNorm(self.embed_dim, eps=1e-6)
            self.norm_AU23 = nn.LayerNorm(self.embed_dim, eps=1e-6)
            self.norm_AU3  = nn.LayerNorm(self.embed_dim, eps=1e-6)
            self.head_AU1= nn.Linear(self.embed_dim, 5)
            self.head_AU21= nn.Linear(self.embed_dim, 1)
            self.head_AU22= nn.Linear(self.embed_dim, 1)
            self.head_AU23= nn.Linear(self.embed_dim, 1)
            self.head_AU3= nn.Linear(self.embed_dim, 14)
        elif num_AU_patch==4:
            self.gap_AU1 = nn.AdaptiveAvgPool2d(1)
            self.gap_AU21 = nn.AdaptiveAvgPool2d(1)
            self.gap_AU23 = nn.AdaptiveAvgPool2d(1)
            self.gap_AU3  = nn.AdaptiveAvgPool2d(1)
            self.norm_AU1 = nn.LayerNorm(self.embed_dim, eps=1e-6)
            self.norm_AU21 = nn.LayerNorm(self.embed_dim, eps=1e-6)
            self.norm_AU23 = nn.LayerNorm(self.embed_dim, eps=1e-6)
            self.norm_AU3  = nn.LayerNorm(self.embed_dim, eps=1e-6)
            self.head_AU1= nn.Linear(self.embed_dim, 4)
            self.head_AU21= nn.Linear(self.embed_dim, 1)
            self.head_AU23= nn.Linear(self.embed_dim, 1)
            self.head_AU3= nn.Linear(self.embed_dim, 7)

        elif num_AU_patch==3:
            self.gap_AU1 = nn.AdaptiveAvgPool2d(1)
            self.gap_AU2 = nn.AdaptiveAvgPool2d(1)
            self.gap_AU3  = nn.AdaptiveAvgPool2d(1)
            self.norm_AU1 = nn.LayerNorm(self.embed_dim, eps=1e-6)
            self.norm_AU2 = nn.LayerNorm(self.embed_dim, eps=1e-6)
            self.norm_AU3 = nn.LayerNorm(self.embed_dim, eps=1e-6)
            self.head_AU1= nn.Linear(self.embed_dim, 5)
            self.head_AU2= nn.Linear(self.embed_dim, 2)
            self.head_AU3= nn.Linear(self.embed_dim, 14)
        elif num_AU_patch==2:
            self.gap_upperAU = nn.AdaptiveAvgPool2d(1)
            self.gap_lowerAU = nn.AdaptiveAvgPool2d(1)
            self.norm_AU_up = nn.LayerNorm(self.embed_dim, eps=1e-6)
            self.norm_AU_low = nn.LayerNorm(self.embed_dim, eps=1e-6)
            self.upperAU_head= nn.Linear(512, 7)
            self.lowerAU_head= nn.Linear(512, 9)
        elif num_AU_patch==1:
            self.gap_AU = nn.AdaptiveAvgPool2d(1)
            self.norm_AU = nn.LayerNorm(self.embed_dim, eps=1e-6)
            self.AU_head= nn.Linear(self.embed_dim, 21)

        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.output_layer = Sequential( Dropout(),
                                        Flatten(),
                                        Linear(base_dims[-1] * heads[-1] * 14 * 14, base_dims[-1] * heads[-1]),
                                        nn.ReLU())
        # Classifier head
        if num_classes > 0:
            self.head = nn.Linear(base_dims[-1] * heads[-1], num_classes)
        else:
            self.head = nn.Identity()

        self.apply(self._init_weights)
        trunc_normal_(self.pos_embed, std=.02)

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def learnable_PosEmbed(self,x):
        B,C,H,W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = x + self.pos_embed
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        if num_classes > 0:
            self.head = nn.Linear(self.embed_dim, num_classes)
        else:
            self.head = nn.Identity()

    def AUBranch(self,AU_x):
        B,C,H,W = AU_x.shape
        if self.num_AU_patch==7:
            AU11=AU_x[:,:, :7 , :7 ]
            AU12=AU_x[:,:, :7 ,4:10]
            AU13=AU_x[:,:, :7 ,7:  ]
            AU21=AU_x[:,:,5:12, :6 ]
            AU22=AU_x[:,:,4:10,4:10]
            AU23=AU_x[:,:,5:12,8:  ]
            AU3 =AU_x[:,:,6:  , :  ]
            AU11=self.head_AU11(self.norm_AU11(self.gap_AU11(AU11).squeeze()))
            AU12=self.head_AU12(self.norm_AU12(self.gap_AU12(AU12).squeeze()))
            AU13=self.head_AU13(self.norm_AU13(self.gap_AU13(AU13).squeeze()))
            AU21=self.head_AU21(self.norm_AU21(self.gap_AU21(AU21).squeeze()))
            AU22=self.head_AU22(self.norm_AU22(self.gap_AU22(AU22).squeeze()))
            AU23=self.head_AU23(self.norm_AU23(self.gap_AU23(AU23).squeeze()))
            AU3 =self.head_AU3(self.norm_AU3(self.gap_AU3(AU3).squeeze()))
            AU1257=torch.maximum(AU11,AU13)
            AU6=torch.maximum(AU21,AU23)
            AU_all=torch.cat((AU1257[:,:2],AU12,AU1257[:,2].view(B,-1),AU6,AU1257[:,3].view(B,-1),AU22,AU3),dim=1)
        elif self.num_AU_patch==5:
            AU1 =AU_x[:,:, :7 , :  ]
            AU21=AU_x[:,:,5:12, :6 ]
            AU22=AU_x[:,:,4:10,4:10]
            AU23=AU_x[:,:,5:12,8:  ]
            AU3 =AU_x[:,:,6:  , :  ]
            AU1=self.head_AU1(self.norm_AU1(self.gap_AU1(AU1).squeeze()))
            AU21=self.head_AU21(self.norm_AU21(self.gap_AU21(AU21).squeeze()))
            AU22=self.head_AU22(self.norm_AU22(self.gap_AU22(AU22).squeeze()))
            AU23=self.head_AU23(self.norm_AU23(self.gap_AU23(AU23).squeeze()))
            AU3 =self.head_AU3(self.norm_AU3(self.gap_AU3(AU3).squeeze()))
            AU6=torch.maximum(AU21,AU23)
            AU_all=torch.cat((AU1[:,:4],AU6,AU1[:,4].view(B,-1),AU22,AU3),dim=1)
        elif self.num_AU_patch==4:
            AU1 =AU_x[:,:, :7 , :  ]
            AU21=AU_x[:,:,5:12, :6 ]
            AU23=AU_x[:,:,5:12,8:  ]
            AU3 =AU_x[:,:,6:  , :  ]
            AU1=self.head_AU1(self.norm_AU1(self.gap_AU1(AU1).squeeze()))
            AU21=self.head_AU21(self.norm_AU21(self.gap_AU21(AU21).squeeze()))
            AU23=self.head_AU23(self.norm_AU23(self.gap_AU23(AU23).squeeze()))
            AU3 =self.head_AU3(self.norm_AU3(self.gap_AU3(AU3).squeeze()))
            AU6=torch.maximum(AU21,AU23)
            AU_all=torch.cat((AU1[:,:3],AU6,AU1[:,3].view(B,-1),AU3),dim=1)

        elif self.num_AU_patch==3:
            AU1=AU_x[:,:, :7 ,: ]
            AU2=AU_x[:,:,4:12,: ]
            AU3=AU_x[:,:,6:  ,: ]
            AU1=self.head_AU1(self.norm_AU1(self.gap_AU1(AU1).squeeze()))
            AU2=self.head_AU2(self.norm_AU2(self.gap_AU2(AU2).squeeze()))
            AU3=self.head_AU3(self.norm_AU3(self.gap_AU3(AU3).squeeze()))
            AU_all=torch.cat((AU1[:,:4],AU2[:,0].view(B,-1),AU1[:,4].view(B,-1),AU2[:,1].view(B,-1),AU3),dim=1)
        elif self.num_AU_patch==2:
            upper_AU=AU_x[:,:, :8 ,:]
            lower_AU=AU_x[:,:, 6: ,:]
            upperAU = self.upperAU_head(self.norm_AU_up(self.gap_upperAU(upper_AU).squeeze()))
            lowerAU = self.lowerAU_head(self.norm_AU_low(self.gap_lowerAU(lower_AU).squeeze()))
            AU_all=torch.cat((upperAU, lowerAU), dim=1)
        elif self.num_AU_patch==1:
            AU_all=self.AU_head(self.norm_AU(self.gap_AU(AU_x).squeeze()))
        return AU_all

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.learnable_PosEmbed(x)
        x = self.pos_drop(x)
        for stage in range(len(self.pools)):
            # x = self.pos_block[stage](x)
            x = self.transformers[stage](x)
            # x = self.pools[stage](x)
        AU_x = x
        AU_output = self.AUBranch(AU_x)
        x = self.transformers[-1](x)
        cls_features = self.norm(self.gap(x).squeeze())
        # cls_features = self.norm(self.output_layer(x))
        return cls_features,AU_output

    def forward(self, x):
        cls_features,AU_output = self.forward_features(x)
        output = self.head(cls_features)
        return output,AU_output

class IR50_ViT(nn.Module):
    def __init__(self, num_classes=7,ir_50_pth=None,num_AU_patch=7):
        super(IR50_ViT, self).__init__()

        self.embed_dim = 256
        self.num_patch = 196

        self.cnn = self.IR_50([112,112])
        # print(self.cnn)
        if ir_50_pth:
            self.cnn=self.load_model_weights(self.cnn,ir_50_pth)
        self.norm = nn.LayerNorm([self.num_patch, self.embed_dim])

        self.vit = PoolingTransformer(
                    num_classes=num_classes,
                    base_dims=[32, 32],
                    depth=[4, 2],
                    heads=[16, 16],
                    mlp_ratio=4,
                    num_AU_patch=num_AU_patch)
        
    def forward(self,x):
        x=self.cnn(x)                     # IR-50
        # x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC  (flatten from H)
        # x = self.norm(x)                  # LayerNorm
        x = self.vit(x)                   # ViT
        return x
    def load_model_weights(self,model,model_path):
        state_dict = torch.load(model_path)
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith( 'output_layer' ):
                continue
            elif k.startswith('body'):
                k = k.strip()
                layer = k.split('.')[1]
                if int(layer)>20:
                    continue
            new_state_dict[k] = v
        # load params
        model.load_state_dict(new_state_dict,strict=True)#使用strict=True，因为没有增加层，只是删减
        return model

    def IR_50(self,input_size):
        """Constructs a ir-50 model.
        """
        model = Backbone(input_size, 50, 'ir')

        return model
