from functools import partial

import timm.models.vision_transformer
from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp, Attention

import torch
import torch.nn as nn

from einops import rearrange


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        bias=True,
    ):
        super().__init__()
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias,
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.proj(x)
        W = x.size(-1)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, T, W


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.1,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        # Temporal Attention Parameters
        self.temporal_norm1 = norm_layer(dim)
        self.temporal_attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.temporal_fc = nn.Linear(dim, dim)

        # Drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, B, T):
        # Temporal
        x = rearrange(x, "b (t n) m -> (b n) t m", b=B, t=(T + 1))
        res_temporal = self.drop_path(self.temporal_attn(self.temporal_norm1(x)))
        res_temporal = self.temporal_fc(res_temporal)
        x = x + res_temporal

        # Spatial
        x = rearrange(x, "(b n) t m -> (b t) n m", b=B, t=(T + 1))
        res_spatial = self.drop_path(self.attn(self.norm1(x)))
        x = x + res_spatial
        x = rearrange(x, "(b t) n m -> b (t n) m", b=B, t=(T + 1))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VideoVisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """Vision Transformer with support for global average pooling"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        global_pool=False,
        num_frames=8,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        qkv_bias=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        **kwargs
    ):
        super(VideoVisionTransformer, self).__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            **kwargs
        )

        self.num_frames = num_frames
        self.patch_embed = PatchEmbed(
            img_size,
            patch_size,
            in_chans,
            embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim),
        )

        self.time_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.time_embed = nn.Parameter(torch.zeros(1, num_frames + 1, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        self.global_pool = global_pool
        if self.global_pool:
            embed_dim = embed_dim
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

        # initialization of temporal attention weights
        i = 0
        for m in self.blocks.modules():
            m_str = str(m)
            if "Block" in m_str:
                if i > 0:
                    nn.init.constant_(m.temporal_fc.weight, 0)
                    nn.init.constant_(m.temporal_fc.bias, 0)
                i += 1

    def forward_features(self, x):
        B = x.shape[0]
        x, T, W = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = rearrange(x, "(b t) n m -> (b n) t m", b=B, t=T)
        time_cls_tokens = self.time_cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((time_cls_tokens, x), dim=1)
        x = x + self.time_embed
        x = self.pos_drop(x)
        x = rearrange(x, "(b n) t m -> b (t n) m", b=B, t=(T + 1))

        for blk in self.blocks:
            x = blk(x, B, T)

        if self.global_pool:
            x = rearrange(x, "b (t n) m -> (b t) n m", b=B, t=(T + 1))
            x = x[:, 1:, :]
            x = rearrange(x, "(b t) n m -> (b n) t m", b=B, t=(T + 1))
            x = x[:, 1:, :]
            x = rearrange(x, "(b n) t m -> b (t n) m", b=B, t=T)

            x = x.mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            x = rearrange(x, "b (t n) m -> b t n m", b=B, t=(T + 1))
            cls_token = x[:, :, 0].mean(dim=1).unsqueeze(0)
            time_cls_token = x[:, 0].mean(dim=1).unsqueeze(0)
            outcome = torch.cat([cls_token, time_cls_token]).mean(dim=0)
            outcome = self.fc_norm(x)

        return outcome

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def video_vit_tiny_patch16(**kwargs):
    model = VideoVisionTransformer(
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def video_vit_base_patch16(**kwargs):
    model = VideoVisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def video_vit_large_patch16(**kwargs):
    model = VideoVisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def video_vit_huge_patch14(**kwargs):
    model = VideoVisionTransformer(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model
