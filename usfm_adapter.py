"""
USFM encoder adapter.

This file keeps only the minimal Vision Transformer encoder pieces needed to
use the official USFM weights as an image feature extractor inside the current
project. It intentionally avoids the original Hydra/Lightning/mmseg runtime.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_


def to_2tuple(value):
    if isinstance(value, tuple):
        return value
    return (value, value)


class DropPath(nn.Module):
    """Stochastic depth."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        window_size=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None

        if window_size is not None:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (
                2 * window_size[1] - 1
            ) + 3
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(self.num_relative_distance, num_heads)
            )

            coords_h = torch.arange(window_size[0])
            coords_w = torch.arange(window_size[1])
            coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += window_size[0] - 1
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1

            relative_position_index = torch.zeros(
                size=(window_size[0] * window_size[1] + 1,) * 2,
                dtype=relative_coords.dtype,
            )
            relative_position_index[1:, 1:] = relative_coords.sum(-1)
            relative_position_index[0, 0:] = self.num_relative_distance - 3
            relative_position_index[0:, 0] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1
            self.register_buffer("relative_position_index", relative_position_index)
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rel_pos_bias=None):
        batch_size, token_count, channels = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (
                    self.q_bias,
                    torch.zeros_like(self.v_bias, requires_grad=False),
                    self.v_bias,
                )
            )

        qkv = F.linear(x, self.qkv.weight, qkv_bias)
        qkv = qkv.reshape(batch_size, token_count, 3, self.num_heads, channels // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        if self.relative_position_bias_table is not None:
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.view(-1)
            ].view(
                self.window_size[0] * self.window_size[1] + 1,
                self.window_size[0] * self.window_size[1] + 1,
                -1,
            )
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)

        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(batch_size, token_count, channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        init_values=None,
        norm_layer=nn.LayerNorm,
        window_size=None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            window_size=window_size,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, hidden_features=int(dim * mlp_ratio), drop=drop)

        if init_values is not None:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
        else:
            self.gamma_1 = None
            self.gamma_2 = None

    def forward(self, x, rel_pos_bias=None):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(
                self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias)
            )
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """Image to patch embedding."""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = self.patch_shape[0] * self.patch_shape[1]
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        _, _, height, width = x.shape
        if (height, width) != self.img_size:
            raise ValueError(
                f"USFM expects input size {self.img_size}, but got {(height, width)}."
            )
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class RelativePositionBias(nn.Module):
    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (
            2 * window_size[1] - 1
        ) + 3
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_distance, num_heads)
        )

        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1

        relative_position_index = torch.zeros(
            size=(window_size[0] * window_size[1] + 1,) * 2,
            dtype=relative_coords.dtype,
        )
        relative_position_index[1:, 1:] = relative_coords.sum(-1)
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self):
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1] + 1,
            self.window_size[0] * self.window_size[1] + 1,
            -1,
        )
        return relative_position_bias.permute(2, 0, 1).contiguous()


class USFMVisionTransformer(nn.Module):
    """
    Minimal ViT encoder used by USFM.

    The default configuration matches the public `configs/model/Cls/vit.yaml`
    in the official USFM repository.
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        init_values=0.1,
        use_abs_pos_emb=False,
        use_rel_pos_bias=True,
        use_shared_rel_pos_bias=False,
        use_mean_pooling=True,
        global_pool="auto",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.global_pool = global_pool
        self.use_rel_pos_bias = use_rel_pos_bias

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = (
            nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
            if use_abs_pos_emb
            else None
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.rel_pos_bias = (
            RelativePositionBias(self.patch_embed.patch_shape, num_heads)
            if use_shared_rel_pos_bias
            else None
        )

        drop_path_values = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path_values[layer_idx],
                    norm_layer=norm_layer,
                    init_values=init_values,
                    window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None,
                )
                for layer_idx in range(depth)
            ]
        )

        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.head = nn.Identity()

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
        self._fix_init_weight()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Conv2d):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def _fix_init_weight(self):
        for layer_id, layer in enumerate(self.blocks):
            layer.attn.proj.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))
            layer.mlp.fc2.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))

    def get_num_layers(self):
        return len(self.blocks)

    def forward_tokens(self, x):
        x = self.patch_embed(x)
        batch_size = x.size(0)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for block in self.blocks:
            x = block(x, rel_pos_bias=rel_pos_bias)

        return self.norm(x)

    def forward_features(self, x):
        tokens = self.forward_tokens(x)

        pool_mode = self.global_pool
        if pool_mode == "auto":
            pool_mode = "avg" if self.fc_norm is not None else "token"

        if pool_mode == "avg":
            patch_tokens = tokens[:, 1:, :]
            if self.fc_norm is not None:
                return self.fc_norm(patch_tokens.mean(dim=1))
            return patch_tokens.mean(dim=1)

        if pool_mode == "token":
            return tokens[:, 0]

        raise ValueError(
            f"Unsupported USFM global pool mode: {self.global_pool}. Use auto/avg/token."
        )

    def forward(self, x):
        return self.forward_features(x)


def _strip_known_prefixes(state_dict: Dict[str, torch.Tensor], prefixes: Iterable[str]):
    stripped = {}
    for key, value in state_dict.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True
        stripped[new_key] = value
    return stripped


def _extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise ValueError("USFM checkpoint must be a state dict or a dict containing one.")

    for candidate_key in ("state_dict", "model_state_dict", "model", "teacher", "module"):
        candidate = checkpoint.get(candidate_key)
        if isinstance(candidate, dict) and candidate:
            checkpoint = candidate
            break

    if not isinstance(checkpoint, dict) or not checkpoint:
        raise ValueError("Unable to find a valid state dict inside the USFM checkpoint.")

    return checkpoint


def _resize_pos_embed_if_needed(model: USFMVisionTransformer, state_dict: Dict[str, torch.Tensor]):
    if "pos_embed" not in state_dict:
        return

    if model.pos_embed is None:
        state_dict.pop("pos_embed", None)
        return

    src_pos_embed = state_dict["pos_embed"]
    dst_pos_embed = model.pos_embed
    if src_pos_embed.shape == dst_pos_embed.shape:
        return

    cls_token = src_pos_embed[:, :1]
    src_patch_tokens = src_pos_embed[:, 1:]
    dst_patch_tokens = dst_pos_embed[:, 1:]

    src_size = int(math.sqrt(src_patch_tokens.shape[1]))
    dst_size = int(math.sqrt(dst_patch_tokens.shape[1]))

    src_patch_tokens = src_patch_tokens.transpose(1, 2).reshape(1, src_pos_embed.shape[-1], src_size, src_size)
    src_patch_tokens = F.interpolate(
        src_patch_tokens,
        size=(dst_size, dst_size),
        mode="bicubic",
        align_corners=False,
    )
    src_patch_tokens = src_patch_tokens.flatten(2).transpose(1, 2)
    state_dict["pos_embed"] = torch.cat([cls_token, src_patch_tokens], dim=1)


def _resize_relative_bias_if_needed(
    model: USFMVisionTransformer, state_dict: Dict[str, torch.Tensor]
):
    if (
        getattr(model, "use_rel_pos_bias", False)
        and "rel_pos_bias.relative_position_bias_table" in state_dict
    ):
        shared_bias = state_dict.pop("rel_pos_bias.relative_position_bias_table")
        for layer_idx in range(model.get_num_layers()):
            state_dict[f"blocks.{layer_idx}.attn.relative_position_bias_table"] = shared_bias.clone()

    for key in list(state_dict.keys()):
        if "relative_position_index" in key:
            state_dict.pop(key)

    model_state = model.state_dict()
    for key in list(state_dict.keys()):
        if "relative_position_bias_table" not in key or key not in model_state:
            continue

        src_bias = state_dict[key]
        dst_bias = model_state[key]
        if src_bias.shape == dst_bias.shape:
            continue

        src_num_pos, num_heads = src_bias.shape
        dst_num_pos, _ = dst_bias.shape
        dst_patch_shape = model.patch_embed.patch_shape
        num_extra_tokens = dst_num_pos - ((dst_patch_shape[0] * 2 - 1) * (dst_patch_shape[1] * 2 - 1))
        src_size = int(math.sqrt(src_num_pos - num_extra_tokens))
        dst_size = int(math.sqrt(dst_num_pos - num_extra_tokens))

        extra_tokens = (
            src_bias[-num_extra_tokens:, :] if num_extra_tokens > 0 else src_bias.new_zeros((0, num_heads))
        )
        bias_tokens = src_bias[:-num_extra_tokens, :] if num_extra_tokens > 0 else src_bias
        bias_tokens = bias_tokens.transpose(0, 1).reshape(1, num_heads, src_size, src_size)
        bias_tokens = F.interpolate(
            bias_tokens,
            size=(dst_size, dst_size),
            mode="bicubic",
            align_corners=False,
        )
        bias_tokens = bias_tokens.reshape(num_heads, -1).transpose(0, 1)
        state_dict[key] = torch.cat([bias_tokens, extra_tokens], dim=0)


def load_usfm_pretrained(model: USFMVisionTransformer, checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _extract_state_dict(checkpoint)
    state_dict = _strip_known_prefixes(
        state_dict,
        prefixes=(
            "module.",
            "model.",
            "backbone.",
            "encoder.",
            "student.",
        ),
    )

    for key in list(state_dict.keys()):
        if key.startswith("head.") or key.startswith("classifier.") or key.startswith("fc.") or key == "mask_token":
            state_dict.pop(key)

    if getattr(model, "fc_norm", None) is not None:
        if "norm.weight" in state_dict and "fc_norm.weight" not in state_dict:
            state_dict["fc_norm.weight"] = state_dict.pop("norm.weight")
        if "norm.bias" in state_dict and "fc_norm.bias" not in state_dict:
            state_dict["fc_norm.bias"] = state_dict.pop("norm.bias")

    _resize_pos_embed_if_needed(model, state_dict)
    _resize_relative_bias_if_needed(model, state_dict)

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    return missing_keys, unexpected_keys


class USFMEncoderAdapter(nn.Module):
    """Thin wrapper that exposes a pooled USFM feature vector."""

    def __init__(self, image_size=224, pretrained_path=None, global_pool="auto"):
        super().__init__()
        self.encoder = USFMVisionTransformer(
            img_size=image_size,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            qkv_bias=True,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            init_values=0.1,
            use_abs_pos_emb=False,
            use_rel_pos_bias=True,
            use_shared_rel_pos_bias=False,
            use_mean_pooling=True,
            global_pool=global_pool,
        )
        self.feature_dim = self.encoder.embed_dim
        self.global_pool = global_pool
        self.pretrained_path = pretrained_path

        if pretrained_path:
            missing_keys, unexpected_keys = load_usfm_pretrained(self.encoder, pretrained_path)
            if unexpected_keys:
                print(f"USFM: ignored {len(unexpected_keys)} unexpected parameters")
            encoder_missing = [
                key
                for key in missing_keys
                if not key.startswith("head.") and not key.endswith("relative_position_index")
            ]
            if encoder_missing:
                print(f"USFM: {len(encoder_missing)} parameters were not restored from pretraining")

    def forward(self, x):
        return self.encoder(x)
