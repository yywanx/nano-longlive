# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

import math

import torch
from torch import nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from wan.modules.attention import attention
from wan.modules.model import WanCrossAttention, WanLayerNorm, WanRMSNorm, rope_params, sinusoidal_embedding_1d
from wan.utils.position_embedding_utils import compute_temporal_freqs


class CausalWanModel(ModelMixin, ConfigMixin):
    """Wan backbone for chunkwise inference with a bounded KV cache."""

    ignore_for_config = ['patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size']

    @register_to_config
    def __init__(
        self,
        model_type='t2v',
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        local_attn_size=-1,
        sink_size=0,
        num_frame_per_block=1,
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
    ):
        super().__init__()
        assert model_type in ['t2v', 'ti2v']
        self.model_type = model_type
        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'), nn.Linear(dim, dim))
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(
                dim,
                ffn_dim,
                num_heads,
                local_attn_size=local_attn_size,
                sink_size=sink_size,
                qk_norm=qk_norm,
                cross_attn_norm=cross_attn_norm,
                eps=eps,
            ) for _ in range(num_layers)
        ])
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        assert dim % num_heads == 0 and dim // num_heads % 2 == 0
        d = dim // num_heads
        # Keep complex frequencies outside buffers so model.to(bfloat16) preserves them.
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
        ], dim=1)
        self.init_weights()
        self.num_frame_per_block = num_frame_per_block
        self.t_scale = 1.0
        self.use_relative_rope = False

    def forward(self, x, t, context, seq_len, kv_cache=None, current_start=0, cache_start=None):
        """Predict flow for x [B, C, F, H, W] and update each layer's cache."""

        if kv_cache is None:
            raise NotImplementedError("AR training without a KV cache is not implemented yet")

        self.freqs = self.freqs.to(x.device)

        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = [tuple(u.shape[2:]) for u in x]
        x = torch.cat([u.flatten(2).transpose(1, 2) for u in x])

        e = sinusoidal_embedding_1d(self.freq_dim, t.flatten())
        e = self.time_embedding(e.unflatten(0, t.shape).type_as(x))
        e0 = self.time_projection(e).unflatten(2, (6, self.dim))

        context = torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context])
        context = self.text_embedding(context)

        cache_update_infos = []
        for block_index, block in enumerate(self.blocks):
            x, cache_update_info = block(
                x,
                e0,
                grid_sizes,
                self.freqs,
                context,
                kv_cache[block_index],
                current_start=current_start,
                t_scale=self.t_scale,
                use_relative_rope=self.use_relative_rope,
            )
            cache_update_infos.append((block_index, cache_update_info))

        # Attention reads temporary K/V; commit the updates after all layers run.
        self._apply_cache_updates(kv_cache, cache_update_infos)
        x = self.head(x, e.unsqueeze(2))
        return torch.stack(self.unpatchify(x, grid_sizes))

    def _apply_cache_updates(self, kv_cache, cache_update_infos):
        for block_index, (current_end, local_end_index, update_info) in cache_update_infos:
            cache = kv_cache[block_index]
            if update_info['action'] == 'roll_and_insert':
                sink_tokens = update_info['sink_tokens']
                num_rolled_tokens = update_info['num_rolled_tokens']
                num_evicted_tokens = update_info['num_evicted_tokens']
                for name in ('k', 'v'):
                    cache[name][:, sink_tokens:sink_tokens + num_rolled_tokens] = cache[name][
                        :, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens
                    ].clone()

            local_start_index = update_info['local_start_index']
            cache['k'][:, local_start_index:local_end_index] = update_info['new_k']
            cache['v'][:, local_start_index:local_end_index] = update_info['new_v']
            cache['global_end_index'].fill_(current_end)
            cache['local_end_index'].fill_(local_end_index)

    def unpatchify(self, x, grid_sizes):
        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        nn.init.zeros_(self.head.head.weight)


class CausalWanAttentionBlock(nn.Module):
    def __init__(self, dim, ffn_dim, num_heads, local_attn_size, sink_size, qk_norm, cross_attn_norm, eps):
        super().__init__()
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(
            dim,
            num_heads,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            qk_norm=qk_norm,
            eps=eps,
        )
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        grid_sizes,
        freqs,
        context,
        kv_cache,
        current_start=0,
        t_scale=1.0,
        use_relative_rope=False,
    ):
        num_frames = e.shape[1]
        frame_seqlen = x.shape[1] // num_frames
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)

        modulated_x = self.norm1(x).unflatten(1, (num_frames, frame_seqlen))
        modulated_x = (modulated_x * (1 + e[1]) + e[0]).flatten(1, 2)
        y, cache_update_info = self.self_attn(
            modulated_x,
            grid_sizes,
            freqs,
            kv_cache,
            current_start=current_start,
            t_scale=t_scale,
            use_relative_rope=use_relative_rope,
        )
        x = x + (y.unflatten(1, (num_frames, frame_seqlen)) * e[2]).flatten(1, 2)

        # The reference recomputes text K/V on every call.
        x = x + self.cross_attn(self.norm3(x), context, context_lens=None)

        ffn_in = self.norm2(x).unflatten(1, (num_frames, frame_seqlen))
        ffn_in = (ffn_in * (1 + e[4]) + e[3]).flatten(1, 2)
        y = self.ffn(ffn_in)
        x = x + (y.unflatten(1, (num_frames, frame_seqlen)) * e[5]).flatten(1, 2)
        return x, cache_update_info


class CausalWanSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, local_attn_size, sink_size=0, qk_norm=True, eps=1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps) if qk_norm else nn.Identity()

    def forward(self, x, grid_sizes, freqs, kv_cache, current_start=0, t_scale=1.0, use_relative_rope=False):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)

        num_new_frames, h, w = grid_sizes[0]
        frame_seqlen = h * w
        num_new_tokens = q.shape[1]
        current_end = current_start + num_new_tokens
        sink_tokens = self.sink_size * frame_seqlen
        kv_cache_size = kv_cache['k'].shape[1]
        _cache_global_end = kv_cache['global_end_index'].item()
        _cache_local_end = kv_cache['local_end_index'].item()

        # Absolute RoPE caches rotated K. Relative RoPE caches raw K.
        if use_relative_rope:
            key_to_cache = k
        else:
            current_start_frame = current_start // frame_seqlen
            roped_query = causal_rope_apply(q, grid_sizes, freqs, current_start_frame, t_scale).type_as(v)
            key_to_cache = causal_rope_apply(k, grid_sizes, freqs, current_start_frame, t_scale).type_as(v)

        temp_k, temp_v = kv_cache['k'].clone(), kv_cache['v'].clone()
        if current_end > _cache_global_end and num_new_tokens + _cache_local_end > kv_cache_size:
            num_evicted_tokens = num_new_tokens + _cache_local_end - kv_cache_size
            num_rolled_tokens = _cache_local_end - num_evicted_tokens - sink_tokens
            local_end_index = _cache_local_end + current_end - _cache_global_end - num_evicted_tokens
            local_start_index = local_end_index - num_new_tokens
            for cache in (temp_k, temp_v):
                cache[:, sink_tokens:sink_tokens + num_rolled_tokens] = cache[
                    :, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens
                ].clone()
            cache_update_info = dict(
                action='roll_and_insert',
                sink_tokens=sink_tokens,
                num_rolled_tokens=num_rolled_tokens,
                num_evicted_tokens=num_evicted_tokens,
            )
        else:
            # Denoising and clean recaching overwrite the current chunk's slots.
            local_end_index = _cache_local_end + current_end - _cache_global_end
            local_start_index = local_end_index - num_new_tokens
            cache_update_info = dict(action='direct_insert')

        temp_k[:, local_start_index:local_end_index] = key_to_cache
        temp_v[:, local_start_index:local_end_index] = v
        cache_update_info.update(
            local_start_index=local_start_index,
            local_end_index=local_end_index,
            new_k=key_to_cache,
            new_v=v,
            current_end=current_end,
        )

        max_attention_size = self.local_attn_size * frame_seqlen
        window_start = max(0, local_end_index - max_attention_size)
        if sink_tokens > 0 and window_start > 0:
            local_window_start = max(sink_tokens, local_end_index - (max_attention_size - sink_tokens))
            window_k = torch.cat([temp_k[:, :sink_tokens], temp_k[:, local_window_start:local_end_index]], dim=1)
            window_v = torch.cat([temp_v[:, :sink_tokens], temp_v[:, local_window_start:local_end_index]], dim=1)
        else:
            window_k = temp_k[:, window_start:local_end_index]
            window_v = temp_v[:, window_start:local_end_index]

        if use_relative_rope:
            window_frames = window_k.shape[1] // frame_seqlen
            window_grid_sizes = [(window_frames, h, w)] * b
            window_k = causal_rope_apply(window_k, window_grid_sizes, freqs, t_scale=t_scale).type_as(v)
            q_start_frame = window_frames - num_new_frames
            roped_query = causal_rope_apply(q, grid_sizes, freqs, q_start_frame, t_scale).type_as(v)

        # Every token sees retained history and the whole current chunk.
        x = attention(roped_query, window_k, window_v)
        x = self.o(x.flatten(2))
        return x, (current_end, local_end_index, cache_update_info)


class CausalHead(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, math.prod(patch_size) * out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        num_frames = e.shape[1]
        frame_seqlen = x.shape[1] // num_frames
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        x = self.norm(x).unflatten(1, (num_frames, frame_seqlen))
        x = x * (1 + e[1]) + e[0]
        return self.head(x)


def causal_rope_apply(x, grid_sizes, freqs, start_frame=0, t_scale=1.0):
    n, c = x.size(2), x.size(3) // 2
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    output = []
    for i, (f, h, w) in enumerate(grid_sizes):
        seq_len = f * h * w
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
        temporal_freqs = compute_temporal_freqs(freqs[0], f, start_frame, t_scale, x.device)
        freqs_i = torch.cat([
            temporal_freqs.view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ], dim=-1).reshape(seq_len, 1, -1)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        output.append(torch.cat([x_i, x[i, seq_len:]]))
    return torch.stack(output).type_as(x)
