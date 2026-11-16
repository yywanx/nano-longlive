"""Load the official LongLive definitions on CPU, and build tiny models shared by the tests."""

import ast
import importlib.util
import math
import os
from pathlib import Path
import types
from typing import List, Optional
import unittest

import torch
from torch import nn
from torch.nn import functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from nano_longlive.inference import CausalDiffusionInferencePipeline
from wan.modules import causal_model, model
from wan.utils.position_embedding_utils import compute_temporal_freqs, select_temporal_offset_for_sample


def reference_root():
    default = Path(__file__).resolve().parents[2] / "LongLive/LongLive2.0"
    root = Path(os.environ.get("LONGLIVE_REFERENCE_DIR", default))
    if not (root / "wan_5b/modules/causal_model.py").is_file():
        raise unittest.SkipTest("reference checkout unavailable")
    return root


def cpu_attention(q, k, v, **kwargs):
    # Both implementations use the same CPU backend; this does not test FlashAttention.
    output = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
    return output.transpose(1, 2).contiguous()


def exec_definitions(path, names, namespace):
    """Execute selected top-level classes/functions from a source file, skipping its imports."""

    tree = ast.parse(path.read_text())
    definitions = [node for node in tree.body if getattr(node, "name", None) in names]
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)


def load_reference():
    """Return the original backbone definitions, without GPU-only FlexAttention/Triton imports."""

    namespace = dict(
        torch=torch, nn=nn, math=math, dist=torch.distributed, BlockMask=object,
        ModelMixin=ModelMixin, ConfigMixin=ConfigMixin, register_to_config=register_to_config,
        WanRMSNorm=model.WanRMSNorm, WanLayerNorm=model.WanLayerNorm, WanCrossAttention=model.WanCrossAttention,
        rope_apply=model.rope_apply, rope_params=model.rope_params, sinusoidal_embedding_1d=model.sinusoidal_embedding_1d,
        attention=cpu_attention, flash_attention=cpu_attention,
        _compute_temporal_freqs=compute_temporal_freqs, select_temporal_offset_for_sample=select_temporal_offset_for_sample,
        _TRITON_ROPE_ENABLED=False, _TRITON_ADALN_ENABLED=False, _CGRAPH_OUTPLACE_KV_ENABLED=False,
        _FREQS_I_CACHE_ENABLED=True, _FREQS_I_CACHE={}, _FREQS_I_CACHE_MAX=16, _CURRENT_GRID_META={},
    )
    names = {
        "causal_rope_apply", "MultiShotT2VCrossAttention", "CausalWanSelfAttention",
        "CausalWanAttentionBlock", "CausalHead", "CausalWanModel",
    }
    exec_definitions(reference_root() / "wan_5b/modules/causal_model.py", names, namespace)
    return namespace


def load_reference_inference():
    """Add the original pipeline, wrapper and schedulers to the backbone namespace."""

    root = reference_root()
    namespace = load_reference()
    namespace.update(
        os=os,
        types=types,
        List=List,
        Optional=Optional,
        tqdm=lambda sequence: sequence,
        DEFAULT_SCENE_CUT_PREFIX="The scene transitions. ",
        _LLV2_TIME=False,
        _LLV2_DUMP_LATENT_DIR="",
        _LLV2_PROFILE_SPEC="",
        _LLV2_PROFILE_OUTPUT_DIR="",
        _LLV2_PROFILE_CALL_COUNTER=0,
    )
    for relative, names in (
        ("utils/scheduler.py", ("FlowMatchScheduler", "SchedulerInterface")),
        ("wan_5b/utils/fm_solvers_unipc.py", ("FlowUniPCMultistepScheduler",)),
    ):
        spec = importlib.util.spec_from_file_location(f"reference_{Path(relative).stem}", root / relative)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        namespace.update({name: getattr(module, name) for name in names})

    for relative, name in (
        ("utils/config.py", "section_get"),
        ("utils/prompt_conditioning.py", "encode_prompt_blocks"),
        ("utils/wan_5b_wrapper.py", "WanDiffusionWrapper"),
        ("pipeline/causal_diffusion_inference.py", "CausalDiffusionInferencePipeline"),
    ):
        exec_definitions(root / relative, {name}, namespace)
    return namespace


def tiny_model(model_class=causal_model.CausalWanModel, local_attn_size=6, sink_size=2):
    return model_class(
        in_dim=2, out_dim=2, dim=24, ffn_dim=48, freq_dim=8, text_dim=12, text_len=4,
        num_heads=2, num_layers=2, patch_size=(1, 2, 2), num_frame_per_block=2,
        local_attn_size=local_attn_size, sink_size=sink_size,
    ).eval().requires_grad_(False)


def make_cache(backbone, dtype, frame_seqlen=2):
    pipeline = CausalDiffusionInferencePipeline(types.SimpleNamespace(model=backbone), None, None, num_frame_per_block=2)
    pipeline.frame_seq_length = frame_seqlen
    return pipeline._initialize_kv_cache(torch.empty(1, dtype=dtype))
