"""Run the official LongLive code and nano side by side on tiny CPU models; outputs must match exactly."""

from contextlib import redirect_stdout
import io
from itertools import product
import os
import unittest
from unittest.mock import patch

import torch
from torch import nn
from diffusers.configuration_utils import FrozenDict

from nano_longlive.inference import CausalDiffusionInferencePipeline
from nano_longlive.wan_adapter import WanDiffusionWrapper
from reference import cpu_attention, load_reference, load_reference_inference, make_cache, tiny_model
from wan.modules import causal_model, model


class ReferenceParityTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.enterContext(patch.object(causal_model, "attention", cpu_attention))
        self.enterContext(patch.object(model, "flash_attention", cpu_attention))
        self.enterContext(torch.inference_mode())

    def test_backbone_matches_reference_through_rollover(self):
        reference = load_reference()
        for dtype in (torch.float32, torch.bfloat16):
            for relative in (False, True):
                for window, sink, extra_frames in ((6, 2, 0), (5, 1, 0), (2, 0, 0), (6, 2, 2)):
                    with self.subTest(dtype=dtype, relative=relative, window=window, sink=sink, extra=extra_frames):
                        original = tiny_model(reference["CausalWanModel"], window, sink).to(dtype)
                        actual = tiny_model(local_attn_size=window, sink_size=sink).to(dtype)
                        # The default zero output head would hide transformer differences.
                        original.head.head.weight.normal_(std=0.1)
                        actual.load_state_dict(original.state_dict(), strict=True)
                        original.use_relative_rope = actual.use_relative_rope = relative
                        for block in original.blocks:
                            block.self_attn.max_attention_size = window * 2
                        expected_cache = make_cache(original, dtype)
                        actual_cache = make_cache(actual, dtype)
                        # A larger buffer exercises sink prepending outside the local attention window.
                        for cache in expected_cache + actual_cache:
                            for key in ("k", "v"):
                                extra = cache[key].new_zeros(1, extra_frames * 2, 2, 12)
                                cache[key] = torch.cat([cache[key], extra], dim=1)
                        context = torch.randn(1, 3, 12).to(dtype)
                        reference["_FREQS_I_CACHE"].clear()
                        reference["_CURRENT_GRID_META"].clear()
                        for chunk in range(6):
                            for step in (999, 500, 0):
                                x = torch.randn(1, 2, 2, 2, 4).to(dtype)
                                t = torch.full((1, 2), step, dtype=torch.float32)
                                kwargs = dict(t=t, context=context, seq_len=4, current_start=chunk * 4)
                                expected = original(x, kv_cache=expected_cache, crossattn_cache=[{}, {}], **kwargs)
                                output = actual(x, kv_cache=actual_cache, **kwargs)
                                torch.testing.assert_close(output, expected, rtol=0, atol=0)
                                for got, want in zip(actual_cache, expected_cache):
                                    for key in got:
                                        torch.testing.assert_close(got[key], want[key], rtol=0, atol=0)

    def test_complete_inference_matches_reference(self):
        reference = load_reference_inference()
        reference["wan_default_config"] = {
            "Wan2.2-TI2V-5B": {"num_heads": 2, "head_dim": 12, "num_transformer_blocks": 2},
        }
        self.enterContext(patch.dict(os.environ, {"LLV2_DEFER_KV_UPDATES": "0"}))

        def snapshot(module, args, kwargs, output):
            record = {name: kwargs[name].clone() for name in ("noisy_image_or_video", "timestep")}
            record["prompt_embeds"] = kwargs["conditional_dict"]["prompt_embeds"].clone()
            record["flow"], record["x0"] = (value.clone() for value in output)
            record["current_start"] = kwargs["current_start"]
            record["cache_start"] = kwargs.get("cache_start", kwargs["current_start"])
            for layer, cache in enumerate(kwargs["kv_cache"]):
                end = cache["local_end_index"].item()
                for key in ("k", "v", "global_end_index", "local_end_index"):
                    value = cache[key][:, :end] if key in ("k", "v") else cache[key]
                    record[f"{layer}:{key}"] = value.clone()
            return record

        profiles = ((4, 5.0, 6, 2), (2, 1.0, 5, 1), (1, 5.0, 2, 0))
        for dtype, relative, guidance, profile in product(
            (torch.float32, torch.bfloat16),
            (False, True),
            (1.0, 2.0),
            profiles,
        ):
            steps, shift, window, sink = profile
            with self.subTest(dtype=dtype, relative=relative, guidance=guidance, profile=profile):
                original = tiny_model(reference["CausalWanModel"], window, sink).to(dtype)
                original.head.head.weight.normal_(std=0.1)
                backbone = tiny_model(local_attn_size=window, sink_size=sink).to(dtype)
                backbone.load_state_dict(original.state_dict(), strict=True)
                backbone.use_relative_rope = relative

                # Supply tiny weights at the original checkpoint-loading boundary.
                with patch.object(reference["CausalWanModel"], "from_pretrained", return_value=original):
                    reference_generator = reference["WanDiffusionWrapper"](
                        is_causal=True,
                        timestep_shift=shift,
                        num_frame_per_block=2,
                        local_attn_size=window,
                        sink_size=sink,
                    )
                generator = WanDiffusionWrapper(backbone, timestep_shift=shift)
                embeddings = {prompt: torch.randn(1, 3, 12).to(dtype) for prompt in ("first", "next", "")}

                def text_encoder(text_prompts):
                    return {"prompt_embeds": torch.cat([embeddings[prompt] for prompt in text_prompts])}

                config = FrozenDict(
                    model_kwargs={},
                    image_or_video_shape=[1, 12, 2, 2, 4],
                    num_frame_per_block=2,
                    timestep_shift=shift,
                    use_relative_rope=relative,
                    inference=dict(
                        sampling_steps=steps,
                        local_attn_size=window,
                        sink_size=sink,
                        guidance_scale=guidance,
                    ),
                )
                with redirect_stdout(io.StringIO()):
                    expected_pipeline = reference["CausalDiffusionInferencePipeline"](
                        config,
                        torch.device("cpu"),
                        generator=reference_generator,
                        text_encoder=text_encoder,
                        vae=nn.Identity(),
                    )
                actual_pipeline = CausalDiffusionInferencePipeline(
                    generator,
                    text_encoder,
                    None,
                    num_frame_per_block=2,
                    guidance_scale=guidance,
                    sampling_steps=steps,
                    timestep_shift=shift,
                )

                actual_calls, expected_calls = [], []
                actual_hook = generator.register_forward_hook(
                    lambda *args: actual_calls.append(snapshot(*args)),
                    with_kwargs=True,
                )
                expected_hook = reference_generator.register_forward_hook(
                    lambda *args: expected_calls.append(snapshot(*args)),
                    with_kwargs=True,
                )
                try:
                    # A changed prompt and noise on request two exercise cache reset.
                    for prompt in ("first", "next"):
                        actual_calls.clear()
                        expected_calls.clear()
                        noise = torch.randn(1, 12, 2, 2, 4).to(dtype)
                        reference["_FREQS_I_CACHE"].clear()
                        reference["_CURRENT_GRID_META"].clear()
                        with redirect_stdout(io.StringIO()):
                            expected = expected_pipeline.inference(
                                noise.clone(),
                                [[prompt] * 6],
                                return_latents=True,
                            )
                        output = actual_pipeline.inference(noise.clone(), [prompt])
                        torch.testing.assert_close(output, expected, rtol=0, atol=0)
                        self.assertEqual(len(actual_calls), 6 * (steps + 1) * (1 if guidance == 1 else 2))
                        self.assertEqual(len(actual_calls), len(expected_calls))
                        for call, (got, want) in enumerate(zip(actual_calls, expected_calls)):
                            for key in got:
                                with self.subTest(prompt=prompt, call=call, field=key):
                                    if torch.is_tensor(got[key]):
                                        torch.testing.assert_close(got[key], want[key], rtol=0, atol=0)
                                    else:
                                        self.assertEqual(got[key], want[key])
                finally:
                    actual_hook.remove()
                    expected_hook.remove()

    def test_rope_offsets_scaling_and_padding_match_reference(self):
        reference = load_reference()
        freqs = tiny_model().freqs
        for dtype in (torch.float32, torch.bfloat16):
            x = torch.randn(2, 15, 2, 12).to(dtype)
            for start in (0, 8, 1020):
                for scale in (1.0, 0.5):
                    with self.subTest(dtype=dtype, start=start, scale=scale):
                        reference["_FREQS_I_CACHE"].clear()
                        grids = [(2, 2, 3), (3, 2, 2)]
                        expected = reference["causal_rope_apply"](x, grids, freqs, start, scale)
                        output = causal_model.causal_rope_apply(x, grids, freqs, start, scale)
                        torch.testing.assert_close(output, expected, rtol=0, atol=0)
                        torch.testing.assert_close(output[:, 12:], x[:, 12:], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
