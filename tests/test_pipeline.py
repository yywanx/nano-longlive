"""Pipeline call order and the Wan adapter boundary, using small stand-in models."""

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch
from torch import nn

from nano_longlive.inference import CausalDiffusionInferencePipeline
from nano_longlive.wan_adapter import WanDiffusionWrapper, load_generator_checkpoint


class PipelineFlowTests(unittest.TestCase):
    """A fake generator returns 1 for the prompt and 0 for the negative prompt; the fake scheduler subtracts flow."""

    def setUp(self):
        self.calls = []
        self.schedulers = []

        def generator(**kwargs):
            self.calls.append(kwargs)
            value = 1 if kwargs["conditional_dict"] == "prompt" else 0
            return torch.full_like(kwargs["noisy_image_or_video"], value), None

        generator.model = SimpleNamespace(
            patch_size=(1, 2, 2), dim=8, num_heads=2,
            blocks=[SimpleNamespace() for _ in range(2)],
            num_frame_per_block=2, local_attn_size=6, sink_size=2,
        )
        schedulers = self.schedulers

        class Pipeline(CausalDiffusionInferencePipeline):
            def _initialize_sample_scheduler(self, noise):
                scheduler = SimpleNamespace(
                    timesteps=[2, 1],
                    step=lambda flow, t, latents, return_dict: (latents - flow,),
                )
                schedulers.append(scheduler)
                return scheduler

        self.pipeline = Pipeline(
            generator,
            lambda text_prompts: text_prompts[0],
            None,
            num_frame_per_block=2,
            guidance_scale=2.0,
        )

    def test_guidance_and_clean_recache_before_next_chunk(self):
        noise = torch.full((1, 4, 1, 2, 2), 10.0)
        output = self.pipeline.inference(noise, ["prompt"])
        self.assertTrue(torch.equal(output, torch.full_like(noise, 6)))
        self.assertTrue(torch.equal(noise, torch.full_like(noise, 10)))
        self.assertEqual(len(self.schedulers), 2)
        self.assertIsNot(self.schedulers[0], self.schedulers[1])
        self.assertEqual([c["current_start"] for c in self.calls], [0]*6 + [2]*6)
        self.assertEqual([c["timestep"][0, 0].item() for c in self.calls],
                         [2, 2, 1, 1, 0, 0]*2)
        for i in (4, 5, 10, 11):
            self.assertTrue(torch.all(self.calls[i]["noisy_image_or_video"] == 6))
        self.assertIsNot(self.calls[0]["kv_cache"], self.calls[1]["kv_cache"])


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([0.25]))
        self.patch_size = (1, 2, 2)

    def forward(self, x, **kwargs):
        self.input = x
        self.kwargs = kwargs
        return x * self.weight


class WanAdapterTests(unittest.TestCase):
    def test_layout_offsets_and_x0(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                model = TinyBackbone().to(dtype)
                generator = WanDiffusionWrapper(model)
                noise = torch.arange(48, dtype=torch.float32).reshape(1, 2, 3, 2, 4).to(dtype)
                timestep = generator.scheduler.timesteps[[200, 700]].reshape(1, 2).to(dtype)
                context = torch.ones(1, 4, 6, dtype=dtype)
                kv_cache = [{"global_end_index": torch.tensor([8]), "local_end_index": torch.tensor([8])}]
                flow, x0 = generator(
                    noise, {"prompt_embeds": context}, timestep,
                    kv_cache=kv_cache, current_start=8,
                )
                self.assertEqual(model.input.shape, (1, 3, 2, 2, 4))
                self.assertIs(model.kwargs["t"], timestep)
                self.assertIs(model.kwargs["context"], context)
                self.assertIs(model.kwargs["kv_cache"], kv_cache)
                self.assertEqual(model.kwargs["current_start"], 8)
                self.assertEqual(model.kwargs["cache_start"], 8)
                self.assertEqual(model.kwargs["seq_len"], 4)
                self.assertEqual(flow.dtype, dtype)
                torch.testing.assert_close(flow, noise * 0.25, rtol=0, atol=0)
                for frame in range(2):
                    index = (generator.scheduler.timesteps.double() - timestep[0, frame]).abs().argmin()
                    sigma = generator.scheduler.sigmas[index].double()
                    expected = (noise[:, frame].double() - sigma * flow[:, frame].double()).to(dtype)
                    torch.testing.assert_close(x0[:, frame], expected, rtol=0, atol=0)

    def test_checkpoint_formats_and_strict_loading(self):
        generator = WanDiffusionWrapper(TinyBackbone())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            for key in (None, "generator", "model"):
                state = {"model.weight": torch.tensor([0.75])}
                torch.save(state if key is None else {key: state}, path)
                load_generator_checkpoint(generator, path)
                self.assertEqual(generator.model.weight.item(), 0.75)
            for state in ({}, {"wrong.weight": torch.ones(1)}, {"model.weight": torch.ones(2)}):
                torch.save(state, path)
                with self.assertRaises(RuntimeError):
                    load_generator_checkpoint(generator, path)
            torch.save({"generator_ema": {"model.weight": torch.tensor(1.0)}}, path)
            with self.assertRaises(ValueError):
                load_generator_checkpoint(generator, path)


if __name__ == "__main__":
    unittest.main()
