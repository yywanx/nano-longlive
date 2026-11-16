"""Record official LongLive and nano inference on the same inputs, then compare the recordings.

prepare: save shared noise and settings.
record:  run one implementation in its own process and save every generator call.
compare: print per-tensor errors between two recordings.
"""

import argparse
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

import torch
from torch import nn

from nano_longlive.utils import generation_parser, make_inputs, pipeline_options


ROOT = Path(__file__).resolve().parents[1]

# Pin the reference to its plain BF16 path: no fused kernels, compilation or alternative attention backends.
REFERENCE_ENV = {
    "LLV2_TRITON_ROPE": "0", "LLV2_TRITON_ADALN": "0", "LLV2_CGRAPH_OUTPLACE_KV": "0",
    "LLV2_DEFER_KV_UPDATES": "0", "LLV2_COMPILE_VAE": "0", "LLV2_USE_FA4": "0",
    "LLV2_USE_FA3": "0", "LLV2_USE_TE_ATTN": "0", "LLV2_TIME": "0", "LLV2_PROFILE": "",
    "LLV2_DUMP_LATENT_DIR": "", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
}


os.environ.update(REFERENCE_ENV)


class ReplayTextEncoder(nn.Module):
    """Return saved prompt embeddings so both runs see identical text conditioning."""

    def __init__(self, embeddings):
        super().__init__()
        self.embeddings = embeddings

    def forward(self, text_prompts):
        return {"prompt_embeds": torch.cat([self.embeddings[prompt] for prompt in text_prompts])}


class Trace:
    """Forward hook that saves each generator call's inputs, outputs and cache state."""

    def __init__(self, directory):
        self.directory = directory
        self.directory.mkdir()
        self.count = 0

    def __call__(self, module, args, kwargs, output):
        caches = kwargs["kv_cache"]
        end = caches[0]["local_end_index"].item()
        positions = torch.linspace(0, end - 1, min(16, end), device=output[0].device).long().unique()
        record = {
            "current_start": kwargs["current_start"],
            "timestep": kwargs["timestep"].cpu(),
            "input": kwargs["noisy_image_or_video"].cpu(),
            "flow": output[0].cpu(),
            "x0": output[1].cpu(),
            "counters": torch.stack([
                torch.cat([cache["global_end_index"], cache["local_end_index"]]) for cache in caches
            ]).cpu(),
        }
        for name in ("k", "v"):
            # A few evenly spaced occupied slots per layer; the full cache is too large to save per call.
            record[name] = torch.stack([cache[name].index_select(1, positions) for cache in caches]).cpu()
        torch.save(record, self.directory / f"{self.count:04d}.pt")
        self.count += 1


@contextmanager
def reference_workspace(reference_dir, model_path):
    """Import the reference from a temporary directory where its hard-coded asset path resolves."""

    old_cwd, old_path = Path.cwd(), sys.path[:]
    with tempfile.TemporaryDirectory(prefix="nano-reference-") as directory:
        assets = Path(directory) / "wan_models"
        assets.mkdir()
        (assets / "Wan2.2-TI2V-5B").symlink_to(model_path, target_is_directory=True)
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(reference_dir))
        os.chdir(directory)
        try:
            yield
        finally:
            os.chdir(old_cwd)
            sys.path[:] = old_path


def reference_pipeline(config, reference_dir):
    from omegaconf import OmegaConf
    from utils.wan_5b_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
    from nano_longlive.wan_adapter import load_generator_checkpoint

    path = reference_dir / "pipeline/causal_diffusion_inference.py"
    spec = importlib.util.spec_from_file_location("longlive_reference_inference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model_kwargs = {name: config[name] for name in (
        "timestep_shift", "local_attn_size", "sink_size", "num_frame_per_block",
    )}
    model_kwargs["model_name"] = "Wan2.2-TI2V-5B"
    generator = WanDiffusionWrapper(**model_kwargs, is_causal=True)
    load_generator_checkpoint(generator, config["generator_ckpt"])
    generator = generator.to(device="cuda", dtype=torch.bfloat16).eval().requires_grad_(False)
    text_encoder = WanTextEncoder().to(device="cuda", dtype=torch.bfloat16)
    vae = WanVAEWrapper().to(device="cuda", dtype=torch.bfloat16)

    inference = {name: config[name] for name in (
        "sampling_steps", "local_attn_size", "sink_size", "guidance_scale", "negative_prompt",
    )}
    inference.update(
        multi_shot_sink=False,
        multi_shot_rope_offset=0.0,
        shot_clean_recache=False,
        independent_first_frame=False,
        streaming_vae=False,
        async_vae=False,
    )
    args = OmegaConf.create({
        "model_kwargs": model_kwargs,
        "inference": inference,
        "num_train_timestep": 1000,
        "num_frame_per_block": config["num_frame_per_block"],
        "timestep_shift": config["timestep_shift"],
        "use_relative_rope": config["use_relative_rope"],
        "kv_quant": False,
        "image_or_video_shape": [1, config["num_frames"], 48, config["height"] // 16, config["width"] // 16],
    })
    return module.CausalDiffusionInferencePipeline(
        args,
        torch.device("cuda"),
        generator=generator,
        text_encoder=text_encoder,
        vae=vae,
    ).eval().requires_grad_(False)


@torch.inference_mode()
def record(args):
    inputs = torch.load(args.inputs, map_location="cpu", weights_only=True)
    config = inputs["config"]
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output_dir.mkdir(parents=True, exist_ok=False)

    if args.implementation == "reference":
        pipeline = reference_pipeline(config, args.reference_dir)
        prompts = [config["prompt"], config["negative_prompt"]]
        embeddings = {p: pipeline.text_encoder(text_prompts=[p])["prompt_embeds"].cpu() for p in prompts}
        # The reference takes one prompt per chunk; single-shot repeats the same prompt.
        text_prompts = [[config["prompt"]] * (config["num_frames"] // config["num_frame_per_block"])]
    else:
        from nano_longlive.inference import CausalDiffusionInferencePipeline

        pipeline = CausalDiffusionInferencePipeline.from_pretrained(**pipeline_options(config))
        embeddings = torch.load(args.embeddings, map_location="cpu", weights_only=True)
        text_prompts = [config["prompt"]]

    torch.save(embeddings, args.output_dir / "embeddings.pt")
    pipeline.text_encoder = ReplayTextEncoder({p: e.cuda() for p, e in embeddings.items()})

    trace = Trace(args.output_dir / "calls")
    handle = pipeline.generator.register_forward_hook(trace, with_kwargs=True)
    noise = inputs["noise"].cuda()
    if args.implementation == "reference":
        latents = pipeline.inference(noise, text_prompts, return_latents=True)
    else:
        latents = pipeline.inference(noise, text_prompts)
    handle.remove()

    torch.save(latents.cpu(), args.output_dir / "latents.pt")
    metadata = {
        "config": config,
        "implementation": args.implementation,
        "calls": trace.count,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {trace.count} calls to {args.output_dir}")


def tensor_error(actual, expected):
    if not actual.is_floating_point():
        return {"max_abs": 0.0 if torch.equal(actual, expected) else float("inf"), "relative_l2": 0.0}
    delta = actual.double() - expected.double()
    return {
        "max_abs": delta.abs().max().item(),
        "relative_l2": (delta.norm() / expected.double().norm().clamp_min(1e-12)).item(),
    }


def compare(args):
    actual_meta = json.loads((args.actual / "metadata.json").read_text())
    expected_meta = json.loads((args.expected / "metadata.json").read_text())
    for name, value in expected_meta["config"].items():
        if name not in ("model_path", "generator_ckpt") and actual_meta["config"][name] != value:
            raise ValueError(f"Runs use different settings: {name}")
    if actual_meta["calls"] != expected_meta["calls"]:
        raise ValueError(f"Call counts differ: {actual_meta['calls']} vs {expected_meta['calls']}")

    rows = [("latents", tensor_error(
        torch.load(args.actual / "latents.pt", weights_only=True),
        torch.load(args.expected / "latents.pt", weights_only=True),
    ))]
    for call in range(expected_meta["calls"]):
        actual = torch.load(args.actual / f"calls/{call:04d}.pt", weights_only=True)
        expected = torch.load(args.expected / f"calls/{call:04d}.pt", weights_only=True)
        if actual["current_start"] != expected["current_start"]:
            raise ValueError(f"Call {call}: current_start differs")
        for name in ("timestep", "input", "flow", "x0", "counters", "k", "v"):
            rows.append((f"call {call:04d} {name}", tensor_error(actual[name], expected[name])))

    print(f"{'tensor':<24}{'max_abs':>14}{'relative_l2':>14}")
    for name, error in rows:
        print(f"{name:<24}{error['max_abs']:>14.3e}{error['relative_l2']:>14.3e}")
    first = next((name for name, error in rows[1:] if error["max_abs"] > 0), None)
    print(f"First differing call tensor: {first or 'none'}")
    if args.output:
        args.output.write_text(json.dumps(dict(rows), indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare_parser = commands.add_parser("prepare", parents=[generation_parser("Shared inputs")], add_help=False)
    prepare_parser.add_argument("--output", type=Path, required=True)

    record_parser = commands.add_parser("record")
    record_parser.add_argument("--implementation", choices=("reference", "nano"), required=True)
    record_parser.add_argument("--inputs", type=Path, required=True)
    record_parser.add_argument("--output-dir", type=Path, required=True)
    record_parser.add_argument("--reference-dir", type=Path, default=ROOT.parent / "LongLive/LongLive2.0")
    record_parser.add_argument("--embeddings", type=Path, help="Reference run's embeddings.pt (required for nano)")

    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("--actual", type=Path, required=True)
    compare_parser.add_argument("--expected", type=Path, required=True)
    compare_parser.add_argument("--output", type=Path, help="Optional JSON copy of the table")

    args = parser.parse_args()
    for name in ("inputs", "output_dir", "embeddings", "actual", "expected", "output"):
        if getattr(args, name, None) is not None:
            setattr(args, name, getattr(args, name).resolve())

    if args.command == "prepare":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(make_inputs(args), args.output)
    elif args.command == "record":
        if args.implementation == "nano":
            if args.embeddings is None:
                parser.error("--embeddings is required for the nano run")
            record(args)
        else:
            args.reference_dir = args.reference_dir.resolve()
            model_path = torch.load(args.inputs, weights_only=True)["config"]["model_path"]
            with reference_workspace(args.reference_dir, model_path):
                record(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()
