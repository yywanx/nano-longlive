import argparse
from pathlib import Path
import subprocess

import torch


def generation_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--generator-ckpt", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-frames", type=int, default=8, help="Number of latent frames, in full chunks")
    parser.add_argument("--height", type=int, default=704, help="Decoded pixel height, divisible by 32")
    parser.add_argument("--width", type=int, default=1280, help="Decoded pixel width, divisible by 32")
    parser.add_argument("--num-frame-per-block", type=int, default=8)
    parser.add_argument("--local-attn-size", type=int, default=32)
    parser.add_argument("--sink-size", type=int, default=8)
    parser.add_argument("--sampling-steps", type=int, default=4)
    parser.add_argument("--timestep-shift", type=float, default=5.0)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--use-relative-rope", action="store_true")
    return parser


def make_inputs(args):
    """Return the run settings and BF16 noise [1, F, 48, H/16, W/16]."""

    if args.height % 32 or args.width % 32:
        raise ValueError("Pixel height and width must be multiples of 32")

    config = {name: getattr(args, name) for name in (
        "prompt", "negative_prompt", "seed", "num_frames", "height", "width", "num_frame_per_block",
        "local_attn_size", "sink_size", "sampling_steps", "timestep_shift", "guidance_scale", "use_relative_rope",
    )}
    config.update(model_path=str(args.model_path.resolve()), generator_ckpt=str(args.generator_ckpt.resolve()))

    # Sample on CPU so the same seed gives the same noise on any machine.
    noise = torch.randn(
        (1, args.num_frames, 48, args.height // 16, args.width // 16),
        generator=torch.Generator(device="cpu").manual_seed(args.seed),
        dtype=torch.float32,
    ).to(torch.bfloat16)
    return {"config": config, "noise": noise}


def pipeline_options(config):
    return {name: config[name] for name in (
        "model_path", "generator_ckpt", "num_frame_per_block", "local_attn_size", "sink_size",
        "sampling_steps", "timestep_shift", "guidance_scale", "negative_prompt", "use_relative_rope",
    )}


def save_video(video, path, fps=24):
    """Write video [1, T, 3, H, W] in [0, 1] to an MP4 with ffmpeg."""

    _, _, _, height, width = video.shape
    frames = video[0].float().clamp(0, 1).mul(255).round().to(torch.uint8).permute(0, 2, 3, 1)
    command = [
        "ffmpeg", "-v", "error", "-n", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0", "-an",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
    ]
    subprocess.run(command, input=frames.contiguous().cpu().numpy().tobytes(), check=True)
