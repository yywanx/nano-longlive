"""Time nano-longlive inference stage by stage on one GPU, and optionally profile one steady-state chunk.

    uv run --no-sync python -m scripts.profile_inference \
      --model-path weights/Wan2.2-TI2V-5B --generator-ckpt weights/LongLive-2.0-5B/model_bf16.pt \
      --prompt "A dog running through a meadow" --num-frames 40 --output-dir runs/profile-40 --trace
"""

import json
import statistics
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from nano_longlive.inference import CausalDiffusionInferencePipeline
from nano_longlive.utils import generation_parser, make_inputs, pipeline_options
from wan.modules.attention import FLASH_ATTN_2_AVAILABLE


class CallTimer:
    """Measure the GPU time of every generator call with CUDA events."""

    def __init__(self, generator):
        self.events = []
        generator.register_forward_pre_hook(self.start)
        generator.register_forward_hook(self.end)

    def start(self, module, args):
        self.events.append((torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)))
        self.events[-1][0].record()

    def end(self, module, args, output):
        self.events[-1][1].record()

    def take_ms(self):
        """Return and clear the call times; call after torch.cuda.synchronize()."""

        times = [start.elapsed_time(end) for start, end in self.events]
        self.events = []
        return times


def timed(function):
    """Run function and return (result, wall-clock seconds), waiting for the GPU before and after."""

    torch.cuda.synchronize()
    start = time.perf_counter()
    result = function()
    torch.cuda.synchronize()
    return result, time.perf_counter() - start


def measure(pipeline, timer, noise, prompt):
    """Time one generation: text encoding, the diffusion loop, and VAE decoding, with peak memory."""

    with torch.inference_mode():
        _, text_s = timed(lambda: pipeline.text_encoder(text_prompts=[prompt]))

    # inference() encodes the prompt again; subtract that to get the diffusion loop alone.
    torch.cuda.reset_peak_memory_stats()
    latents, inference_s = timed(lambda: pipeline.inference(noise, [prompt]))
    inference_peak = torch.cuda.max_memory_allocated()
    call_ms = timer.take_ms()

    torch.cuda.reset_peak_memory_stats()
    _, decode_s = timed(lambda: pipeline.decode(latents))
    decode_peak = torch.cuda.max_memory_allocated()

    return {
        "text_encoder_s": text_s,
        "diffusion_s": inference_s - text_s,
        "decode_s": decode_s,
        "call_ms": call_ms,
        "inference_peak_gib": inference_peak / 2**30,
        "decode_peak_gib": decode_peak / 2**30,
    }


def summarize(runs, calls_per_chunk):
    """Take the median of each number over the timed runs; group generator calls into chunks."""

    summary = {name: statistics.median(run[name] for run in runs) for name in runs[0] if name != "call_ms"}
    call_ms = [statistics.median(times) for times in zip(*(run["call_ms"] for run in runs))]
    summary["call_ms"] = call_ms
    summary["chunk_ms"] = [
        sum(call_ms[start:start + calls_per_chunk]) for start in range(0, len(call_ms), calls_per_chunk)
    ]
    return summary


def profile_last_chunk(pipeline, noise, prompt, calls_per_chunk, output_dir):
    """Run one generation and record a torch.profiler trace of its last chunk only."""

    num_calls = noise.shape[1] // pipeline.num_frame_per_block * calls_per_chunk
    first_profiled_call = num_calls - calls_per_chunk
    profiler = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
    )
    calls_done = 0

    def start(module, args):
        if calls_done == first_profiled_call:
            profiler.start()

    def end(module, args, output):
        nonlocal calls_done
        calls_done += 1
        if calls_done == num_calls:
            torch.cuda.synchronize()
            profiler.stop()

    handles = [
        pipeline.generator.register_forward_pre_hook(start),
        pipeline.generator.register_forward_hook(end),
    ]
    pipeline.inference(noise, [prompt])
    for handle in handles:
        handle.remove()

    profiler.export_chrome_trace(str(output_dir / "trace.json"))
    table = profiler.key_averages().table(sort_by="cuda_time_total", row_limit=40)
    (output_dir / "profile.txt").write_text(table)
    print(table)


def main():
    parser = generation_parser("Time and profile nano-longlive inference")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3, help="Timed runs after one warm-up run")
    parser.add_argument("--trace", action="store_true", help="Also profile the last chunk with torch.profiler")
    args = parser.parse_args()

    inputs = make_inputs(args)
    config = inputs["config"]
    args.output_dir.mkdir(parents=True, exist_ok=False)

    pipeline, load_s = timed(lambda: CausalDiffusionInferencePipeline.from_pretrained(**pipeline_options(config)))
    weights_gib = torch.cuda.memory_allocated() / 2**30
    timer = CallTimer(pipeline.generator)
    noise = inputs["noise"].cuda()
    prompt = config["prompt"]

    # One denoising call per step plus the clean recache, for each guidance branch.
    calls_per_chunk = (args.sampling_steps + 1) * (2 if args.guidance_scale != 1.0 else 1)

    measure(pipeline, timer, noise, prompt)
    runs = [measure(pipeline, timer, noise, prompt) for _ in range(args.repeats)]
    summary = summarize(runs, calls_per_chunk)

    result = {
        "config": config,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "flash_attention": FLASH_ATTN_2_AVAILABLE,
        "load_s": load_s,
        "weights_gib": weights_gib,
        "calls_per_chunk": calls_per_chunk,
        "summary": summary,
        "runs": runs,
    }
    (args.output_dir / "timing.json").write_text(json.dumps(result, indent=2))

    print(f"GPU {result['gpu']}, torch {result['torch']}, FlashAttention {result['flash_attention']}")
    print(f"load {load_s:.1f} s, weights {weights_gib:.1f} GiB")
    print(f"text encoder {summary['text_encoder_s']:.3f} s")
    print(f"diffusion    {summary['diffusion_s']:.3f} s")
    print(f"decode       {summary['decode_s']:.3f} s")
    print("chunks (ms)  " + " ".join(f"{ms:.0f}" for ms in summary["chunk_ms"]) + "  (generator calls only)")
    print(f"peak memory  inference {summary['inference_peak_gib']:.1f} GiB (weights included)")
    print(f"             decode {summary['decode_peak_gib']:.1f} GiB")

    if args.trace:
        profile_last_chunk(pipeline, noise, prompt, calls_per_chunk, args.output_dir)

    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
