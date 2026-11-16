# nano-longlive

A minimal reimplementation of [LongLive 2.0](https://github.com/NVlabs/LongLive) chunkwise autoregressive video generation.

The official code covers many features (multi-shot, NVFP4, streaming decode, compiled kernels) across thousands of lines. nano-longlive keeps only the core algorithm: single-prompt BF16 text-to-video, denoised chunk by chunk over a bounded KV cache with an attention sink. The goal is to make the algorithm easy to learn.

## Setup

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and the environment:

   ```sh
   uv sync
   ```

2. Install [FlashAttention 2](https://github.com/Dao-AILab/flash-attention) from a prebuilt wheel. On the [releases page](https://github.com/Dao-AILab/flash-attention/releases), copy the link of the wheel whose name contains `cu12`, `torch2.8`, `cxx11abiTRUE`, `cp311` and `linux_x86_64`, and install it, for example:

   ```sh
   uv pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
   uv run --no-sync python -c "import flash_attn"    # check it imports
   ```

3. Install ffmpeg:

   ```sh
   apt-get install -y ffmpeg
   ```

4. Download the weights:

   ```sh
   uv run --no-sync hf download Wan-AI/Wan2.2-TI2V-5B --local-dir weights/Wan2.2-TI2V-5B
   uv run --no-sync hf download Efficient-Large-Model/LongLive-2.0-5B model_bf16.pt --local-dir weights/LongLive-2.0-5B
   ```

FlashAttention is not in `uv.lock`, so run everything with `uv run --no-sync`; a plain `uv run` would uninstall it.

## Generate a video

```sh
uv run --no-sync python generate.py \
  --model-path weights/Wan2.2-TI2V-5B \
  --generator-ckpt weights/LongLive-2.0-5B/model_bf16.pt \
  --prompt "A dog running through a meadow" \
  --num-frames 8 --output-dir runs/demo
```

The video is written to `runs/demo/video.mp4`. `--num-frames` counts latent frames in multiples of 8; 8 gives about 1.2 s of video at 24 fps, 40 about 6.5 s. `--help` lists the other options.

## Compare with the official pipeline

Check out [NVlabs/LongLive](https://github.com/NVlabs/LongLive) at `../LongLive/LongLive2.0`, then:

```sh
uv run --no-sync python -m scripts.compare_reference prepare \
  --model-path weights/Wan2.2-TI2V-5B --generator-ckpt weights/LongLive-2.0-5B/model_bf16.pt \
  --prompt "A dog running through a meadow" --num-frames 8 --output runs/inputs-8.pt
uv run --no-sync python -m scripts.compare_reference record \
  --implementation reference --inputs runs/inputs-8.pt --output-dir runs/reference-8
uv run --no-sync python -m scripts.compare_reference record \
  --implementation nano --inputs runs/inputs-8.pt --output-dir runs/nano-8 \
  --embeddings runs/reference-8/embeddings.pt
uv run --no-sync python -m scripts.compare_reference compare \
  --actual runs/nano-8 --expected runs/reference-8 --output runs/comparison-8.json
```

## Tests

```sh
uv run --no-sync python -m unittest discover -s tests -v
```
