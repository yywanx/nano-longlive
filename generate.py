from pathlib import Path

import torch

from nano_longlive.inference import CausalDiffusionInferencePipeline
from nano_longlive.utils import generation_parser, make_inputs, pipeline_options, save_video


def main():
    parser = generation_parser("BF16 single-shot video generation with nano-longlive")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--latent-only", action="store_true", help="Skip VAE decoding and MP4 export")
    args = parser.parse_args()

    inputs = make_inputs(args)
    config = inputs["config"]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.save(inputs, args.output_dir / "inputs.pt")

    pipeline = CausalDiffusionInferencePipeline.from_pretrained(**pipeline_options(config))

    latents = pipeline.inference(inputs["noise"].cuda(), [config["prompt"]])
    torch.save(latents.cpu(), args.output_dir / "latents.pt")

    if not args.latent_only:
        video = pipeline.decode(latents)
        save_video(video, args.output_dir / "video.mp4")

    print(f"Saved generation to {args.output_dir}")


if __name__ == "__main__":
    main()
