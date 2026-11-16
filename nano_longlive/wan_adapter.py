from pathlib import Path

import torch
from torch import nn

from wan.utils.scheduler import FlowMatchScheduler


MEAN = [
    -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557,
    -0.1382, 0.0542, 0.2813, 0.0891, 0.157, -0.0098, 0.0375, -0.1825,
    -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108, -0.2158, 0.2502,
    -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.123,
    -0.0313, -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.052, 0.3748,
    0.0152, 0.1957, 0.1433, -0.2944, 0.3573, -0.0548, -0.1681, -0.0667,
]

STD = [
    0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.499, 0.4818, 0.5013,
    0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
    0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
    0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
    0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
    0.3971, 1.06, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
]


def load_generator_checkpoint(generator, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "generator" in checkpoint:
        checkpoint = checkpoint["generator"]
    elif "model" in checkpoint:
        checkpoint = checkpoint["model"]
    elif "generator_ema" in checkpoint:
        raise ValueError("Expected non-EMA generator weights")
    generator.load_state_dict(checkpoint, strict=True)


class WanDiffusionWrapper(nn.Module):
    def __init__(self, model, timestep_shift=5.0):
        super().__init__()
        self.model = model
        # This schedule maps timesteps to sigma for x0 conversion; UniPC performs sampling.
        self.scheduler = FlowMatchScheduler(shift=timestep_shift, sigma_min=0.0, extra_one_step=True)
        self.scheduler.set_timesteps(1000, training=True)

    @classmethod
    def from_pretrained(
        cls,
        model_path,
        generator_ckpt,
        num_frame_per_block=8,
        local_attn_size=32,
        sink_size=8,
        timestep_shift=5.0,
        use_relative_rope=False,
    ):
        model_path = Path(model_path)
        from wan.modules.causal_model import CausalWanModel

        model = CausalWanModel.from_pretrained(
            str(model_path),
            local_files_only=True,
            num_frame_per_block=num_frame_per_block,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
        )
        model.use_relative_rope = use_relative_rope
        generator = cls(model, timestep_shift=timestep_shift)
        load_generator_checkpoint(generator, generator_ckpt)
        return generator.eval().requires_grad_(False)

    def forward(
        self,
        noisy_image_or_video,
        conditional_dict,
        timestep,
        kv_cache=None,
        current_start=0,
        cache_start=None,
    ):
        _, num_frames, _, height, width = noisy_image_or_video.shape
        patch_size = self.model.patch_size
        frame_seq_length = height * width // (patch_size[1] * patch_size[2])

        # Pipeline tensors are [B, F, C, H, W]; Wan consumes [B, C, F, H, W].
        x = noisy_image_or_video.permute(0, 2, 1, 3, 4)
        flow_pred = self.model(
            x,
            t=timestep,
            context=conditional_dict["prompt_embeds"],
            seq_len=num_frames * frame_seq_length,
            kv_cache=kv_cache,
            current_start=current_start,
            cache_start=current_start if cache_start is None else cache_start,
        )
        flow_pred = flow_pred.permute(0, 2, 1, 3, 4)

        # Fold batch and frames together for one sigma per latent frame.
        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred.flatten(0, 1),
            noisy_image_or_video.flatten(0, 1),
            timestep.flatten(),
        )
        pred_x0 = pred_x0.unflatten(0, flow_pred.shape[:2])
        return flow_pred, pred_x0

    def _convert_flow_pred_to_x0(self, flow_pred, xt, timestep):
        timesteps = self.scheduler.timesteps.to(device=flow_pred.device, dtype=torch.float64)
        sigmas = self.scheduler.sigmas.to(device=flow_pred.device, dtype=torch.float64)
        timestep_id = (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs().argmin(dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        return (xt.double() - sigma_t * flow_pred.double()).to(flow_pred.dtype)


class WanTextEncoder(nn.Module):
    def __init__(self, text_encoder, tokenizer):
        super().__init__()
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, model_path):
        from wan.modules.t5 import umt5_xxl
        from wan.modules.tokenizers import HuggingfaceTokenizer

        model_path = Path(model_path)
        text_encoder = umt5_xxl(encoder_only=True, return_tokenizer=False, dtype=torch.float32, device="cpu")
        text_encoder.load_state_dict(
            torch.load(
                model_path / "models_t5_umt5-xxl-enc-bf16.pth",
                map_location="cpu",
                weights_only=True,
            ),
            strict=True,
        )
        tokenizer = HuggingfaceTokenizer(
            str(model_path / "google/umt5-xxl"),
            seq_len=512,
            clean="whitespace",
            local_files_only=True,
        )
        return cls(text_encoder, tokenizer).eval().requires_grad_(False)

    def forward(self, text_prompts):
        device = next(self.text_encoder.parameters()).device
        ids, mask = self.tokenizer(text_prompts, return_mask=True, add_special_tokens=True)
        ids, mask = ids.to(device), mask.to(device)
        context = self.text_encoder(ids, mask)
        for u, length in zip(context, mask.gt(0).sum(dim=1)):
            u[length:] = 0
        return {"prompt_embeds": context}


class WanVAEWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.mean = torch.tensor(MEAN, dtype=torch.float32)
        self.std = torch.tensor(STD, dtype=torch.float32)

    @classmethod
    def from_pretrained(cls, model_path):
        from wan.modules.vae2_2 import _video_vae

        model = _video_vae(pretrained_path=str(Path(model_path) / "Wan2.2_VAE.pth"))
        return cls(model).eval().requires_grad_(False)

    def decode_to_pixel(self, latent):
        scale = [self.mean.to(latent), 1.0 / self.std.to(latent)]
        zs = latent.permute(0, 2, 1, 3, 4)
        output = []
        for u in zs:
            decoded = self.model.decode(u.unsqueeze(0), scale)
            decoded = decoded.float().clamp_(-1, 1).squeeze(0)
            output.append(decoded)
        return torch.stack(output).permute(0, 2, 1, 3, 4)
