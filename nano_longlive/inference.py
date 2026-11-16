import torch

from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


class CausalDiffusionInferencePipeline:
    """Chunkwise inference with separate caches for each guidance branch."""

    def __init__(
        self,
        generator,
        text_encoder,
        vae,
        num_frame_per_block=8,
        guidance_scale=1.0,
        negative_prompt="",
        sampling_steps=4,
        timestep_shift=5.0,
    ):
        self.generator = generator
        self.text_encoder = text_encoder
        self.vae = vae
        self.num_frame_per_block = num_frame_per_block
        self.guidance_scale = guidance_scale
        self.negative_prompt = negative_prompt
        self.sampling_steps = sampling_steps
        self.shift = timestep_shift
        self.num_train_timesteps = 1000

    @classmethod
    def from_pretrained(
        cls,
        model_path,
        generator_ckpt,
        device="cuda",
        dtype=torch.bfloat16,
        num_frame_per_block=8,
        local_attn_size=32,
        sink_size=8,
        use_relative_rope=False,
        sampling_steps=4,
        timestep_shift=5.0,
        guidance_scale=1.0,
        negative_prompt="",
    ):
        from nano_longlive.wan_adapter import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper

        generator = WanDiffusionWrapper.from_pretrained(
            model_path,
            generator_ckpt,
            num_frame_per_block=num_frame_per_block,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            timestep_shift=timestep_shift,
            use_relative_rope=use_relative_rope,
        ).to(device=device, dtype=dtype)
        text_encoder = WanTextEncoder.from_pretrained(model_path).to(device=device, dtype=dtype)
        vae = WanVAEWrapper.from_pretrained(model_path).to(device=device, dtype=dtype)
        return cls(
            generator,
            text_encoder,
            vae,
            num_frame_per_block=num_frame_per_block,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            sampling_steps=sampling_steps,
            timestep_shift=timestep_shift,
        )

    def inference(self, noise, text_prompts):
        """Generate latents [1, F, C, H, W] from noise of the same shape and one text prompt."""

        batch_size, num_frames, _, height, width = noise.shape
        patch_frames, patch_height, patch_width = self.generator.model.patch_size
        if batch_size != 1 or len(text_prompts) != 1:
            raise ValueError("Expected one sample and one text prompt")
        if num_frames % self.num_frame_per_block:
            raise ValueError("Noise must contain full chunks")
        if patch_frames != 1 or height % patch_height or width % patch_width:
            raise ValueError("Noise must divide into spatial patches, with temporal patch size 1")

        with torch.inference_mode():
            conditional_dict = self.text_encoder(text_prompts=text_prompts)
            unconditional_dict = None
            if self.guidance_scale != 1.0:
                unconditional_dict = self.text_encoder(text_prompts=[self.negative_prompt])

            # Frame offsets below count latent frames; the model takes token offsets.
            self.frame_seq_length = (height // patch_height) * (width // patch_width)
            self._initialize_cache(noise)
            output = torch.empty_like(noise)

            for current_start_frame in range(0, num_frames, self.num_frame_per_block):
                current_end_frame = current_start_frame + self.num_frame_per_block
                current_start = current_start_frame * self.frame_seq_length
                latents = noise[:, current_start_frame:current_end_frame]

                # Each chunk starts a new UniPC history.
                sample_scheduler = self._initialize_sample_scheduler(noise)
                timestep = torch.ones(
                    (batch_size, self.num_frame_per_block),
                    device=noise.device,
                    dtype=torch.float32,
                )

                for t in sample_scheduler.timesteps:
                    flow_pred = self._predict_flow(
                        latents,
                        timestep * t,
                        conditional_dict,
                        unconditional_dict,
                        current_start,
                    )
                    latents = sample_scheduler.step(flow_pred, t, latents, return_dict=False)[0]

                output[:, current_start_frame:current_end_frame] = latents

                # Replace noisy K/V with clean context before the next chunk.
                self._predict_flow(
                    latents,
                    timestep * 0,
                    conditional_dict,
                    unconditional_dict,
                    current_start,
                )

            return output

    def decode(self, latents):
        """Decode latents to video [1, T, 3, H, W] in [0, 1]."""

        with torch.inference_mode():
            video = self.vae.decode_to_pixel(latents)
        return (video * 0.5 + 0.5).clamp(0, 1)

    def _predict_flow(self, latents, timestep, conditional_dict, unconditional_dict, current_start):
        """Predict flow and update K/V, including when the returned flow is unused."""

        flow_pred, _ = self.generator(
            noisy_image_or_video=latents,
            conditional_dict=conditional_dict,
            timestep=timestep,
            kv_cache=self.kv_cache_pos,
            current_start=current_start,
        )
        if unconditional_dict is not None:
            flow_pred_uncond, _ = self.generator(
                noisy_image_or_video=latents,
                conditional_dict=unconditional_dict,
                timestep=timestep,
                kv_cache=self.kv_cache_neg,
                current_start=current_start,
            )
            flow_pred = flow_pred_uncond + self.guidance_scale * (flow_pred - flow_pred_uncond)
        return flow_pred

    def _initialize_cache(self, noise):
        model = self.generator.model
        if model.num_frame_per_block != self.num_frame_per_block:
            raise ValueError("Pipeline and backbone chunk sizes must match")
        if model.sink_size < 0 or model.local_attn_size < model.sink_size + self.num_frame_per_block:
            raise ValueError("Attention window must fit a nonnegative sink and a full chunk")

        # Release previous requests before allocating their replacement.
        self.kv_cache_pos = self.kv_cache_neg = None
        self.kv_cache_pos = self._initialize_kv_cache(noise)
        if self.guidance_scale != 1.0:
            self.kv_cache_neg = self._initialize_kv_cache(noise)

    def _initialize_kv_cache(self, noise):
        model = self.generator.model
        kv_cache_size = model.local_attn_size * self.frame_seq_length
        shape = (noise.shape[0], kv_cache_size, model.num_heads, model.dim // model.num_heads)
        return [{
            "k": noise.new_zeros(shape),
            "v": noise.new_zeros(shape),
            "global_end_index": torch.zeros(1, dtype=torch.long, device=noise.device),
            "local_end_index": torch.zeros(1, dtype=torch.long, device=noise.device),
        } for _ in model.blocks]

    def _initialize_sample_scheduler(self, noise):
        sample_scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False,
        )
        sample_scheduler.set_timesteps(self.sampling_steps, device=noise.device, shift=self.shift)
        return sample_scheduler
