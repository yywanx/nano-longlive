# Reused Wan / LongLive components

Copied from [NVlabs/LongLive](https://github.com/NVlabs/LongLive) (`LongLive2.0/`), commit `2aa857f595d35c81526938d603ae7b1082922ac6`.

| File | Upstream path | Changes |
|---|---|---|
| `modules/t5.py`, `modules/tokenizers.py`, `modules/vae2_2.py` | `wan_5b/modules/` | none |
| `utils/fm_solvers_unipc.py` | `wan_5b/utils/` | none |
| `utils/scheduler.py`, `utils/position_embedding_utils.py` | `utils/` | none |
| `modules/model.py` | `wan_5b/modules/` | kept norms, RoPE, time embedding and cross-attention; removed the non-causal `WanModel` |
| `modules/attention.py` | `wan_5b/modules/` | kept FlashAttention 2 and the PyTorch SDPA fallback; removed FA3/FA4/TransformerEngine |
| `modules/causal_model.py` | `wan_5b/modules/` | inference-only: dense bounded KV cache, eager RoPE/adaLN; removed training, multi-shot, quantized cache and compilation paths |
