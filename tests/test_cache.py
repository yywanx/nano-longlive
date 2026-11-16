"""KV cache behavior of one attention layer, checked against hand-computed expectations."""

import unittest
from unittest.mock import patch

import torch

from reference import cpu_attention, make_cache, tiny_model
from wan.modules import causal_model


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(causal_model, "attention", cpu_attention))
        self.enterContext(torch.inference_mode())

    def test_cache_keeps_sink_and_recent_tokens_without_early_mutation(self):
        backbone = tiny_model()
        attn = backbone.blocks[0].self_attn
        attn.v.weight.copy_(torch.eye(24))
        attn.v.bias.zero_()
        cache = make_cache(backbone, torch.float32)[0]
        history = []
        for chunk in range(7):
            for write in range(3):
                value = chunk * 10 + write + 1
                x = torch.full((1, 4, 24), float(value))
                before = {key: tensor.clone() for key, tensor in cache.items()}
                _, update = attn(x, [(2, 1, 2)], backbone.freqs, cache, current_start=chunk * 4)
                for key in cache:
                    torch.testing.assert_close(cache[key], before[key], rtol=0, atol=0)
                backbone._apply_cache_updates([cache], [(0, update)])
                if write == 0:
                    history.extend([value] * 4)
                else:
                    history[-4:] = [value] * 4
                visible = history if len(history) <= 12 else history[:4] + history[-8:]
                end = len(visible)
                expected = torch.tensor(visible).view(1, end, 1, 1).expand(1, end, 2, 12).float()
                torch.testing.assert_close(cache["v"][:, :end], expected, rtol=0, atol=0)
                self.assertEqual(cache["global_end_index"].item(), (chunk + 1) * 4)
                self.assertEqual(cache["local_end_index"].item(), end)

    def test_current_chunk_attends_all_its_tokens(self):
        backbone = tiny_model()
        attn = backbone.blocks[0].self_attn
        attn.q.weight.zero_()
        attn.q.bias.zero_()
        attn.v.weight.copy_(torch.eye(24))
        attn.v.bias.zero_()
        attn.o.weight.copy_(torch.eye(24))
        attn.o.bias.zero_()
        x = torch.arange(4).float().view(1, 4, 1).expand(1, 4, 24)
        output, _ = attn(x, [(2, 1, 2)], backbone.freqs, make_cache(backbone, torch.float32)[0])
        torch.testing.assert_close(output, torch.full_like(x, 1.5), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
