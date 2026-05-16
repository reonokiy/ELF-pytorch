# ELF: Embedded Language Flows — PyTorch

[![arXiv](https://img.shields.io/badge/arXiv-2605.10938-b31b1b.svg)](https://arxiv.org/abs/2605.10938)&nbsp;
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)&nbsp;
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-reonokiy-blue.svg)](https://huggingface.co/reonokiy)&nbsp;

Unofficial **PyTorch port** of [ELF: Embedded Language Flows](https://arxiv.org/abs/2605.10938). The original JAX/Flax implementation is at [lillian039/ELF](https://github.com/lillian039/ELF).

Weights are converted from the official checkpoints and verified numerically (max |JAX − PT| < 1e-5 on matched inputs).

---

ELF is a continuous diffusion language model based on flow matching. It operates in the T5 embedding space throughout denoising, discretizing to tokens only at the final step via a shared-weight factored decoder head.

<p align="center">
  <img src="assets/teaser.gif" alt="Conceptual illustration of ELF" width="100%"/>
</p>

## Checkpoints

All converted PyTorch checkpoints are on HuggingFace under [`reonokiy`](https://huggingface.co/reonokiy).

<table><tbody>
<td valign="bottom">OpenWebText (unconditional)</td>
<td valign="bottom" align="center">ELF-B (105M)</td>
<td valign="bottom" align="center">ELF-M (342M)</td>
<td valign="bottom" align="center">ELF-L (652M)</td>
<tr><td align="left">PyTorch checkpoint</td>
<td align="center"><a href="https://huggingface.co/reonokiy/ELF-B-owt">reonokiy/ELF-B-owt</a></td>
<td align="center"><a href="https://huggingface.co/reonokiy/ELF-M-owt">reonokiy/ELF-M-owt</a></td>
<td align="center"><a href="https://huggingface.co/reonokiy/ELF-L-owt">reonokiy/ELF-L-owt</a></td>
</tr>
<tr><td align="left">Gen. PPL ↓ (paper)</td>
<td align="center">24.1</td>
<td align="center">21.7</td>
<td align="center">23.3</td>
</tr>
<tr><td align="left">Entropy ↑ (paper)</td>
<td align="center">5.15</td>
<td align="center">5.18</td>
<td align="center">5.28</td>
</tr>
</tbody></table>

<table><tbody>
<td valign="bottom">Conditional generation (ELF-B)</td>
<td valign="bottom" align="center">WMT14 De-En</td>
<td valign="bottom" align="center" colspan="3">XSum</td>
<tr><td align="left">PyTorch checkpoint</td>
<td align="center"><a href="https://huggingface.co/reonokiy/ELF-B-de-en">reonokiy/ELF-B-de-en</a></td>
<td align="center" colspan="3"><a href="https://huggingface.co/reonokiy/ELF-B-xsum">reonokiy/ELF-B-xsum</a></td>
</tr>
<tr><td align="left">Metric</td>
<td align="center">BLEU ↑</td>
<td align="center">ROUGE-1 ↑</td>
<td align="center">ROUGE-2 ↑</td>
<td align="center">ROUGE-L ↑</td>
</tr>
<tr><td align="left">Score (paper)</td>
<td align="center">26.4</td>
<td align="center">36.0</td>
<td align="center">12.2</td>
<td align="center">27.8</td>
</tr>
</tbody></table>

## Installation

```bash
pixi install
```

Or with pip:

```bash
pip install torch einops transformers huggingface-hub
```

## Usage

```python
import torch
from huggingface_hub import hf_hub_download
from src.modules.model import ELF_models

# Load model
model = ELF_models["ELF-B"](
    text_encoder_dim=512,   # t5-small d_model
    max_length=512,
    num_time_tokens=4,
    num_self_cond_cfg_tokens=4,
    num_model_mode_tokens=4,
    bottleneck_dim=128,
    vocab_size=32128,
)

ckpt_path = hf_hub_download("reonokiy/ELF-B-owt", "pytorch_model.bin")
model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
model.eval()

# Forward pass: x is normalized T5 embeddings (B, S, 512)
# For self-conditioning, concatenate [z, x_pred] along last dim → (B, S, 1024)
B, S = 2, 64
x = torch.randn(B, S, 512)        # noisy latent
t = torch.tensor([0.5, 0.5])      # timestep in [0, 1]
sc_scale = torch.tensor([3.0, 3.0])

with torch.no_grad():
    x_pred, decoder_logits = model(x, t, self_cond_cfg_scale=sc_scale)
# x_pred: predicted clean latent (B, S, 512)
# decoder_logits: None unless decoder_step_active=True
```

## Model Architecture

```
Input (B, S, 512 or 1024)
  └─ self_cond_proj       [if 2×enc_dim input: 1024 → 512]
  └─ text_proj            [bottleneck: 512 → 128 → hidden]
  └─ prefix tokens        [time (4) + SC-CFG (4) tokens]
  └─ Transformer blocks   [RoPE, RMSNorm, SDPA, SwiGLU]
  └─ final_layer          [RMSNorm + Linear: hidden → 512]
Output (B, S, 512)

Optional decoder head (decoder_step_active=True):
  └─ GELU(x @ proj_kernel) @ unembed_kernel → (B, S, vocab_size)
```

| Model | Depth | Hidden | Heads | Params |
|-------|-------|--------|-------|--------|
| ELF-B | 12 | 768 | 12 | 105M |
| ELF-M | 24 | 1056 | 16 | 342M |
| ELF-L | 32 | 1280 | 16 | 652M |

All models use a frozen [t5-small](https://huggingface.co/google-t5/t5-small) encoder (512-dim) for text conditioning.

## Converting Checkpoints

To re-run the JAX → PyTorch conversion yourself:

```bash
pixi install -e convert
JAX_PLATFORMS=cpu pixi run -e convert python src/convert_all.py
```

This clones the upstream JAX repo, converts all 5 checkpoints, and uploads to HuggingFace.

## Citation

```bibtex
@article{elf2026,
  title={ELF: Embedded Language Flows},
  author={Hu, Keya and Qiu, Linlu and Lu, Yiyang and Zhao, Hanhong and Li, Tianhong and Kim, Yoon and Andreas, Jacob and He, Kaiming},
  journal={arXiv preprint arXiv:2605.10938},
  year={2026}
}
```
