"""
Convert ELF-B-owt JAX/Flax checkpoint → PyTorch state dict, then verify
numerical equivalence between the two models on random inputs.

Usage:
    JAX_PLATFORMS=cpu python src/convert_and_verify.py
"""
import os
import sys
import copy
import numpy as np

# ── path setup ──────────────────────────────────────────────────────────────
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC  = os.path.join(REPO, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# ── JAX imports ─────────────────────────────────────────────────────────────
import jax
import jax.numpy as jnp
import optax
from transformers import AutoTokenizer

from modules.model import ELF_models as JAX_ELF_models
from modules.t5_encoder import get_encoder
from utils.checkpoint_utils import load_encoder_checkpoint, load_checkpoint
from utils.train_utils import TrainState

# ── PyTorch imports ──────────────────────────────────────────────────────────
import torch
from modules_pt.model import ELF_models as PT_ELF_models


# ============================================================================
# Weight conversion
# ============================================================================

def _v(arr):
    """JAX array → float32 torch tensor."""
    return torch.from_numpy(np.array(arr, dtype=np.float32))

def _t(arr):
    """Flax Dense kernel (in, out) → PyTorch Linear weight (out, in)."""
    return _v(arr).T


def convert_elf_params(p, depth):
    """Convert flat Flax param dict to PyTorch state dict."""
    sd = {}

    # ── learnable tokens / raw parameters ───────────────────────────────────
    sd['t_emb_tokens']           = _v(p['t_emb_tokens'])
    sd['self_cond_cfg_tokens']   = _v(p['self_cond_cfg_tokens'])
    sd['mode_tokens']            = _v(p['mode_tokens'])
    sd['proj_kernel']            = _v(p['proj_kernel'])
    sd['proj_bias']              = _v(p['proj_bias'])
    sd['unembed_kernel']         = _v(p['unembed_kernel'])
    sd['unembed_bias']           = _v(p['unembed_bias'])

    # ── self_cond_proj ───────────────────────────────────────────────────────
    sd['self_cond_proj.weight']  = _t(p['self_cond_proj']['kernel'])
    sd['self_cond_proj.bias']    = _v(p['self_cond_proj']['bias'])

    # ── text_proj (BottleneckTextProj) ───────────────────────────────────────
    sd['text_proj.proj1.weight'] = _t(p['text_proj']['proj1']['kernel'])
    sd['text_proj.proj2.weight'] = _t(p['text_proj']['proj2']['kernel'])
    sd['text_proj.proj2.bias']   = _v(p['text_proj']['proj2']['bias'])

    # ── t_embedder ───────────────────────────────────────────────────────────
    for name in ('t_embedder', 'self_cond_cfg_embedder'):
        sd[f'{name}.mlp_0.weight'] = _t(p[name]['mlp_0']['kernel'])
        sd[f'{name}.mlp_0.bias']   = _v(p[name]['mlp_0']['bias'])
        sd[f'{name}.mlp_2.weight'] = _t(p[name]['mlp_2']['kernel'])
        sd[f'{name}.mlp_2.bias']   = _v(p[name]['mlp_2']['bias'])

    # ── final_layer ──────────────────────────────────────────────────────────
    sd['final_layer.norm_final.weight'] = _v(p['final_layer']['norm_final']['weight'])
    sd['final_layer.linear.weight']     = _t(p['final_layer']['linear']['kernel'])
    sd['final_layer.linear.bias']       = _v(p['final_layer']['linear']['bias'])

    # ── transformer blocks ───────────────────────────────────────────────────
    for i in range(depth):
        bp  = p[f'blocks_{i}']
        pfx = f'blocks.{i}'

        sd[f'{pfx}.norm1.weight']           = _v(bp['norm1']['weight'])
        sd[f'{pfx}.norm2.weight']           = _v(bp['norm2']['weight'])

        sd[f'{pfx}.attn.qkv.weight']        = _t(bp['attn']['qkv']['kernel'])
        sd[f'{pfx}.attn.qkv.bias']          = _v(bp['attn']['qkv']['bias'])
        sd[f'{pfx}.attn.proj.weight']       = _t(bp['attn']['proj']['kernel'])
        sd[f'{pfx}.attn.proj.bias']         = _v(bp['attn']['proj']['bias'])
        sd[f'{pfx}.attn.q_norm.weight']     = _v(bp['attn']['q_norm']['weight'])
        sd[f'{pfx}.attn.k_norm.weight']     = _v(bp['attn']['k_norm']['weight'])

        sd[f'{pfx}.mlp.w12.weight']         = _t(bp['mlp']['w12']['kernel'])
        sd[f'{pfx}.mlp.w12.bias']           = _v(bp['mlp']['w12']['bias'])
        sd[f'{pfx}.mlp.w3.weight']          = _t(bp['mlp']['w3']['kernel'])
        sd[f'{pfx}.mlp.w3.bias']            = _v(bp['mlp']['w3']['bias'])

    return sd


# ============================================================================
# Load JAX checkpoint
# ============================================================================

def load_jax_model(model_name='ELF-B', checkpoint='embedded-language-flows/ELF-B-owt',
                   encoder_model='t5-small', max_length=64):
    tokenizer = AutoTokenizer.from_pretrained(encoder_model)
    enc_cfg, _, _ = get_encoder(encoder_model, jnp.float32)
    enc_params = load_encoder_checkpoint(
        'embedded-language-flows/t5_small_encoder_jax/t5_small_encoder_jax.pkl'
    )

    rng = jax.random.PRNGKey(0)
    rng, init_rng, drop_rng = jax.random.split(rng, 3)

    text_enc_dim = enc_cfg.d_model
    dummy_x = jnp.ones((1, max_length, 2 * text_enc_dim))
    dummy_t = jnp.ones((1,))
    dummy_sc = jnp.ones((1,))

    jax_model = JAX_ELF_models[model_name](
        text_encoder_dim=text_enc_dim,
        max_length=max_length,
        num_time_tokens=4,
        num_self_cond_cfg_tokens=4,
        num_model_mode_tokens=4,
        bottleneck_dim=128,
        vocab_size=tokenizer.vocab_size,
    )

    elf_params = jax_model.init(
        init_rng, x=dummy_x, t=dummy_t, self_cond_cfg_scale=dummy_sc, deterministic=True
    )

    state = TrainState.create(
        apply_fn=jax_model.apply,
        params=elf_params['params'],
        tx=optax.adamw(1e-4),
        dropout_rng=drop_rng,
        ema_params1=copy.deepcopy(elf_params['params']),
    )
    state, step = load_checkpoint(checkpoint, state)
    print(f'JAX checkpoint loaded: step={step}')
    return jax_model, state.ema_params1, tokenizer, enc_cfg


# ============================================================================
# Build PyTorch model + load weights
# ============================================================================

def build_pt_model(jax_params, model_name, vocab_size, enc_dim, max_length):
    depth_map = {'ELF-B': 12, 'ELF-M': 24, 'ELF-L': 32}
    depth = depth_map[model_name]

    pt_model = PT_ELF_models[model_name](
        text_encoder_dim=enc_dim,
        max_length=max_length,
        num_time_tokens=4,
        num_self_cond_cfg_tokens=4,
        num_model_mode_tokens=4,
        bottleneck_dim=128,
        vocab_size=vocab_size,
    )

    sd = convert_elf_params(jax_params, depth)
    missing, unexpected = pt_model.load_state_dict(sd, strict=True)
    assert not missing,    f'Missing keys: {missing}'
    assert not unexpected, f'Unexpected keys: {unexpected}'
    pt_model.eval()
    print(f'PyTorch model built and weights loaded ({sum(p.numel() for p in pt_model.parameters()):,} params)')
    return pt_model


# ============================================================================
# Numerical verification
# ============================================================================

def verify(jax_model, jax_params, pt_model, enc_dim, max_length, batch=2):
    rng = jax.random.PRNGKey(42)
    rng, xr, tr, scr = jax.random.split(rng, 4)

    # shared random inputs
    x_np  = np.random.RandomState(0).randn(batch, max_length, 2 * enc_dim).astype(np.float32)
    t_np  = np.array([0.3, 0.7], dtype=np.float32)[:batch]
    sc_np = np.array([1.5, 2.0], dtype=np.float32)[:batch]
    mask_np = np.ones((batch, max_length), dtype=np.float32)

    # ── JAX forward ─────────────────────────────────────────────────────────
    x_jax  = jnp.array(x_np)
    t_jax  = jnp.array(t_np)
    sc_jax = jnp.array(sc_np)
    mask_jax = jnp.array(mask_np)

    out_jax, _ = jax_model.apply(
        {'params': jax_params}, x_jax, t_jax,
        attention_mask=mask_jax,
        self_cond_cfg_scale=sc_jax,
        deterministic=True,
    )
    out_jax_np = np.array(out_jax)

    # ── PyTorch forward ──────────────────────────────────────────────────────
    x_pt   = torch.from_numpy(x_np)
    t_pt   = torch.from_numpy(t_np)
    sc_pt  = torch.from_numpy(sc_np)
    mask_pt = torch.from_numpy(mask_np)

    with torch.no_grad():
        out_pt, _ = pt_model(x_pt, t_pt, attention_mask=mask_pt, self_cond_cfg_scale=sc_pt)
    out_pt_np = out_pt.numpy()

    # ── compare ──────────────────────────────────────────────────────────────
    abs_diff = np.abs(out_jax_np - out_pt_np)
    max_err  = abs_diff.max()
    mean_err = abs_diff.mean()
    rel_err  = (abs_diff / (np.abs(out_jax_np) + 1e-8)).max()

    print(f'\n=== Numerical Verification ===')
    print(f'Output shape: JAX={out_jax_np.shape}  PT={out_pt_np.shape}')
    print(f'Max  |JAX - PT|  = {max_err:.6e}')
    print(f'Mean |JAX - PT|  = {mean_err:.6e}')
    print(f'Max  rel error   = {rel_err:.6e}')
    ok = max_err < 1e-3
    print(f'PASS ✓' if ok else f'FAIL ✗  (max error {max_err:.3e} exceeds 1e-3)')
    return ok


# ============================================================================
# Main
# ============================================================================

def main():
    MODEL_NAME = 'ELF-B'
    CHECKPOINT = 'embedded-language-flows/ELF-B-owt'
    MAX_LENGTH  = 64   # smaller for faster CPU verification

    print('Loading JAX model and checkpoint ...')
    jax_model, jax_params, tokenizer, enc_cfg = load_jax_model(
        MODEL_NAME, CHECKPOINT, max_length=MAX_LENGTH
    )

    print('\nBuilding PyTorch model ...')
    pt_model = build_pt_model(
        jax_params, MODEL_NAME, tokenizer.vocab_size, enc_cfg.d_model, MAX_LENGTH
    )

    # Save PyTorch checkpoint
    out_path = os.path.join(REPO, 'src', 'elf_b_owt_pt.pt')
    torch.save(pt_model.state_dict(), out_path)
    print(f'\nPyTorch checkpoint saved → {out_path}')

    print('\nRunning numerical verification ...')
    verify(jax_model, jax_params, pt_model, enc_cfg.d_model, MAX_LENGTH)


if __name__ == '__main__':
    main()
