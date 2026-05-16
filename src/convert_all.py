"""
Convert all ELF JAX/Flax checkpoints to PyTorch and upload to HuggingFace.

Usage:
    JAX_PLATFORMS=cpu pixi run -e convert python src/convert_all.py
"""
import os
import sys
import copy
import subprocess
import numpy as np
import torch
from huggingface_hub import HfApi, create_repo

# ── paths ─────────────────────────────────────────────────────────────────────
REPO    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUR_SRC = os.path.join(REPO, "src")
JAX_SRC = "/tmp/elf_jax_src"

# ── clone upstream JAX code if needed ─────────────────────────────────────────
if not os.path.exists(JAX_SRC):
    print("Cloning upstream JAX source...")
    subprocess.run(
        ["git", "clone", "--depth=1", "https://github.com/lillian039/ELF", JAX_SRC],
        check=True,
    )

# ── import JAX modules first (before our src shadows 'modules') ───────────────
# Python adds the script's directory (src/) to sys.path[0] automatically;
# remove it so our PT modules don't shadow the JAX ones during import.
for _p in [OUR_SRC, "", "."]:
    while _p in sys.path:
        sys.path.remove(_p)
sys.path.insert(0, os.path.join(JAX_SRC, "src"))

import jax
import jax.numpy as jnp
import optax
from transformers import AutoTokenizer

import modules.model          as _jax_model_mod
import modules.t5_encoder     as _jax_t5_mod
import utils.checkpoint_utils as _jax_ckpt_mod
import utils.train_utils      as _jax_train_mod

JAX_ELF_models          = _jax_model_mod.ELF_models
get_encoder             = _jax_t5_mod.get_encoder
load_checkpoint         = _jax_ckpt_mod.load_checkpoint
load_encoder_checkpoint = _jax_ckpt_mod.load_encoder_checkpoint
TrainState              = _jax_train_mod.TrainState

# ── switch to our PT modules ──────────────────────────────────────────────────
sys.path.remove(os.path.join(JAX_SRC, "src"))
for key in list(sys.modules.keys()):
    if key.startswith(("modules", "utils", "configs")):
        del sys.modules[key]

sys.path.insert(0, OUR_SRC)
from modules.model import ELF_models as PT_ELF_models


# ── weight conversion ─────────────────────────────────────────────────────────
def _v(arr):
    return torch.from_numpy(np.array(arr, dtype=np.float32))

def _t(arr):
    return _v(arr).T

def convert_elf_params(p, depth):
    sd = {}
    sd["t_emb_tokens"]           = _v(p["t_emb_tokens"])
    sd["self_cond_cfg_tokens"]   = _v(p["self_cond_cfg_tokens"])
    sd["mode_tokens"]            = _v(p["mode_tokens"])
    sd["proj_kernel"]            = _v(p["proj_kernel"])
    sd["proj_bias"]              = _v(p["proj_bias"])
    sd["unembed_kernel"]         = _v(p["unembed_kernel"])
    sd["unembed_bias"]           = _v(p["unembed_bias"])

    sd["self_cond_proj.weight"]  = _t(p["self_cond_proj"]["kernel"])
    sd["self_cond_proj.bias"]    = _v(p["self_cond_proj"]["bias"])

    sd["text_proj.proj1.weight"] = _t(p["text_proj"]["proj1"]["kernel"])
    sd["text_proj.proj2.weight"] = _t(p["text_proj"]["proj2"]["kernel"])
    sd["text_proj.proj2.bias"]   = _v(p["text_proj"]["proj2"]["bias"])

    for name in ("t_embedder", "self_cond_cfg_embedder"):
        sd[f"{name}.mlp_0.weight"] = _t(p[name]["mlp_0"]["kernel"])
        sd[f"{name}.mlp_0.bias"]   = _v(p[name]["mlp_0"]["bias"])
        sd[f"{name}.mlp_2.weight"] = _t(p[name]["mlp_2"]["kernel"])
        sd[f"{name}.mlp_2.bias"]   = _v(p[name]["mlp_2"]["bias"])

    sd["final_layer.norm_final.weight"] = _v(p["final_layer"]["norm_final"]["weight"])
    sd["final_layer.linear.weight"]     = _t(p["final_layer"]["linear"]["kernel"])
    sd["final_layer.linear.bias"]       = _v(p["final_layer"]["linear"]["bias"])

    for i in range(depth):
        bp  = p[f"blocks_{i}"]
        pfx = f"blocks.{i}"
        sd[f"{pfx}.norm1.weight"]       = _v(bp["norm1"]["weight"])
        sd[f"{pfx}.norm2.weight"]       = _v(bp["norm2"]["weight"])
        sd[f"{pfx}.attn.qkv.weight"]    = _t(bp["attn"]["qkv"]["kernel"])
        sd[f"{pfx}.attn.qkv.bias"]      = _v(bp["attn"]["qkv"]["bias"])
        sd[f"{pfx}.attn.proj.weight"]   = _t(bp["attn"]["proj"]["kernel"])
        sd[f"{pfx}.attn.proj.bias"]     = _v(bp["attn"]["proj"]["bias"])
        sd[f"{pfx}.attn.q_norm.weight"] = _v(bp["attn"]["q_norm"]["weight"])
        sd[f"{pfx}.attn.k_norm.weight"] = _v(bp["attn"]["k_norm"]["weight"])
        sd[f"{pfx}.mlp.w12.weight"]     = _t(bp["mlp"]["w12"]["kernel"])
        sd[f"{pfx}.mlp.w12.bias"]       = _v(bp["mlp"]["w12"]["bias"])
        sd[f"{pfx}.mlp.w3.weight"]      = _t(bp["mlp"]["w3"]["kernel"])
        sd[f"{pfx}.mlp.w3.bias"]        = _v(bp["mlp"]["w3"]["bias"])
    return sd


# ── checkpoint list ───────────────────────────────────────────────────────────
ENCODER_MODEL = "t5-small"
DEPTH_MAP     = {"ELF-B": 12, "ELF-M": 24, "ELF-L": 32}

CHECKPOINTS = [
    {"model": "ELF-B", "src": "embedded-language-flows/ELF-B-owt",   "dst": "reonokiy/ELF-B-owt"},
    {"model": "ELF-M", "src": "embedded-language-flows/ELF-M-owt",   "dst": "reonokiy/ELF-M-owt"},
    {"model": "ELF-L", "src": "embedded-language-flows/ELF-L-owt",   "dst": "reonokiy/ELF-L-owt"},
    {"model": "ELF-B", "src": "embedded-language-flows/ELF-B-de-en", "dst": "reonokiy/ELF-B-de-en"},
    {"model": "ELF-B", "src": "embedded-language-flows/ELF-B-xsum",  "dst": "reonokiy/ELF-B-xsum"},
]


# ── per-checkpoint pipeline ───────────────────────────────────────────────────
def load_jax_params(model_name, checkpoint, max_length=64):
    tokenizer = AutoTokenizer.from_pretrained(ENCODER_MODEL)
    enc_cfg, _, _ = get_encoder(ENCODER_MODEL, jnp.float32)
    enc_params = load_encoder_checkpoint(
        "embedded-language-flows/t5_small_encoder_jax/t5_small_encoder_jax.pkl"
    )
    enc_dim = enc_cfg.d_model

    rng = jax.random.PRNGKey(0)
    rng, init_rng, drop_rng = jax.random.split(rng, 3)

    jax_model = JAX_ELF_models[model_name](
        text_encoder_dim=enc_dim,
        max_length=max_length,
        num_time_tokens=4,
        num_self_cond_cfg_tokens=4,
        num_model_mode_tokens=4,
        bottleneck_dim=128,
        vocab_size=tokenizer.vocab_size,
    )
    dummy_x  = jnp.ones((1, max_length, 2 * enc_dim))
    dummy_t  = jnp.ones((1,))
    dummy_sc = jnp.ones((1,))
    elf_params = jax_model.init(
        init_rng, x=dummy_x, t=dummy_t, self_cond_cfg_scale=dummy_sc, deterministic=True
    )
    state = TrainState.create(
        apply_fn=jax_model.apply,
        params=elf_params["params"],
        tx=optax.adamw(1e-4),
        dropout_rng=drop_rng,
        ema_params1=copy.deepcopy(elf_params["params"]),
    )
    state, step = load_checkpoint(checkpoint, state)
    print(f"  JAX checkpoint loaded: step={step}")
    return state.ema_params1, tokenizer.vocab_size, enc_dim


def build_pt_model(jax_params, model_name, vocab_size, enc_dim):
    pt_model = PT_ELF_models[model_name](
        text_encoder_dim=enc_dim,
        max_length=512,
        num_time_tokens=4,
        num_self_cond_cfg_tokens=4,
        num_model_mode_tokens=4,
        bottleneck_dim=128,
        vocab_size=vocab_size,
    )
    sd = convert_elf_params(jax_params, DEPTH_MAP[model_name])
    missing, unexpected = pt_model.load_state_dict(sd, strict=True)
    assert not missing,    f"Missing keys: {missing}"
    assert not unexpected, f"Unexpected keys: {unexpected}"
    pt_model.eval()
    n = sum(p.numel() for p in pt_model.parameters())
    print(f"  PT model: {n:,} params")
    return pt_model


def upload_to_hf(pt_model, dst_repo, src_repo, out_dir="/tmp/elf_pt_ckpts"):
    os.makedirs(out_dir, exist_ok=True)
    filename  = "pytorch_model.bin"
    local_path = os.path.join(out_dir, dst_repo.replace("/", "_") + ".bin")
    torch.save(pt_model.state_dict(), local_path)

    api = HfApi()
    create_repo(dst_repo, repo_type="model", exist_ok=True, private=False)
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=filename,
        repo_id=dst_repo,
        repo_type="model",
        commit_message=f"Add PyTorch checkpoint converted from {src_repo}",
    )
    print(f"  Uploaded → https://huggingface.co/{dst_repo}")
    os.remove(local_path)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    for cfg in CHECKPOINTS:
        model_name, src, dst = cfg["model"], cfg["src"], cfg["dst"]
        print(f"\n{'='*60}")
        print(f"{src}  →  {dst}")
        print(f"{'='*60}")

        print("  Loading JAX checkpoint...")
        jax_params, vocab_size, enc_dim = load_jax_params(model_name, src)

        print("  Converting to PyTorch...")
        pt_model = build_pt_model(jax_params, model_name, vocab_size, enc_dim)

        print("  Uploading to HuggingFace...")
        upload_to_hf(pt_model, dst, src)

        del jax_params, pt_model

    print("\nAll done!")


if __name__ == "__main__":
    main()
