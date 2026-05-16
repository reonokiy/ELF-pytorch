"""Minimal script: download ELF-B-owt checkpoint and verify it loads."""
import os
import sys
import copy
import logging

import jax
import jax.numpy as jnp
import optax
from transformers import AutoTokenizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from modules.model import ELF_models
from modules.t5_encoder import get_encoder
from utils.checkpoint_utils import load_encoder_checkpoint, load_checkpoint
from utils.train_utils import TrainState

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
log = logging.getLogger(__name__)

HF_MODEL = "embedded-language-flows/ELF-B-owt"
ENCODER_MODEL = "t5-small"
ENCODER_CKPT = "embedded-language-flows/t5_small_encoder_jax/t5_small_encoder_jax.pkl"
MAX_LENGTH = 128  # use 128 instead of 1024 for faster init on CPU

def main():
    log.info(f"JAX devices: {jax.devices()}")

    # --- Tokenizer ---
    log.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(ENCODER_MODEL)

    # --- T5 encoder ---
    log.info("Loading T5 encoder config...")
    encoder_config, _, _ = get_encoder(ENCODER_MODEL, jnp.float32)
    log.info("Downloading T5 encoder weights from HF...")
    encoder_params = load_encoder_checkpoint(ENCODER_CKPT)
    log.info(f"Encoder loaded. d_model={encoder_config.d_model}")

    # --- ELF-B model init ---
    log.info("Initializing ELF-B model on CPU...")
    rng = jax.random.PRNGKey(0)
    rng, init_rng, dropout_rng = jax.random.split(rng, 3)

    # self_cond_prob=0.5 → input dim is 2x encoder dim
    text_enc_dim = encoder_config.d_model
    dummy_x = jnp.ones((1, MAX_LENGTH, 2 * text_enc_dim))
    dummy_t = jnp.ones((1,))
    dummy_sc_cfg = jnp.ones((1,))  # num_self_cond_cfg_tokens > 0

    model = ELF_models["ELF-B"](
        text_encoder_dim=text_enc_dim,
        max_length=MAX_LENGTH,
        num_time_tokens=4,
        num_self_cond_cfg_tokens=4,
        num_model_mode_tokens=4,
        bottleneck_dim=128,
        vocab_size=tokenizer.vocab_size,
    )

    elf_params = model.init(init_rng, x=dummy_x, t=dummy_t,
                            self_cond_cfg_scale=dummy_sc_cfg, deterministic=True)
    total = sum(x.size for x in jax.tree_util.tree_leaves(elf_params))
    log.info(f"ELF-B initialized. Parameters: {total:,}")

    # --- Train state template ---
    state = TrainState.create(
        apply_fn=model.apply,
        params=elf_params["params"],
        tx=optax.adamw(learning_rate=1e-4),
        dropout_rng=dropout_rng,
        ema_params1=copy.deepcopy(elf_params["params"]),
    )

    # --- Load checkpoint from HF ---
    log.info(f"Downloading checkpoint from HF: {HF_MODEL}")
    state, step = load_checkpoint(HF_MODEL, state)
    log.info(f"Checkpoint loaded successfully! step={step}")

    # Quick shape check
    param_leaves = jax.tree_util.tree_leaves(state.params)
    log.info(f"Loaded {len(param_leaves)} parameter arrays")
    log.info("Done.")

if __name__ == "__main__":
    main()
