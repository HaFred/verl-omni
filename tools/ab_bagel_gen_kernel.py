#!/usr/bin/env python
"""A/B the untouched BAGEL-7B-MoT checkpoint across GEN kernels.

Isolates the two kernel divergences that make the RL rollout look washed:

    cfg_text_scale      4.0 (canonical Bagel)  vs  1.0 (our Joint-Training
                            kernel -- ``bagel_corl_trainer.py`` pins this, CFG-free)
    num_inference_steps 50  (canonical Bagel)  vs  15  (our RL rollout)

Runs a 2x2 factorial on the REAL rollout prompts and writes a labelled montage
plus a laplacian-variance readout per cell.

'Canonical' here means canonical literally: this drives the OFFICIAL BAGEL
inferencer (``/scratch/fq9hpsac/zyd/Bagel``, ``inferencer.py``), whose own
defaults are cfg_text_scale=4.0, cfg_img_scale=1.5, cfg_interval=(0.4, 1.0),
num_timesteps=50, timestep_shift=3.0.

Usage:
    CUDA_VISIBLE_DEVICES=<gpu> .venv/bin/python tools/ab_bagel_gen_kernel.py \
        --out /tmp/ab_bagel_cfg.png
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from PIL import Image

BAGEL_REPO = os.environ.get("BAGEL_REPO", "/scratch/fq9hpsac/zyd/Bagel")
MODEL_PATH = os.environ.get(
    "BAGEL_MODEL_PATH",
    "/scratch/fq9hpsac/huggingface/hub/models--ByteDance-Seed--BAGEL-7B-MoT"
    "/snapshots/5019f57d168e5816e8f3f701b17cc816bb7cf24b",
)

# The prompts actually used by the step_000001 rollout dump, so the A/B is on
# the same inputs the user is judging.
PROMPTS = [
    "a destroyed school in an anime style",
    "Chaos anime-style cartoon of a destroyed school",
    "Full body portrait of a handsome bald male with a head tattoo, sporting a "
    "pink fancy mustache. He has a lean build and is dressed like a monk, "
    "embodying the character of a D&D hero",
]

# (label, cfg_text_scale, num_inference_steps)
ARMS = [
    ("canonical  cfg=4.0 / 50 steps", 4.0, 50),
    ("CFG-FREE   cfg=1.0 / 50 steps", 1.0, 50),
    ("cfg=4.0 / 15 steps (ours)", 4.0, 15),
    ("OUR KERNEL cfg=1.0 / 15 steps", 1.0, 15),
]

# Everything else pinned to the canonical Bagel values.
COMMON = dict(
    cfg_img_scale=1.5,
    cfg_interval=(0.4, 1.0),
    cfg_renorm_min=0.0,
    cfg_renorm_type="global",
    timestep_shift=3.0,
    image_shapes=(512, 512),  # matches the RL rollout resolution
    think=False,
    understanding_output=False,
)


def _install_transformers5_rope_compat() -> None:
    """Re-register the RoPE ``"default"`` initialiser that transformers 5.x removed.

    BAGEL's vendored ``modeling/qwen2/modeling_qwen2.py`` is a copy of the
    transformers 4.x ``Qwen2RotaryEmbedding``, which resolves its init function
    via ``ROPE_INIT_FUNCTIONS[self.rope_type]`` with ``rope_type == "default"``
    whenever ``config.rope_scaling`` is unset (which it is here -- the
    checkpoint's ``llm_config.json`` has no ``rope_scaling``).  transformers
    5.14's registry is only
    ``{dynamic, linear, llama3, longrope, proportional, yarn}``, so the lookup
    raises ``KeyError: 'default'`` and the model cannot even be constructed.

    The body below is the transformers 4.x ``_compute_default_rope_parameters``,
    which is plain un-scaled RoPE (``attention_factor = 1.0``).  Signature must
    stay ``(config, device, **kwargs)`` because that is how the copied class
    calls it.  No-op if a ``"default"`` entry already exists.
    """
    try:
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    except ImportError:  # pragma: no cover - only relevant on transformers 5.x
        return
    if "default" in ROPE_INIT_FUNCTIONS:
        return

    import torch as _torch

    def _compute_default_rope_parameters(config=None, device=None, seq_len=None, **kwargs):  # noqa: ARG001
        base = config.rope_theta
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", None) or (config.hidden_size // config.num_attention_heads)
        dim = int(head_dim * partial_rotary_factor)
        inv_freq = 1.0 / (
            base ** (_torch.arange(0, dim, 2, dtype=_torch.int64).float().to(device) / dim)
        )
        return inv_freq, 1.0

    ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters


def build_inferencer():
    """Load the untouched checkpoint exactly as the official ``app.py`` does."""
    _install_transformers5_rope_compat()
    sys.path.insert(0, BAGEL_REPO)

    from accelerate import infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch

    from data.data_utils import add_special_tokens
    from data.transforms import ImageTransform
    from inferencer import InterleaveInferencer
    from modeling.autoencoder import load_ae
    from modeling.bagel import Bagel, BagelConfig
    from modeling.bagel import Qwen2Config, Qwen2ForCausalLM
    from modeling.bagel import SiglipVisionConfig, SiglipVisionModel
    from modeling.qwen2 import Qwen2Tokenizer

    llm_config = Qwen2Config.from_json_file(os.path.join(MODEL_PATH, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"
    # Version shim: the checkpoint's llm_config.json carries no ``pad_token_id``
    # and newer transformers no longer leaves a ``None`` class default, but
    # BAGEL's ``Qwen2Model.__init__`` reads ``config.pad_token_id`` straight into
    # ``self.padding_idx``.  Generation is unpadded, so any in-vocab id is inert;
    # mirror the tokenizer's pad (``<|endoftext|>`` == bos_token_id 151643).
    if getattr(llm_config, "pad_token_id", None) is None:
        llm_config.pad_token_id = int(llm_config.bos_token_id)
    # Same era of drift: the copied 4.x rotary module reads these off the config
    # and 5.x no longer guarantees any of them exist.
    if getattr(llm_config, "rope_scaling", "MISSING") == "MISSING":
        llm_config.rope_scaling = None
    if getattr(llm_config, "rope_theta", None) is None:
        llm_config.rope_theta = 1000000.0
    if getattr(llm_config, "max_position_embeddings", None) is None:
        llm_config.max_position_embeddings = 32768
    if getattr(llm_config, "head_dim", None) is None:
        llm_config.head_dim = llm_config.hidden_size // llm_config.num_attention_heads

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(MODEL_PATH, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers -= 1

    vae_model, vae_config = load_ae(local_path=os.path.join(MODEL_PATH, "ae.safetensors"))

    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        latent_patch_size=2,
        max_latent_size=64,
    )

    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    tokenizer = Qwen2Tokenizer.from_pretrained(MODEL_PATH)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    device_map = infer_auto_device_map(
        model,
        max_memory={i: "80GiB" for i in range(torch.cuda.device_count())},
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )

    same_device_modules = [
        "language_model.model.embed_tokens",
        "time_embedder",
        "latent_pos_embed",
        "vae2llm",
        "llm2vae",
        "connector",
        "vit_pos_embed",
    ]
    first = device_map.get(same_device_modules[0], "cuda:0")
    for k in same_device_modules:
        device_map[k] = device_map.get(k, first)

    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=os.path.join(MODEL_PATH, "ema.safetensors"),
        device_map=device_map,
        offload_buffers=True,
        offload_folder="offload",
        dtype=torch.bfloat16,
        force_hooks=True,
    ).eval()

    device = next(model.parameters()).device if hasattr(model, "parameters") else torch.device("cuda:0")
    vae_model = vae_model.to(device=device, dtype=torch.float32).eval()

    # Diagnostics: with accelerate dispatch the module placement is not obvious,
    # and the VAE/latent device pairing is what broke the previous attempt.
    _devs: dict[str, int] = {}
    for p in model.parameters():
        key = str(p.device)
        _devs[key] = _devs.get(key, 0) + 1
    print(f"[ab] model param placement: {_devs}")
    print(f"[ab] vae device={next(vae_model.parameters()).device} "
          f"dtype={next(vae_model.parameters()).dtype}")

    inferencer = InterleaveInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=ImageTransform(1024, 512, 16),
        vit_transform=ImageTransform(980, 224, 14),
        new_token_ids=new_token_ids,
    )

    # ``interleave_inference`` wraps generation in ``torch.autocast(cuda, bf16)``.
    # The VAE is fp32, and autocast casts a conv2d *input* to bf16 while leaving
    # the fp32 bias alone, so ``decode_image`` dies with
    #   RuntimeError: Input type (float) and bias type (c10::BFloat16) should be the same.
    # Decode with autocast off and an fp32 latent instead -- this keeps the VAE
    # in its native fp32 precision (no bf16 decode artefacts) and both A/B arms
    # go through the identical path.
    _orig_decode_image = inferencer.decode_image
    _vae_param = next(vae_model.parameters())
    _vae_device = _vae_param.device

    def _decode_image_fp32(latent, image_shape):
        # ``accelerate``'s dispatch can hand back a CPU latent, while the VAE
        # lives on the accelerator; pin both device and dtype explicitly.
        with torch.autocast(device_type="cuda", enabled=False):
            return _orig_decode_image(
                latent.to(device=_vae_device, dtype=torch.float32), image_shape
            )

    inferencer.decode_image = _decode_image_fp32

    return inferencer


def lapvar(img: Image.Image) -> float:
    import numpy as np

    g = np.asarray(img.convert("L"), dtype=np.float32)
    lap = (
        -4 * g
        + np.roll(g, 1, 0) + np.roll(g, -1, 0)
        + np.roll(g, 1, 1) + np.roll(g, -1, 1)
    )
    return float(lap.var())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/ab_bagel_cfg.png")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    print(f"[ab] repo     = {BAGEL_REPO}")
    print(f"[ab] ckpt     = {MODEL_PATH}")
    print(f"[ab] device   = visible {torch.cuda.device_count()} GPU(s)")
    inferencer = build_inferencer()

    cells: list[tuple[str, str, Image.Image]] = []
    cells_dir = Path(args.out).parent / "ab_bagel_cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    for pi, prompt in enumerate(PROMPTS):
        for label, cfg, steps in ARMS:
            # Same seed per (prompt, arm) -> both arms draw the same initial
            # latent, so the delta is CFG and step-count only.
            torch.manual_seed(args.seed + pi)
            torch.cuda.manual_seed_all(args.seed + pi)
            out = inferencer(
                text=prompt,
                cfg_text_scale=cfg,
                num_timesteps=steps,
                **COMMON,
            )
            img = out["image"] if isinstance(out, dict) else out[-1]
            cells.append((prompt, label, img))
            # Persist at native 512x512: the montage below downscales for
            # display, and that resample alone shifts Laplacian variance, so
            # metrics must always be computed from these files, never the montage.
            safe = label.replace("/", "-").replace(" ", "_")
            img.save(cells_dir / f"p{pi}_{safe}.png")
            print(f"[ab] p{pi} {label:<32} lap-var={lapvar(img):8.1f} size={img.size}")

    # ---- montage: rows = arms, cols = prompts -----------------------------
    from PIL import ImageDraw, ImageFont

    tile = 470
    cap = 34
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    canvas = Image.new("RGB", (tile * len(PROMPTS), (tile + cap) * len(ARMS)), "white")
    d = ImageDraw.Draw(canvas)
    for ai, (label, _, _) in enumerate(ARMS):
        y = ai * (tile + cap)
        d.rectangle([0, y, canvas.width, y + cap], fill=(0, 0, 0))
        d.text((8, y + 9), label, fill=(255, 255, 255), font=font)
        for pi in range(len(PROMPTS)):
            _, _, img = cells[ai * len(PROMPTS) + pi]
            canvas.paste(img.convert("RGB").resize((tile, tile)), (pi * tile, y + cap))
    canvas.save(args.out)
    print(f"[ab] wrote {args.out} {canvas.size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
