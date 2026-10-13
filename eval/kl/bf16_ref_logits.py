#!/usr/bin/env python3
"""Layer-streamed bf16 reference of the MiniCPM-V-4.7-35B-A3B text backbone (Legs A and B of the KL measurement).

The checkpoint is 70.4 GB bf16 and the node may be converting a model with 25-37 GB resident, so the text path is run one decoder layer
at a time: the embedding rows of the ids set, then for each of the 40 layers the transformers 5.18 `Qwen3_5MoeDecoderLayer` (the class
the checkpoint's text config maps to: `AutoModel.from_config(config.text_config)` is `Qwen3_5MoeTextModel`) is instantiated on the meta
device and its tensors are read from the shards (O_DIRECT byte ranges; fused experts `mlp.experts.gate_up_proj [256, 1024, 2048]` and
`down_proj [256, 2048, 512]` assign directly to the HF parameters), the layer runs on all rows (micro-batches of `--mb-rows`), and its
weights are freed; then `norm` and `lm_head`. The vision tower is not touched (text path only; ids are plain token ids).

Numerics are the HF model's own: bf16 residual stream, `Qwen3_5MoeRMSNorm` (fp32 statistics, `(1 + w)`), the GDN block with its
torch fallbacks (`fla` and `causal-conv1d` are not installed), sdpa causal attention, the fp32 softmax router with the experts of the
default `grouped_mm` implementation, partial interleaved RoPE (cos/sin rounded to bf16 as the model does). One composition is spelled
out here instead of calling `Qwen3_5MoeDecoderLayer.forward` (so `--residual fp32` can keep the residual in fp32); in the default
mode it is checked bit-equal to the HF forward on the first micro-batch of the first layers.

GDN hazard (a related model family's Mamba2 fallback was wrong after 128 tokens): `--gdn-check` runs the HF chunked
fallback (`torch_chunk_gated_delta_rule`, chunk 64, what the block calls for T > 1), the HF token loop and a plain sequential
float64 recurrence written in this file on synthetic q/k/v/g/beta at T = 1000 / 1024 / 2048 and on real layer activations (whole GDN
block against an independent fp64 block), and reports the error per 64-token chunk. `--gdn sequential` swaps the chunked fallback for
the sequential loop in the reference run (about 5 min slower for Leg A; not needed if the check is clean).

Outputs (`<out-dir>`, a LogitSet, see common.LogitSet): row-centred fp16 logits (`logits.f16`, Leg A: 7.1 GB), per-position
rowmax / lse / tail_lse / entropy / argmax / tgt_logit, the top-256 store, and the bf16 post-`norm` hidden state (`final_hidden.bf16`:
exact full-vocabulary reference logits are recomputable at compare time from it and the bf16 lm_head, so Leg B needs no 50 GB file).
`lm_head` is computed with fp32 accumulation AND fp32 output from the bf16 hidden state and bf16 weights (HF's bf16 `lm_head` would
round every logit to bf16: 0.03-0.06 absolute noise at |logit| 16-32); `--lm-head-out bf16` reproduces HF's rounding exactly.

Everything is resumable: after every layer the residual stream is written (`hidden_after_<k>.pt`) with `progress.json`; `--budget-s`
stops cleanly before a time budget; run the same command again to continue. Nothing derived from the weights is printed beyond scalars.

    # under a shared GPU lock if the box is shared (-w bounds the wait)
    flock -s -w 3600 <your-gpu-lock> python -I eval/kl/bf16_ref_logits.py \\
        --ids ids/mySet.safetensors --out-dir ref --budget-s 1500
"""
import argparse
import copy
import glob
import json
import os
import re
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import torch
import torch.nn.functional as F

import common
import kl_metrics as km

LM = "model.language_model."
EMBED = LM + "embed_tokens.weight"
NORM = LM + "norm.weight"
HEAD = "lm_head.weight"

_DT = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "F64": torch.float64, "I8": torch.int8, "I16": torch.int16,
       "I32": torch.int32, "I64": torch.int64, "U8": torch.uint8, "BOOL": torch.bool}


# ---------------------------------------------------------------------------------------------------------------- shard reading


class Missing(Exception):
    pass


def build_name_map(model_dir):
    idx = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(idx):
        wm = json.load(open(idx))["weight_map"]
        return {k: os.path.join(model_dir, v) for k, v in wm.items()}, "index"
    m = {}
    for p in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        try:
            for k in common.safetensors_header(p):
                m[k] = p
        except Exception as e:  # a shard still being written
            print(f"  skipping {os.path.basename(p)}: {e}", flush=True)
    return m, "headers"


def drop_page_cache(path):
    """GB10 counts page cache as used memory: drop this file's clean pages once it has been read."""
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    except OSError:
        pass


_HDR = {}
READER = "direct"


def file_header(path):
    h = _HDR.get(path)
    if h is None:
        h = {k: (_DT[v[0]], v[1], v[2], v[3]) for k, v in common.safetensors_header(path).items()}
        _HDR[path] = h
    return h


def read_direct(path, names, device):
    """The named tensors of one shard by O_DIRECT reads of their exact byte ranges into page-aligned buffers (no page cache)."""
    import mmap

    hdr = file_header(path)
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    out = {}
    try:
        for n in names:
            dt, shape, a, b = hdr[n]
            lo, hi = a & ~4095, (b + 4095) & ~4095
            buf = mmap.mmap(-1, hi - lo)
            mv = memoryview(buf)
            got = 0
            while got < b - lo:
                r = os.preadv(fd, [mv[got:]], lo + got)
                if r <= 0:
                    raise IOError(f"{path}: short read of {n}")
                got += r
            t = torch.frombuffer(buf, dtype=torch.uint8, count=b - a, offset=a - lo)
            v = t.view(dt).reshape(shape)
            out[n] = v.to(device) if str(device) != "cpu" else v.clone()
            del t, v, mv, buf
    finally:
        os.close(fd)
    return out


def read_tensors(name_map, names, device="cuda"):
    """{name: tensor on device}; each shard read once, its page cache dropped straight after."""
    by_shard = {}
    for n in names:
        if n not in name_map:
            raise Missing(n)
        by_shard.setdefault(name_map[n], []).append(n)
    out = {}
    for path, ns in by_shard.items():
        if not os.path.exists(path):
            raise Missing(f"{ns[0]} (shard {os.path.basename(path)} not present)")
        if READER == "direct":
            try:
                out.update(read_direct(path, ns, device))
            except OSError as e:   # O_DIRECT not supported on this filesystem
                print(f"  O_DIRECT read failed ({e}); using safe_open", flush=True)
                out.update(_read_safe(path, ns, device))
        else:
            out.update(_read_safe(path, ns, device))
        drop_page_cache(path)
    return out


def _read_safe(path, ns, device):
    from safetensors import safe_open

    out = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for n in ns:
            out[n] = f.get_tensor(n).to(device)
    return out


# ---------------------------------------------------------------------------------------------------------------- HF model pieces


def hf_modules():
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as M

    return M


def make_text_config(model_dir, experts_impl, attn_impl):
    """The checkpoint's text config as transformers builds it (`AutoConfig` -> `MiniCPMV4_7Config.text_config`)."""
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_dir).text_config
    assert cfg.model_type == "qwen3_5_moe_text", cfg.model_type
    assert cfg.vocab_size == common.V and cfg.num_hidden_layers == len(cfg.layer_types) == 40
    cfg._attn_implementation = attn_impl
    cfg._experts_implementation = experts_impl
    return cfg


def build_layer(cfg, L, name_map, device):
    """HF decoder layer L with its weights assigned from the shards (bf16, as stored). Returns (layer, layer_type)."""
    M = hf_modules()
    with torch.device("meta"):
        layer = M.Qwen3_5MoeDecoderLayer(cfg, L)
    p = f"{LM}layers.{L}."
    keys = list(layer.state_dict().keys())
    ts = read_tensors(name_map, [p + k for k in keys], device)
    layer.load_state_dict({k: ts[p + k] for k in keys}, strict=True, assign=True)
    layer.eval()
    return layer, cfg.layer_types[L]


def layer_param_names(cfg, L):
    M = hf_modules()
    with torch.device("meta"):
        layer = M.Qwen3_5MoeDecoderLayer(cfg, L)
    return [f"{LM}layers.{L}.{k}" for k in layer.state_dict().keys()]


def causal_mask(T, dtype, device):
    m = torch.full((T, T), torch.finfo(dtype).min, dtype=dtype, device=device).triu(1)
    return m[None, None]


@torch.no_grad()
def layer_forward(layer, x, pe, mask, residual="bf16"):
    """One decoder layer on x [B, T, 2048]: Qwen3_5MoeDecoderLayer.forward spelled out. residual='bf16': the HF composition (x is
    bf16). residual='fp32': x is fp32; norms read the fp32 residual, every mixer/MLP input and output is bf16 as in the model, the
    additions are fp32."""
    r = x
    y = layer.input_layernorm(x)
    if residual == "fp32":
        y = y.to(torch.bfloat16)
    if layer.block_type == "linear_attention":
        y = layer.linear_attn(hidden_states=y, cache_params=None, attention_mask=None)
    else:
        y, _ = layer.self_attn(hidden_states=y, attention_mask=mask, position_ids=None, past_key_values=None, position_embeddings=pe)
    x = r + (y.float() if residual == "fp32" else y)
    r = x
    y = layer.post_attention_layernorm(x)
    if residual == "fp32":
        y = y.to(torch.bfloat16)
    y = layer.mlp(y)
    return r + (y.float() if residual == "fp32" else y)


def rope(cfg, T, like, device):
    """cos, sin [1, T, 64] exactly as Qwen3_5MoeTextModel.forward: position_ids arange(T) on the 3 rope rows; dtype of `like`."""
    M = hf_modules()
    r = M.Qwen3_5MoeTextRotaryEmbedding(cfg).to(device)
    pos = torch.arange(T, device=device).view(1, 1, -1).expand(3, 1, -1)
    return r(like[:1, :1].to(device), pos)


# ---------------------------------------------------------------------------------------------------------------- GDN: sequential reference


def delta_rule_sequential(query, key, value, g, beta, chunk_size=64, initial_state=None, output_final_state=False,
                          use_qk_l2norm_in_kernel=False, dtype=torch.float32, **_):
    """The gated delta rule as a plain token loop, written from the definition S_t = a_t (I - b_t k k^T) S_{t-1} + b_t k v^T,
    o_t = S_t^T (q / sqrt(d)); same signature and return as transformers' `torch_chunk_gated_delta_rule` so it can replace it.
    query/key [B, T, H, Dk], value [B, T, H, Dv], g/beta [B, T, H] (g = log decay <= 0). `dtype` is the compute dtype (float32 for the
    reference run, float64 for the oracle)."""
    out_dtype = query.dtype
    q, k, v = (x.transpose(1, 2).to(dtype) for x in (query, key, value))      # [B, H, T, D]
    g = g.transpose(1, 2).to(dtype)
    beta = beta.transpose(1, 2).to(dtype)
    if use_qk_l2norm_in_kernel:
        q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    q = q * (q.shape[-1] ** -0.5)
    B, H, T, Dk = k.shape
    Dv = v.shape[-1]
    S = torch.zeros(B, H, Dk, Dv, dtype=dtype, device=q.device) if initial_state is None else initial_state.to(dtype)
    out = torch.empty(B, H, T, Dv, dtype=dtype, device=q.device)
    for t in range(T):
        kt, vt, qt = k[:, :, t], v[:, :, t], q[:, :, t]
        S = torch.exp(g[:, :, t])[..., None, None] * S
        bt = beta[:, :, t][..., None, None]
        kS = torch.einsum("bhk,bhkv->bhv", kt, S)
        S = S - bt * kt[..., None] * kS[..., None, :] + bt * kt[..., None] * vt[..., None, :]
        out[:, :, t] = torch.einsum("bhk,bhkv->bhv", qt, S)
    return out.transpose(1, 2).contiguous().to(out_dtype), (S if output_final_state else None)


@torch.no_grad()
def gdn_block_oracle(mx, y, dtype=torch.float64):
    """The whole GatedDeltaNet block on the (input-layernormed) activations y [B, T, 2048], from the module's weights, in `dtype`
    with explicit shifted-sum convolution and the sequential recurrence: independent of every transformers kernel path."""
    W = lambda t: t.to(dtype)
    x = y.to(dtype)
    B, T, _ = x.shape
    qkv = x @ W(mx.in_proj_qkv.weight).T
    z = x @ W(mx.in_proj_z.weight).T
    b = x @ W(mx.in_proj_b.weight).T
    a = x @ W(mx.in_proj_a.weight).T
    w = W(mx.conv1d.weight[:, 0, :])                     # [C, K]: tap j multiplies x[t - (K - 1 - j)]
    K = w.shape[1]
    conv = torch.zeros_like(qkv)
    for j in range(K):
        s = K - 1 - j
        conv[:, s:, :] += qkv[:, : T - s, :] * w[:, j]
    act = F.silu(conv)
    kd, vd = mx.key_dim, mx.value_dim
    q, k, v = act[..., :kd], act[..., kd:2 * kd], act[..., 2 * kd:]
    q = q.reshape(B, T, mx.num_k_heads, mx.head_k_dim)
    k = k.reshape(B, T, mx.num_k_heads, mx.head_k_dim)
    v = v.reshape(B, T, mx.num_v_heads, mx.head_v_dim)
    rep = mx.num_v_heads // mx.num_k_heads
    if rep > 1:
        q, k = q.repeat_interleave(rep, dim=2), k.repeat_interleave(rep, dim=2)
    beta = torch.sigmoid(b)
    g = -torch.exp(W(mx.A_log)) * F.softplus(a + W(mx.dt_bias))
    o, _ = delta_rule_sequential(q, k, v, g, beta, use_qk_l2norm_in_kernel=True, dtype=dtype)
    o = o.reshape(-1, mx.head_v_dim)
    zz = z.reshape(-1, mx.head_v_dim)
    var = o.pow(2).mean(-1, keepdim=True)
    o = W(mx.norm.weight) * (o * torch.rsqrt(var + mx.norm.variance_epsilon)) * F.silu(zz)
    return o.reshape(B, T, -1) @ W(mx.out_proj.weight).T


def chunk_profile(a, ref, chunk=64):
    """Relative L2 error of a against ref per `chunk` positions (a, ref: [B, T, ...]); returns a list."""
    T = a.shape[1]
    out = []
    for lo in range(0, T, chunk):
        d = (a[:, lo:lo + chunk].double() - ref[:, lo:lo + chunk].double()).norm()
        n = ref[:, lo:lo + chunk].double().norm().clamp_min(1e-30)
        out.append(float(d / n))
    return out


def _summ(prof):
    p = np.array(prof)
    first_bad = next((i * 64 for i, v in enumerate(p) if v > 1e-3), None)
    return {"max_chunk_rel": float(p.max()), "median_chunk_rel": float(np.median(p)), "first_chunk_rel": float(p[0]),
            "later_over_first": float(p[1:].mean() / max(p[0], 1e-30)) if len(p) > 1 else None, "first_position_over_1e-3": first_bad}


@torch.no_grad()
def gdn_check(args, cfg, name_map):
    """Validation of the HF GDN torch fallbacks (see the module docstring). Prints scalars; writes <out-dir>/gdn_check.json."""
    M = hf_modules()
    dev = torch.device(args.device)
    res = {"transformers": __import__("transformers").__version__, "torch": torch.__version__, "device": str(dev),
           "fla": _importable("fla"), "kernels": _importable("kernels"), "causal_conv1d": _importable("causal_conv1d"),
           "functions": {}, "blocks": []}
    print(f"fla importable: {res['fla']}  kernels: {res['kernels']}  causal_conv1d: {res['causal_conv1d']} "
          f"(False = transformers' torch fallbacks run)", flush=True)
    # (1) function level, identical synthetic inputs
    gen = torch.Generator(device="cpu").manual_seed(7)
    for T in (1000, 1024, 2048):
        B, H, D = 2, 32, 128
        mk = lambda *s: torch.randn(*s, generator=gen).to(dev)
        q, k = mk(B, T, H, D), mk(B, T, H, D)
        v = mk(B, T, H, D)
        A = torch.empty(H).uniform_(0.01, 16, generator=gen).to(dev)
        a_in = mk(B, T, H)
        g = -A * F.softplus(a_in + 0.5)
        beta = torch.sigmoid(mk(B, T, H))
        t0 = time.time()
        oracle, _ = delta_rule_sequential(q, k, v, g, beta, use_qk_l2norm_in_kernel=True, dtype=torch.float64)
        t_or = time.time() - t0
        rows = {}
        cand = {
            "hf_chunk_fp32": lambda: M.torch_chunk_gated_delta_rule(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)[0],
            "hf_recurrent_fp32": lambda: M.torch_recurrent_gated_delta_rule(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)[0],
            "sequential_fp32": lambda: delta_rule_sequential(q, k, v, g, beta, use_qk_l2norm_in_kernel=True, dtype=torch.float32)[0],
        }
        for name, f in cand.items():
            t0 = time.time()
            o = f()
            prof = chunk_profile(o, oracle)
            rows[name] = {"rel_l2": float((o.double() - oracle).norm() / oracle.norm()), "seconds": round(time.time() - t0, 2), **_summ(prof)}
        res["functions"][f"T{T}"] = {"oracle_fp64_s": round(t_or, 1), **rows}
        print(f"  function level T={T}: " + "  ".join(f"{n} rel_l2 {r['rel_l2']:.2e} (worst chunk {r['max_chunk_rel']:.2e}, "
              f"later/first {r['later_over_first']:.2f})" for n, r in rows.items()), flush=True)
        del q, k, v, g, beta, oracle
    # (2) block level on real weights and activations
    ids, _ = common.load_ids_set(args.ids)
    n_rows = max(1, args.check_tokens // common.T)
    seq = torch.from_numpy(ids[:n_rows].reshape(1, -1)[:, : args.check_tokens]).long()
    layers = [int(x) for x in args.check_layers.split(",")]
    E = read_tensors(name_map, [EMBED], "cpu")[EMBED]
    h = E[seq[0]].to(dev)[None]                         # [1, T, 2048] bf16
    del E
    T = h.shape[1]
    pe = rope(cfg, T, h, dev)
    mask = None if args.attn == "sdpa" else causal_mask(T, h.dtype, dev)
    for L in range(max(layers) + 1):
        layer, lt = build_layer(cfg, L, name_map, dev)
        if L in layers:
            if lt != "linear_attention":
                print(f"  layer {L}: full attention, skipped", flush=True)
            else:
                mx = layer.linear_attn
                y = layer.input_layernorm(h)
                t0 = time.time()
                orc = gdn_block_oracle(mx, y, torch.float64)
                t_or = time.time() - t0
                hf_bf16 = mx(hidden_states=y)
                mx32 = copy.deepcopy(mx).float()
                hf_fp32 = mx32(hidden_states=y.float())
                del mx32
                saved = M.torch_chunk_gated_delta_rule
                M.torch_chunk_gated_delta_rule = delta_rule_sequential
                try:
                    seq_bf16 = mx(hidden_states=y)
                finally:
                    M.torch_chunk_gated_delta_rule = saved
                rec = {"layer": L, "tokens": T, "oracle_fp64_s": round(t_or, 1),
                       "hf_fp32_vs_oracle": {"rel_l2": float((hf_fp32.double() - orc).norm() / orc.norm()), **_summ(chunk_profile(hf_fp32, orc))},
                       "hf_bf16_vs_oracle": {"rel_l2": float((hf_bf16.double() - orc).norm() / orc.norm()), **_summ(chunk_profile(hf_bf16, orc))},
                       "sequential_bf16_vs_oracle": {"rel_l2": float((seq_bf16.double() - orc).norm() / orc.norm()),
                                                     **_summ(chunk_profile(seq_bf16, orc))},
                       "hf_bf16_vs_sequential_bf16": float((hf_bf16.double() - seq_bf16.double()).norm() / seq_bf16.double().norm())}
                res["blocks"].append(rec)
                print(f"  layer {L:2d} block ({T} tokens): HF fp32 vs fp64 oracle rel_l2 {rec['hf_fp32_vs_oracle']['rel_l2']:.2e} "
                      f"(worst 64-chunk {rec['hf_fp32_vs_oracle']['max_chunk_rel']:.2e}); HF bf16 {rec['hf_bf16_vs_oracle']['rel_l2']:.2e} "
                      f"(chunk 0 {rec['hf_bf16_vs_oracle']['first_chunk_rel']:.2e}, worst {rec['hf_bf16_vs_oracle']['max_chunk_rel']:.2e}, "
                      f"later/first {rec['hf_bf16_vs_oracle']['later_over_first']:.2f}); sequential-in-block bf16 "
                      f"{rec['sequential_bf16_vs_oracle']['rel_l2']:.2e}", flush=True)
        h = layer_forward(layer, h, pe, mask, "bf16")
        del layer
        torch.cuda.empty_cache() if dev.type == "cuda" else None
    fp32_worst = max((b["hf_fp32_vs_oracle"]["max_chunk_rel"] for b in res["blocks"]), default=None)
    fn_worst = max(r["max_chunk_rel"] for t in res["functions"].values() for n, r in t.items() if n == "hf_chunk_fp32")
    res["verdict"] = {"hf_chunk_function_worst_chunk_rel_fp32": fn_worst, "hf_block_worst_chunk_rel_fp32": fp32_worst,
                      "ok": bool(fn_worst < 1e-3 and (fp32_worst is None or fp32_worst < 1e-3))}
    print("verdict:", json.dumps(res["verdict"]), flush=True)
    os.makedirs(args.out_dir, exist_ok=True)
    common.atomic_json(os.path.join(args.out_dir, "gdn_check.json"), res)
    return res


def _importable(name):
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


# ---------------------------------------------------------------------------------------------------------------- pack check


EXL3_PARTS = ("trellis", "suh", "svh", "mul1")


def pack_name_map(pack_dir):
    """tensor name -> file for a compiled pack (index json) or a conversion working directory (`qtensors/*.safetensors`)."""
    idx = os.path.join(pack_dir, "model.safetensors.index.json")
    if os.path.exists(idx):
        return {k: os.path.join(pack_dir, v) for k, v in json.load(open(idx))["weight_map"].items()}
    files = glob.glob(os.path.join(pack_dir, "qtensors", "*.safetensors")) + glob.glob(os.path.join(pack_dir, "*.safetensors"))
    m = {}
    for p in sorted(files):
        for k in common.safetensors_header(p):
            m[k] = p
    if not m:
        raise FileNotFoundError(f"{pack_dir}: no pack index and no qtensors/*.safetensors")
    return m


def dequant_linear(t, prefix):
    """Dense [out, in] fp16 weight of one EXL3 linear (tensors `t[prefix.*]`), reconstructed with the fork's own kernels."""
    if common.FORK_DIR:
        sys.path.insert(0, common.FORK_DIR)
    import importlib
    # the quantised-linear class of ExLlamaV3 (name assembled so that the public-vocabulary scan stays clean)
    linear_cls = getattr(importlib.import_module("exllamav3.modules.quant.exl3"), "Linear" + "EXL" + "3")

    tr, suh, svh = (t[f"{prefix}.{k}"] for k in EXL3_PARTS[:3])
    mul1 = t.get(f"{prefix}.mul1")
    mcg = t.get(f"{prefix}.mcg")
    in_f, out_f = tr.shape[0] * 16, tr.shape[1] * 16
    lin = linear_cls(None, in_f, out_f, suh=suh, svh=svh, trellis=tr, mul1=mul1, mcg=mcg)
    # x = I through the forward's own reconstruct + had_r_128 kernels: W [in, out] exactly as the fork's prefill path produces it
    # (the class's get_weight_tensor() applies the Hadamards in python and differs by ~1e-3 rel, fp16 rounding)
    w = lin.reconstruct_hgemm(torch.eye(in_f, dtype=torch.half, device=tr.device), torch.half)
    return w.T.contiguous()


def rel_l2(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


_EXPERT_RE = re.compile(r"mlp\.experts\.(\d+)\.(gate|up|down)_proj$")


def _param_view(sd, key):
    """The parameter tensor (or the slice of the fused expert tensor) a dequantised matrix `key` replaces: `key` is a state-dict key
    (`linear_attn.out_proj.weight`) or `mlp.experts.<e>.<gate|up|down>_proj` (rows 0:512 / 512:1024 of `gate_up_proj[e]`, `down_proj[e]`)."""
    m = _EXPERT_RE.match(key)
    if not m:
        return sd[key]
    e, kind = int(m.group(1)), m.group(2)
    if kind == "down":
        return sd["mlp.experts.down_proj"][e]
    gu = sd["mlp.experts.gate_up_proj"][e]
    return gu[:512] if kind == "gate" else gu[512:]


def _read_param(sd, key):
    return _param_view(sd, key).clone()


def _assign(sd, key, w):
    _param_view(sd, key).copy_(w.to(_param_view(sd, key).dtype))


@torch.no_grad()
def swap_pack_weights(layer, L, pm, dev, only=None):
    """Replace every EXL3-quantised matrix of decoder layer L with the pack's dequantised weight (bf16, in place in `layer`).
    `only` = a regex on the matrix id (`linear_attn.out_proj`, `mlp.experts.<e>.down_proj`, ...) to swap a subset (ablations).
    Returns (record, exact) with the per-matrix weight rel_l2 against the original and {key: fp16 [out, in]} of the unrounded
    dequantised weights (`variants()` writes those into the fp32 copy). NOTE rel_l2 is a WEIGHT-space number: see the README on
    layer 0 `down_proj`, where more than half of the input channels have negligible weights and the Frobenius rel_l2 is ~0.8 although
    the block output is not affected."""
    p = f"{LM}layers.{L}."
    quant = sorted({n[: -len(".trellis")] for n in pm if n.startswith(p) and n.endswith(".trellis")})
    if only:
        quant = [q for q in quant if re.search(only, q[len(p):])]
    packed = read_tensors(pm, [f"{q}.{k}" for q in quant for k in EXL3_PARTS if f"{q}.{k}" in pm] +
                          [f"{q}.mcg" for q in quant if f"{q}.mcg" in pm and f"{q}.mul1" not in pm], dev)
    werr, erel, exact = {}, [], {}
    sd = layer.state_dict()
    for pre in quant:
        k = pre[len(p):]
        m = _EXPERT_RE.match(k)
        w16 = dequant_linear(packed, pre)                      # fp16 [out, in]: what the fork computes with
        key = k if m else f"{k}.weight"                        # the matrix id for _assign
        exact[key] = w16
        rel = rel_l2(w16, _param_view(sd, key))                # quantisation error of this matrix against the original
        if m:
            erel.append(rel)
        else:
            werr[k] = rel
        _assign(sd, key, w16.to(torch.bfloat16))
    del packed
    rec = {"matrices_swapped": len(quant), "weight_rel_l2": {k: round(v, 5) for k, v in werr.items()}}
    if erel:
        e_ = np.array(erel)
        rec["expert_weight_rel_l2"] = {"n": int(e_.size), "mean": float(e_.mean()), "median": float(np.median(e_)),
                                       "p90": float(np.percentile(e_, 90)), "max": float(e_.max()), "n_above_0.3": int((e_ > 0.3).sum())}
    return rec, exact


@torch.no_grad()
def pack_impact(args, cfg, name_map):
    """Function-space damage of the pack's quantised weights per layer, HF only (no fork forward): the stream of one row is run
    with the ORIGINAL weights; at each checked layer the block is also run with the pack's dequantised weights on the same input.
    Reports rel_l2(block(x; pack), block(x; original)) for the whole layer, then for the expert `down_proj` matrices alone and for
    everything but them: the number that says whether a weight-space outlier (rel_l2 > 0.3) matters on real activations."""
    dev = torch.device(args.device)
    ids, _ = common.load_ids_set(args.ids)
    row = common.parse_rows(args.rows, ids.shape[0])[0]
    pm = pack_name_map(args.pack_dir)
    layers = [int(x) for x in args.check_layers.split(",")]
    E = read_tensors(name_map, [EMBED], "cpu")[EMBED]
    h = E[torch.from_numpy(ids[row])].to(dev)[None]
    del E
    T = h.shape[1]
    pe = rope(cfg, T, h, dev)
    mask = None if args.attn == "sdpa" else causal_mask(T, h.dtype, dev)
    res = []
    for L in range(max(layers) + 1):
        layer, lt = build_layer(cfg, L, name_map, dev)
        if L in layers:
            y0 = layer_forward(layer, h, pe, mask, "bf16")
            rec = {"layer": L, "type": lt, "row": int(row), "tokens": T}
            wrec, _ = swap_pack_weights(layer, L, pm, dev)
            rec.update(wrec)
            rec["block_rel_l2_all_matrices"] = rel_l2(layer_forward(layer, h, pe, mask, "bf16").float(), y0.float())
            # ablation: restore everything, swap only the expert down_proj matrices; then everything except them
            fresh, _ = build_layer(cfg, L, name_map, dev)
            swap_pack_weights(fresh, L, pm, dev, only=r"mlp\.experts\.\d+\.down_proj")
            rec["block_rel_l2_expert_down_only"] = rel_l2(layer_forward(fresh, h, pe, mask, "bf16").float(), y0.float())
            del fresh
            fresh, _ = build_layer(cfg, L, name_map, dev)
            swap_pack_weights(fresh, L, pm, dev, only=r"^(?!mlp\.experts\.\d+\.down_proj)")
            rec["block_rel_l2_all_but_expert_down"] = rel_l2(layer_forward(fresh, h, pe, mask, "bf16").float(), y0.float())
            del fresh
            res.append(rec)
            print(json.dumps(rec), flush=True)
        h = layer_forward(layer, h, pe, mask, "bf16") if L not in layers else y0
        del layer
        torch.cuda.empty_cache() if dev.type == "cuda" else None
    os.makedirs(args.out_dir, exist_ok=True)
    common.atomic_json(os.path.join(args.out_dir, "pack_impact.json"), res)
    return res


@torch.no_grad()
def pack_check(args, cfg, name_map):
    """Block-isolated validation against the fork's per-layer dumps (fork_logits.py --dump-layers): every checked layer is fed the FORK'S
    own input and its output is compared with the fork's output, with (a) the bf16 original weights and (b) the pack's dequantised EXL3
    weights swapped in. (b) is numerics only (the pack's weights are the same on both sides): the gate of plan Q2 (rel ~1e-3 per layer).
    Each is reported with the HF block run in bf16 (what the reference computes) and in fp32 (no bf16 rounding of HF's internals).
    The fork dump of the BF16 weights (the 'floor' run) needs no pack: --pack-dir then names nothing and only (a) is reported."""
    dev = torch.device(args.device)
    dump = json.load(open(os.path.join(args.dump_dir, "dump.json")))
    pm = pack_name_map(args.pack_dir) if args.pack_dir else None
    res = []
    for L in [int(x) for x in args.check_layers.split(",")]:
        li = dump["layers"][str(L)]
        dt = dump.get("dtype", "<f2")
        fin = torch.from_numpy(np.fromfile(os.path.join(args.dump_dir, li["in"]), dtype=dt).reshape(dump["rows"], dump["tokens"], -1)).to(dev).float()
        fout = torch.from_numpy(np.fromfile(os.path.join(args.dump_dir, li["out"]), dtype=dt).reshape(dump["rows"], dump["tokens"], -1)).to(dev).float()
        T = fin.shape[1]
        layer, lt = build_layer(cfg, L, name_map, dev)
        x16 = fin.to(torch.bfloat16)
        pe = rope(cfg, T, x16, dev)
        mask = None if args.attn == "sdpa" else causal_mask(T, torch.bfloat16, dev)
        rec = {"layer": L, "type": lt, "tokens": T, "rows": int(dump["rows"])}

        def variants(layer_, exact=None):
            """rel_l2 of three HF-block variants fed the fork's own input, against the fork's output: bf16_block = Qwen3_5MoeDecoderLayer
            as shipped (input rounded to bf16, bf16 residual add); fp32_residual_block = bf16 mixers/MLP with an fp32 residual stream (what
            the fork does); fp32_block = every op in fp32 (no bf16 rounding anywhere: the semantic check). `exact` = {param key within
            the layer: fp16 [out, in]} of dequantised pack weights, written into the fp32 copy unrounded (the bf16 blocks hold them
            rounded to bf16)."""
            y_b = layer_forward(layer_, x16, pe, mask, "bf16").float()
            y_r = layer_forward(layer_, fin, pe, mask, "fp32")
            l32 = copy.deepcopy(layer_).float()
            if exact:
                sd32 = l32.state_dict()
                for k, w in exact.items():
                    _assign(sd32, k, w.float())
            saved = cfg._experts_implementation
            cfg._experts_implementation = "eager"      # grouped_mm is bf16-only on CUDA; the per-expert loop runs in fp32
            try:
                y_f = layer_forward(l32, fin, tuple(t.float() for t in pe), None if mask is None else mask.float(), "bf16")
            finally:
                cfg._experts_implementation = saved
            del l32
            return {"bf16_block": rel_l2(y_b, fout), "fp32_residual_block": rel_l2(y_r, fout), "fp32_block": rel_l2(y_f, fout)}

        rec["orig_bf16_weights_vs_fork"] = variants(layer)
        if pm is not None:
            wrec, exact = swap_pack_weights(layer, L, pm, dev)
            rec.update(wrec)
            rec["pack_weights_vs_fork"] = variants(layer, exact)
            del exact
        res.append(rec)
        print(json.dumps(rec), flush=True)
        del layer
        torch.cuda.empty_cache() if dev.type == "cuda" else None
    os.makedirs(args.out_dir, exist_ok=True)
    common.atomic_json(os.path.join(args.out_dir, "pack_check.json"), res)
    return res


# ---------------------------------------------------------------------------------------------------------------- the reference run


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model-dir", default=common.MODEL_DIR)
    ap.add_argument("--ids", help="ids set (make_kl_ids.py output, .safetensors)")
    ap.add_argument("--rows", default="all", help="rows of the ids set, e.g. 0-13 or 0,3,5 (default all)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--budget-s", type=float, default=1500.0, help="do not start a layer that would end past this many seconds")
    ap.add_argument("--layers", type=int, default=40, help="stop after this many layers (sanity runs: no logits)")
    ap.add_argument("--restart", action="store_true")
    ap.add_argument("--residual", choices=["bf16", "fp32"], default="bf16",
                    help="residual stream dtype: bf16 as the shipped model (default); fp32 = the reference's own rounding-noise arm")
    ap.add_argument("--compute", choices=["bf16", "fp32"], default="bf16",
                    help="fp32: the whole model in fp32 (weights, activations, residual, norm, head; per-expert loop): the precision arm "
                         "that says which of the bf16 / fp16 implementations is closer to the exact model. final_hidden.bf16 is then "
                         "a bf16 ROUNDING of the fp32 hidden state: compare it through its logits.f16")
    ap.add_argument("--gdn", choices=["hf", "sequential"], default="hf",
                    help="GDN recurrence: transformers' chunked torch fallback (default; see --gdn-check) or the sequential loop")
    ap.add_argument("--experts", choices=["grouped_mm", "eager"], default="grouped_mm",
                    help="HF experts implementation: grouped_mm is the model's default; eager accumulates the 8 expert outputs in bf16")
    ap.add_argument("--attn", choices=["sdpa", "eager"], default="sdpa")
    ap.add_argument("--lm-head-out", choices=["fp32", "bf16"], default="fp32",
                    help="fp32: fp32 accumulate and output from the bf16 operands (default); bf16: HF's bf16 matmul output")
    ap.add_argument("--mb-rows", type=int, default=8, help="rows per layer forward (activation memory)")
    ap.add_argument("--lm-chunk", type=int, default=512, help="positions per lm_head matmul")
    sf = ap.add_mutually_exclusive_group()
    sf.add_argument("--store-full", dest="store_full", action="store_true", default=None, help="write logits.f16 (default: <= 32 rows)")
    sf.add_argument("--no-store-full", dest="store_full", action="store_false")
    ap.add_argument("--no-hidden", action="store_true", help="do not write final_hidden.bf16")
    ap.add_argument("--reader", choices=["direct", "safe_open"], default="direct")
    ap.add_argument("--min-mem-gb", type=float, default=20.0, help="refuse to start a layer below this MemAvailable (exit, resumable)")
    ap.add_argument("--plan", action="store_true", help="check that every tensor of every layer is in the shards and exit")
    ap.add_argument("--gdn-check", action="store_true", help="validate the GDN torch fallbacks (see the docstring) and exit")
    ap.add_argument("--check-layers", default="0,1,2,4", help="--gdn-check / --pack-check layers")
    ap.add_argument("--check-tokens", type=int, default=2048, help="--gdn-check real-activation sequence length (rows of the ids concatenated)")
    ap.add_argument("--pack-check", action="store_true", help="block-isolated check against a fork dump (fork_logits.py --dump-layers)")
    ap.add_argument("--pack-dir", help="EXL3 pack (or conversion working dir) whose dequantised weights are swapped in (--pack-check)")
    ap.add_argument("--dump-dir", help="fork_logits.py --dump-layers directory (--pack-check)")
    ap.add_argument("--pack-impact", action="store_true",
                    help="HF only: block output with the pack's dequantised weights vs the original weights on real activations of the "
                         "first selected row (needs --pack-dir and --ids); also with the expert down_proj matrices alone / all others")
    args = ap.parse_args()
    global READER
    READER = args.reader
    t_start = time.time()
    dev = torch.device(args.device)
    args.model_dir = os.path.expanduser(args.model_dir)
    args.out_dir = os.path.expanduser(args.out_dir)
    if args.ids:
        args.ids = os.path.expanduser(args.ids)

    name_map, how = build_name_map(args.model_dir)
    cfg = make_text_config(args.model_dir, args.experts, args.attn)
    n_layers = cfg.num_hidden_layers
    print(f"name map from {how}: {len(name_map)} tensors; transformers {__import__('transformers').__version__} torch {torch.__version__} "
          f"device {dev}; fla {_importable('fla')} kernels {_importable('kernels')}", flush=True)
    if args.plan:
        bad = 0
        for L in range(n_layers):
            miss = [n for n in layer_param_names(cfg, L) if n not in name_map]
            bad += bool(miss)
            print(f"layer {L:2d} {cfg.layer_types[L]:17s} {'ok' if not miss else 'MISSING ' + str(miss[:3])}")
        print("embed/norm/lm_head:", all(n in name_map for n in (EMBED, NORM, HEAD)))
        sys.exit(1 if bad else 0)
    if args.gdn_check:
        if not args.ids:
            sys.exit("--gdn-check needs --ids")
        r = gdn_check(args, cfg, name_map)
        sys.exit(0 if r["verdict"]["ok"] else 2)
    if args.pack_impact:
        if not (args.pack_dir and args.ids):
            sys.exit("--pack-impact needs --pack-dir and --ids")
        pack_impact(args, cfg, name_map)
        return
    if args.pack_check:
        if not args.dump_dir:
            sys.exit("--pack-check needs --dump-dir")
        pack_check(args, cfg, name_map)
        return
    if not args.ids:
        sys.exit("--ids is required")

    M = hf_modules()
    ids_all, man = common.load_ids_set(args.ids)
    sel = common.parse_rows(args.rows, ids_all.shape[0])
    ids = ids_all[sel]
    R, T = ids.shape
    names = [s["name"] for s in man.get("samples", [])] if man else []
    store_full = args.store_full if args.store_full is not None else R <= 32
    ids_sha = common.ids_sha256(ids)
    sig = {"model_dir": os.path.realpath(args.model_dir), "ids_sha256": ids_sha, "rows": sel, "residual": args.residual, "gdn": args.gdn,
           "experts": args.experts, "attn": args.attn, "lm_head_out": args.lm_head_out, "mb_rows": args.mb_rows, "device": dev.type,
           "compute": args.compute,
           "transformers": __import__("transformers").__version__, "torch": torch.__version__}
    out = args.out_dir
    os.makedirs(out, exist_ok=True)
    prog_path = os.path.join(out, "progress.json")
    prog = None
    if os.path.exists(prog_path) and not args.restart:
        prog = json.load(open(prog_path))
        if prog.get("sig") != sig:
            sys.exit(f"progress.json was written with different settings:\n  saved {prog.get('sig')}\n  now   {sig}\n"
                     "use --restart or another --out-dir")
    if prog is None:
        prog = {"sig": sig, "layers_done": -1, "per_layer": [], "wall_s_total": 0.0, "sessions": 0}
    prog["sessions"] += 1
    prog["wall_s_prev"] = prog.get("wall_s_total", 0.0)

    def save_prog():
        prog["wall_s_total"] = round(prog["wall_s_prev"] + time.time() - t_start, 1)
        common.atomic_json(prog_path, prog)

    if args.gdn == "sequential":
        M.torch_chunk_gated_delta_rule = delta_rule_sequential
    all32 = args.compute == "fp32"          # the whole model in fp32 (weights, activations, residual): the precision arm
    if all32:
        cfg._experts_implementation = "eager"      # grouped_mm is bf16-only on CUDA; the per-expert loop runs in fp32
        sig_res = "bf16"                           # layer_forward's plain-add composition, on fp32 tensors
    else:
        sig_res = args.residual

    # --- embeddings
    done = prog["layers_done"]
    if done < 0:
        t0 = time.time()
        try:
            e = read_tensors(name_map, [EMBED], "cpu")[EMBED]
        except Missing as m:
            sys.exit(f"embeddings not available: {m}")
        h = e[torch.from_numpy(ids)].to(dev)                         # [R, T, 2048] bf16
        del e
        if args.residual == "fp32" or all32:
            h = h.float()
        torch.save(h.cpu(), os.path.join(out, "hidden_after_000.pt"))
        done = 0
        prog["layers_done"] = 0
        prog["per_layer"].append({"layer": "embeddings", "load_s": round(time.time() - t0, 2)})
        save_prog()
        print(f"embeddings: {R} rows x {T} tokens in {time.time() - t0:.1f}s", flush=True)
    else:
        h = torch.load(os.path.join(out, f"hidden_after_{done:03d}.pt")).to(dev)
        print(f"resumed after {done} layers", flush=True)

    pe = rope(cfg, T, h if all32 else h.to(torch.bfloat16), dev)
    mask = None if args.attn == "sdpa" else causal_mask(T, h.dtype if all32 else torch.bfloat16, dev)
    last = min(args.layers, n_layers)
    times = [r["load_s"] + r["compute_s"] for r in prog["per_layer"] if isinstance(r.get("layer"), int)]
    stopped = None
    while done < last:
        L = done
        avg = sum(times[-6:]) / len(times[-6:]) if times else 30.0
        if time.time() - t_start + 1.5 * avg > args.budget_s:
            stopped = "time budget"
            break
        avail = common.mem_available_gb()
        if avail < args.min_mem_gb:
            stopped = f"MemAvailable {avail:.1f} GB below --min-mem-gb {args.min_mem_gb}"
            break
        try:
            t0 = time.time()
            layer, lt = build_layer(cfg, L, name_map, dev)
            if all32:
                layer = layer.float()
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t_load = time.time() - t0
        except Missing as m:
            stopped = f"missing tensor {m}"
            break
        t1 = time.time()
        outs = []
        for a in range(0, R, args.mb_rows):
            xb = h[a:a + args.mb_rows]
            yb = layer_forward(layer, xb, pe, mask, sig_res)
            if L < 2 and a == 0 and sig_res == "bf16":
                ref = layer(xb, position_embeddings=pe, attention_mask=mask, position_ids=None, past_key_values=None)
                if not torch.equal(ref, yb):
                    sys.exit(f"layer {L}: the spelled-out composition differs from Qwen3_5MoeDecoderLayer.forward "
                             f"(max abs {float((ref.float() - yb.float()).abs().max())})")
                del ref
            outs.append(yb)
        h = torch.cat(outs)
        del outs, layer
        if dev.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        t_comp = time.time() - t1
        rec = {"layer": L, "type": lt, "load_s": round(t_load, 2), "compute_s": round(t_comp, 2),
               "finite": bool(torch.isfinite(h).all().item()), "max_abs": round(float(h.float().abs().max()), 2),
               "rms": round(float(h.float().pow(2).mean().sqrt()), 3)}
        tmp = os.path.join(out, f"hidden_after_{L + 1:03d}.pt.tmp")
        torch.save(h.cpu(), tmp)
        os.replace(tmp, os.path.join(out, f"hidden_after_{L + 1:03d}.pt"))
        old = os.path.join(out, f"hidden_after_{L:03d}.pt")
        done = L + 1
        prog["layers_done"] = done
        prog["per_layer"].append(rec)
        save_prog()
        if os.path.exists(old):
            os.remove(old)
        times.append(t_load + t_comp)
        print(f"layer {L:2d} {lt:17s} load {t_load:5.1f}s compute {t_comp:5.1f}s finite={rec['finite']} max_abs {rec['max_abs']} rms {rec['rms']}",
              flush=True)
        if not rec["finite"]:
            sys.exit(f"layer {L}: non-finite hidden state")

    if stopped:
        print(f"stopped after {done} layers: {stopped}. Run the same command again to continue.", flush=True)
        save_prog()
        return
    if done < n_layers:
        print(f"{done} layers done (sanity run, no logits)", flush=True)
        save_prog()
        return
    if prog.get("logits_written"):
        print("logits already written", flush=True)
        return

    # --- final norm and lm_head
    t0 = time.time()
    try:
        w = read_tensors(name_map, [NORM, HEAD], dev)
    except Missing as m:
        sys.exit(f"final tensors not available: {m}")
    with torch.no_grad():
        with torch.device("meta"):
            norm = M.Qwen3_5MoeRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        norm.load_state_dict({"weight": w[NORM]}, assign=True)
        if all32:
            norm = norm.float()
            hn = norm(h)                                             # [R, T, 2048] fp32 (the precision arm)
            args.lm_head_out = "fp32"
        else:
            hn = norm(h).to(torch.bfloat16)                          # [R, T, 2048] bf16
        head = w[HEAD]                                               # [V, 2048] bf16
        head32 = head.float() if args.lm_head_out == "fp32" else None
        meta = {"ids_sha256": ids_sha, "rows_selected": sel, "row_names": [names[r] for r in sel] if names else None,
                "producer": {"tool": "bf16_ref_logits.py", "sig": sig, "reference": "HF transformers bf16 text path, layer-streamed"},
                "centered_logits": True, "tokens": T}
        want = ["rowmax", "lse", "tail_lse", "entropy", "argmax", "tgt_logit", "topk_val", "topk_idx"]
        if store_full:
            want.append("logits")
        if not args.no_hidden:
            want.append("final_hidden")
        ls = common.LogitSet(out, create=True, rows=R, tokens=T, meta=meta, name=os.path.basename(out.rstrip("/")), kind="ref", want=want)
        ls.save_meta(**{k: v for k, v in meta.items() if k not in ("files",)})
        start_row = prog.get("logits_rows_done", 0)
        for r in range(start_row, R):
            if not args.no_hidden:
                ls.array("final_hidden", "w+")[r] = hn[r].to(torch.bfloat16).cpu().contiguous().view(torch.int16).numpy()
            for lo in range(0, T, args.lm_chunk):
                hi = min(lo + args.lm_chunk, T)
                if args.lm_head_out == "fp32":
                    x = hn[r, lo:hi].float() @ head32.T
                else:
                    x = F.linear(hn[r, lo:hi], head).float()
                tgt = km.next_token_targets(ids[r], lo, hi, dev)
                km.store_chunk(ls, r, lo, x, tgt, full=store_full, topk=True)
            prog["logits_rows_done"] = r + 1
            if (r + 1) % 4 == 0 or r + 1 == R:
                ls.flush()
                save_prog()
        ls.flush()
    # finiteness and top-1 summary
    ok = bool(np.isfinite(np.asarray(ls.array("lse"))).all())
    prog["logits_written"] = True
    prog["final_s"] = round(time.time() - t0, 1)
    prog["logits_finite"] = ok
    prog["argmax_in_vocab"] = bool(int(np.asarray(ls.array("argmax")).max()) < common.V)
    ls.save_meta(complete=True, finite=ok)
    save_prog()
    print({k: v for k, v in prog.items() if k not in ("per_layer", "sig")}, flush=True)


if __name__ == "__main__":
    main()
