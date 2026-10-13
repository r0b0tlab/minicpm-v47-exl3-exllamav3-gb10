"""KL / agreement metrics of the MiniCPM-V-4.7 text measurement: pure torch + numpy, CPU or CUDA, unit-tested on synthetic logits
(`test_kl_metrics.py`). Used by `kl_compare.py` (two logit sets on disk) and `fork_logits.py --compare-ref` (candidate logits
streamed from the model straight into the metrics, nothing stored).

Conventions (see the pack's model card): fp32 log-softmax over the full 248,144-token vocabulary, KL terms in fp32 summed in fp64,
nats. "p" is the reference (bf16 HF), "q" the candidate. kl_pq = KL(p || q), kl_qp = KL(q || p). Top-1 = argmax equality; the top-k
overlap is |topk(p) & topk(q)| / k for k = 1..5 ("top-5 overlap" for k = 5); `ex<k>` is the stricter "the ordered top-k
lists are equal" of the fork's eval/model_diff.py. nll is -log q(next token), so ppl = exp(mean nll) over the 1,023 positions of a
row that have a next token.

Exact vs top-k reference. `position_metrics` needs the reference's full logits (stored fp16, or recomputed from the stored post-norm
hidden state and the bf16 lm_head: `HiddenSource`). `lumped_metrics` needs only the reference's top-K logits + its logsumexp and the
logsumexp of everything outside the top-K (`tail_lse`): both distributions are coarse-grained into the K reference-top-K tokens plus one
"everything else" bucket and KL is computed on that K+1 outcome space. By the data-processing inequality that is a LOWER bound of the
true KL in each direction; the bound is tight when the reference's tail mass (outside the top-256) is small, and the reverse direction
additionally cannot see where q puts tail mass relative to p's tail shape. `kl_compare.py --check-lumped` measures the ratio
lumped / exact on rows that have both (Leg A) and writes it next to the Leg B numbers.
"""
import math

import numpy as np
import torch

KS = (1, 2, 3, 4, 5)
CH = 64            # positions per chunk: 64 x 248,144 x 4 B = 63.5 MB per fp32 array
V = 248144
HIDDEN_DIM = 2048

PER_POSITION_KEYS = ["kl_pq", "kl_qp", "top1"] + [f"ov{k}" for k in KS] + [f"ex{k}" for k in KS] + [
    "nll_p", "nll_q", "ent_p", "ent_q", "gap_p", "mad"]


# ---------------------------------------------------------------------------------------------------------------- per-position metrics


def _topk_agreement(p_idx, q_idx):
    """p_idx, q_idx [n, 5] -> (overlap [n, 5], exact [n, 5]) for k = 1..5."""
    eq = p_idx[:, :, None] == q_idx[:, None, :]  # [n, 5, 5]: eq[:, i, j] = p's i-th equals q's j-th
    ov, ex = [], []
    for k in KS:
        hits = eq[:, :k, :k].any(dim=-1).sum(dim=-1)  # how many of p's top-k are inside q's top-k
        ov.append(hits.to(torch.float64) / k)
        ex.append((p_idx[:, :k] == q_idx[:, :k]).all(dim=-1).to(torch.float64))
    return torch.stack(ov, dim=-1), torch.stack(ex, dim=-1)


def _row_stats(x):
    """log-softmax (fp32), logsumexp, probabilities of a [n, V] fp32 logit tensor."""
    lse = torch.logsumexp(x, dim=-1)
    lp = x - lse[:, None]
    return lse, lp


def _target_nll(x, lse, tgt):
    """-log softmax(x)[tgt] with tgt [n] int64 (negative = no target -> NaN)."""
    safe = tgt.clamp_min(0)
    v = lse - x.gather(1, safe[:, None])[:, 0]
    return torch.where(tgt >= 0, v.to(torch.float64), torch.full_like(v, float("nan"), dtype=torch.float64))


@torch.no_grad()
def position_metrics(p, q, tgt=None):
    """Exact metrics of candidate logits `q` against reference logits `p`, both [n, V] float32 (raw or any common shift per row for
    everything except `mad`). tgt [n] int64 = next-token ids (-1 where none). Returns {key: float64 tensor [n]}."""
    assert p.shape == q.shape and p.dim() == 2, (p.shape, q.shape)
    n = p.shape[0]
    if tgt is None:
        tgt = torch.full((n,), -1, dtype=torch.long, device=p.device)
    lse_p, lp = _row_stats(p)
    lse_q, lq = _row_stats(q)
    pp = lp.exp()
    qq = lq.exp()
    kl_pq = (pp * (lp - lq)).sum(-1, dtype=torch.float64)
    kl_qp = (qq * (lq - lp)).sum(-1, dtype=torch.float64)
    tp, ip = torch.topk(p, 5, dim=-1)
    tq, iq = torch.topk(q, 5, dim=-1)
    ov, ex = _topk_agreement(ip, iq)
    out = {
        "kl_pq": kl_pq,
        "kl_qp": kl_qp,
        "top1": (ip[:, 0] == iq[:, 0]).to(torch.float64),
        "nll_p": _target_nll(p, lse_p, tgt),
        "nll_q": _target_nll(q, lse_q, tgt),
        "ent_p": -(pp * lp).sum(-1, dtype=torch.float64),
        "ent_q": -(qq * lq).sum(-1, dtype=torch.float64),
        "gap_p": (tp[:, 0] - tp[:, 1]).to(torch.float64),
        "mad": (p - q).abs().mean(-1, dtype=torch.float64),
    }
    for j, k in enumerate(KS):
        out[f"ov{k}"] = ov[:, j]
        out[f"ex{k}"] = ex[:, j]
    return out


@torch.no_grad()
def lumped_metrics(p_val, p_idx, p_lse, p_tail_lse, p_ent, p_tgt_logit, q, tgt=None):
    """Metrics against a TOP-K reference store (see the module docstring): p_val [n, K] fp32 raw logits (descending), p_idx [n, K] int64,
    p_lse [n] the reference's logsumexp over all V, p_tail_lse [n] logsumexp over the entries OUTSIDE the K, p_ent [n] its entropy,
    p_tgt_logit [n] its raw logit of the next token (NaN where none); q [n, V] float32 candidate logits.
    kl_pq / kl_qp are the K+1-outcome (top-K + tail bucket) KLs: lower bounds of the exact values. `mad` is NaN."""
    n, K = p_val.shape
    if tgt is None:
        tgt = torch.full((n,), -1, dtype=torch.long, device=q.device)
    lse_q, lq = _row_stats(q)
    lp_top = (p_val - p_lse[:, None]).double()                     # log p_i, i in top-K
    lq_top = lq.gather(1, p_idx).double()                           # log q_i
    lq_tail = (torch.logsumexp(q.scatter(1, p_idx, float("-inf")), dim=-1) - lse_q).double()   # log of q's mass outside the set
    lp_tail = (p_tail_lse - p_lse).double()
    p_top, q_top = lp_top.exp(), lq_top.exp()
    p_t, q_t = lp_tail.exp(), lq_tail.exp()

    def xlr(a, la, lb):
        # a * (log a - log b) with 0 * anything = 0 (an empty tail bucket has log mass -inf)
        return torch.where(a > 0, a * (la - lb), torch.zeros_like(a))

    kl_pq = (p_top * (lp_top - lq_top)).sum(-1) + xlr(p_t, lp_tail, lq_tail)
    kl_qp = (q_top * (lq_top - lp_top)).sum(-1) + xlr(q_t, lq_tail, lp_tail)
    tq, iq = torch.topk(q, 5, dim=-1)
    ov, ex = _topk_agreement(p_idx[:, :5], iq)
    qq = lq.exp()
    safe = tgt.clamp_min(0)
    nll_p = torch.where(tgt >= 0, (p_lse - p_tgt_logit).double(), torch.full_like(kl_pq, float("nan")))
    out = {
        "kl_pq": kl_pq.clamp_min(0.0),
        "kl_qp": kl_qp.clamp_min(0.0),
        "top1": (p_idx[:, 0] == iq[:, 0]).to(torch.float64),
        "nll_p": nll_p,
        "nll_q": _target_nll(q, lse_q, tgt),
        "ent_p": p_ent.double(),
        "ent_q": -(qq * lq).sum(-1, dtype=torch.float64),
        "gap_p": (p_val[:, 0] - p_val[:, 1]).double(),
        "mad": torch.full_like(kl_pq, float("nan")),
    }
    for j, k in enumerate(KS):
        out[f"ov{k}"] = ov[:, j]
        out[f"ex{k}"] = ex[:, j]
    return out


# ---------------------------------------------------------------------------------------------------------------- reference sources


class SetSource:
    """Serves chunks of one LogitSet on `device`. mode: 'full' (logits.f16 + rowmax), 'hidden' (final_hidden.bf16 x the bf16 lm_head,
    fp32 matmul: exact full-vocabulary logits of the reference at no storage), 'topk' (stats only: lumped metrics), or 'auto'."""

    def __init__(self, ls, device="cpu", mode="auto", lm_head=None):
        self.ls = ls
        self.device = torch.device(device)
        if mode == "auto":
            mode = "full" if ls.has("logits") else ("hidden" if (ls.has("final_hidden") and lm_head is not None) else "topk")
        self.mode = mode
        self.exact = mode in ("full", "hidden")
        self._w = None
        if mode == "hidden":
            if lm_head is None or not ls.has("final_hidden"):
                raise ValueError("mode 'hidden' needs final_hidden.bf16 in the set and --lm-head")
            self._w = (lm_head if torch.is_tensor(lm_head) else torch.load(lm_head)).to(self.device).float()   # [V, 2048]
        if mode == "topk" and not (ls.has("topk_val") and ls.has("topk_idx") and ls.has("lse") and ls.has("tail_lse")):
            raise ValueError(f"{ls.path}: top-k mode needs topk_val/topk_idx/lse/tail_lse")
        if mode == "full" and not ls.has("logits"):
            raise ValueError(f"{ls.path}: no logits.f16")

    def logits(self, r, lo, hi):
        if self.mode == "full":
            x = torch.from_numpy(np.array(self.ls.array("logits")[r, lo:hi])).to(self.device).float()
            return x + torch.from_numpy(np.array(self.ls.array("rowmax")[r, lo:hi])).to(self.device)[:, None]
        if self.mode == "hidden":
            h = np.array(self.ls.array("final_hidden")[r, lo:hi])
            h = torch.from_numpy(h).to(self.device).view(torch.bfloat16).float()
            return h @ self._w.T
        raise RuntimeError("top-k source has no full logits")

    def topk(self, r, lo, hi):
        g = lambda k: torch.from_numpy(np.array(self.ls.array(k)[r, lo:hi])).to(self.device)
        return {"val": g("topk_val"), "idx": g("topk_idx").long(), "lse": g("lse"), "tail_lse": g("tail_lse"), "ent": g("entropy"),
                "tgt_logit": g("tgt_logit")}


def compare_chunk(ref, r, lo, hi, q, tgt):
    """Metrics of candidate logits q [hi - lo, V] float32 (on ref.device) against `ref` (a SetSource) for row r positions lo..hi."""
    if ref.exact:
        return position_metrics(ref.logits(r, lo, hi), q, tgt)
    t = ref.topk(r, lo, hi)
    return lumped_metrics(t["val"], t["idx"], t["lse"], t["tail_lse"], t["ent"], t["tgt_logit"], q, tgt)


@torch.no_grad()
def store_chunk(ls, r, lo, x, tgt, full=True, topk=True, topk_k=256):
    """Write one chunk of a producer's logits `x` [n, V] float32 (raw) into LogitSet `ls` (row r, positions lo..): the per-position
    statistics always (rowmax, lse, tail_lse, entropy, argmax, tgt_logit, from the fp32 values before any fp16 rounding), the top-K
    store when `topk`, the row-centred fp16 logits when `full`. tgt [n] int64 = next-token ids (-1 where none)."""
    n = x.shape[0]
    rowmax = x.max(dim=-1).values
    lse = torch.logsumexp(x, dim=-1)
    lp = x - lse[:, None]
    ent = -(lp.exp() * lp).sum(-1)
    tv, ti = torch.topk(x, topk_k, dim=-1)
    tail = torch.logsumexp(x.scatter(1, ti, float("-inf")), dim=-1)
    tl = x.gather(1, tgt.clamp_min(0)[:, None])[:, 0]
    tl = torch.where(tgt >= 0, tl, torch.full_like(tl, float("nan")))
    w = lambda k: ls.array(k, "w+")
    sl = slice(lo, lo + n)
    cpu = lambda t: t.detach().cpu().numpy()
    w("rowmax")[r, sl] = cpu(rowmax)
    w("lse")[r, sl] = cpu(lse)
    w("tail_lse")[r, sl] = cpu(tail)
    w("entropy")[r, sl] = cpu(ent)
    w("argmax")[r, sl] = cpu(ti[:, 0].to(torch.int32))
    w("tgt_logit")[r, sl] = cpu(tl)
    if topk:
        w("topk_val")[r, sl, :topk_k] = cpu(tv)
        w("topk_idx")[r, sl, :topk_k] = cpu(ti.to(torch.int32))
    if full:
        w("logits")[r, sl] = cpu((x - rowmax[:, None]).half())


def next_token_targets(ids_row, lo, hi, device="cpu"):
    """int64 [hi - lo]: ids[p + 1] for position p, -1 at the last position of the row."""
    n = len(ids_row)
    t = np.full(hi - lo, -1, dtype=np.int64)
    for j, p in enumerate(range(lo, hi)):
        if p + 1 < n:
            t[j] = int(ids_row[p + 1])
    return torch.from_numpy(t).to(device)


def fresh_arrays(rows, tokens):
    return {k: np.full((rows, tokens), np.nan, dtype=np.float64) for k in PER_POSITION_KEYS}


def put(arrays, r, lo, m):
    for k, v in m.items():
        arrays[k][r, lo:lo + v.shape[0]] = v.double().cpu().numpy()


# ---------------------------------------------------------------------------------------------------------------- aggregation


def stats(v):
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    return {"mean": float(v.mean()), "median": float(np.median(v)), "p95": float(np.percentile(v, 95)),
            "p99": float(np.percentile(v, 99)), "max": float(v.max())}


def _mean(v):
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else None


def _nanmean_cols(x):
    """Column means of [R, T] ignoring NaN; NaN where a column has no value (no warnings)."""
    ok = np.isfinite(x)
    cnt = ok.sum(0)
    s = np.where(ok, x, 0.0).sum(0)
    return np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)


def summarize(m, row_names=None, row_kinds=None, rows=None, bucket=64, mask=None):
    """Aggregate per-position arrays {key: [R, T]} into the kl.json block of one comparison. `rows` selects row indices (default
    all). `mask` [R, T] bool (over ALL rows of `m`) restricts the statistics to those positions (Leg C: continuation rows only); NaN
    entries (not measured) are always ignored. Returns a JSON-able dict."""
    R, T = m["kl_pq"].shape
    sel = list(range(R)) if rows is None else list(rows)
    g = {k: np.asarray(v)[sel] for k, v in m.items()}
    valid = np.isfinite(g["kl_pq"])
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)[sel]
    for k in g:
        g[k] = np.where(valid, g[k], np.nan)
    n = int(valid.sum())
    if n == 0:
        return {"rows": len(sel), "tokens_per_row": T, "positions": 0}
    kl, klr, ent = g["kl_pq"][valid], g["kl_qp"][valid], g["ent_p"][valid]
    top1 = g["top1"][valid]
    out = {"rows": len(sel), "tokens_per_row": T, "positions": n}
    out["kl_ref_to_q"] = stats(kl)
    out["kl_q_to_ref"] = stats(klr)
    out["top1_agreement"] = _mean(top1)
    out["topk_overlap"] = {str(k): _mean(g[f"ov{k}"][valid]) for k in KS}
    out["topk_exact_order"] = {str(k): _mean(g[f"ex{k}"][valid]) for k in KS}
    out["mean_abs_logit_diff"] = _mean(g["mad"][valid])
    nllp, nllq = g["nll_p"], g["nll_q"]
    ok = np.isfinite(nllp) & np.isfinite(nllq)
    if ok.any():
        mp, mq = float(nllp[ok].mean()), float(nllq[ok].mean())
        out["ppl"] = {"ref": math.exp(mp), "q": math.exp(mq), "ratio_q_over_ref": math.exp(mq - mp), "log_ratio": mq - mp, "tokens": int(ok.sum())}
    out["ref_entropy_nats"] = stats(ent)
    buckets = {"entropy<0.1": ent < 0.1, "0.1<=entropy<1": (ent >= 0.1) & (ent < 1.0), "entropy>=1": ent >= 1.0}
    out["by_ref_entropy"] = {}
    for name, bm in buckets.items():
        if bm.any():
            out["by_ref_entropy"][name] = {"positions": int(bm.sum()), "share_of_positions": float(bm.mean()),
                                           "kl_ref_to_q_mean": float(kl[bm].mean()), "kl_q_to_ref_mean": float(klr[bm].mean()),
                                           "top1_agreement": float(top1[bm].mean()),
                                           "share_of_total_kl": float(kl[bm].sum() / max(kl.sum(), 1e-300))}
    srt = np.sort(kl)[::-1]
    out["share_of_total_kl_in_worst_1pct_positions"] = float(srt[: max(1, int(round(0.01 * n)))].sum() / max(srt.sum(), 1e-300))
    flips = top1 == 0
    out["top1_flips"] = {"count": int(flips.sum()), "of_positions": n}
    if flips.any():
        gap = g["gap_p"][valid][flips]
        out["top1_flips"]["with_ref_top1_top2_gap_below_0.5"] = int((gap < 0.5).sum())
        out["top1_flips"]["median_ref_gap_at_flips"] = float(np.median(gap))
    # per row
    pr = []
    for j, r in enumerate(sel):
        vj = valid[j]
        base = {"row": int(r), "name": None if row_names is None else row_names[r], "kind": None if row_kinds is None else row_kinds[r],
                "positions": int(vj.sum())}
        if vj.any():
            base.update({"kl_ref_to_q_mean": _mean(g["kl_pq"][j][vj]), "kl_ref_to_q_median": float(np.median(g["kl_pq"][j][vj])),
                         "kl_ref_to_q_max": float(g["kl_pq"][j][vj].max()), "kl_q_to_ref_mean": _mean(g["kl_qp"][j][vj]),
                         "top1_agreement": _mean(g["top1"][j][vj]), "top5_overlap": _mean(g["ov5"][j][vj]),
                         "nll_ref": _mean(g["nll_p"][j]), "nll_q": _mean(g["nll_q"][j]), "ref_entropy_mean": _mean(g["ent_p"][j][vj])})
        pr.append(base)
    out["per_row"] = pr
    rm = [(p["kl_ref_to_q_mean"], p["positions"]) for p in pr if p.get("kl_ref_to_q_mean") is not None]
    if len(rm) > 1:
        # a handful of chaotic positions in one row can carry most of the mean (Leg A chat row): show the means without it
        w = max(range(len(rm)), key=lambda i: rm[i][0])
        rest = [x for i, x in enumerate(rm) if i != w]
        out["row_mean_kl_median"] = float(np.median([x[0] for x in rm]))
        out["mean_kl_excluding_worst_row"] = float(sum(x[0] * x[1] for x in rest) / max(sum(x[1] for x in rest), 1))
        out["worst_row"] = pr[[i for i, p in enumerate(pr) if p.get("kl_ref_to_q_mean") == rm[w][0]][0]]["row"]
    if row_kinds is not None:
        kinds = {}
        for j, r in enumerate(sel):
            if valid[j].any():
                kinds.setdefault(row_kinds[r], []).append(j)
        out["per_kind"] = {k: {"rows": len(js), "kl_ref_to_q_mean": float(np.nanmean(g["kl_pq"][js])),
                               "kl_q_to_ref_mean": float(np.nanmean(g["kl_qp"][js])), "top1_agreement": float(np.nanmean(g["top1"][js]))}
                           for k, js in sorted(kinds.items())}
    # per position: the mean over rows at every position, and bucket means
    pos_kl = _nanmean_cols(g["kl_pq"])
    pos_top1 = _nanmean_cols(g["top1"])
    edges = sorted({min(e, T) for e in [0, 8, 64] + list(range(64 + bucket, T + bucket, bucket)) + [T]})

    def bmean(x, a, b):
        v = x[a:b]
        v = v[np.isfinite(v)]
        return float(v.mean()) if v.size else None

    out["per_position"] = {
        "buckets": [{"from": int(a), "to": int(b), "kl_ref_to_q_mean": bmean(pos_kl, a, b), "top1_agreement": bmean(pos_top1, a, b)}
                    for a, b in zip(edges[:-1], edges[1:])],
        "kl_ref_to_q_mean_by_position": [None if not np.isfinite(x) else round(float(x), 6) for x in pos_kl],
    }
    return out


def paired_diff(a, b, mask=None):
    """Mean and standard error of the per-position KL difference a - b ([R, T] arrays), rows as the unit of independence."""
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    if mask is not None:
        diff = np.where(np.asarray(mask, dtype=bool), diff, np.nan)
    d = _nanmean_cols(diff.T)
    d = d[np.isfinite(d)]
    return {"mean": float(d.mean()), "se": float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else None, "rows": int(len(d))}
