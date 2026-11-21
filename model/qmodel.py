"""Integer-exact TTLM model.

The forward pass below uses only operations the Gigatron can do cheaply:
ternary weights (adds/subs), shifts, clamps, small lookup tables and 8-bit
multiplies (quarter-square table) inside attention. Every intermediate value
is an integer below 2^24, so the float32 forward is *exactly* the integer
computation: this file is the specification that the C code must match bit
for bit. Training uses straight-through estimators (STE) for every rounding.

Units
- residual stream h : int, |h| <= 2047, value = h / 16
- activations a     : int8 [-127, 127] (q, k: int6 [-31, 31])
- weights           : {-1, 0, +1}, one power-of-two output shift per matrix
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

CTX = 24
HMAX = 2047          # residual clamp (12 bits)
QMAX = 31            # q, k range (int6) so q.k over 16 dims fits int16
ZMAX = 31            # softmax table length - 1
EXP = torch.tensor([round(64 * 2 ** (-z / 4)) for z in range(ZMAX + 1)], dtype=torch.float32)
RCP = torch.tensor([round(16384 / (8 * w + 4)) for w in range(CTX * 64 // 8 + 1)], dtype=torch.float32)
RCP[:8] = RCP[8]     # W >= 64 always (the max position has weight 64)


EXACT = True        # False: same architecture, no rounding (diagnostics only)


def rnd(x):
    return x + (torch.round(x) - x).detach() if EXACT else x


def flr(x):
    return x + (torch.floor(x) - x).detach() if EXACT else x


def tern(w):
    """Weights in {-1, 0, +1} (BitNet b1.58 rounding, scale dropped: it lives in the shifts)."""
    s = w.abs().mean().clamp(min=1e-5)
    q = (w / s).round().clamp(-1, 1)
    return w / s + (q - w / s).detach()


def shift(y, ls):
    """y * 2^round(ls), floored: an arithmetic shift on the Gigatron."""
    return flr(y * torch.pow(2.0, rnd(ls)))


def lookup(table, idx_cont, soft):
    """Table lookup with STE: forward = table[floor(idx)], backward = smooth soft()."""
    s = soft(idx_cont)
    if not EXACT:
        return s
    i = torch.floor(idx_cont).clamp(0, len(table) - 1).long().detach()
    return s + (table.to(s.device)[i] - s).detach()


def norm(h):
    """Power-of-two mean-abs norm: a = h * 2^(10 - floor(log2(sum|h|))), int8."""
    S = h.abs().sum(-1, keepdim=True).clamp(min=1)
    e = torch.floor(torch.log2(S.detach()))
    return flr(h * torch.pow(2.0, 10 - e)).clamp(-127, 127)


class TLinear(nn.Module):
    def __init__(self, i, o, ls=-3.0):
        super().__init__()
        self.w = nn.Parameter(torch.randn(o, i) / math.sqrt(i))
        self.ls = nn.Parameter(torch.tensor(ls))

    def forward(self, a):            # raw integer sums, |y| <= in * 127
        return F.linear(a, tern(self.w))


class Block(nn.Module):
    def __init__(self, d, h, m):
        super().__init__()
        self.h, self.dh = h, d // h
        self.qkv = TLinear(d, 3 * d)
        self.lq = nn.Parameter(torch.tensor(-3.0))
        self.lk = nn.Parameter(torch.tensor(-3.0))
        self.lv = nn.Parameter(torch.tensor(-2.0))
        self.latt = nn.Parameter(torch.tensor(7.0))   # z = (max - s) >> latt
        self.o = TLinear(d, d)
        self.up = TLinear(d, m)
        self.down = TLinear(m, d)

    def attend(self, a):
        B, T, D = a.shape
        y = self.qkv(a)
        q, k, v = y.split(D, -1)
        q = shift(q, self.lq).clamp(-QMAX, QMAX)
        k = shift(k, self.lk).clamp(-QMAX, QMAX)
        v = shift(v, self.lv).clamp(-127, 127)
        sh = lambda t: t.view(B, T, self.h, self.dh).transpose(1, 2)     # B h T dh
        q, k, v = sh(q), sh(k), sh(v)
        s = q @ k.transpose(-1, -2)                                         # exact ints, |s| <= 15376
        mask = torch.ones(T, T, dtype=torch.bool, device=a.device).tril()
        s = s.masked_fill(~mask, -1e9)
        m = s.max(-1, keepdim=True).values.detach()
        zc = (m - s) * torch.pow(2.0, -rnd(self.latt))
        e = lookup(EXP, zc.clamp(max=ZMAX + 0.5), lambda z: 64 * torch.pow(2.0, -z / 4))
        e = e.masked_fill(~mask, 0)
        W = e.sum(-1, keepdim=True)
        R = lookup(RCP, W / 8, lambda w8: 16384 / (8 * w8 + 4))
        p = flr(e * R / 256)
        o = flr((p @ v) / 64).clamp(-127, 127)
        return o.transpose(1, 2).reshape(B, T, D)

    def forward(self, h):
        h = (h + shift(self.o(self.attend(norm(h))), self.o.ls)).clamp(-HMAX, HMAX)
        u = shift(F.relu(self.up(norm(h))), self.up.ls).clamp(0, 127)
        return (h + shift(self.down(u), self.down.ls)).clamp(-HMAX, HMAX)


class QMoth(nn.Module):
    def __init__(self, V, d=32, L=2, h=2, m=64):
        super().__init__()
        self.cfg = dict(V=V, d=d, L=L, h=h, m=m)
        self.emb = nn.Parameter(torch.randn(V, d))
        self.lemb = nn.Parameter(torch.tensor(4.0))      # token embedding -> residual: +-2^lemb
        self.pos = nn.Parameter(torch.randn(CTX, d) * 8)  # int8, residual units
        self.blocks = nn.ModuleList(Block(d, h, m) for _ in range(L))
        self.lout = nn.Parameter(torch.tensor(-7.0))     # logit temperature (training only)

    def logits_int(self, idx):
        E = tern(self.emb)
        T = idx.shape[1]
        P = rnd(self.pos[:T]).clamp(-127, 127)
        h = (E[idx] * torch.pow(2.0, rnd(self.lemb)) + P).clamp(-HMAX, HMAX)
        for b in self.blocks:
            h = b(h)
        return norm(h) @ E.t()                             # exact ints

    def forward(self, idx):
        return self.logits_int(idx) * torch.pow(2.0, self.lout)

    # --- export -----------------------------------------------------------------
    @torch.no_grad()
    def export(self):
        """Integer tensors and shifts: everything the Gigatron needs."""
        t = lambda w: (w / w.abs().mean().clamp(min=1e-5)).round().clamp(-1, 1).to(torch.int8)
        r = lambda p: int(torch.round(p))
        out = {"cfg": self.cfg, "emb": t(self.emb), "lemb": r(self.lemb),
               "pos": torch.round(self.pos).clamp(-127, 127).to(torch.int16), "blocks": []}
        for b in self.blocks:
            out["blocks"].append({
                "qkv": t(b.qkv.w), "o": t(b.o.w), "up": t(b.up.w), "down": t(b.down.w),
                "lq": r(b.lq), "lk": r(b.lk), "lv": r(b.lv), "latt": r(b.latt),
                "lo": r(b.o.ls), "lup": r(b.up.ls), "ldown": r(b.down.ls)})
        return out
