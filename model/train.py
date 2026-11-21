"""Train the TTLM moth: a tiny decoder-only transformer with ternary weights.

Usage: python train.py --d 32 --layers 2 --heads 2 --mlp 64 --steps 3000 --out moth.pt
"""
import argparse
import itertools
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import INTENTS, UNKNOWN_WORDS, build_vocab, make_dataset

CTX = 24


def ternary(w):
    """BitNet b1.58 style: scale by mean |w|, round to {-1, 0, +1}, straight-through."""
    s = w.abs().mean().clamp(min=1e-5)
    q = (w / s).round().clamp(-1, 1) * s
    return w + (q - w).detach()


class TLinear(nn.Linear):
    def __init__(self, i, o, tern):
        super().__init__(i, o, bias=False)
        self.tern = tern

    def forward(self, x):
        return F.linear(x, ternary(self.weight) if self.tern else self.weight)


class RMSNorm(nn.Module):
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)


class Block(nn.Module):
    def __init__(self, d, h, m, tern):
        super().__init__()
        self.h = h
        self.n1, self.n2 = RMSNorm(), RMSNorm()
        self.qkv = TLinear(d, 3 * d, tern)
        self.o = TLinear(d, d, tern)
        self.up = TLinear(d, m, tern)
        self.down = TLinear(m, d, tern)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).split(D, -1)
        sh = lambda t: t.view(B, T, self.h, D // self.h).transpose(1, 2)
        y = F.scaled_dot_product_attention(sh(q), sh(k), sh(v), is_causal=True)
        x = x + self.o(y.transpose(1, 2).reshape(B, T, D))
        return x + self.down(F.relu(self.up(self.n2(x))))


class Moth(nn.Module):
    def __init__(self, V, d, L, h, m, tern):
        super().__init__()
        self.tern = tern
        self.emb = nn.Parameter(torch.randn(V, d) * 0.5)
        self.pos = nn.Parameter(torch.randn(CTX, d) * 0.1)
        self.blocks = nn.ModuleList(Block(d, h, m, tern) for _ in range(L))
        self.nf = RMSNorm()
        self.out_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, idx):
        E = ternary(self.emb) if self.tern else self.emb
        x = E[idx] + self.pos[: idx.shape[1]]
        for b in self.blocks:
            x = b(x)
        return self.out_scale * self.nf(x) @ E.t()


def encode(words, stoi):
    return [stoi.get(w, stoi["<unk>"]) for w in words]


def batchify(data, stoi):
    xs, ys = [], []
    for _, q, a in data:
        seq = [stoi["<q>"]] + encode(q, stoi) + [stoi["<a>"]] + encode(a, stoi) + [stoi["<e>"]]
        seq = seq[:CTX + 1]
        n_q = len(q) + 2  # tokens up to and including <a>
        x = seq[:-1] + [0] * (CTX - len(seq) + 1)
        y = [-100] * (n_q - 1) + seq[n_q:] + [-100] * (CTX - len(seq) + 1)
        xs.append(x[:CTX]); ys.append(y[:CTX])
    return torch.tensor(xs), torch.tensor(ys)


def valid_answers():
    out = {}
    for name, (_, alts) in INTENTS.items():
        s = set()
        for alt in alts:
            opts = [frag if isinstance(frag, list) else [frag] for frag in alt]
            for combo in itertools.product(*opts):
                s.add(" ".join(" ".join(combo).split()))
        out[name] = s
    return out


@torch.no_grad()
def generate(model, q, stoi, itos, greedy=True, temp=0.7):
    seq = [stoi["<q>"]] + encode(q, stoi) + [stoi["<a>"]]
    out = []
    while len(seq) < CTX:
        logits = model(torch.tensor([seq]))[0, -1]
        t = int(logits.argmax()) if greedy else int(torch.multinomial(F.softmax(logits / temp, -1), 1))
        if itos[t] == "<e>":
            break
        out.append(itos[t]); seq.append(t)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--mlp", type=int, default=64)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--float", action="store_true", help="no ternary quantization")
    ap.add_argument("--out", default="moth.pt")
    a = ap.parse_args()

    torch.manual_seed(0)
    vocab = build_vocab(); stoi = {w: i for i, w in enumerate(vocab)}
    train = make_dataset(40000, seed=1)
    model = Moth(len(vocab), a.d, a.layers, a.heads, a.mlp, not a.float)
    n = sum(p.numel() for p in model.parameters())
    print(f"vocab {len(vocab)}  params {n}  ternary {not a.float}")
    X, Y = batchify(train, stoi)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.1)
    for step in range(a.steps):
        i = torch.randint(0, len(X), (128,))
        loss = F.cross_entropy(model(X[i]).view(-1, len(vocab)), Y[i].view(-1), ignore_index=-100)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 500 == 0 or step == a.steps - 1:
            print(f"step {step:5d}  loss {loss.item():.3f}")

    model.eval()
    valid = valid_answers()
    test = make_dataset(400, seed=99)
    ok = sum(" ".join(generate(model, q, stoi, vocab)) in valid[it] for it, q, _ in test)
    print(f"valid greedy answers: {ok}/{len(test)}")
    for q in ["hi moth", "who are you", "where is the cpu", "do you like the light", "what is money",
              "are you an ai", "what do you eat", "tell me a joke", "good night moth", "can you code"]:
        print(f"  you> {q:24s} moth> {' '.join(generate(model, q.split(), stoi, vocab))}")
    torch.save({"model": model.state_dict(), "vocab": vocab, "args": vars(a)}, a.out)


if __name__ == "__main__":
    main()
