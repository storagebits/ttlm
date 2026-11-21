"""Pure-integer TTLM inference, token by token with a KV cache.

This is the reference the Gigatron C code must match bit for bit. Python ints
only (no torch in the hot path), written the way the C code will be written.
"""
from qmodel import CTX, EXP, HMAX, QMAX, RCP, VMAX, ZMAX

EXP_T = [int(x) for x in EXP]
RCP_T = [int(x) for x in RCP]


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def shl(y, s):               # y * 2^s, floored (arithmetic shift)
    return y << s if s >= 0 else y >> -s


def norm(h):
    S = max(sum(abs(x) for x in h), 1)
    sh = 10 - (S.bit_length() - 1)
    return [clamp(shl(x, sh), -127, 127) for x in h]


def matvec(W, a):            # ternary rows: adds and subtracts only
    return [sum(x if w > 0 else -x for w, x in zip(row, a) if w) for row in W]


class Ref:
    def __init__(self, ex):
        c = ex["cfg"]
        self.d, self.h, self.dh = c["d"], c["h"], c["d"] // c["h"]
        L = lambda t: t.tolist()
        self.E, self.lemb = L(ex["emb"]), ex["lemb"]
        self.P = [[shl(p, ex["lpos"]) for p in row] for row in L(ex["pos"])]
        self.B = [{k: (L(v) if hasattr(v, "tolist") else v) for k, v in b.items()} for b in ex["blocks"]]
        self.reset()

    def reset(self):
        self.t = 0
        self.K = [[] for _ in self.B]
        self.V = [[] for _ in self.B]

    def attend(self, li, a):
        b, d, dh = self.B[li], self.d, self.dh
        y = matvec(b["qkv"], a)
        q = [clamp(shl(x, b["lq"]), -QMAX, QMAX) for x in y[:d]]
        k = [clamp(shl(x, b["lk"]), -QMAX, QMAX) for x in y[d:2 * d]]
        v = [clamp(shl(x, b["lv"]), -VMAX, VMAX) for x in y[2 * d:]]
        self.K[li].append(k); self.V[li].append(v)
        out = []
        for hd in range(self.h):
            sl = slice(hd * dh, (hd + 1) * dh)
            s = [sum(qi * ki for qi, ki in zip(q[sl], kj[sl])) for kj in self.K[li]]
            m = max(s)
            e = [EXP_T[min((m - sj) >> b["latt"], ZMAX)] for sj in s]
            R = RCP_T[min(sum(e) >> 3, len(RCP_T) - 1)]
            p = [(ej * R) >> 8 for ej in e]
            for i in range(dh):
                acc = sum(pj * vj[sl][i] for pj, vj in zip(p, self.V[li]))
                out.append(clamp(acc >> 6, -127, 127))
        return out

    def step(self, tok):
        """Feed one token, return the integer logits for the next one."""
        h = [clamp(shl(e, self.lemb) + p, -HMAX, HMAX) for e, p in zip(self.E[tok], self.P[self.t])]
        for li, b in enumerate(self.B):
            o = matvec(b["o"], self.attend(li, norm(h)))
            h = [clamp(x + shl(y, b["lo"]), -HMAX, HMAX) for x, y in zip(h, o)]
            u = [clamp(shl(max(x, 0), b["lup"]), 0, 127) for x in matvec(b["up"], norm(h))]
            dn = matvec(b["down"], u)
            h = [clamp(x + shl(y, b["ldown"]), -HMAX, HMAX) for x, y in zip(h, dn)]
        self.t += 1
        a = norm(h)
        return [sum(x if w > 0 else -x for w, x in zip(row, a) if w) for row in self.E]


if __name__ == "__main__":
    import sys
    import torch
    from data import build_vocab, make_dataset
    from qmodel import QMoth
    from train import encode, valid_answers

    ck = torch.load(sys.argv[1])
    vocab = ck["vocab"]; stoi = {w: i for i, w in enumerate(vocab)}
    model = QMoth(len(vocab)); model.load_state_dict(ck["model"]); model.eval()
    ref = Ref(model.export())

    # 1) bit-exactness against the torch model on full sequences
    mism = 0
    for _, q, a in make_dataset(50, seed=7):
        seq = [stoi["<q>"]] + encode(q, stoi) + [stoi["<a>"]] + encode(a, stoi)
        seq = seq[:CTX]
        with torch.no_grad():
            full = model.logits_int(torch.tensor([seq]))[0].round().long().tolist()
        ref.reset()
        for t, tok in enumerate(seq):
            if ref.step(tok) != full[t]:
                mism += 1
    print(f"logit mismatches vs torch: {mism}")

    # 2) quality of the pure-integer greedy decoder
    def chat(words):
        ref.reset()
        seq = [stoi["<q>"]] + encode(words, stoi) + [stoi["<a>"]]
        for tok in seq:
            lg = ref.step(tok)
        out = []
        while ref.t < CTX:
            t = max(range(len(lg)), key=lg.__getitem__)
            if vocab[t] == "<e>":
                break
            out.append(vocab[t]); lg = ref.step(t)
        return out

    valid = valid_answers()
    test = make_dataset(400, seed=99)
    print("valid:", sum(" ".join(chat(q)) in valid[it] for it, q, _ in test), "/", len(test))
    for q in ["hi moth", "where is the cpu", "are you an ai", "what is money"]:
        print(f"  you> {q:20s} moth> {' '.join(chat(q.split()))}")
