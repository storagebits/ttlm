"""Write gigatron/src/tests.h (prompts) and the expected greedy outputs from ref.py."""
import sys, torch
from qmodel import CTX, QMoth
from ref import Ref
from train import encode
ck = torch.load(sys.argv[1]); vocab = ck["vocab"]; stoi = {w: i for i, w in enumerate(vocab)}
m = QMoth(len(vocab)); m.load_state_dict(ck["model"]); ref = Ref(m.export())
prompts = ["hi moth", "where is the cpu", "are you an ai", "what is money", "do you like the light",
           "can you code"][: int(sys.argv[3]) if len(sys.argv) > 3 else 6]
lines, exp = [], []
for i, q in enumerate(prompts):
    seq = [stoi["<q>"]] + encode(q.split(), stoi) + [stoi["<a>"]]
    lines.append(f"unsigned char t{i}[] = {{{','.join(map(str, seq))},255}};\n")
    ref.reset()
    for tok in seq: lg = ref.step(tok)
    out = []
    while True:
        t = max(range(len(lg)), key=lg.__getitem__)
        if t == stoi["<e>"] or ref.t >= CTX: break
        out.append(t); lg = ref.step(t)
    exp.append(f"{i}: " + " ".join(map(str, out)) if out else f"{i}:")
open(sys.argv[2], "w").write("".join(lines) + f"#define NTESTS {len(prompts)}\nunsigned char *tests[NTESTS] = {{{','.join(f't{i}' for i in range(len(prompts)))}}};\n")
print("\n".join(e.replace(": ", ": ", 1) for e in exp))
