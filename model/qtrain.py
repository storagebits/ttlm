"""Train the integer-exact moth (qmodel.QMoth)."""
import argparse
import torch
import torch.nn.functional as F
from data import build_vocab, make_dataset
from qmodel import QMoth
from train import batchify, generate, valid_answers

ap = argparse.ArgumentParser()
ap.add_argument("--d", type=int, default=32); ap.add_argument("--layers", type=int, default=2)
ap.add_argument("--heads", type=int, default=2); ap.add_argument("--mlp", type=int, default=64)
ap.add_argument("--steps", type=int, default=4000); ap.add_argument("--lr", type=float, default=3e-3)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--soft", action="store_true"); ap.add_argument("--init", default=None); ap.add_argument("--tern-t", type=float, default=1.0); ap.add_argument("--out", default="qmoth.pt")
a = ap.parse_args()
torch.manual_seed(a.seed)
import qmodel
qmodel.EXACT = not a.soft; qmodel.TERN_T = a.tern_t
vocab = build_vocab(); stoi = {w: i for i, w in enumerate(vocab)}
model = QMoth(len(vocab), a.d, a.layers, a.heads, a.mlp)
if a.init: model.load_state_dict(torch.load(a.init)["model"])
print(f"vocab {len(vocab)}  params {sum(p.numel() for p in model.parameters())}")
X, Y = batchify(make_dataset(40000, seed=1), stoi)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.1)
for step in range(a.steps):
    i = torch.randint(0, len(X), (128,))
    loss = F.cross_entropy(model(X[i]).view(-1, len(vocab)), Y[i].view(-1), ignore_index=-100)
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if step % 500 == 0 or step == a.steps - 1:
        print(f"step {step:5d}  loss {loss.item():.3f}", flush=True)
model.eval()
valid = valid_answers()
test = make_dataset(400, seed=99)
ok = sum(" ".join(generate(model, q, stoi, vocab)) in valid[it] for it, q, _ in test)
print(f"valid greedy answers: {ok}/{len(test)}")
ex = model.export(); nz = sum(int((b[k] != 0).sum()) for b in ex["blocks"] for k in ["qkv", "o", "up", "down"]) + int((ex["emb"] != 0).sum())
print(f"nonzero weights: {nz}")
for q in ["hi moth", "who are you", "where is the cpu", "do you like the light", "what is money",
          "are you an ai", "what do you eat", "tell me a joke", "good night moth", "can you code"]:
    print(f"  you> {q:24s} moth> {' '.join(generate(model, q.split(), stoi, vocab))}")
ex = model.export()
print("shifts:", ex["lemb"], [{k: v for k, v in b.items() if k.startswith('l')} for b in ex["blocks"]])
torch.save({"model": model.state_dict(), "vocab": vocab, "args": vars(a)}, a.out)
