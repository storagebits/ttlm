"""What is on the screen: a byte-by-byte map of the brain (screen rows 0..ROWS-1).

The Gigatron draws its picture straight from RAM, and TTLM's model lives in the
top rows of the screen. This reads the linker's fragment map and says, for each
visible byte, what it holds: which weights, the KV cache, code, tables...

Usage: python tools/brainmap.py build_v4.frags [rows=72]   -> JSON on stdout
Used by `make web` (web/ttlm.json) and tools/xray.py (the README picture).
"""
import json
import re
import sys

XROW = 71   # activation page (0x4f00), see gigatron/src/row.s

# key: (label, color). Order = legend order.
KINDS = {
    "emb":  ("Word embeddings (also the output layer)", "#ff5555"),
    "pos":  ("Position embeddings", "#ff55ff"),
    "att1": ("Layer 1: attention weights", "#55aaff"),
    "mlp1": ("Layer 1: MLP weights", "#55ffaa"),
    "att2": ("Layer 2: attention weights", "#aa77ff"),
    "mlp2": ("Layer 2: MLP weights", "#bbff44"),
    "kv":   ("KV cache: the conversation so far", "#ffff55"),
    "act":  ("Activations (the line that flickers)", "#00ffff"),
    "tab":  ("Softmax and multiplication tables", "#ffaa44"),
    "voc":  ("Vocabulary (155 words)", "#ffffff"),
    "code": ("Program code", "#8a8a8a"),
    "var":  ("Program variables", "#4f7fa8"),
    "free": ("Unused", "#1c1c1c"),
}
MATS = {"q": "queries", "k": "keys", "v": "values", "o": "output", "up": "up", "down": "down"}
LINE = re.compile(r"([0-9a-f]{4})-([0-9a-f]{4}) \(\d+ bytes?\)\s+(\w+)\s+(\S+)")


def classify(kind, name):
    """-> (kind key, human label) for one linker fragment."""
    if kind == "CODE":
        return "code", f"code: {name}()"
    m = re.fullmatch(r"m_(q|k|v|o|up|down)(\d)(x?)(?:_\d+)?", name)
    if m:
        mat, layer = m.group(1), int(m.group(2)) + 1
        part = "attention" if mat in "qkvo" else "MLP"
        key = ("att" if part == "attention" else "mlp") + str(layer)
        return key, f"layer {layer}, {part} {MATS[mat]} weights"
    if re.fullmatch(r"m_embx?(_\d+)?", name):
        return "emb", "word embeddings"
    if re.fullmatch(r"m_posx?(_\d+)?", name):
        return "pos", "position embeddings"
    m = re.fullmatch(r"(k|v)c_(\d)_(\d+)", name)
    if m:
        what = "keys" if m.group(1) == "k" else "values"
        return "kv", f"KV cache, layer {int(m.group(2)) + 1}, {what} of word {int(m.group(3)) + 1}"
    if re.fullmatch(r"(k|v)c(_\d)?", name):
        return "kv", "KV cache index"
    m = re.fullmatch(r"m_(q|k|v|o|up|down)x", name)
    if m:
        return "var", f"pointers to the {MATS[m.group(1)]} weights"
    if name.startswith("sh_"):
        return "var", "scale shifts"
    if name.startswith(".") or name.startswith("__"):
        return "var", "program constants"
    if re.fullmatch(r"exp_t|rcp(_t|_\d+)?", name):
        return "tab", "softmax table"
    if name in ("sqlo", "sqhi"):
        return "tab", "quarter-square multiplication table"
    if re.fullmatch(r"words(_t|_first|_\d+)?", name):
        return "voc", "vocabulary"
    return "var", f"variable: {name}"


def brainmap(frags_path, rows=72):
    frags = []
    for line in open(frags_path):
        m = LINE.search(line)
        if not m:
            continue
        a, b = int(m.group(1), 16), int(m.group(2), 16)
        if b < 0x0800 or a >= (0x08 + rows) << 8:
            continue
        key, label = classify(m.group(3), m.group(4))
        frags.append((a, b, key, label))
    labels, keys = [], list(KINDS)
    cells = [[None] * 160 for _ in range(rows)]
    for a, b, key, label in frags:
        if label not in labels:
            labels.append(label)
        li = labels.index(label)
        for addr in range(a, b + 1):
            r, x = (addr >> 8) - 0x08, addr & 0xff
            if 0 <= r < rows and x < 160:
                cells[r][x] = (keys.index(key), li)
    labels += ["activations: x and -x for the ternary dot product", "unused"]
    for x in range(160):
        cells[XROW][x] = (keys.index("act"), len(labels) - 2)
    for r in range(rows):
        for x in range(160):
            if cells[r][x] is None:
                cells[r][x] = (keys.index("free"), len(labels) - 1)
    count = {k: 0 for k in keys}
    for r in range(rows):
        for x in range(160):
            count[keys[cells[r][x][0]]] += 1
    return {
        "rows": rows,
        # kinds with no visible byte (code, embeddings) live in the 96 off-screen bytes of each line
        "kinds": [{"key": k, "label": KINDS[k][0], "color": KINDS[k][1], "bytes": count[k]} for k in keys],
        "labels": labels,
        # one entry per visible byte, row by row: kind index, label index
        "kind": "".join(chr(48 + c[0]) for row in cells for c in row),
        "label": [c[1] for row in cells for c in row],
    }


if __name__ == "__main__":
    print(json.dumps(brainmap(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 72), separators=(",", ":")))
