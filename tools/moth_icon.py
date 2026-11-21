"""Draw the TTLM moth as pixel art in the Gigatron palette (2 bits per channel).

Usage: python tools/moth_icon.py docs/img/moth.ppm [scale]
"""
import sys

SPRITE = [
    "....a......a....",
    ".....a....a.....",
    "......a..a......",
    "..YY...bb...YY..",
    ".YWWY..BB..YWWY.",
    "YWWOWY.BB.YWOWWY",
    "YWWWWWYBBYWWWWWY",
    "YWWWWWWBBWWWWWWY",
    ".YWWWWWBBWWWWWY.",
    "..YWWWYBBYWWWY..",
    "..YWWWYBBYWWWY..",
    ".YWWWY.BB.YWWWY.",
    ".YWOY..BB..YOWY.",
    "..YY...BB...YY..",
    ".......bb.......",
    "................",
]
# Gigatron colors: each channel is 0, 85, 170 or 255
COL = {".": None, "a": (170, 170, 170), "b": (85, 85, 85), "B": (170, 170, 170),
       "Y": (170, 85, 0), "W": (255, 170, 85), "O": (85, 0, 0)}


def main(out, scale=16):
    scale = int(scale)
    n = len(SPRITE)
    W = H = n * scale
    px = bytearray()
    for y in range(H):
        for x in range(W):
            c = COL[SPRITE[y // scale][x // scale]]
            px += bytes(c if c else (0, 0, 0))
    open(out, "wb").write(f"P6 {W} {H} 255\n".encode() + px)


if __name__ == "__main__":
    main(*sys.argv[1:])
