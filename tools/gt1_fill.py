"""Fill the unused bytes of the brain (screen rows 0..ROWS-1) with black in a GT1 file.

The linker scatters the model data over the top of the screen; whatever it
does not cover would show leftovers of the main menu. Extra GT1 segments cost
no RAM: the ROM Loader simply writes them. Rows from 72 down are left alone
(the Loader itself runs there).

Usage: python tools/gt1_fill.py in.gt1 out.gt1 [rows=72]
"""
import sys


def read_gt1(data):
    segs, i = [], 0
    while True:
        hi = data[i]
        if hi == 0 and segs:
            break
        lo, n = data[i + 1], data[i + 2] or 256
        segs.append(((hi << 8) | lo, data[i + 3:i + 3 + n]))
        i += 3 + n
    return segs, data[i + 1:i + 3]


def main(src, dst, rows=72):
    data = open(src, "rb").read()
    segs, start = read_gt1(data)
    used = set()
    for a, d in segs:
        used.update(range(a, a + len(d)))
    extra = []
    for r in range(int(rows)):
        page = (0x08 + r) << 8
        x = 0
        while x < 160:
            if page + x in used:
                x += 1
                continue
            x0 = x
            while x < 160 and page + x not in used:
                x += 1
            extra.append((page + x0, bytes(x - x0)))
    out = bytearray()
    zp = [sg for sg in segs if sg[0] < 0x100]   # a zero-page segment must stay first
    rest = [sg for sg in segs if sg[0] >= 0x100]
    for a, d in zp + extra + rest:         # fill before the real data
        out += bytes([a >> 8, a & 255, len(d) & 255]) + d
    out += b"\x00" + start
    open(dst, "wb").write(out)
    print(f"{dst}: {len(extra)} fill segments, {sum(len(d) for _, d in extra)} bytes")


if __name__ == "__main__":
    main(*sys.argv[1:])
