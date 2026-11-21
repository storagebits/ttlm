"""The README picture: the Gigatron screen next to its X-ray (what each pixel holds).

Usage: python tools/xray.py screen.ppm build_v4.frags out.png
  screen.ppm: a 640x480 screen from tools/gtshot (s<file>), after an answer.
Needs Pillow.
"""
import sys

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from brainmap import brainmap  # noqa: E402

BG, DIM, TEXT, MOTH = (12, 11, 10), (156, 148, 136), (233, 227, 216), (255, 170, 85)


def font(size):
    for f in ("/System/Library/Fonts/Menlo.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def hexrgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def main(ppm, frags, out):
    screen = Image.open(ppm).convert("RGB")
    bm = brainmap(frags)
    rows, kinds = bm["rows"], bm["kinds"]
    xray = screen.copy()
    px, sx = xray.load(), screen.load()
    for r in range(rows):
        for x in range(160):
            c = hexrgb(kinds[ord(bm["kind"][r * 160 + x]) - 48]["color"])
            p = sx[4 * x + 1, 4 * r + 1]
            lum = sum(p) / (3 * 255)                       # keep the texture of the data
            k = 0.35 + 0.65 * lum if kinds[ord(bm["kind"][r * 160 + x]) - 48]["key"] != "free" else 1
            col = tuple(int(v * k) for v in c)
            for dy in range(4):
                for dx in range(4):
                    px[4 * x + dx, 4 * r + dy] = col
    for y in range(4 * rows, 480):                         # the chat area, dimmed
        for x in range(640):
            px[x, y] = tuple(v // 3 for v in sx[x, y])

    shown = [k for k in kinds if k["bytes"]]
    pad, gap, lh = 40, 40, 30
    legend_h = ((len(shown) + 1) // 2) * lh + 40
    W, H = 2 * 640 + gap + 2 * pad, 110 + 480 + 30 + legend_h + pad
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((pad, 28), "The noise is the model.", font=font(34), fill=MOTH)
    d.text((pad, 72), "Left: the Gigatron's screen. Right: what each pixel holds. "
           "No video memory: the picture is the RAM.", font=font(17), fill=DIM)
    img.paste(screen, (pad, 110))
    img.paste(xray, (pad + 640 + gap, 110))
    y0 = 110 + 480 + 30
    half = (len(shown) + 1) // 2
    for i, k in enumerate(shown):
        x = pad + (i // half) * (640 + gap)
        y = y0 + (i % half) * lh
        d.rectangle((x, y + 3, x + 18, y + 21), fill=hexrgb(k["color"]), outline=(60, 56, 50))
        d.text((x + 30, y), k["label"], font=font(17), fill=TEXT)
        d.text((x + 640 - 4, y), f"{k['bytes']:,} B", font=font(17), fill=DIM, anchor="ra")
    d.text((pad, y0 + half * lh + 12), "Not visible: the word embeddings and the program code, tucked into the 96 bytes "
           "at the end of each line that the video never draws.", font=font(15), fill=DIM)
    img.save(out, optimize=True)
    print(out, img.size)


if __name__ == "__main__":
    main(*sys.argv[1:4])
