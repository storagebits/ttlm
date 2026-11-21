"""Wiring diagram: Arduino Mega 2560 (or Uno) running BabelFish -> Gigatron controller port.

Usage: python tools/wiring.py out.png    (needs Pillow)
"""
import sys

from PIL import Image, ImageDraw, ImageFont

BG, PANEL, LINE, TEXT, DIM, MOTH = (14, 13, 12), (24, 22, 20), (60, 56, 50), (233, 227, 216), (156, 148, 136), (255, 170, 85)
BOARD, HDR, PIN = (24, 70, 140), (20, 20, 20), (200, 190, 150)
WIRES = {  # arduino label -> (db9 pin, color, signal)
    "GND": (8, (150, 150, 150), "ground"),
    "13": (2, (255, 220, 60), "data"),
    "12": (3, (80, 220, 120), "latch"),
    "11": (4, (80, 160, 255), "pulse"),
}


def font(size, bold=False):
    for f in ("/System/Library/Fonts/Menlo.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"):
        try:
            return ImageFont.truetype(f, size, index=1 if bold else 0)
        except OSError:
            pass
    return ImageFont.load_default()


def bezier(p0, p1, p2, p3, n=60):
    pts = []
    for i in range(n + 1):
        t = i / n
        a, b, c, d = (1 - t) ** 3, 3 * t * (1 - t) ** 2, 3 * t * t * (1 - t), t ** 3
        pts.append((a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0], a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1]))
    return pts


def main(out):
    W, H = 1600, 1160
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((60, 40), "Loading TTLM: Arduino → Gigatron", font=font(40, True), fill=MOTH)
    d.text((60, 98), "4 jumper wires, female end on the Gigatron, male end on the Arduino. "
           "Gigatron switched OFF while wiring.", font=font(20), fill=DIM)

    # --- Arduino Mega, seen from above, USB jack on the left ---
    bx, by, bw, bh = 60, 260, 640, 300
    d.rounded_rectangle((bx, by, bx + bw, by + bh), 18, fill=BOARD, outline=(70, 120, 200), width=3)
    d.rectangle((bx - 30, by + 40, bx + 50, by + 120), fill=(170, 170, 175), outline=(90, 90, 95), width=2)
    d.text((bx - 22, by + 130), "USB", font=font(18, True), fill=TEXT)
    d.rectangle((bx - 20, by + 200, bx + 40, by + 270), fill=(30, 30, 30))
    d.text((bx + 70, by + 150), "ELEGOO MEGA 2560", font=font(26, True), fill=(220, 230, 255))
    d.text((bx + 70, by + 190), "BabelFish flashed on it", font=font(18), fill=(170, 190, 230))
    # top-left header: SCL SDA AREF GND 13 12 11 10 9 8
    labels = ["SCL", "SDA", "AREF", "GND", "13", "12", "11", "10", "9", "8"]
    hx, hy, step = bx + 110, by + 14, 46
    d.rectangle((hx - 18, hy - 4, hx + step * 9 + 18, hy + 34), fill=HDR)
    pinpos = {}
    for i, lab in enumerate(labels):
        x = hx + i * step
        used = lab in WIRES
        d.rectangle((x - 7, hy + 8, x + 7, hy + 22), fill=(60, 60, 60) if not used else WIRES[lab][1])
        d.text((x, hy + 58), lab, font=font(17, used), fill=TEXT if used else DIM, anchor="mm")
        pinpos[lab] = (x, hy + 15)
    d.text((hx + step * 4.5, hy + 92), "the header in the corner next to the USB jack",
           font=font(16), fill=DIM, anchor="mm")
    d.text((bx + bw - 20, by + bh - 40), "(on a Uno: same pins, same place)", font=font(16), fill=(170, 190, 230), anchor="ra")

    # --- Gigatron controller port, DB9, seen from the front (facing the pins) ---
    cx, cy = 1240, 470
    shell = [(cx - 230, cy - 110), (cx + 230, cy - 110), (cx + 185, cy + 110), (cx - 185, cy + 110)]
    d.polygon(shell, fill=(150, 150, 155), outline=(90, 90, 95))
    inner = [(cx - 200, cy - 85), (cx + 200, cy - 85), (cx + 162, cy + 85), (cx - 162, cy + 85)]
    d.polygon(inner, fill=(25, 25, 25))
    dbpos = {}
    for n in range(1, 10):
        if n <= 5:
            x, y = cx - 140 + (n - 1) * 70, cy - 35
        else:
            x, y = cx - 105 + (n - 6) * 70, cy + 40
        used = [k for k, v in WIRES.items() if v[0] == n]
        col = WIRES[used[0]][1] if used else (120, 110, 90)
        d.ellipse((x - 14, y - 14, x + 14, y + 14), fill=col, outline=(240, 230, 200) if used else None, width=2)
        d.text((x, y), str(n), font=font(15, True), fill=(10, 10, 10), anchor="mm")
        dbpos[n] = (x, y)
    d.text((cx, cy + 145), "GIGATRON  ·  Game Controller Port", font=font(22, True), fill=TEXT, anchor="mm")
    d.text((cx, cy + 175), "the right-hand connector at the back (the left one is Video Out)", font=font(15), fill=DIM, anchor="mm")
    d.text((cx, cy + 200), "seen from the front: row of 5 pins on top, pin 1 on the left", font=font(15), fill=DIM, anchor="mm")

    # --- wires ---
    order = ["GND", "13", "12", "11"]
    for k, lab in enumerate(order):
        n, col, sig = WIRES[lab]
        a, b = pinpos[lab], dbpos[n]
        lift = 150 + 22 * k
        pts = bezier(a, (a[0], a[1] - lift), (b[0] - 260, b[1] - 280 + 30 * k), (b[0], b[1] - 16))
        d.line(pts, fill=(0, 0, 0), width=11)
        d.line(pts, fill=col, width=7)

    # --- table ---
    ty = 740
    d.text((60, ty), "Connections", font=font(26, True), fill=MOTH)
    rows = [("Arduino", "Gigatron DB9", "signal", None)] + [(lab, f"pin {WIRES[lab][0]}", WIRES[lab][2], WIRES[lab][1]) for lab in ["13", "12", "11", "GND"]]
    for i, (a, b, c, col) in enumerate(rows):
        y = ty + 50 + i * 38
        f = font(20, i == 0)
        if col:
            d.rectangle((60, y + 4, 80, y + 22), fill=col)
        d.text((100, y), a, font=f, fill=TEXT if i else DIM)
        d.text((260, y), b, font=f, fill=TEXT if i else DIM)
        d.text((460, y), c, font=f, fill=TEXT if i else DIM)

    sx = 760
    d.text((sx, ty), "Then", font=font(26, True), fill=MOTH)
    steps = [
        "1. Arduino on the Mac's USB. Gigatron powered, monitor on.",
        "2. On the Mac:",
        "   python3 sendFile.py gigatron/ttlm.gt1",
        "   (it resets the Gigatron and starts the Loader itself)",
        "3. About 10 s of loading, then the moth appears.",
        "4. Type through BabelFish, or a PS/2 keyboard on the",
        "   Arduino (pin 3 = clock, pin 4 = data).",
        "",
        "The game controller cannot stay plugged in at the same time.",
    ]
    for i, s in enumerate(steps):
        d.text((sx, ty + 50 + i * 34), s, font=font(19), fill=TEXT if not s.startswith("The game") else DIM)
    img.save(out, optimize=True)
    print(out, img.size)


if __name__ == "__main__":
    main(sys.argv[1])
