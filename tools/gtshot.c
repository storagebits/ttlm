/* gtshot: headless Gigatron for TTLM tests and screenshots.
 *
 * CPU core: Docs/gtemu.c from kervinck/gigatron-rom.
 *   Copyright (c) 2017,2018,2019, Marcel van Kervinck. All rights reserved.
 *   BSD 2-Clause License: redistribution and use in source and binary forms, with or
 *   without modification, are permitted provided that source redistributions retain
 *   this copyright notice, this list of conditions and the disclaimer of the original
 *   (THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND), and binary
 *   redistributions reproduce them in the documentation.
 * GT1 loading: same trick as gtsim (lb3361/gigatron-lcc): the second SYS_Exec
 * (the main menu) is replaced by the GT1 file.
 *
 * Usage: gtshot ROM GT1 [cmd ...]
 *   w<sec>    run for <sec> emulated seconds
 *   t<text>   type text through the serial/controller input ('|' = Enter)
 *   s<file>   write the screen memory (videoTable view) as a 640x480 PPM
 *   v<file>   write what a VGA monitor shows (video modes, scanlines) as a 640x480 PPM
 *   R<file>   raw VGA capture (200 cycles x 480 lines) for calibration
 *   c<n>:<prefix>  from now on, save one VGA frame every n frames as <prefix>00000.ppm, ...
 *             (c0 stops)
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { uint16_t PC; uint8_t IR, D, AC, X, Y, OUT, undef; } CpuState;
static uint8_t ROM[1 << 16][2], RAM[1 << 15], IN = 0xff;

static CpuState cpuCycle(const CpuState S)
{
  CpuState T = S;
  T.IR = ROM[S.PC][0]; T.D = ROM[S.PC][1];
  int ins = S.IR >> 5, mod = (S.IR >> 2) & 7, bus = S.IR & 3, W = (ins == 6), J = (ins == 7);
  uint8_t lo = S.D, hi = 0, *to = NULL; int incX = 0;
  if (!J) switch (mod) {
#define E(p) (W ? 0 : p)
    case 0: to = E(&T.AC); break;
    case 1: to = E(&T.AC); lo = S.X; break;
    case 2: to = E(&T.AC); hi = S.Y; break;
    case 3: to = E(&T.AC); lo = S.X; hi = S.Y; break;
    case 4: to = &T.X; break;
    case 5: to = &T.Y; break;
    case 6: to = E(&T.OUT); break;
    case 7: to = E(&T.OUT); lo = S.X; hi = S.Y; incX = 1; break;
  }
  uint16_t addr = (hi << 8) | lo;
  int B = S.undef;
  switch (bus) {
    case 0: B = S.D; break;
    case 1: if (!W) B = RAM[addr & 0x7fff]; break;
    case 2: B = S.AC; break;
    case 3: B = IN; break;
  }
  if (W) RAM[addr & 0x7fff] = B;
  uint8_t ALU = 0;
  switch (ins) {
    case 0: ALU = B; break; case 1: ALU = S.AC & B; break; case 2: ALU = S.AC | B; break;
    case 3: ALU = S.AC ^ B; break; case 4: ALU = S.AC + B; break; case 5: ALU = S.AC - B; break;
    case 6: ALU = S.AC; break; case 7: ALU = -S.AC; break;
  }
  if (to) *to = ALU;
  if (incX) T.X = S.X + 1;
  T.PC = S.PC + 1;
  if (J) {
    if (mod != 0) { int cond = (S.AC >> 7) + 2 * (S.AC == 0); if (mod & (1 << cond)) T.PC = (S.PC & 0xff00) | B; }
    else T.PC = (S.Y << 8) | B;
  }
  return T;
}

static uint16_t deek(int a) { return RAM[a] | RAM[a + 1] << 8; }
static void doke(int a, int v) { RAM[a] = v; RAM[a + 1] = v >> 8; }

static int loadGt1(const char *f)
{
  FILE *fp = fopen(f, "rb"); int c, len; uint16_t addr;
  if (!fp) { perror(f); exit(1); }
  c = getc(fp);
  do {
    addr = c << 8; addr |= getc(fp);
    len = getc(fp); if (len == 0) len = 256;
    while (len--) { RAM[addr & 0x7fff] = getc(fp); addr = (addr & 0xff00) | ((addr + 1) & 0xff); }
    c = getc(fp);
  } while (c > 0);
  addr = getc(fp) << 8; addr |= getc(fp);
  fclose(fp);
  return addr;
}

static CpuState S;
static uint8_t vga[525][200];   /* raw VGA: one byte per CPU cycle (4 dots), per scanline */
static int vgaX, vgaY;
static long long t;
static long frames;
static const char *gt1;

static void cycle(void)
{
  if (t < 0) S.PC = 0;
  if (S.PC == 0x3b4 && deek(0x22) == 0x00ad) {        /* SYS_Exec_88 */
    static int n = 0; int pc = deek(0x16);
    if (!(n == 0 && pc >= 0x1f0 && pc < 0x1f8) && ++n == 1) {
      int ex = loadGt1(gt1);
      doke(0x16, (ex & 0xff00) | ((ex - 2) & 0xff));
      doke(0x1a, ex);
      S.IR = 0x00; S.D = 0xf8; S.PC = 0x3cb;           /* REENTER */
    }
  }
  CpuState T = cpuCycle(S);
  if (vgaY >= 0 && vgaY < 525 && vgaX < 200) vga[vgaY][vgaX] = S.OUT;
  vgaX++;
  if ((T.OUT & 0x80) < (S.OUT & 0x80)) { frames++; vgaY = -36; }   /* falling vSync */
  if ((T.OUT & 0x40) > (S.OUT & 0x40)) { vgaX = 0; vgaY++; }       /* rising hSync */
  S = T; t++;
}

static int cap_every;                /* capture one VGA frame every N frames (0 = off) */
static char cap_prefix[200];
static int cap_n;
static void vsave(const char *f);
static void run_frames(long n)
{
  long f = frames + n, last = frames;
  while (frames < f) {
    cycle();
    if (frames != last) {
      last = frames;
      if (cap_every && frames % cap_every == 0) {
        char name[256];
        snprintf(name, sizeof name, "%s%05d.ppm", cap_prefix, cap_n++);
        vsave(name);
      }
    }
  }
}

static void type(const char *s)
{
  for (; *s; s++) {
    IN = (*s == '|') ? '\n' : *s;
    run_frames(3);
    IN = 0xff;
    run_frames(3);
  }
}

static void shot(const char *f)
{
  FILE *fp = fopen(f, "wb"); int x, y, dy, dx;
  fprintf(fp, "P6 640 480 255\n");
  for (y = 0; y < 120; y++)
    for (dy = 0; dy < 4; dy++)
      for (x = 0; x < 160; x++) {
        int page = RAM[0x100 + 2 * y], xo = RAM[0x101 + 2 * y];
        uint8_t c = RAM[((page << 8) | ((x + xo) & 0xff)) & 0x7fff];
        uint8_t rgb[3] = { (c & 3) * 85, ((c >> 2) & 3) * 85, ((c >> 4) & 3) * 85 };
        for (dx = 0; dx < 4; dx++) fwrite(rgb, 1, 3, fp);
      }
  fclose(fp);
}

/* What a VGA monitor shows: 640x480 from the captured signal (cropped). */
#define VX0 12
#define VY0 0
static void vshot(const char *f, int raw);
static void vsave(const char *f) { vshot(f, 2); }
static void vshot(const char *f, int raw)
{
  if (raw != 2) run_frames(1);                        /* a complete frame */
  else raw = 0;
  FILE *fp = fopen(f, "wb"); int x, y, dx;
  int W = raw ? 200 : 160, H = raw ? 525 : 480;
  fprintf(fp, "P6 %d 480 255\n", W * 4);
  for (y = 0; y < 480; y++)
    for (x = 0; x < W; x++) {
      uint8_t c = raw ? vga[y][x] : vga[y + VY0][x + VX0];
      if (!raw && (c & 0xc0) != 0xc0) c = 0;          /* blanking */
      uint8_t rgb[3] = { (c & 3) * 85, ((c >> 2) & 3) * 85, ((c >> 4) & 3) * 85 };
      for (dx = 0; dx < 4; dx++) fwrite(rgb, 1, 3, fp);
    }
  (void)H;
  fclose(fp);
}

int main(int argc, char **argv)
{
  if (argc < 3) { fprintf(stderr, "usage: gtshot ROM GT1 [w<sec>|t<text>|s<file>] ...\n"); return 1; }
  FILE *fp = fopen(argv[1], "rb");
  if (!fp || fread(ROM, 1, sizeof ROM, fp) != sizeof ROM) { fprintf(stderr, "bad rom\n"); return 1; }
  fclose(fp);
  gt1 = argv[2];
  srand(1);
  for (size_t i = 0; i < sizeof RAM; i++) RAM[i] = rand();
  t = -2;
  for (int i = 3; i < argc; i++) {
    const char *a = argv[i];
    if (a[0] == 'w') run_frames((long)(atof(a + 1) * 59.98));
    else if (a[0] == 't') type(a + 1);
    else if (a[0] == 's') { shot(a + 1); fprintf(stderr, "[%.1fs] %s\n", t / 6.25e6, a + 1); }
    else if (a[0] == 'v') { vshot(a + 1, 0); fprintf(stderr, "[%.1fs] %s\n", t / 6.25e6, a + 1); }
    else if (a[0] == 'R') vshot(a + 1, 1);
    else if (a[0] == 'c') { cap_every = atoi(a + 1); const char *p = strchr(a, ':'); if (p) snprintf(cap_prefix, sizeof cap_prefix, "%s", p + 1); }
  }
  return 0;
}
