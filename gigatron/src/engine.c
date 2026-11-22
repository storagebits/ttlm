/* TTLM inference engine for the Gigatron: integer only, bit-exact with model/ref.py.
 *
 * Ternary matrices are stored as index lists, so a matrix-vector product is
 * nothing but additions (row.s). Attention uses int6 queries/keys, a 32-entry
 * exp table, a reciprocal table and quarter-square tables for the 6-bit
 * products: no division anywhere.
 */
#include "engine.h"

#define HMAX 2047
#define QMAX 31
#define VMAX 63

/* Activation page (screen row 71): x at +0, -x at +128 (see row.s). */
#define XB ((int *)0x4f00)
#pragma glcc segment(0x4f00, 0x5000, "")

extern int row(unsigned char **pp) __attribute__((quickcall));

int h[D];              /* residual stream, value = h / 16 */
int a[MLP];            /* int8 activations, stored as int */
int y[MLP];            /* raw matrix-vector products */
int att[D];            /* attention output */
unsigned char t_pos;   /* tokens already in the KV cache */

static int shl(int x, signed char s) { return s >= 0 ? x << s : x >> -s; }

static int clampi(int x, int lo, int hi) { return x < lo ? lo : x > hi ? hi : x; }

static void load_x(int *src, unsigned char n)
{
  int *p = XB, *q = XB + 64;
  while (n--) { *p++ = *src; *q++ = -*src++; }
}

/* out = W x for a ternary W (x must be loaded with load_x) */
static void matvec(unsigned char **chunks, int nrows, int *out)
{
  unsigned char *p, r;
  while (nrows > 0) {
    p = *chunks++;
    r = *p++;
    nrows -= r;
    while (r--) *out++ = row(&p);
  }
}

/* dst[i] = clamp(y[i] * 2^sh, lo, hi) for the D outputs of a matvec */
static void requant(signed char *dst, signed char sh, int lo, int hi)
{
  unsigned char i;
  for (i = 0; i < D; i++) dst[i] = clampi(shl(y[i], sh), lo, hi);
}

/* h += y * 2^sh, clamped: the residual connection */
static void residual(signed char sh)
{
  unsigned char i;
  for (i = 0; i < D; i++) h[i] = clampi(h[i] + shl(y[i], sh), -HMAX, HMAX);
}

/* power-of-two mean-abs norm: a = h * 2^(10 - floor(log2(sum|h|))), then load it */
static void norm(void)
{
  unsigned S = 0, t;
  signed char e, i;
  for (i = 0; i < D; i++) S += h[i] < 0 ? -h[i] : h[i];
  if (!S) S = 1;
  e = -1; for (t = S; t; t >>= 1) e++;
  e = 10 - e;
  for (i = 0; i < D; i++) a[i] = clampi(shl(h[i], e), -127, 127);
  load_x(a, D);
}

/* x*y = sq(x+y) - sq(x-y), sq(n) = floor(n*n/4) in two byte tables (|n| < 128) */
static int qmul(int x, int y)
{
  int u = x + y, v = x - y;
  if (u < 0) u = -u;
  if (v < 0) v = -v;
  return (sqlo[u] - sqlo[v]) + ((sqhi[u] - sqhi[v]) << 8);
}

static int dot(signed char *x, signed char *y)
{
  unsigned char i;
  int s = 0;
  for (i = 0; i < DH; i++) s += qmul(x[i], y[i]);
  return s;
}

static void attend(unsigned char l)
{
  static int s[CTX];
  static unsigned char p[CTX];
  static signed char q[D];
  signed char *v, latt = sh_latt[l];
  unsigned char j, i, hd, e, R;
  unsigned W, z;
  int m, *o;

  matvec(m_qx[l], D, y); requant(q, sh_lq[l], -QMAX, QMAX);
  matvec(m_kx[l], D, y); requant(kc[l][t_pos], sh_lk[l], -QMAX, QMAX);
  matvec(m_vx[l], D, y); requant(vc[l][t_pos], sh_lv[l], -VMAX, VMAX);

  for (hd = 0; hd < D; hd += DH) {
    m = -32767;
    for (j = 0; j <= t_pos; j++) {
      s[j] = dot(q + hd, kc[l][j] + hd);
      if (s[j] > m) m = s[j];
    }
    W = 0;
    for (j = 0; j <= t_pos; j++) {
      z = (unsigned)(m - s[j]) >> latt;
      W += p[j] = exp_t[z < NEXP ? z : NEXP - 1];
    }
    W >>= 3;
    if (W > NRCP - 1) W = NRCP - 1;
    R = rcp_t[W / RCP_CHUNK][W % RCP_CHUNK];
    o = att + hd;
    for (i = 0; i < DH; i++) o[i] = 0;
    for (j = 0; j <= t_pos; j++) {
      e = ((unsigned)p[j] * R) >> 8;        /* softmax weight, sums to ~64 */
      if (!e) continue;                     /* sharp attention: most weights are 0 */
      v = vc[l][j] + hd;
      for (i = 0; i < DH; i++) o[i] += qmul(e, v[i]);
    }
    for (i = 0; i < DH; i++) o[i] = clampi(o[i] >> 6, -127, 127);
  }
}

/* h += unit * row r of a ternary matrix */
static void add_row(unsigned char **chunks, int r, int unit)
{
  unsigned char *p, n, b;
  for (;;) {
    p = *chunks++; n = *p++;
    if (r >= n) { r -= n; continue; }
    while (r--) { n = *p++; p += n; }
    n = *p++;
    while (n--) {
      b = *p++;
      if (b & 0x80) h[(b & 0x7f) >> 1] -= unit;
      else h[b >> 1] += unit;
    }
    return;
  }
}

void reset(void) { t_pos = 0; }

int top_tok[TOPK], top_val[TOPK];

/* Feed one token. If want, compute the output head and return the best next
 * token (top_tok/top_val hold the TOPK best); otherwise return -1. */
int step(int tok, int want)
{
  unsigned char l, i, k, nrows;
  unsigned char *p, **chunks;
  int acc, v;

  for (i = 0; i < D; i++) h[i] = 0;
  add_row(m_emb, tok, 1 << LEMB);
  add_row(m_pos, t_pos, 1 << LPOS);
  for (i = 0; i < D; i++) h[i] = clampi(h[i], -HMAX, HMAX);

  for (l = 0; l < NL; l++) {
    norm();
    attend(l);
    load_x(att, D);
    matvec(m_ox[l], D, y);
    residual(sh_lo[l]);
    norm();
    matvec(m_upx[l], MLP, y);
    for (i = 0; i < MLP; i++) a[i] = y[i] > 0 ? clampi(shl(y[i], sh_lup[l]), 0, 127) : 0;
    load_x(a, MLP);
    matvec(m_downx[l], D, y);
    residual(sh_ldown[l]);
  }
  t_pos++;
  if (!want) return -1;
  norm();

  /* output head: logits = E . a, keep the TOPK best */
  for (k = 0; k < TOPK; k++) { top_tok[k] = -1; top_val[k] = -32767; }
  chunks = m_emb; v = 0;
  while (v < V) {
    p = *chunks++; nrows = *p++;
    while (nrows--) {
      acc = row(&p);
      if (acc > top_val[TOPK - 1]) {
        for (k = TOPK - 1; k > 0 && acc > top_val[k - 1]; k--) {
          top_val[k] = top_val[k - 1]; top_tok[k] = top_tok[k - 1];
        }
        top_val[k] = acc; top_tok[k] = v;
      }
      v++;
    }
  }
  return top_tok[0];
}
