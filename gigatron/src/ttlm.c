/* TTLM: chat with the 1947 moth on a Gigatron (no CPU, only TTL chips).
 *
 * Screen layout (160x120):
 *   rows   0..70  the brain: model weights and KV cache, shown as pixels
 *   row       71  activations (they flicker while the moth thinks)
 *   rows  72..79  title bar
 *   rows  80..119 chat, 5 lines of 26 characters (the ROM Loader lives here while loading)
 *
 * Input comes from the keyboard (PS/2 via Pluggy, or the Mac via BabelFish).
 * With no input for a while, the moth runs a demo conversation by itself.
 */
#include <string.h>
#include <gigatron/console.h>
#include <gigatron/sys.h>
#include "engine.h"

#define COLS 26
#define TITLE_ROW 72
#define FGBG(fg, bg) ((fg) << 8 | (bg))   /* Gigatron colors: 6 bits, BBGGRR */
#define C_TITLE FGBG(0x00, 0x0f)          /* black on yellow */
#define C_YOU FGBG(0x3f, 0x00)            /* white on black */
#define C_MOTH FGBG(0x0f, 0x00)           /* yellow on black */
#define C_DIM FGBG(0x15, 0x00)            /* grey on black */
#define MAXQ 7            /* max words in a question */
#define IDLE_FRAMES 1800  /* 30 s without a key: demo mode */

/* The console lives in the 5 bottom text lines; it never touches the brain. */
const struct console_info_s console_info = { 5, COLS, { 160, 176, 192, 208, 224 } };

void _console_setup(void)
{
  int i, *t = (int *)videoTable;
  for (i = 8; i != 128; i++) *t++ = i;          /* identity video table */
  _console_clear((char *)((0x08 + TITLE_ROW) << 8), 0, 48); /* clear UI rows only */
  _console_clear((char *)((0x08 + TITLE_ROW) << 8), 0x0f, 8); /* title bar */
}

static char *demo[] = {
  "hi moth", "who are you", "where is the cpu", "are you an ai", "what do you eat",
  "what is money", "do you like the light", "can you code", "good night moth"
};
#define NDEMO (sizeof(demo) / sizeof(demo[0]))

static char line[40];
static unsigned char q[MAXQ + 3];

static void title(void)
{
  static char t[] = " TTLM  22k params  0 CPU  ";
  _console_printchars(C_TITLE, (char *)((0x08 + TITLE_ROW) << 8) + 2, t, COLS);
}

/* Print one word with wrapping; punctuation sticks to the previous word. */
static void say(const char *w)
{
  int n = strlen(w);
  char punct = (n == 1 && (w[0] == '.' || w[0] == ',' || w[0] == '?' || w[0] == '!'));
  if (!punct) {
    if (console_state.cx + n + 1 > COLS) console_print("\n  ", 3);
    else if (console_state.cx > 2) console_print(" ", 1);
  }
  console_print(w, n);
}

/* Vocabulary: packed "w\0w\0..." chunks (weights.c) */
static char *vword(int i)
{
  unsigned char c = 0;
  char *s;
  while (words_first[c + 1] <= i) c++;
  s = words_t[c];
  for (i -= words_first[c]; i; i--) s += strlen(s) + 1;
  return s;
}

static int lookup(const char *w, int n)
{
  unsigned char c;
  int i;
  char *s;
  for (c = 0; c < NWCHUNK; c++)
    for (s = words_t[c], i = words_first[c]; i < words_first[c + 1]; i++, s += strlen(s) + 1)
      if (!strncmp(s, w, n) && s[n] == 0) return i;
  return TOK_UNK;
}

/* Lowercase, split on spaces, keep '?' as a token, drop other punctuation. */
static int tokenize(char *s)
{
  int n = 0, k;
  char *w;
  q[n++] = TOK_Q;
  while (*s && n < MAXQ + 1) {
    while (*s == ' ') s++;
    if (!*s) break;
    if (*s == '?') { q[n++] = lookup("?", 1); s++; continue; }
    w = s;
    while (*s && *s != ' ' && *s != '?') {
      if (*s >= 'A' && *s <= 'Z') *s += 32;
      s++;
    }
    k = s - w;
    while (k && (w[k - 1] == '.' || w[k - 1] == ',' || w[k - 1] == '!')) k--;
    if (k) q[n++] = lookup(w, k);
  }
  q[n++] = TOK_A;
  return n;
}

/* Read a line; returns 0 if idle too long (then the demo takes over). */
static int readline(void)
{
  int n = 0, c;
  unsigned idle = 0;
  unsigned char f = frameCount;
  console_state.fgbg = C_YOU;
  console_print("> ", 2);
  for (;;) {
    c = console_getkey();
    if (c < 0 || c == 255) {
      if (frameCount != f) { f = frameCount; if (++idle > IDLE_FRAMES && n == 0) return 0; }
      continue;
    }
    idle = 0;
    if (c == '\n' || c == '\r') { if (n) break; continue; }
    if ((c == 8 || c == 127) && n) { n--; console_print("\b \b", 3); continue; }
    if (c >= 32 && c < 127 && n < 34 && console_state.cx < COLS - 1) {
      line[n++] = c; console_print((char *)&line[n - 1], 1);
    }
  }
  line[n] = 0;
  console_print("\n", 1);
  return n;
}

static void type_demo(const char *s)
{
  unsigned char f;
  console_state.fgbg = C_YOU;
  console_print("> ", 2);
  strcpy(line, s);
  while (*s) {
    console_print(s++, 1);
    f = frameCount; while ((unsigned char)(frameCount - f) < 6);
  }
  console_print("\n", 1);
}

static void answer(void)
{
  int n = tokenize(line), i, t;
  console_state.fgbg = C_MOTH;
  console_print("  ", 2);
  SYS_SetMode(3);                       /* thinking: show 1 line in 4, compute 10x faster */
  reset();
  for (i = 0; i < n; i++) {
    t = step(q[i], i == n - 1);
    console_state.fgbg = C_DIM;  /* one dim dot per word read */
    console_print(".", 1);
  }
  while (console_state.cx > 2) console_print("\b \b", 3);
  console_state.fgbg = C_MOTH;
  while (t != TOK_E && t_pos < CTX) {
    say(vword(t));
    t = step(t, 1);
  }
  SYS_SetMode(0);
  console_print("\n", 1);
}

int main(void)
{
  int d = 0;
  console_state.wrapy = 1;              /* scroll the chat */
  title();
  for (;;) {
    if (!readline()) {
      type_demo(demo[d]);
      if (++d == NDEMO) d = 0;
    }
    answer();
  }
  return 0;
}
