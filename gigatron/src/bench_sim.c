/* Time per token (in 1/60 s frames) for one prompt + answer. */
#include <stdio.h>
#include "engine.h"
#include "tests.h"
#define FRAMES (*(volatile unsigned char*)0x0e)
int main(void)
{
  unsigned char f0, k; int t; unsigned tot = 0, n = 0, f;
  unsigned char *pr = tests[0];
  reset();
  for (k = 0; pr[k] != 255; k++) { f0 = FRAMES; t = step(pr[k], 1); f = (unsigned char)(FRAMES - f0); tot += f; n++; printf("%u ", f); }
  while (t != TOK_E && t_pos < CTX) { f0 = FRAMES; t = step(t, 1); f = (unsigned char)(FRAMES - f0); tot += f; n++; printf("%u ", f); }
  printf("\ntokens %u frames %u\n", n, tot);
  return 0;
}
