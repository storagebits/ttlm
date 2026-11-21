/* Bit-exactness test in gtsim: prompts from tests.h, greedy decoding, prints token ids. */
#include <stdio.h>
#include "engine.h"
#include "tests.h"

int main(void)
{
  int i, k, t;
  for (i = 0; i < NTESTS; i++) {
    unsigned char *pr = tests[i];
    reset();
    for (k = 0; pr[k] != 255; k++) t = step(pr[k], 1);
    printf("%d:", i);
    while (t != TOK_E && t_pos < CTX) { printf(" %d", t); t = step(t, 1); }
    printf("\n");
  }
  return 0;
}
