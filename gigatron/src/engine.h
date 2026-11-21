#ifndef ENGINE_H
#define ENGINE_H
#include "weights.h"
#define TOPK 4
extern int top_tok[TOPK], top_val[TOPK];
extern unsigned char t_pos;
void reset(void);
int step(int tok, int want);
#endif
