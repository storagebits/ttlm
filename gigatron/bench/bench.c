/* TTLM speed bench: how fast can the Gigatron do ternary dot products? */
#include <stdio.h>
#define FRAMES (*(volatile unsigned char*)0x0e)
#define N 32
signed char x[N];
unsigned char idx[N];          /* index list variant */
unsigned char packed[N/4];     /* 4 ternary weights per byte, base-3 code 0..80 */
int lut[N/4][81];              /* per-chunk partial sums */

static unsigned elapsed(unsigned char t0){ return (unsigned char)(FRAMES - t0); }

int main(void){
  int i,r,c,acc; unsigned char t0; unsigned f; long ops;
  for(i=0;i<N;i++){ x[i]=i-16; idx[i]=(i*7)&31; }
  for(i=0;i<N/4;i++) packed[i]=(i*13)%81;

  /* A: index-list add (one add per nonzero weight) */
  t0=FRAMES; acc=0;
  for(r=0;r<64;r++) for(i=0;i<N;i++) acc+=x[idx[i]];
  f=elapsed(t0); printf("A idx-add   : 2048 adds in %u frames (acc %d)\n",f,acc);

  /* B: int8 multiply-accumulate (software multiply on v4) */
  t0=FRAMES; acc=0;
  for(r=0;r<16;r++) for(i=0;i<N;i++) acc+=x[i]*(signed char)idx[i];
  f=elapsed(t0); printf("B int8 MAC  : 512 macs in %u frames (acc %d)\n",f,acc);

  /* C: LUT build (81 sums per chunk of 4 inputs) + lookups */
  t0=FRAMES;
  for(c=0;c<N/4;c++){
    int *L=lut[c]; signed char a=x[4*c],b=x[4*c+1],d=x[4*c+2],e=x[4*c+3];
    int k=0,w0,w1,w2,w3;
    for(w3=-1;w3<=1;w3++) for(w2=-1;w2<=1;w2++) for(w1=-1;w1<=1;w1++){
      int s=(w3<0?-e:w3?e:0)+(w2<0?-d:w2?d:0)+(w1<0?-b:w1?b:0);
      L[k++]=s-a; L[k++]=s; L[k++]=s+a;
    }
  }
  f=elapsed(t0); printf("C lut build : %d chunks in %u frames\n",N/4,f);
  t0=FRAMES; acc=0;
  for(r=0;r<256;r++) for(c=0;c<N/4;c++) acc+=lut[c][packed[c]];
  f=elapsed(t0); printf("C lut rows  : 256 rows x32 in %u frames (acc %d)\n",f,acc);
  return 0;
}
