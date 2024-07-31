#include<immintrin.h>
#include <stdalign.h>
#include "i.h"
#include "boxfilter.h"

void matmul(const float *x1, const float *x2, float *y, size_t n, size_t ld)
{

  size_t i, ld_red;

  float_packed v1, v2, vy;

  ld_red = ld >> POW_V;

  const float *a1 = x1;
  const float *a2 = x2;
  float *b  = y;

  i = n * ld_red;
  do
    {
      v1 = LOAD(a1[0]);
      v2 = LOAD(a2[0]);
      vy = MUL(v1, v2);
      STORE(b[0], vy);
      a1 += V;
      a2 += V;
      b  += V;
    } while (i -= 1);
}



void diffmatmul(const float *x1, const float *x2, const float * x3, float *y, size_t n, size_t ld)
{

  size_t i, ld_red;

  float_packed v1, v2, v3, vy;

  ld_red = ld >> POW_V;

  const float *a1 = x1;
  const float *a2 = x2;
  const float *a3 = x3;
  float *b  = y;

  i = n * ld_red;
  do
    {

      v2 = LOAD(a2[0]);
      v3 = LOAD(a3[0]);
      vy = MUL(v2, v3);
      v1 = LOAD(a1[0]);
      vy = SUB(v1, vy);
      STORE(b[0], vy);
      a1 += V;
      a2 += V;
      a3 += V;
      b  += V;
    } while (i -= 1);
}


void addmatmul(const float *x1, const float *x2, const float * x3, float *y, size_t n, size_t ld)
{

  size_t i, ld_red;

  float_packed v1, v2, v3, vy;

  ld_red = ld >> POW_V;

  const float *a1 = x1;
  const float *a2 = x2;
  const float *a3 = x3;
  float *b  = y;

  i = n * ld_red;
  do
    {

      v2 = LOAD(a2[0]);
      v3 = LOAD(a3[0]);
      vy = MUL(v2, v3);
      v1 = LOAD(a1[0]);
      vy = ADD(v1, vy);
      STORE(b[0], vy);
      a1 += V;
      a2 += V;
      a3 += V;
      b  += V;
    } while (i -= 1);
}



void matdivconst(const float *x1, const float *x2, float *y, size_t n, size_t ld, float e)
{

  size_t i, ld_red;

  float_packed v1, v2, vy, ve;


  ve = BROADCAST(e);
  
  ld_red = ld >> POW_V;

  const float *a1 = x1;
  const float *a2 = x2;
  float *b  = y;

  i = n * ld_red;
  do
    {
      v1 = LOAD(a1[0]);
      v2 = LOAD(a2[0]);
      v2 = ADD(v2, ve);
      vy = DIV(v1, v2);
      STORE(b[0], vy);
      a1 += V;
      a2 += V;
      b  += V;
    } while (i -= 1);
}

