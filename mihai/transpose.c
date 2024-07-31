#include<immintrin.h>
#include <stdalign.h>
#include "i.h"
#include "boxfilter.h"


//#include <stdio.h>


void transpose_8x8(float * a, float * b, size_t n)
{
  float_packed v0, v1, v2, v3, v4, v5, v6, v7;
  float_packed s0, s1, s2, s3, s4, s5, s6, s7;
  

  v0 = LOAD(a[0]);
  v1 = LOAD(a[n]);
  v2 = LOAD(a[2*n]);
  v3 = LOAD(a[3*n]);
  v4 = LOAD(a[4*n]);
  v5 = LOAD(a[5*n]);
  v6 = LOAD(a[6*n]);
  v7 = LOAD(a[7*n]);
  
  s0 = _mm256_unpacklo_ps(v0, v1);
  s1 = _mm256_unpackhi_ps(v0, v1);
  s2 = _mm256_unpacklo_ps(v2, v3);
  s3 = _mm256_unpackhi_ps(v2, v3);
  s4 = _mm256_unpacklo_ps(v4, v5);
  s5 = _mm256_unpackhi_ps(v4, v5);
  s6 = _mm256_unpacklo_ps(v6, v7);
  s7 = _mm256_unpackhi_ps(v6, v7);
  
  v0 = _mm256_shuffle_ps(s0,s2,_MM_SHUFFLE(1,0,1,0));  
  v1 = _mm256_shuffle_ps(s0,s2,_MM_SHUFFLE(3,2,3,2));
  v2 = _mm256_shuffle_ps(s1,s3,_MM_SHUFFLE(1,0,1,0));
  v3 = _mm256_shuffle_ps(s1,s3,_MM_SHUFFLE(3,2,3,2));
  v4 = _mm256_shuffle_ps(s4,s6,_MM_SHUFFLE(1,0,1,0));
  v5 = _mm256_shuffle_ps(s4,s6,_MM_SHUFFLE(3,2,3,2));
  v6 = _mm256_shuffle_ps(s5,s7,_MM_SHUFFLE(1,0,1,0));
  v7 = _mm256_shuffle_ps(s5,s7,_MM_SHUFFLE(3,2,3,2));
  
  s0 = _mm256_permute2f128_ps(v0, v4, 0x20);
  s1 = _mm256_permute2f128_ps(v1, v5, 0x20);
  s2 = _mm256_permute2f128_ps(v2, v6, 0x20);
  s3 = _mm256_permute2f128_ps(v3, v7, 0x20);
  s4 = _mm256_permute2f128_ps(v0, v4, 0x31);
  s5 = _mm256_permute2f128_ps(v1, v5, 0x31);
  s6 = _mm256_permute2f128_ps(v2, v6, 0x31);
  s7 = _mm256_permute2f128_ps(v3, v7, 0x31);
  
  STORE(b[0],s0);
  STORE(b[n],s1);
  STORE(b[2*n],s2);
  STORE(b[3*n],s3);
  STORE(b[4*n],s4);
  STORE(b[5*n],s5);
  STORE(b[6*n],s6);
  STORE(b[7*n],s7);
}


void transpose(float * in, float * out, size_t n, size_t ld)
{
  /*
    For the moment works only for n divisible by V
  */
  
  size_t ni = n>>POW_BL;
  size_t rest;

  float *a0 = in;
  float *b0 = out;

  float *a;
  float *b;

  size_t k, l, i, j;


  for (k = 0; k < ni; k++)
    {
      for (l = 0; l < ni; l++)
      {
	a = a0;
	b = b0;
	for (i = 0; i<BL_V; i++)
	  {
	    for (j = 0; j<BL_V; j++)
	    {
	      transpose_8x8(a, b, ld);
       	      a += V;
	      b += ld*V;
	    }
	    a -= BL_V * V;
	    a += ld * V;
            b -= BL_V * ld * V;
	    b += V;
	  }
	    
	a0 += BL_V * V;
	b0 += ld * BL_V * V;
      }

      a0 -= ni * BL_V * V;
      a0 += ld * BL_V * V;

      b0 -= ni * ld * BL_V * V;
      b0 += BL_V * V;
    }

  rest = n - (ni << POW_BL);

  if (rest == 0) return;

  rest = rest >> POW_V;


  a0 = in  + ni * BL_V * V;
  b0 = out + ni * BL_V * V * ld;


  for (k = 0; k < ni; k++)
    {
      a = a0;
      b = b0;
      for (i = 0; i<BL_V; i++)
	{
	  for (j = 0; j<rest; j++)
	    {
	      transpose_8x8(a, b, ld);
	      a += V;
	      b += ld*V;
	    }
	  a -= rest * V;
	  a += ld * V;
	  b -= rest * ld * V;
	  b += V;
	}
      a0 += ld * BL_V * V;
      b0 += BL_V * V;
	  
    }


  a0 = in  + ni * BL_V * V * ld;
  b0 = out + ni * BL_V * V;

  for (k = 0; k < ni; k++)
    {
      a = a0;
      b = b0;
      for (i = 0; i<BL_V; i++)
	{
	  for (j = 0; j<rest; j++)
	    {
	      transpose_8x8(a, b, ld);
	      a += ld * V;
	      b += V;
	    }
	  a -= rest * ld * V;
	  a += V;
	  b -= rest * V;
	  b += ld * V;

	}
      a0 += BL_V * V;
      b0 += ld * BL_V * V;
	  
    }
 
  a = in  + BL_V * ni * V * ld + BL_V * ni * V ;
  b = out + BL_V * ni * V * ld + BL_V * ni * V ;

  for (i = 0; i<rest; i++)
    {
      for (j = 0; j<rest; j++)
	{
	  transpose_8x8(a, b, ld);
	  a += V;
	  b += ld*V;
	}
      a -= rest * V;
      a += ld * V;
      b -= rest * ld * V;
      b += V;
    }
  
		   
}

