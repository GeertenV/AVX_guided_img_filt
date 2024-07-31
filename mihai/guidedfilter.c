#include<immintrin.h>
#include <stdalign.h>

#include "i.h"
#include "boxfilter.h"

int guidedfilter(const float *I, const float *p, float *q, size_t r, size_t n, size_t ld, float eps)
{
  
  if (n < 2 * r + 1)
    {
      return -1;
    }
  if ((ld >> POW_V) << POW_V != ld)
    {
      return -2; //ld not divisible by V=8
    }

  
   float * work    = aligned_alloc(32, (ld*(ld+2))*sizeof(float));
  
  float * mean_I   = aligned_alloc(32, (ld*n)*sizeof(float));
  float * mean_p   = aligned_alloc(32, (ld*n)*sizeof(float));
  float * Ip       = aligned_alloc(32, (ld*n)*sizeof(float));
  float * mean_Ip  = aligned_alloc(32, (ld*n)*sizeof(float));
  float * cov_Ip   = aligned_alloc(32, (ld*n)*sizeof(float));

  float * II       = aligned_alloc(32, (ld*n)*sizeof(float));
  float * mean_II  = aligned_alloc(32, (ld*n)*sizeof(float));
  float * var_I    = aligned_alloc(32, (ld*n)*sizeof(float));
  float * a        = aligned_alloc(32, (ld*n)*sizeof(float));
  float * b        = aligned_alloc(32, (ld*n)*sizeof(float));

  float * mean_a   = aligned_alloc(32, (ld*n)*sizeof(float));
  float * mean_b   = aligned_alloc(32, (ld*n)*sizeof(float));

  

    
  
  boxfilter(I, mean_I, r, n, ld, work);

  boxfilter(p, mean_p, r, n, ld, work);
  
  
  matmul(I, p, Ip, n, ld);

  boxfilter(Ip, mean_Ip, r, n, ld, work);
  

  diffmatmul(mean_Ip, mean_I, mean_p, cov_Ip, n, ld);

  //  II = Ip;
  
  matmul(I, I, II, n, ld);

  boxfilter(II, mean_II, r, n, ld, work);
  
  diffmatmul(mean_II, mean_I, mean_I, var_I, n, ld);

  
  matdivconst(cov_Ip, var_I, a, n, ld, eps);

  diffmatmul(mean_p, a, mean_I, b, n, ld);

  boxfilter(a, mean_a, r, n, ld, work);

  boxfilter(b, mean_b, r, n, ld, work);

  addmatmul(mean_b, mean_a, I, q, n, ld);
  

  free(work);
  free(mean_I);
  free(mean_p);
  free(Ip);
  free(mean_Ip);
  free(cov_Ip);
  free(mean_II);
  free(var_I);
  free(a);
  free(b);
  free(mean_a);
  free(mean_b);
  free(II);
  
  
  return 0;

  
}
