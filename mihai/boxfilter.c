#include<immintrin.h>
#include <stdalign.h>
#include "i.h"
#include "boxfilter.h"

//#include <cblas.h>

alignas(32) const float zero = 0;


int boxfilter(const float *x_in, float *x_out, size_t r, size_t n, size_t ld, float * work)

{

    /*
    - x_in  = pointer to 2D image n x n as 1D array with leading dimension ld
    
    - x_out = pointer to output 2D image n x n as 1D array with leading dimension ld
            = result of filter applied in 1 dimension to x_in
    
    - r     = radius of filter, made of 2*r+1 ones
    
    - n     = size of image, should be >= 2 * r + 1
    
    - ld    = leading dimension for x_in and x_out, should be divisible by V! 

    - work  = pointer to a work array ld x ld+2 of leading dimension ld divisible by 8 and greater than n

     - return values:
            0     ok
           -1     n < 2 * r + 1, image smaller than filter.
           -2     ld is not divisible by V, vector register length (for floats!)
	 
    */

  float * t;
  float * ai;
  float * bi;
  
 
  if (n < 2 * r + 1)
    {
      return -1;
    }
  if ((ld >> POW_V) << POW_V != ld)
    {
      return -2; //ld not divisible by V 
    }

  t = work;

  ai = work + ld*ld;

  bi = work + (ld+1)*ld;

  
  size_t i;
  
  for (i = 0; i < r+1; i++)
    {
      ai[i] = 1./(r+1+i)/(2*r+1);
      bi[i] = 1./(r+1+i)*(2*r+1);
    }
  
  for (i = r+1; i < n - (r+1); i++)
    {
      ai[i] = 1./(2*r+1)/(2*r+1);
    }
  
  for (i = n - (r+1); i < n; i++)
    {
      ai[i] = ai[n - 1 - i];
    }
  
  boxfilter1D(x_in, t, r, n, ld);
  
  // cblas_somatcopy(CblasRowMajor, CblasTrans, n, n, 1.0f, t, ld, x_out, ld); 

  transpose(t, x_out, n, ld); 

  boxfilter1D_norm(x_out, t, r, n, ld, ai, bi);
   
  // cblas_somatcopy(CblasRowMajor, CblasTrans, n, n, 1.0f, t, ld, x_out, ld); 
  
  transpose(t, x_out, n, ld);
  
  
  return 0;
  
}
