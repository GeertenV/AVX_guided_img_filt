#include<immintrin.h>
#include <stdalign.h>
#include "i.h"
#include "boxfilter.h"

//#include <stdio.h>


void boxfilter1D_norm(const float *x_in, float *x_out, size_t r, size_t n, size_t ld, const float * a_norm, const float * b_norm)
{
  
  /*
    - x_in  = pointer to 2D image n x n as 1D array
    
    - x_out = pointer to output 2D image n x n as 1D array
            = result of filter applied in 1 dimension to x_in 
	      AND final normalization by multiplication with matrix 1./N 
    
    - r     = radius of filter, made of 2*r+1 ones
    
    - n     = size of image, should be >= 2 * r + 1
    
    - ld    = leading dimension for x_in and x_out, should be divisible by V! 

    - a_norm = pointer to a vector of length ld
    
    - b_norm = pointer to a vector of length r+1

    Note: a_norm @ b_norm = 1./N, where @ is the tensor product and N
    is the normalization matrix.

    b_norm(k) = 1 for k > r && k < n - r 

    These values for b are not saved, as multiplication with 1 can be
    omitted!

    See further comments in boxfilter1D.c

  */

      
  float_packed v[NR];
  float_packed s[NR];
  float_packed va, vb;
 
  const float * a0 =  x_in;
  const float * a_diff = x_in;
  float * b0 = x_out;

  const float * a;
  float * b;

  const float * ai =  a_norm;
  
  size_t i, j, k, rest, ni;

  //Loop using NR registers with V values, with POW_T = log2(V*NR) 
  
  ni =  n >> POW_T; // ni = n/V/NR;
  
  //  k = ni;

  // printf("%ld\n", ni);
  
  for (k = 0; k<ni; k++)
    {
      a = a0;
      b = b0;
      a_diff = a0;
     
      s[0] = BROADCAST(zero);
 
      for (i = 1; i<NR; i++)
	{
	  s[i] = s[0];
	}
      
      for (j = 0; j < r; j++)
	{
	  for (i = 0; i<NR; i++)
	    {
	      v[i] = LOAD(a[i*V]);
	      s[i] = ADD(s[i], v[i]);
	    }
          a += ld;	  
	} 
      
      for (j = 0; j < r + 1; j++)
	{
	  vb = BROADCAST(b_norm[j]);
	  for (i = 0; i<NR; i++)
	    {
	      v[i] = LOAD(a[i*V]);
	      s[i] = ADD(s[i], v[i]);
	      va   = LOAD(ai[i*V]);
	      v[i] = MUL(s[i], va);
	      v[i] = MUL(v[i], vb);
	      STORE(b[i*V], v[i]);
	    }
          a += ld;
	  b += ld;
	} 

      for (j = 0; j < n - 2 * r - 1; j++)
	{
	  for (i = 0; i<NR; i++)
	    {
	      v[i] = LOAD(a[i*V]);
	      s[i] = ADD(s[i], v[i]);
	      v[i] = LOAD(a_diff[i*V]);
	      s[i] = SUB(s[i], v[i]);
	      va   = LOAD(ai[i*V]);
	      v[i] = MUL(s[i], va);
	      STORE(b[i*V], v[i]);
	    }
          a += ld;
	  b += ld;
	  a_diff += ld;
	};

      for (j = 0; j < r; j++)
	{
	  vb = BROADCAST(b_norm[r-1-j]);
	  for (i = 0; i<NR; i++)
	    {
	      v[i] = LOAD(a_diff[i*V]);
	      s[i] = SUB(s[i], v[i]);
	      va   = LOAD(ai[i*V]);
	      v[i] = MUL(s[i], va);
	      v[i] = MUL(v[i], vb);
	      STORE(b[i*V], v[i]);
	    }
	  b += ld;
	  a_diff += ld;
	  
	}

      a0 += NR*V;
      b0 += NR*V;
      ai += NR*V;
            
    }

  
  rest  = n - (ni<<POW_T);

  if (rest == 0) return;

  ni = rest >> POW_V;  //ni = rest/V

  // printf("%ld\n", ni);

  if (rest - (ni << POW_V) > 0)  ni = ni + 1; //based on leading dimension being divisible by V!
  
  
  a = a0;
  b = b0;
  a_diff = a0;
  
  s[0] = BROADCAST(zero);
  for (i = 1; i<ni; i++)
    {
      s[i] = s[0];
    }

  for (j = 0; j < r; j++)
    {
      for (i = 0; i<ni; i++)
	{
	  v[i] = LOAD(a[i*V]);
	  s[i] = ADD(s[i], v[i]);
	    }
      a += ld;	  
    }
  
  for (j = 0; j < r + 1; j++)
    {
      vb = BROADCAST(b_norm[j]);
      for (i = 0; i<ni; i++)
	{
	  v[i] = LOAD(a[i*V]);
	  s[i] = ADD(s[i], v[i]);
	  va   = LOAD(ai[i*V]);
	  v[i] = MUL(s[i], va);
	  v[i] = MUL(v[i], vb);
	  STORE(b[i*V], v[i]);
	}
      a += ld;
      b += ld;
    } while (j -= 1);
  
  for (j = 0; j < n - 2 * r - 1; j++)
    {
      for (i = 0; i<ni; i++)
	{
	  v[i] = LOAD(a[i*V]);
	  s[i] = ADD(s[i], v[i]);
	  v[i] = LOAD(a_diff[i*V]);
	  s[i] = SUB(s[i], v[i]);
	  va   = LOAD(ai[i*V]);
	  v[i] = MUL(s[i], va);
	  STORE(b[i*V], v[i]);
	}
      a += ld;
      b += ld;
      a_diff += ld;
    }
  
  for (j = 0; j < r; j++)
    {
      vb = BROADCAST(b_norm[r-1-j]);
      for (i = 0; i<ni; i++)
	{
	  v[i] = LOAD(a_diff[i*V]);
	  s[i] = SUB(s[i], v[i]);
	  va   = LOAD(ai[i*V]);
	  v[i] = MUL(s[i], va);
	  v[i] = MUL(v[i], vb);
	  STORE(b[i*V], v[i]);
	}
      b += ld;
      a_diff += ld;
      
    }

  return;
}
