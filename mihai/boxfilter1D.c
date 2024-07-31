#include<immintrin.h>
#include <stdalign.h>
#include "i.h"
#include "boxfilter.h"

#include <stdio.h>

void boxfilter1D(const float *x_in, float *x_out, size_t r, size_t n, size_t ld)
{
  
  /*
    - x_in  = pointer to 2D image n x n as 1D array
    
    - x_out = pointer to output 2D image n x n as 1D array
            = result of filter applied in 1 dimension to x_in
    
    - r     = radius of filter, made of 2*r+1 ones
    
    - n     = size of image, should be >= 2 * r + 1
    
    - ld    = leading dimension for x_in and x_out, should be divisible by V! 



    NOTICE 1: Both x_in and x_out should be aligned to 32bytes for
    avx2 and 64bytes for avx512.  This can be done for example using
    aligned_alloc instead of malloc:
    
    float *x_in = aligned_alloc(32, (n*n)*sizeof(float));

    or using alignas(32) float x_in[N*N] for static arrays. 
    
    NOTICE 2: If n is not divisible by V, consider a leading dimension
    ld divisible by V and save x_in as a submatrix of size n x n in a
    matrix ld x n (column-major) or n x ld (row-major).
    

    The function works for both column-major and row-major 2D arrays
    by using index arithmetic in a 1D array.
   
    
    
    Consider the row-major case.

    NR = number of registers, each with V float values (NR = 16, V = 8
    for avx2 with 256b)
    
    Outer loop over NR*V columns
      
      Inner loop: sliding window over all lines

      Go to next NR*V columns (via a0 for x_in and b0 for x_out)

    End outer loop

    Repeat once the outer loop for the rest of columns up to next number divisible by V greater than n,
    therefore still using vector acceleration but less than NR registers. This works as leading dimension ld is 
    divisible by 8!

    Example for n = 1005, leading dimension ld = 1008, NR = 16, V = 8, NR*V = 128

    Outer loop with NR registers and V values - 7 times = 896 lines

    Outer loop with 13+1 registers and V values - once = 112 lines (max (NR-1)*V lines)
    
    Total: 1008 lines. Note that the last 3 lines are present via ld but never used in x_in and x_out.



    The inner loop over columns is divided in 4 loops - first r columns, next r+1, central part, last r columns
 
    The pointers a and a_diff for x_in and b for x_out start with the first column, current lines (via a0 and b0)
   

    Inner central loop (with j = n - 2 * r - 1):
    
        s contains NR*V values saved already in x_out.
    
        Read NR*V values from the next line starting from pointer a
    
        Add them to s

        Read NR*V values from line - (2*r-1)

        Substract them from s

        Save s to x_out, starting from pointer b
    
        Point a and b go the next column, same lines, a = a + n, b = b + n

    End central loop


  */

      
  float_packed v[NR];
  float_packed s[NR];

  const float * a0 =  x_in;
  const float * a_diff = x_in;
  float * b0 = x_out;

  const float * a;
  float * b;
  
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
	  for (i = 0; i<NR; i++)
	    {
	      v[i] = LOAD(a[i*V]);
	      s[i] = ADD(s[i], v[i]);
	      STORE(b[i*V], s[i]);
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
	      STORE(b[i*V], s[i]);
	    }
          a += ld;
	  b += ld;
	  a_diff += ld;
	};

      for (j = 0; j < r; j++)
	{
	  for (i = 0; i<NR; i++)
	    {
	      v[i] = LOAD(a_diff[i*V]);
	      s[i] = SUB(s[i], v[i]);
	      STORE(b[i*V], s[i]);
	    }
	  b += ld;
	  a_diff += ld;
	  
	}

      a0 += NR*V;
      b0 += NR*V;	
            
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
      for (i = 0; i<ni; i++)
	{
	  v[i] = LOAD(a[i*V]);
	  s[i] = ADD(s[i], v[i]);
	  STORE(b[i*V], s[i]);
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
	  STORE(b[i*V], s[i]);
	}
      a += ld;
      b += ld;
      a_diff += ld;
    }
  
  for (j = 0; j < r; j++)
    {
      for (i = 0; i<ni; i++)
	{
	  v[i] = LOAD(a_diff[i*V]);
	  s[i] = SUB(s[i], v[i]);
	  STORE(b[i*V], s[i]);
	}
      b += ld;
      a_diff += ld;
      
    }
}
