#include <stdlib.h>
#include<assert.h>
#include <stdalign.h>
#include<stdio.h>

#include "boxfilter.h"


int main()
{

  const _Alignas(32) float cat[] = {
#include "cat.csv"			
  };
  
  static_assert(sizeof(cat)/sizeof(float) == 73984, "Wrong initializer for cat");
  
  
 int r = 4;

 int n = 272;

 int ld = n;

 float eps = 0.2*0.2;

 const float * I = cat;
 const float * p = cat;
 
 float *q = aligned_alloc(32, (ld * n)*sizeof(float));


  
  guidedfilter(I, p, q, r, n, ld, eps);
 

  int i, j; 
 for (i=0; i<n; i++)
    {
      for (j = 0; j<n; j++)
	{
	  printf("%f, ", q[i*ld + j]);
	}
      printf("\n");
    }
 

 free(q);
 

}
