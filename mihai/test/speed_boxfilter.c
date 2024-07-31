#include<stdio.h>
#include<immintrin.h>
#include <stdalign.h>
#include "timing.h"
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

#include "boxfilter.h"


#define N 1024

#define R 5

#define N_TESTS 125  //Note: each test is repeated 8 times, total # tests = 8 * N_TESTS

int main()
{

  int i=0;
   
  
  double result[N_TESTS*8];

  timing start;
  timing finish;
  timing t[2];

  timing_now(&start);


  float *x_in = aligned_alloc(32, (N*N)*sizeof(float));
  
  float *x_out = aligned_alloc(32, (N*N)*sizeof(float));

  float *work = aligned_alloc(32, (N*(N+2))*sizeof(float));

  //double sum = 0;

  for (i=0; i<N*N; i++)
    {
      x_in[i] = (float) i / N;
    }


  for (i=0; i<N_TESTS*8; i++)
    {
      timing_now(&t[0]);
      boxfilter(x_in, x_out, R, N, N, work);
      timing_now(&t[1]);
      
      result[i] = timing_diff(&t[1],&t[0]);
    }
 
  
  /* printf("%d ", N); */
  /* for (i=0; i<2*8; i++) */
  /*   { */
  /*     printf(" %7.0f", result[i+(N_TESTS-2)*8]); */
  /*   } */
  /* printf("\n"); */
 

  timing_now(&finish);

  
  double mean = 0;
  double min = 1e12;
  double max = 1;

  int nb = 0;
  int nb2 = 0;
  
  for (i=0; i<N_TESTS*8; i++)
    {
      if (result[i] < 0 || result[i] > 1e7)
	{

	}
      else
	{
	  nb = nb + 1;
	  mean = mean +  result[i];
	  if (result[i] < min) min = result[i];
	  if (result[i] > max) max = result[i];
	}
    }
  mean = mean/nb;

  double std_sq = 0;
  for (i=0; i<N_TESTS*8; i++)
    {
      if (result[i] < 0 || result[i] > 1e7)
	{

	}
      else
	{
	  nb2 = nb2 + 1;
	  std_sq = std_sq + (result[i]-mean)*(result[i]-mean);
	}
    }
  
  printf("size: %d x %d, mean: %g, std: %g, min: %g, max: %g, nb: %d\n",  N, N, mean, sqrt(std_sq/(nb2-1)), min, max, nb);

  
  for (i=0; i<4*N; i++)
    {
      //printf("%f, ", x_out[i]);
    }
  //printf("\n");

  free(x_in);
  free(x_out);
  free(work);
  
}
